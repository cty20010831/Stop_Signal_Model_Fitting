# Load the SimSST package
library(SimSST)

library(truncnorm)

# Simulation settings
n_participants <- 100
blocks_per_participant <- 4
trials_per_block <- 64
stops_per_block <- trials_per_block / 4
start_ssd <- 200
total_rows <- n_participants * blocks_per_participant

# Create vectors for inputs
pid <- rep(paste0(1:n_participants), each = blocks_per_participant)
block <- rep(1:blocks_per_participant, times = n_participants)
n <- rep(trials_per_block, n_participants * blocks_per_participant)
m <- rep(stops_per_block, n_participants * blocks_per_participant)
SSD.b <- rep(start_ssd, n_participants * blocks_per_participant)
dist.go <- rep("ExG", n_participants * blocks_per_participant)
dist.stop <- rep("ExG", n_participants * blocks_per_participant)

# Sample parameters 
set.seed(0)

# === Sample group-level hyperparameters ===
set.seed(1)

# Group-level means
mu_mu_go       <- runif(1, 300, 600)
sigma_mu_go    <- runif(1, 10, 40)
mu_sigma_go    <- runif(1, 30, 80)
sigma_sigma_go <- runif(1, 5, 20)
mu_tau_go      <- runif(1, 50, 150)
sigma_tau_go   <- runif(1, 5, 20)

mu_mu_stop       <- runif(1, 150, 300)
sigma_mu_stop    <- runif(1, 10, 30)
mu_sigma_stop    <- runif(1, 30, 80)
sigma_sigma_stop <- runif(1, 5, 20)
mu_tau_stop      <- runif(1, 30, 100)
sigma_tau_stop   <- runif(1, 5, 20)

# Subject-level parameters (truncated normal)
mu_go    <- rtruncnorm(n_participants, mean = mu_mu_go, sd = sigma_mu_go, lower = 0.001, upper = 1000)
sigma_go <- rtruncnorm(n_participants, mean = mu_sigma_go, sd = sigma_sigma_go, lower = 1, upper = 500)
tau_go   <- rtruncnorm(n_participants, mean = mu_tau_go, sd = sigma_tau_go, lower = 1, upper = 500)

mu_stop    <- rtruncnorm(n_participants, mean = mu_mu_stop, sd = sigma_mu_stop, lower = 0.001, upper = 600)
sigma_stop <- rtruncnorm(n_participants, mean = mu_sigma_stop, sd = sigma_sigma_stop, lower = 1, upper = 350)
tau_stop   <- rtruncnorm(n_participants, mean = mu_tau_stop, sd = sigma_tau_stop, lower = 1, upper = 350)

# === Sample subject-level parameters ===
mu_go    <- rtruncnorm(n_participants, mean = mu_mu_go, sd = sigma_mu_go, lower = 0.001, upper = 1000)
sigma_go <- rtruncnorm(n_participants, mean = mu_sigma_go, sd = sigma_sigma_go, lower = 1, upper = 500)
tau_go   <- rtruncnorm(n_participants, mean = mu_tau_go, sd = sigma_tau_go, lower = 1, upper = 500)

mu_stop    <- rtruncnorm(n_participants, mean = mu_mu_stop, sd = sigma_mu_stop, lower = 0.001, upper = 600)
sigma_stop <- rtruncnorm(n_participants, mean = mu_sigma_stop, sd = sigma_sigma_stop, lower = 1, upper = 350)
tau_stop   <- rtruncnorm(n_participants, mean = mu_tau_stop, sd = sigma_tau_stop, lower = 1, upper = 350)

theta.go <- cbind(
  rep(mu_go, each = blocks_per_participant),
  rep(sigma_go, each = blocks_per_participant),
  rep(tau_go, each = blocks_per_participant)
)
theta.stop <- cbind(
  rep(mu_stop, each = blocks_per_participant),
  rep(sigma_stop, each = blocks_per_participant),
  rep(tau_stop, each = blocks_per_participant)
)

# ----- Save subject-level parameters -----
param_df <- data.frame(
  participant_id = 1:n_participants,
  mu_go = mu_go,
  sigma_go = sigma_go,
  tau_go = tau_go,
  mu_stop = mu_stop,
  sigma_stop = sigma_stop,
  tau_stop = tau_stop
)
write.csv(param_df, "SST_simulated_parameters.csv", row.names = FALSE)

# ----- Simulate data -----
SST_data <- as.data.frame(
  simsstrack(
    pid = pid,
    block = block,
    n = n,
    m = m,
    SSD.b = SSD.b,
    dist.go = dist.go,
    theta.go = theta.go,
    dist.stop = dist.stop,
    theta.stop = theta.stop
  )
)

# ----- Convert the format to original pymc code expects -----
# Extract and rename relevant columns
pymc_df <- SST_data[, c("Participant.id", "Trial", "Inhibition", "GORT", "SSRT", "SRRT", "SSD")]
colnames(pymc_df) <- c("participant_id", "trial_type", "inhibited", "go_rt", "ss_rt", "sr_rt", "ssd")

# Change participant_id to zero-index based
pymc_df$participant_id <- as.numeric(pymc_df$participant_id) - 1

# Recode trial_type to 'go' or 'stop'
pymc_df$trial_type <- ifelse(pymc_df$trial_type == "Go", "go", "stop")

# Determine ssrt
pymc_df$ss_rt[pymc_df$ss_rt == -999] <- NA # Replace -999 with NA

# Determine the final observed RT
pymc_df$observed_rt <- pymc_df$go_rt
pymc_df$observed_rt[pymc_df$trial_type == "stop" & pymc_df$inhibited == 0] <- pymc_df$sr_rt
pymc_df$observed_rt[pymc_df$observed_rt < 0] <- NA  # Replace -999 with NA

# Determine outcome
pymc_df$outcome <- ifelse(pymc_df$trial_type == "go", "go",
                     ifelse(pymc_df$inhibited == 1, "successful inhibition", "stop-respond"))

# Clean and reorder
pymc_df <- pymc_df[, c("trial_type", "ssd", "observed_rt", "ss_rt", "outcome", "participant_id")]
pymc_df$ssd[pymc_df$ssd < 0] <- NA  # replace -999 SSD with NA
pymc_df$ssd <- as.numeric(format(pymc_df$ssd, nsmall = 1))  # force .0
pymc_df$observed_rt <- as.numeric(format(pymc_df$observed_rt, nsmall = 1))  # force .0

# Write to CSV
write.csv(pymc_df, "test_staircase_pymc.csv", row.names = FALSE, quote = FALSE)

# ----- Convert the format to BEESTS expects -----
Datatemp <- SST_data 
subj_idx <- as.numeric(Datatemp[, 1])
ss_presented <- recode(Datatemp[,3], 'Stop' = 1, 'Go' = 0)
inhibited <- as.numeric(Datatemp[,4])
ssd <- as.numeric(Datatemp[,8])
rt <- as.numeric(Datatemp[,5])
srrt <- as.numeric(Datatemp[,7])
Data <- cbind.data.frame(subj_idx, ss_presented, inhibited, ssd, rt, srrt)
Data$rt[Data$inhibited == 0] <- Data$srrt[Data$inhibited == 0] 
myBEESTSdata <- (Data[,-6])

# Remove out trials with ssd equal to zero?
# myBEESTSdata <- myBEESTSdata[as.numeric(myBEESTSdata$ssd) > 0, ]

# Save the (full) data
write.csv(myBEESTSdata, file = "test_staircase_SimSST.csv", row.names = FALSE, quote = FALSE)

# Save the (small-sample) data
myBEESTSdata_small <- myBEESTSdata[myBEESTSdata$subj_idx <= 5,]
write.csv(myBEESTSdata_small, file = "test_staircase_SimSST_n_5.csv", row.names = FALSE, quote = FALSE)