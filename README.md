# Stop_Signal_Model_Fitting

This repository serves as the testing base for implementing the [**stop signal respond task**](https://cambridgecognition.com/stop-signal-task-sst/) in the [`pymc`](https://www.pymc.io/welcome.html) ecosystem, before incorporating into the [`hssm`](https://lnccbrown.github.io/HSSM/) architecture (also built upon `pymc`).

This project is in collaboration with Michael J. Frank and Alex Fengler from Brown University.

## Requirements
It is recommended to run the code on the local machine by setting up a virtual environment (e.g., [venv](https://docs.python.org/3/library/venv.html) or [conda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)). For example, you can create the virtual `venv`
environment named **venv**:
```bash
# Create the virtual environment:
python3.11 -m venv venv

# Activate the virtual environment:
source venv/bin/activate
```

Then, download the required packages:
```bash
pip install -r requirements.txt
```

The [requirements.txt](requirements.txt) is generated using: 
```bash
pip freeze
```

## Stop Signal Task
As a classifical experimental paradigm to measure impulse control, the experimental design of the stop signal task (for model fitting) follows the one proposed by Frederick Verbruggen (charactrized by 25% of stop trials and staircase ssd). For more details, please refer to [his jspsych implementation of this task](https://github.com/fredvbrug/STOP-IT/tree/master/jsPsych_version). For the simulation (and relatedly, posterior predictive checks), we follow the his experimental 
settings. It is a staircase design for SSD adjustment, where the SSD increases by 50 ms after a successful inhibition trial and decreases by 50 ms after a failed inhibition trial. It also contains 25% of stop trials (trial sequence randomly assigned). In terms of the starting SSDs, they are set to be a fixed value of 200 in the jsPsych version. That said, these settings (except for staircase, as opposed to fixed SSDs) can be flexibly modified in the [simSST.py](simulation/simSST.py) file.

## (Top-Level) Repository Structure
    .
    ├── BEESTS/               # a GUI software for model fitting on stop signal task
    ├── data_for_paper/       # real data for model fitting
    ├── model_fitting/        # hierarchical and individual level model fitting
    ├── progress_report/      # progress reports for this project
    ├── simulation/           # (forward) simulator of stop signal test
    ├── .gitignore
    ├── README.md
    └── requirementd.txt   

### `BEESTS` directory
BEESTS (use this [link](https://osf.io/482fv/) to download the software) is a GUI-based package enabling Bayesian hierarchical estimation of response time models for the stop signal task ([Matzke et al., 2013](https://doi.org/10.3389/fpsyg.2013.00918)). In this repository, it mainly serves to perform sanity check (also help debugging) for our own pymc model. 

In order to run it smoothly on your local machine after downloading the software, it is important to check (oftentimes overwrite) the currernt permission to run this software. For instance, for mac users:
```bash
# Allow software downloading from anywhere (replace `disable` into `enable` after if you do not want to always allow your computer to download anything)
sudo spctl --master-disable

# Grant permission
sudo chmod -R 755 <path to software>
```

We also included [convert_format.py](BEESTS/convert_format.py) file that helps convert data format from the one our pymc model expects to the one BEESTS expects. 

#### Fitting data using BEESTS
One thing we did is to fit data (simulated or real) using BEESTS, which not only serves as a sanity check to validate our model fitting results using pymc, but also helps us potential issues with integration while defining the likelihood in sucessful inhibition trials. 

After converting the format to the one BEESTS expects, open **File** section on the top left to upload the data the BEESTS software, and then click **Run** under the **Analysis** section. It will create a directory at the same level of where the data is located, including what analysis is performed under the hood (under `analysis.txt`), deviance statistics and estimated parameter values (`.csv` files), and database (`.db` files). 
<img src="BEESTS/screeshots/BEESTS_analysis.png">

One thing to notice is that under the **Advanced** section, users can specify whether they want to include trigger failure in the model, alongside prior settings for all parameters and upper and lower bounds of integration.
<img src="BEESTS/screeshots/BEESTS_advanced.png">

Due to the mismatch of `libRblas.dylib` file with my mac operation system (silicon vs. intel-based mac), I adapted the [analysis script](BEESTS/analysis.R), which should originally be called to run after clicking **Run** under the **Analysis** section. Users can now run:
```bash
cd BEESTS

Rscript analysis.R <path_to_analysisDir> <summary_statistics> <posterior_distributions> <mcmc_chains> <posterior_predictors>
# E.g., Rscript analysis.R real_data/real_data.csv_241016-090124 TRUE TRUE TRUE TRUE

cd .
```
The results will be saved in `output.pdf file` under the <path_to_analysisDir>. In addition, I added a [function](BEESTS/compute_deviance.py) to compute deviance (from deviance.csv fules) averaged across all chains (for simple model comparison):
```bash
python BEESTS/compute_deviance.py --data <for test data or real data> --data_analysis_name <Name of the directory storing data analysis (model fitting) results>
# E.g., python BEESTS/compute_deviance.py --data real --data_analysis_name real_data.csv_241017-220245
```

```bash
python BEESTS/generate_sub_param.py --data <for test data or real data> --data_file_name <Name of the data file name (used for model fitting)> --data_analysis_name <Name of the directory storing data analysis (model fitting) results> --with_trigger_failure <whether including trigger failure>
# E.g., python BEESTS/generate_sub_param.py --data real --data_file_name real_data.csv --data_analysis_name real_data.csv_241017-220245 --with_trigger_failure True
```

### `model_fitting` directory
This is the main directory to test our model fitting results on real data. It includes [util.py](model_fitting/util.py) (utility functions used in model fitting notebooks, including likelihood defintion for different trial types and posterior predictive checks).

Right now, there are three main versions: 1) traditional pymc implementation, 2) Cython-wrapped likelihood implementation (mimicing the cython likelihood definition in BEESTS), and 3) JAX-based implementation without gradient support (for now).

#### Archive_Model_Fitting
This directory includes archived model fitting notebooks and utility functions that are no longer actively maintained.

#### Numerical Integration
To ensure that our numerical integration using pre-computed Legendre quadrature matches the result using [gsl](https://www.gnu.org/software/gsl/) (specifically, `gsl_integration_qag` used under the hood of BEESTS, which is QAG adaptive integration), we compiled the `.cpp` code and then compare the results from `.cpp` files (including instructions on how to compile and run the code inside) and results in [numerical_integration.ipynb](model_fitting/numerical_integration/numerical_integration.ipynb). Overall, the two groups of results match. 

### `simulation` directory
1. [archive/](simulation/archive/): archived simulation scripts
2. [simSST.py](simulation/simSST.py): main forward simulator of stop signal task
3. [simulation_psuedo_code.tt](simulation/simulation_pseudo_code.txt): pseudo code of the forward simulator
