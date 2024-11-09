#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <vector>
#include <limits>

// Helper constants for NaN/Inf detection
const double NEG_INF = -std::numeric_limits<double>::infinity();
const double POS_INF = std::numeric_limits<double>::infinity();

// Ex-Gaussian PDF function
double ExGauss_pdf(double value, double mu, double sigma, double tau) {
    if (tau > 0.05 * sigma) {
        double z = value - mu - (sigma * sigma) / tau;
        return -std::log(tau) - (z + (sigma * sigma) / (2 * tau)) / tau
               + std::log(gsl_cdf_gaussian_P(z / sigma, 1.0));
    } else {
        return std::log(gsl_ran_gaussian_pdf(value - mu, sigma));
    }
}

// Ex-Gaussian CDF function
double ExGauss_cdf(double value, double mu, double sigma, double tau) {
    if (tau > 0.05 * sigma) {
        double z = value - mu - (sigma * sigma) / tau;
        double part1 = gsl_cdf_gaussian_P((value - mu) / sigma, 1.0);
        double part2 = gsl_cdf_gaussian_P(z / sigma, 1.0) 
                       * std::exp(((mu + (sigma * sigma) / tau) * (mu + (sigma * sigma) / tau) - 
                                  mu * mu - 2 * value * (sigma * sigma) / tau) / 
                                  (2 * sigma * sigma));
        return std::log(1.0 - (part1 - part2));
    } else {
        return std::log(1.0 - gsl_cdf_gaussian_P((value - mu) / sigma, 1.0));
    }
}

// Function to evaluate integrand for GSL
double eval_cexgauss(double x, void* params) {
    double* p = static_cast<double*>(params);
    double imu_go = p[0], isigma_go = p[1], itau_go = p[2];
    double imu_stop = p[3], isigma_stop = p[4], itau_stop = p[5];
    double ip_tf = p[6], issd = p[7];

    double pdf_value = std::exp(ExGauss_pdf(x, imu_go, isigma_go, itau_go));
    double cdf_value = std::exp(ExGauss_cdf(x - issd, imu_stop, isigma_stop, itau_stop));

    return pdf_value * (1 - cdf_value);
    // return pdf_value * cdf_value * (1.0 - ip_tf);
}

// GSL-based integration function
double integrate_cexgauss(double lower, double upper, 
                          double imu_go, double isigma_go, double itau_go, 
                          double imu_stop, double isigma_stop, double itau_stop, 
                          double ip_tf, int issd) {
    gsl_integration_workspace* W = gsl_integration_workspace_alloc(5000);

    double params[8] = {imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ip_tf, static_cast<double>(issd)};
    gsl_function F;
    F.function = &eval_cexgauss;
    F.params = &params;

    double result, error;
    gsl_integration_qag(&F, lower, upper, 1e-4, 1e-4, 5000, GSL_INTEG_GAUSS41, W, &result, &error);
    gsl_integration_workspace_free(W);

    return result;
}

// Ex-Gaussian Go log-likelihood function
double Go(const std::vector<double>& values, double imu_go, double isigma_go, double itau_go) {
    double sum_logp = 0.0;
    for (double value : values) {
        double p = ExGauss_pdf(value, imu_go, isigma_go, itau_go);
        if (std::isinf(p) || std::isnan(p)) return NEG_INF;
        sum_logp += p;
    }
    return sum_logp;
}

// SRRT log-likelihood function
double SRRT(const std::vector<double>& values, const std::vector<int>& issd, 
            double imu_go, double isigma_go, double itau_go, 
            double imu_stop, double isigma_stop, double itau_stop, 
            double ip_tf) {
    double sum_logp = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        double p1 = std::exp(ExGauss_pdf(values[i], imu_go, isigma_go, itau_go)) * ip_tf;
        double p2 = std::exp(ExGauss_pdf(values[i], imu_go, isigma_go, itau_go)) * 
                    std::exp(ExGauss_cdf(values[i] - issd[i], imu_stop, isigma_stop, itau_stop)) * 
                    (1.0 - ip_tf);
        double p = std::log(p1 + p2);
        if (std::isinf(p) || std::isnan(p)) return NEG_INF;
        sum_logp += p;
    }
    return sum_logp;
}

// Main function for testing
int main() {
    // Sample parameters for testing
    double imu_go = 1000, isigma_go = 50, itau_go = 100;
    double imu_stop = 1000, isigma_stop = 30, itau_stop = 80;
    double ip_tf = 0.5;
    int issd = 0;

    // Test integration
    double integral_result = integrate_cexgauss(0, 10000, imu_go, isigma_go, itau_go, 
                                                imu_stop, isigma_stop, itau_stop, 
                                                ip_tf, issd);
    std::cout << "Integral Result: " << integral_result << std::endl;

    // Test Go log-likelihood
    // std::vector<double> go_values = {350, 400, 450};
    std::vector<double> go_values;
    go_values.push_back(350);
    go_values.push_back(400);
    go_values.push_back(450);
    double go_log_likelihood = Go(go_values, imu_go, isigma_go, itau_go);
    std::cout << "Go Log-Likelihood: " << go_log_likelihood << std::endl;

    // Test SRRT log-likelihood
    // std::vector<int> ssd_values = {250, 300, 350};
    std::vector<int> ssd_values;
    ssd_values.push_back(250);
    ssd_values.push_back(300);
    ssd_values.push_back(350);
    double srrt_log_likelihood = SRRT(go_values, ssd_values, imu_go, isigma_go, itau_go, 
                                      imu_stop, isigma_stop, itau_stop, ip_tf);
    std::cout << "SRRT Log-Likelihood: " << srrt_log_likelihood << std::endl;

    return 0;
}

// Example Usage
/// Compile: g++ test_integration_successful_inhibition.cpp -o gsl_integration_successful_inhibition -I/opt/homebrew/Cellar/gsl/2.8/include -L /opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas -lm
/// Run: ./gsl_integration_successful_inhibition