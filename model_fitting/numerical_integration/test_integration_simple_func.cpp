#include <iostream>
#include <gsl/gsl_integration.h>

double integrand(double x, void* params) {
    return 1.0 / (x * x + 1.0);
}

int main() {
    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1000);
    double result, error;

    gsl_function F;
    F.function = &integrand;
    F.params = nullptr;

    // Adaptive general-purpose integration routines
    gsl_integration_qag(&F, 0, 1, 0, 1e-7, 1000, 4, workspace, &result, &error);

    std::cout << "Integral result: " << result << std::endl;

    gsl_integration_workspace_free(workspace);
    return 0;
}

// Example Usage
/// Compile: g++ test_integration_simple_func.cpp -o gsl_integration_qag_simple_func -I/opt/homebrew/Cellar/gsl/2.8/include -L /opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas -lm
/// Run: ./gsl_integration_qag_simple_func