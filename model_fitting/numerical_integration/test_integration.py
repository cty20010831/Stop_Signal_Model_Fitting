import gsl_integration

# Set integration limits
a, b = 0, 10

# Perform the integration
result = gsl_integration.integrate_with_gsl(a, b)
print("Integral result: {}".format(result))