# gsl_integration.pyx
import ctypes
from libc.math cimport exp

# Load the GSL shared library
gsl = ctypes.CDLL("/usr/local/lib/libgsl.dylib") # Use "libgsl.dylib" on macOS; Use "libgsl.so" on Windows

# Define function type for integrand
FUNC = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_void_p)

class GSLFunction(ctypes.Structure):
    _fields_ = [("function", FUNC), ("params", ctypes.c_void_p)]

def integrand(x, params):
    return 1.0 / (x * x + 1.005)

def integrate_with_gsl(a, b, tol=1e-7):
    workspace = gsl.gsl_integration_workspace_alloc(1000)
    result = ctypes.c_double()
    error = ctypes.c_double()

    gsl_func = FUNC(integrand)
    func = GSLFunction(function=gsl_func, params=None)

    gsl.gsl_integration_qag.argtypes = [
        ctypes.POINTER(GSLFunction), ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    gsl.gsl_integration_qag.restype = ctypes.c_int

    status = gsl.gsl_integration_qag(
        ctypes.byref(func), a, b, tol, tol, 1000, 6, 
        workspace, ctypes.byref(result), ctypes.byref(error)
    )

    if status != 0:
        raise RuntimeError(f"GSL integration failed with status code {status}")

    gsl.gsl_integration_workspace_free(workspace)
    return result.value