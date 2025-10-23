'''
This setup script is used to compile the Cython extension for the stop_likelihoods_wtf module.
'''

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="stop_likelihoods_wtf",
        sources=["stop_likelihoods_wtf.pyx"],
        include_dirs=[
            numpy.get_include(),
            "/opt/homebrew/include",  # <- GSL headers path for Mac
        ],
        library_dirs=["/opt/homebrew/lib"],  # <- GSL lib path
        libraries=["gsl", "gslcblas", "m"],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="stop_likelihoods_wtf",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)

# Example usage:
# python setup.py build_ext --inplace