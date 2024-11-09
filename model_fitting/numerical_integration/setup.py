# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# tar -xvzf gsl-2.7.tar.gz
# cd gsl-2.7
# ./configure --prefix=/usr/local
# make
# sudo make install

# Use the correct paths based on your gsl-config output
ext = Extension(
    "gsl_integration",
    sources=["gsl_integration.pyx"],
    libraries=["gsl", "gslcblas", "m"],  # GSL libraries
    library_dirs=["/usr/local/lib"],  # Path from gsl-config --libs
    include_dirs=[numpy.get_include(), "/usr/local/include"],  # Path from gsl-config --cflags
)

setup(
    ext_modules=cythonize([ext], language_level=3),
)

# Example clean (rebuild)
# python setup.py clean --all 
# python setup.py build_ext --inplace