from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

my_integrals = [Extension('slowquant.molecularintegrals.MIcython',['slowquant/molecularintegrals/MIcython.pyx'])]

setup(ext_modules=cythonize(my_integrals))
