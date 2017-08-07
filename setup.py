from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
#import Cython.Compiler.Options
#Cython.Compiler.Options.get_directive_defaults()['cdivision'] = True
#Cython.Compiler.Options.get_directive_defaults()['boundscheck'] = False
#Cython.Compiler.Options.get_directive_defaults()['wraparound'] = False

my_integrals = [Extension('slowquant.molecularintegrals.runMIcython',['slowquant/molecularintegrals/runMIcython.pyx'])]

setup(ext_modules=cythonize(my_integrals), include_dirs=[numpy.get_include()])
