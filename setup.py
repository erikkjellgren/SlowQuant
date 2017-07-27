from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import Cython.Compiler.Options

Cython.Compiler.Options.get_directive_defaults()['profile'] = True
Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
Cython.Compiler.Options.get_directive_defaults()['binding'] = True

my_integrals = [Extension('slowquant.molecularintegrals.runMIcython',['slowquant/molecularintegrals/runMIcython.pyx'],define_macros=[('CYTHON_TRACE', '1')])]

setup(ext_modules=cythonize(my_integrals,compiler_directives={'linetrace': True, 'profile' :True, 'binding' : True}), include_dirs=[numpy.get_include()])
