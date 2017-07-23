

SlowQuant
===========

SlowQuant is a molecular quantum chemistry program written in python. Even the computational demanding parts are written in python, so it lacks speed, thus the name SlowQuant. The program is run as:

::
  
  python SlowQuant.py MOLECULE SETTINGS
  
As a ready to run example:

::
  
  python SlowQuant.py H2O.csv settingExample.csv

SlowQuant have the following requirements:

- Python 3.5 or above
- numpy
- scipy
- numba
- cython
- gcc

.. toctree::
   :maxdepth: 2
   :caption: How to use
   
   install.rst
   keywords.rst
   Examples.rst
   issues.rst
   illustrativecalc.rst

.. toctree::
   :maxdepth: 2
   :caption: Working equations and functions 
   
   General.rst
   MolecularIntegral.rst
   IntegralTrans.rst
   HFMethods.rst
   DIIS.rst
   MPn.rst
   Properties.rst
   GeoOpt.rst
   CI.rst
   CC.rst
   
