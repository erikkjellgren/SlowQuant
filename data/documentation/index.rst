

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
- numpy 1.13.1 
- scipy 0.19.1  
- numba 0.34.0
- cython 0.25.2
- gcc 5.4.0

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
   BOMD.rst
   
