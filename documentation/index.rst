

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

Installation
------------

This installation guide is made for Ubunutu. 

Python 3.6 can be installed with conda running the following command lines:

::
  
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

Then run:

::
  
  bash Miniconda3-latest-Linux-x86_64.sh

The python packages can now be installed by:

::
  
  conda install numpy scipy numba cython

The gcc compiler can be installed by:

::
  
  sudo apt-get install gcc

The SlowQuant program can be downloaded by:

::
  
  git clone https://github.com/Melisius/SlowQuant.git

Inside the SlowQuant folder run:

::
  
  python python setup.py build_ext --inplace

To test the installation install pytest:

::
  
  conda install pytest

And then run:

::
  
  pytest tests.py

All of the tests should succeed.

Windows 10
----------

For Windows 10, an Ubunutu can be installed following the guide in the following link:

https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/

Here after all the above steps can be followed.

Compiling documentation
-----------------------

The documentation is compiled using sphinx:

::
  
  conda install sphinx

To make the equations LaTeX is needed, and can be installed by:

::
  
  sudo apt-get install texlive-full

The theme is downloaded by:

::
  
  pip install sphinx_rtd_theme

The documentation is now compiled by:

::
  
  sphinx-build documentation docs 

.. toctree::
   :maxdepth: 2
   :caption: How to use
   
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
   
