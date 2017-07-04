

Example input files
===================

::

  10;;; 
  1;1.63803684;1.136548823;0
  8;0;-0.143225817;0
  1;-1.63803684;1.136548823;0

Example of inputfile. The molecule is water.

Hartree-Fock calculation
------------------------

::

  basisset;3-21G
  Initial Method;HF

Example of settingsfile. Performs a Hartree-Fock calculation with the basisset 3-21G

MP2 calculation
---------------

::

  basisset;3-21G
  Initial Method;HF
  MPn;MP2

Example of settingsfile. Performs a MP2 calculation with the basisset 3-21G

Geometry optimization
---------------------

::

  basisset;STO3G
  Initial Method;HF
  GeoOpt;Yes

Example of settingsfile. Performs a geometry optimization calculation with the basisset STO3G
