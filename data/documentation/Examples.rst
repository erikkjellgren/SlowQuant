

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

BOMD simulation
---------------
::

   basisset;3-21G
   Initial method;BOMD
   SCF Energy Threshold;1e-12
   SCF RMSD Threshold;1e-12
   steps;100
   stepsize;1.0
   SCF Max iterations;1000

Example of settingsfile. Perfoms a BOMD simulation of 100 steps, with a time step of 1.0 a.u.
