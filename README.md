[![Build Status](https://travis-ci.org/Melisius/Hartree-Fock.svg?branch=master)](https://travis-ci.org/Melisius/Hartree-Fock)
[![Coverage Status](https://coveralls.io/repos/github/Melisius/Hartree-Fock/badge.svg?branch=master)](https://coveralls.io/github/Melisius/Hartree-Fock?branch=master)

# SlowQuant

![SlowQuant logo](https://cloud.githubusercontent.com/assets/11976167/26658726/5e125b02-466c-11e7-8790-8412789fc9fb.jpg)

SlowQuant is a molecular quantum chemistry program written in python. Even the computational demanding parts are written in python, so it lacks speed, thus the name SlowQuant. The program is run from HFrun.py by specifing an input file and a settings file in the buttom of the script.

### Input file:

The input file have the following format:

-----------------------------------------

10;;;

1;1.63803684;1.136548823;0

8;0;-0.143225817;0

1;-1.63803684;1.136548823;0

-----------------------------------------

The first number is the number of electrons (have to be even since the program is only for closed shell). 

The first coloum is the atom given as number in the periodic system.

The three next coloums is the coordinates in a.u. (bohr).


### Setting file:

The setting file have the following format:

-----------------------------------------

basisset;STO3G

MPn;MP2

-----------------------------------------

When a calculation is run, first the Standardsettings.csv is loaded in. The user given settings file will then overrule the specified settings. The above file will make a calculation using the basisset STO3G. It will perform a MP2 calculation.
## Features:

As of now the program have the following features and settings.

### Requirements:

Python 3.4 or above (might work for lower versions)

Numpy

Scipy

### Settings:

        ; INPUT SETTINGS
        
basisset;STO3G - See basissets below

        ; SCF SETTINGS
        
SCF Energy Threshold;12 - Threshold for convergence of the SCF, given as 10^-x

SCF RMSD Threshold;12 - Threshold for convergence of the SCF, given as 10^-x

SCF Max iterations;100 - Maximum SCF iterations

DIIS;Yes - Activation of DISS

Keep Steps;6 - Number of steps saved in the DIIS algorithm

        ; PROPERTY SETTINGS
        
Charge;Mulliken - Type of charge to be calculated

Dipole;Yes - Calculation of molecular dipolemoment

MPn;No - Møller-Plesset energy correction, MP2, is the key to run it

        ; FITTING SETTINGS
        
Multipolefit;No - What moment to fit, Charge, Dipole, is the key to run it

Griddensity;1 - Density of points on a spherical surface, points/au^2

vdW scaling;1.4 - Scaling of the radius of the spherical surface

Constraint charge;Yes - Constrain charges

Qtot;0 - Total charge of molecule

Constraint dipole;Yes - Constain dipole moments

Write ESP;Yes - Write ESP to a PBD file

        ; GEOMOETRY OPTIMIZATION SETTINGS
        
GeoOpt;No - Geometry optimizatoin, Yes, is the key to run it

Max iteration GeoOpt;100 - Maximum iterations

Geometry Tolerance;3 - Tolerance for convergence, given as 10^-x

Gradient Decent Step;1.0 - Gradient scaling factor in Gradient Decent algorithm

Force Numeric;No - Choose to evaluate Forces numerically, Yes, is the key to run it


### Basis sets:

STO2G, STO3G, DZ, DZP, 3-21G - For up to the first 20 elements.

### Properties:

HF energy, MP2 energy

Mulliken charges 

Molecular dipole moment

Geometry Optimization for HF, analytic gradient

### Fitted properties:

ESP fitted charges

ESP fitted dipoles

### Schemes:

SCF, DIIS

Overlap integral, Obara-Saika

Electronic kinetic energy integral, Obara-Saika

Multipole integral, Obara-Saika

One electron coulomb integral, MacMurchie-Davidson

ERI, MacMurchie-Davidson



