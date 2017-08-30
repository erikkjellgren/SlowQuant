
List of keywords
================

Settings are given in the setting inputfile by using the below keywords. The order of the keywords in the setting inputfile does not matter at all.

- basisset;x

x=bassiset used in calculation. STO2G, STO3G, DZ, DZP, 3-21G, 6-31ppGss (6-31++G**), 6-31Gs (6-31G*), 6-31pGs (6-31+G*).

Integrals
---------

- None

Initial Method
--------------

- Initial Method;x

Method to use during SCF. x=HF for Hartree-Fock calculation. x=UHF for unrestricted Hartree-Fock calculation.

- UHF mix guess;x

Mix initial coefficients to break spin symmetry

SCF
---

- SCF Energy Threshold;x

Threshold for convergence of the SCF, given as x

- SCF RMSD Threshold;x

Threshold for convergence of the SCF, given as x

- SCF Max iterations;x

Maximum SCF iterations

- DIIS;x

Activation of DISS, x=Yes

- Keep Steps;x

Number of steps saved in the DIIS algorithm

Perturbation
------------

- MPn;x

Møller-Plesset energy correction, x=MP2


Properties
----------

- Charge;x

Calculation of atomic charges. x=Mulliken gives Mulliken charges. x=Lowdin gives Lowdin charges.

- Dipole;x

Calcultion of molecular dipolemomemnt. x=Yes for calcultion.

- Excitation energy;x

Calculation of excitation energies. x=RPA for using TDHF/RPA.


Geomoetry Optimization
----------------------

- GeoOpt;x

Turns on geometry optimization. x=Yes to turn it on. Only works for Hartree-Fock.

- Max iteration GeoOpt;x

Maximum geometry optimization steps. 

- Geometry Tolerance;x

Tolerance for convergence, given as x

- Gradient Descent Step;x

Gradient scaling factor in Gradient Descent algorithm

- Force Numeric;x

Choose to evaluate Forces numerically. x=Yes to activate. Only works for Hartree-Fock.

Configuration Interaction
-------------------------

- CI;x

To get exciation energies with CI singles. x=CIS

Couple Cluster
--------------

- CC;x

To run CCSD, x=CCSD. To run CCSD(T), x=CCSD(T)

- CC Max iterations;x

Maximaum CC amplitudes iterations. 

- CC RMSD Threshold;10

RMSD Threshold for T1 and T2, given as x

- CC Energy Threshold;10

Energy change Threshold for CC, given as x

Ab-Inition Molecular Dynamics
-----------------------------

- stepsize;x

Integration stepsize for the movement of the nuclei.

- steps;20

Number of steps for the dynamics simulation
