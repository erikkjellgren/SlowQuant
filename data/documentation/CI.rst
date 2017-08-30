
Configuration Interaction
=========================

In this section configuration interaction methods will be described.

CI Singles
----------

The CI singles are done by building a Hamiltonian and diagonalizing it. The Hamiltonian build accourding to:

.. math::
   H_{ia,jb} =f_{ab}\delta_{ij}-f_{ij}\delta_{ab}+\left\langle aj\left|\right|ib\right\rangle 
   
Here everything is in spin orbital basis.

FUNCTION:

- CI.CIS(occ, F, C, VeeMOspin)
- return Exc

Input:

- occ, number of occupied MOs in spinbasis
- F, fock matrix in spatial basis
- C, MO coeffcients in spatial basis
- VeeMOspin, MO integrals in spin basis

Output:

- Exc, single excitation energies

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project12