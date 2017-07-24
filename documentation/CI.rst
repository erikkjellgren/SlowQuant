
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

- CI.CIS(F, C, input, results)

Input:

- F, fock matrix in spatial basis
- C, MO coeffcients in spatial basis
- input, inputfile object
- results, results object

Output:

- results, results obejct with added entries
- results['CIS Exc'] = Exc

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project12