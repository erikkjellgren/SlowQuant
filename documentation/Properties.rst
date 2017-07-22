Properties
==========

This section containts information about atomic and molecular properties that can be calculated.

Molecular dipole
----------------

The molecular dipole in one dimension is found as:

.. math::
   \mu_{x}=-\sum_{i}\sum_{j}D_{i,j}\left(i\left|x\right|j\right)+\sum_{A}Z_{A}X_{A}

FUNCTION:

- Properties.dipolemoment(basis, input, D, results)
- return results

Input:

- basis, basisset object
- input, inputfile object
- D, density matrix
- results, results object

Output:

- results, results obejct with added entries
- results['dipolex'] = ux
- results['dipoley'] = uy
- results['dipolez'] = uz
- results['dipoletot'] = u 

Refrence:

- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory

Mulliken charges
----------------

The atomic charge of the A'th atom can be found as:

.. math::
   q_{A}=Z_{A}-\sum_{i\in A}\left(D\cdot S\right)_{i,i}

FUNCTION:

- Properties.MulCharge(basis, input, D)
- None

Input:

- basis, basisset object
- input, inputfile object
- D, density matrix

Output:

- None

Refrence:

- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory


Lowdin charges
--------------

The atomic charge of the A'th atom can be found as:

.. math::
   q_{A}=Z_{A}-\sum_{i\in A}\left(S^{1/2}\cdot D\cdot S^{1/2}\right)_{i,i}

FUNCTION:

- Properties.LowdinCharge(basis, input, D)
- None

Input:

- basis, basisset object
- input, inputfile object
- D, density matrix

Output:

- None

Refrence:

- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory

Random-Phase Approximation Excitation energy
--------------------------------------------

The excitation energies can be calculated by using the random-phase approximation also known as time dependent Hartree-Fock. The exciation energy is found by diagonalizing the following equation:

.. math::
   \left(A+B\right)\left(A-B\right)X=E^{2}X
 
with the elements given as:

.. math::
   A_{ia,jb}=f_{ab}\delta_{ij}-f_{ij}\delta_{ab}+\left\langle aj\left|\right|ib\right\rangle 
   
   B_{ia,jb}=\left\langle ab\left|\right|ij\right\rangle 
  
All of the elements are in spin basis.

FUNCTION:

- Properties.RPA(F, C, input, results)

Input:

- F, fock matrix in spatial basis
- C, MO coeffcients in spatial basis
- input, inputfile object
- results, results object

Output:

- results, results obejct with added entries
- results['RPA Exc'] = Exc

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project12

