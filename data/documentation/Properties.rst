Properties
==========

This section containts information about atomic and molecular properties that can be calculated.

Molecular dipole
----------------

The molecular dipole in one dimension is found as:

.. math::
   \mu_{x}=-\sum_{i}\sum_{j}D_{i,j}\left(i\left|x\right|j\right)+\sum_{A}Z_{A}X_{A}

FUNCTION:

- Properties.dipolemoment(input, D, mux, muy, muz)
- return ux, uy, uz, u

Input:

- input, inputfile object
- D, density matrix
- mux, dipolemoment integrals in x direction
- muy, dipolemoment integrals in y direction
- muz, dipolemoment integrals in z direction

Output:

- ux, dipolemoment in x direction
- uy, dipolemoment in y direction
- uz, dipolemoment in z direction
- u, total dipolemoment

Refrence:

- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory

Mulliken charges
----------------

The atomic charge of the A'th atom can be found as:

.. math::
   q_{A}=Z_{A}-\sum_{i\in A}\left(D\cdot S\right)_{i,i}

FUNCTION:

- Properties.MulCharge(basis, input, D, S)
- return qvec

Input:

- basis, basisset object
- input, inputfile object
- D, density matrix
- S, overlap matrix

Output:

- qvec, vector of Mulliken charges

Refrence:

- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory


Lowdin charges
--------------

The atomic charge of the A'th atom can be found as:

.. math::
   q_{A}=Z_{A}-\sum_{i\in A}\left(S^{1/2}\cdot D\cdot S^{1/2}\right)_{i,i}

FUNCTION:

- Properties.LowdinCharge(basis, input, D, S)
- return qvec

Input:

- basis, basisset object
- input, inputfile object
- D, density matrix
- S, overlap matrix

Output:

- qvec, vector of Lowdin charges


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
   
.. math::
   B_{ia,jb}=\left\langle ab\left|\right|ij\right\rangle 
  
All of the elements are in spin basis.

FUNCTION:

- Properties.RPA(occ, F, C, VeeMOspin)
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

