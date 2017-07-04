
DIIS
====

Direct Inversion in the Iterative Subspace (DIIS).
Makes new F' guesses bassed on previous guesses.


The error vector in DIIS is given as:

.. math::
    e_{i}=F_{i}D_{i}S - SD_{i}F_{i}



It is wanted that the sum of error vectors is zero:

.. math::
    e'=\sum_{i}c_{i}e_{i}=0

And now with the requirement that the sum of all c is zero, the following matrix eqution is solwed:

.. math:: 
    B_{i,j}c_{i}=b0

Here:

.. math::
   B_{ij}=\mathrm{tr}\left(e_{i}\cdot e_{j}\right)

and,

.. math::
   b0=\left(\begin{array}{c} 0\\ 0\\ ...\\ -1 \end{array}\right)

Finally the new F' is constructed as:

.. math::
   F'=\sum_{i}c_{i}F_{i}

FUNCTION:

- DIIS.DIIS(F,D,S,Efock,Edens,basis,numbF)
- return Fprime, Efock, Edens, Emax

Input:

- F, Fock matrix
- D, Density matrix
- S, overlap matrix
- Efock, saved fock matrices
- Edens, saved density matrices
- basis, basis set
- numbF, number of matrixes to save

Output:

- Fprime, F' guess
- Efock, saved fock matrices
- Edens, saved density matrices
- Emax, Maximum error in error vector

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project8
- P. Pulay, Chem. Phys. Lett. 73, 393 (1980).

