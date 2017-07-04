

Perturbations
==============

Contains information about perturbative methods. 

Møller-Plesset, second order
----------------------------

The second order Møller-Plesset correction to the energy is given as:

.. MATH::
   E_{\mathrm{MP2}}=\sum_{i,j}^{\mathrm{occ}}\sum_{a,b}^{\mathrm{unocc}}\frac{\left(ia\left|jb\right.\right)\left[2\left(ia\left|jb\right.\right)-\left(ib\left|ja\right.\right)\right]}{\epsilon_{i}+\epsilon_{j}-\epsilon_{a}-\epsilon_{b}}

The above equation is valid for closed shell systems. The integrals is in Mulliken notation, and the denominator is the orbital energies in MO, obtained from an SCF procedure.

FUNCTION:

- MPn.MP2(basis, input, F, C)
- return EMP2

Input:

- basis, basis set object
- input, input file object
- F, Fock matrix
- C, MO coefficient matrix

Output:

- EMP2, MP2 correction to the energy

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project4
- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory

