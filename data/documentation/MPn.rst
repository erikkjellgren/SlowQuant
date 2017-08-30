

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

- MPn.MP2(occ, F, C, VeeMO)
- return EMP2

Input:

- occ, number of occupied MOs
- F, Fock matrix
- C, MO coefficient matrix
- VeeMO, twoelecron integral matrix

Output:

- EMP2, MP2 correction to the Hartree-Fock energy

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project4
- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory

Møller-Plesset, third order
---------------------------

The third order Møller-Plesset correction to the energy is given as:

.. MATH::
   E^{(3)}=\frac{1}{8}\sum_{abcdrs}\frac{\left\langle ab\left|\right|rs\right\rangle \left\langle cd\left|\right|ab\right\rangle \left\langle rs\left|\right|cd\right\rangle }{\left(\epsilon_{a}+\epsilon_{b}-\epsilon_{r}-\epsilon_{s}\right)\left(\epsilon_{c}+\epsilon_{d}-\epsilon_{r}-\epsilon_{s}\right)}
   
   +\frac{1}{8}\sum_{abrstu}\frac{\left\langle ab\left|\right|rs\right\rangle \left\langle rs\left|\right|tu\right\rangle \left\langle tu\left|\right|ab\right\rangle }{\left(\epsilon_{a}+\epsilon_{b}-\epsilon_{r}-\epsilon_{s}\right)\left(\epsilon_{a}+\epsilon_{b}-\epsilon_{t}-\epsilon_{u}\right)}
   
   +\sum_{abcrst}\frac{\left\langle ab\left|\right|rs\right\rangle \left\langle cs\left|\right|tb\right\rangle \left\langle rt\left|\right|ac\right\rangle }{\left(\epsilon_{a}+\epsilon_{b}-\epsilon_{r}-\epsilon_{s}\right)\left(\epsilon_{a}+\epsilon_{c}-\epsilon_{r}-\epsilon_{t}\right)}
   
FUNCTION:

- MPn.MP3(occ, F, C, VeeMO)
- return EMP3

Input:

- occ, number of occupied MOs in spinbasis
- F, Fock matrix
- C, MO coefficient matrix
- VeeMOspin, twoelecron integral matrix in spinbasis

Output:

- EMP3, MP3 correction to the energy

References:

- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory

Degeneracy-corrected perturbation, second order
-----------------------------------------------

The second order Degeneracy-corrected correction to the energy is given as:

.. MATH::
   E^{(2)}=\frac{1}{2}\left(D_{abij}-\sqrt{D_{abij}^{2}+4\left\langle \left.\left.ij\right|ab\right|\right\rangle ^{2}}\right)+\frac{1}{4}\left(D_{abij}-\sqrt{D_{abij}^{2}+4\left(\left\langle \left.\left.ij\right|ab\right|\right\rangle -\left\langle \left.\left.ij\right|ba\right|\right\rangle \right)^{2}}\right)

with:

.. MATH:
   D_{abij}=\epsilon_{a}+\epsilon_{b}-\epsilon_{i}-\epsilon_{j}
   
FUNCTION:

- MPn.DCPT2(occ, F, C, VeeMO)
- return EDCPT2

Input:

- occ, number of occupied MOs
- F, Fock matrix
- C, MO coefficient matrix
- VeeMO, twoelecron integral matrix

Output:

- EDCPT2, DCPT2 correction to the Hartree-Fock energy

References:

- Xavier Assfeld, Jan E Almlöf, and Donald G Truhlar. Degeneracy-corrected perturbation theory for electronic structure calculations. Chemical physics letters, 241(4):438–444, 1995   