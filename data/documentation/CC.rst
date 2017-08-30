
Coupled Cluster
===============

In this section the working equations for Coupled Cluster can be found.

Coupled Cluster Singles Double
------------------------------

For CCSD the T1 (tia) and T2 (tijab) amplitudes is constructed to calculate the energy. These are constructed by the fisrt creating some intermediates:

.. math::
   F_{ae}=\left(1-\delta_{ae}\right)f_{ae}-\frac{1}{2}\sum_{m}^{occ}f_{me}t_{m}^{a}+\sum_{m,f}^{occ,virt}t_{m}^{f}\left\langle ma\left|\right|fe\right\rangle -\frac{1}{2}\sum_{m,n,f}^{occ,occ,virt}\tilde{\tau}_{mn}^{af}\left\langle mn\left|\right|ef\right\rangle 

.. math::
   F_{mi}=\left(1-\delta_{mi}\right)f_{mi}+\frac{1}{2}\sum_{e}^{virt}t_{i}^{e}f_{me}+\sum_{e,n}^{virt,occ}\left\langle mn\left|\right|ie\right\rangle +\frac{1}{2}\sum_{n,e,f}^{occ,virt,virt}\tilde{\tau}_{in}^{ef}\left\langle mn\left|\right|ef\right\rangle 
   
.. math::
   F_{me}=f_{me}+\sum_{n,f}^{occ,virt}t_{n}^{f}\left\langle mn\left|\right|ef\right\rangle 
   
.. math::
   W_{mnij}=\left\langle mn\left|\right|ij\right\rangle +P_{-}\left(ij\right)\sum_{e}^{virt}t_{j}^{e}\left\langle mn\left|\right|ie\right\rangle +\frac{1}{4}\sum_{e,f}^{virt,virt}\tau_{ij}^{ef}\left\langle mn\left|\right|ef\right\rangle 
   
.. math::
   W_{abef}=\left\langle ab\left|\right|ef\right\rangle -P_{-}\left(ab\right)\sum_{m}^{occ}t_{m}^{b}\left\langle ma\left|\right|ef\right\rangle +\frac{1}{4}\sum_{m,n}^{occ,occ}\tau_{mn}^{ab}\left\langle mn\left|\right|ef\right\rangle 
   
.. math::
   W_{mbej}=\left\langle mb\left|\right|ej\right\rangle +\sum_{f}^{virt}\left\langle mb\left|\right|ef\right\rangle -\sum_{n}^{occ}t_{n}^{b}\left\langle mn\left|\right|ej\right\rangle -\sum_{n,f}^{occ,virt}\left(\frac{1}{2}t_{jn}^{fb}+t_{j}^{f}t_{n}^{b}\right)\left\langle mn\left|\right|ef\right\rangle 
   
In the above equations the following definitions is used:

.. math::
   \tilde{\tau}_{ij}^{ab}=t_{ij}^{ab}+\frac{1}{2}\left(t_{i}^{a}t_{j}^{b}-t_{i}^{b}t_{j}^{a}\right)

.. math::
   \tau=t_{ij}^{ab}+t_{i}^{a}t_{j}^{b}-t_{i}^{b}t_{j}^{a}
   
.. math::
   P_{-}\left(ij\right)=1-P\left(ij\right)

The T1 and T2 is the constructed as:

.. math::
   t_{i}^{a}D_{i}^{a}=f_{ia}+\sum_{e}^{occ}t_{i}^{e}F_{ae}-\sum_{m}^{occ}t_{m}^{a}F_{mi}+\sum_{m,e}^{occ,virt}t_{im}^{ae}F_{me}-\sum_{n,f}^{occ,virt}t_{n}^{f}\left\langle na\left|\right|if\right\rangle 
   
   -\frac{1}{2}\sum_{m,e,f}^{occ,virt,virt}t_{im}^{ef}\left\langle ma\left|\right|ef\right\rangle -\frac{1}{2}\sum_{m,e,n}^{occ,virt,occ}t_{mn}^{ae}\left\langle mn\left|\right|ei\right\rangle 
   
.. math::
   t_{ij}^{ab}D_{ij}^{ab}=\left\langle ij\left|\right|ab\right\rangle +P_{-}\left(ab\right)\sum_{e}^{virt}t_{ij}^{ae}\left(F_{be}-\frac{1}{2}\sum_{m}^{occ}t_{m}^{b}F_{me}\right)+\frac{1}{2}\sum_{e,f}^{virt,virt}\tau_{ij}^{ef}W_{abef}
   
   -P_{-}\left(ij\right)\sum_{m}^{occ}t_{im}^{ab}\left(F_{mj}+\frac{1}{2}\sum_{e}^{virt}t_{j}^{e}F_{me}\right)+\frac{1}{2}\sum_{m,n}^{occ,occ}\tau_{mn}^{ab}W_{mnij}
   
   +P_{-}\left(ij\right)P_{-}\left(ab\right)\sum_{m,e}^{occ,virt}\left(t_{im}^{ae}W_{mbej}-t_{i}^{e}t_{m}^{a}\left\langle mb\left|\right|ej\right\rangle \right)
   
   +P_{-}\left(ij\right)\sum_{e}^{virt}t_{i}^{e}\left\langle ab\left|\right|ej\right\rangle -P_{-}\left(ab\right)\sum_{m}^{occ}t_{m}^{a}\left\langle mb\left|\right|ij\right\rangle 

Here:

.. math::
   D_{i}^{a}=f_{ii}-f_{aa}
 
.. math::
   D_{ij}^{ab}=f_{ii}+f_{jj}-f_{aa}-f_{bb}

It can be noted that the T1 and T2 equations depends on T1 and T2. Thus it have to be solver iteratively. The initial guess is given as:

.. math::
   t_{i}^{a}=0
   
.. math::
   t_{ij}^{ab}=\frac{\left\langle ij\left|\right|ab\right\rangle }{D_{ij}^{ab}}

The CCSD energy is then found as:

.. math::
   
   E_{\mathrm{CCSD}}=\sum_{i,a}^{occ,virt}f_{ia}t_{i}^{a}+\sum_{i,j,a,b}^{occ,occ,virt,virt}\left\langle ij\left|\right|ab\right\rangle \left(\frac{1}{4}t_{ij}^{ab}+\frac{1}{2}t_{i}^{a}t_{j}^{b}\right)
   
FUNCTION:

- CC.CCSD(occ, F, C, VeeMOspin, maxiter, deTHR, rmsTHR, runCCSDT=0)
- return EMP2, ECCSD

Input:

- F, fock matrix in spatial basis
- C, MO coeffcients in spatial basis
- VeeMOspin, two electron integrals in spinbasis
- deTHR, change in energy check for convergence
- rmsTHR, check for change in T1 and T2
- runCCSDT, 0 for CCSD and 1 for CCSD(T)

Output:

- EMP2, MP2 energy
- ECCSD, CCSD energy

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project5
- J.F. Stanton, J. Gauss, J.D. Watts, and R.J. Bartlett, J. Chem. Phys. volume 94, pp. 4334-4345 (1991)

Perturbative Triples Correction
-------------------------------

To find the perturbative triples correction, the disconnected and connected T3 have to be calculated. The disconnected is found as:

.. math::
   
   D_{ijk}^{abc}t_{ijk,\mathrm{disconnected}}^{abc}=P\left(i/jk\right)P\left(a/bc\right)t_{i}^{a}\left\langle jk\left|\right|bc\right\rangle 

And the connected is found as:

.. math::
   
   D_{ijk}^{abc}t_{ijk,\mathrm{connected}}^{abc}=P\left(i/jk\right)P\left(a/bc\right)\left[\sum_{e}^{virt}t_{jk}^{ae}\left\langle ei\left|\right|bc\right\rangle -\sum_{m}^{occ}t_{im}^{bc}\left\langle ma\left|\right|jk\right\rangle \right]

In the above equations the following definitions is used:

.. math::
   D_{ijk}^{abc}=f_{ii}+f_{jj}+f_{kk}-f_{aa}-f_{bb}-f_{cc}

.. math::
   P\left(i/jk\right)f\left(i,j,k\right)=f\left(i,j,k\right)-f\left(j,i,k\right)-f\left(k,j,i\right)

The energy correction can now be found as:

.. math::
   
   E_{\mathrm{\left(T\right)}}=\frac{1}{36}\sum_{i,j,k,a,b,c}^{occ,occ,occ,virt,virt,virt}t_{ijk,\mathrm{connected}}^{abc}D_{ijk}^{abc}\left(t_{ijk,\mathrm{connected}}^{abc}+t_{ijk,\mathrm{disconnected}}^{abc}\right)

FUNCTION:

- see CCSD function above
- return EMP2, ECCSD, ET

Input:

- runCCSDT=1

Output:

- EMP2, MP2 energy
- ECCSD, CCSD energy
- ET, perturbative triples corrections 
   
References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project6