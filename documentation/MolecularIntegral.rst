
Molecular Integrals
===================

Contains the information about how the integrals are calculated. In the equations in this section the following definitions is used.

.. math::
   p=a+b

.. math::
   \mu=\frac{ab}{a+b}

.. math::
   P_{x}=\frac{aA_{x}+bB_{x}}{p}

.. math::
   X_{AB}=A_{x}-B_{x}

Here a and b are Gaussian exponent factors. Ax and Bx are the position of the Gaussians in one dimension. Further the basisset functions is of Gaussian kind and given as:

.. math::
   \phi_{A}\left(r\right)=N\left(x-A_{x}\right)^{l}\left(y-A_{y}\right)^{m}\left(z-A_{z}\right)^{n}\exp\left(-\zeta\left(\vec{r}-\vec{A}\right)^{2}\right)

with a normalization constant given as:

.. math::
   N=\left(\frac{2\alpha}{\pi}\right)^{3/4}\left[\frac{\left(8\alpha\right)^{l+m+n}l!m!n!}{\left(2l\right)!\left(2m\right)!\left(2n\right)!}\right]

If the basis functions is constracted the normalization of the contraction is given as:

.. math::
   N_{\mathrm{contracted}}=\left[\frac{\pi^{3/2}\left(2l-1\right)!!\left(2m-1\right)!!\left(2n-1\right)!!}{2^{l+m+n}}\sum_{i,j}^{basis}\frac{c_{i}N_{i}c_{j}N_{j}}{\left(\alpha_{i}+\alpha_{j}\right)^{l+m+n+3/2}}\right]^{-1/2}

Boys Function
-------------

The Boys function is given as:

.. math::
    F_{n}\left(x\right)=\int_{0}^{1}\exp\left(-xt^{2}\right)t^{2n}dt

FUNCTION:

- MolecularIntegrals.boys(m,T)
- return value

Input:

- m, subscript of the Boys function
- T, argument of the Boys function

Output:

- value, value corrosponding to given m and T

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory

Expansion coefficients
----------------------

The expansion coefficient is found by the following recurrence relation:

.. math::
   E_{t}^{i,j}=0,\,\,\,\,t<0\,\,\mathrm{or}\,\,t>i+j

   E_{t}^{i+1,j}=\frac{1}{2p}E_{t-1}^{i,j}+X_{\mathrm{PA}}E_{t}^{i,j}+\left(t+1\right)E_{t+1}^{i,j}

   E_{t}^{i,j+1}=\frac{1}{2p}E_{t-1}^{i,j}+X_{\mathrm{PB}}E_{t}^{i,j}+\left(t+1\right)E_{t+1}^{i,j}

With the boundary condition that:

.. math::
   E_{0}^{0,0}=\exp\left(-pX_{\mathrm{AB}}^{2}\right)

FUNCTION:

- MolecularIntegrals.E(i,j,t,Qx,a,b,XPA,XPB,XAB)
- return val

Input:

- i, input values
- j, input values
- t, input values
- Qx, input values
- a, input values
- b, input values
- XPA, input values
- XPB, input values
- XAB, input values

Output:

- val, value corrosponding to the given input

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory
- http://joshuagoings.com/assets/integrals.pdf


Hermite coulomb integral
------------------------

The hermite coulomb integrals is given as the following recurrence relations:

.. math::
   R_{t+1,u,v}^{n}\left(p,R_{\mathrm{PC}}\right)=tR_{t-1,u,v}^{n+1}\left(p,R_{\mathrm{PC}}\right)+X_{\mathrm{PC}}R_{t,u,v}^{n+1}\left(p,R_{\mathrm{PC}}\right)

   R_{t,u+1,v}^{n}\left(p,R_{\mathrm{PC}}\right)=uR_{t,u-1,v}^{n+1}\left(p,R_{\mathrm{PC}}\right)+Y_{\mathrm{PC}}R_{t,u,v}^{n+1}\left(p,R_{\mathrm{PC}}\right)

   R_{t,u,v+1}^{n}\left(p,R_{\mathrm{PC}}\right)=vR_{t,u,v-1}^{n+1}\left(p,R_{\mathrm{PC}}\right)+Z_{\mathrm{PC}}R_{t,u,v}^{n+1}\left(p,R_{\mathrm{PC}}\right)

With the boundary condition:

.. math::
   R_{0,0,0}^{n}\left(p,R_{\mathrm{PC}}\right)=\left(-2p\right)^{n}F_{n}\left(pR_{\mathrm{PC}}^{2}\right)

FUNCTION:

- MolecularIntegrals.R(t,u,v,n,p,PCx,PCy,PCz,RPC)
- return val

Input:

- t, input value
- u, input value
- v, input value
- n, input value 
- p, input value
- PCx, input value
- PCy, input value
- PCz, input value
- RPC, input value

Output:

- val, value corrosponding to the given input

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory
- http://joshuagoings.com/assets/integrals.pdf

Overlap
-------

The overlap integrals are solved by the following recurrence relation:

.. math::

   S_{i+1,j}=X_{PA}S_{ij}+\frac{1}{2p}\left(iS_{i-1,j}+jS_{i,j-1}\right)

   S_{i,j+1}=X_{PB}S_{ij}+\frac{1}{2p}\left(iS_{i-1,j}+jS_{i,j-1}\right)

With the boundary condition that:

.. math::

   S_{00}=\sqrt{\frac{\pi}{p}}\exp\left(-\mu X_{AB}^{2}\right)

FUNCTION:

- MolecularIntegrals.Overlap(a, b, la, lb, Ax, Bx)
- return Sij

Input:

- a, Gaussian exponent factor
- b, Gaussian exponent factor
- la, angular momentum quantum number
- lb, angular momentum quantum number
- Ax, position along one axis
- Bx, position along one axis

Output:

- Sij, non-normalized overlap element in one dimension

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory
 
Kinetic energy
--------------

The kinetic energy integrals are solved by the following recurrence relation:

.. math::

   T_{i+1,j}=X_{\mathrm{PA}}T_{i,j}+\frac{1}{2p}\left(iT_{i-1,j}+jT_{i,j-1}\right)+\frac{b}{p}\left(2aS_{i+1,j}-iS_{i-1,j}\right)

   T_{i,j+1}=X_{\mathrm{PB}}T_{i,j}+\frac{1}{2p}\left(iT_{i-1,j}+jT_{i,j-1}\right)+\frac{a}{p}\left(2bS_{i,j+1}-iS_{i,j-1}\right)

With the boundary condition that:

.. math::

   T_{00}=\left[a-2a^{2}\left(X_{\mathrm{PA}}^{2}+\frac{1}{2p}\right)\right]S_{00}

FUNCTION:

- Kin(a, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb, N1, N2, c1, c2)
- return Tij, Sij

Input:

- a, Gaussian exponent factor
- b, Gaussian exponent factor
- Ax, position along the x-axis
- Bx, position along the x-axis
- Ay, position along the y-axis
- By, position along the y-axis
- Az, position along the z-axis
- Bz, position along the z-axis
- la, angular momentum quantum number
- lb, angular momentum quantum number
- ma, angular momentum quantum number
- mb, angular momentum quantum number
- na, angular momentum quantum number
- nb, angular momentum quantum number
- N1, normalization constant
- N2, normalization constant
- c1, Gaussian prefactor
- c2, Gaussian prefactor

Output:

- Tij, normalized kinetic energy matrix element
- Sij, normalized overlap matrix element

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory


Electron-nuclear attraction
----------------------------

The electron-nuclear interaction integral is given as:

.. math::
   V_{ijklmn}^{000}=\frac{2\pi}{p}\sum_{t}^{i+j}E_{t}^{ij}\sum_{u}^{k+l}E_{u}^{kl}\sum_{v}^{m+n}E_{v}^{mn}R_{tuv}

FUNCTION:

- MolecularIntegrals.elnuc(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input)
- return Vij

Input:

- a, Gaussian exponent factor
- b, Gaussian exponent factor
- Ax, position along the x-axis
- Ay, position along the y-axis
- Az, position along the z-axis
- Bx, position along the x-axis
- By, position along the y-axis
- Bz, position along the z-axis
- l1, angular momentum quantum number
- l2, angular momentum quantum number
- m1, angular momentum quantum number
- n1, angular momentum quantum number
- n2, angular momentum quantum number
- N1, normalization constant
- N2, normalization constant
- c1, Gaussian prefactor
- c2, Gaussian prefactor
- input, inputfile object 

Output:

- Vij, normalized electron-nuclei attraction matrix element

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory
- http://joshuagoings.com/assets/integrals.pdf


Electron-nuclear field
----------------------

The electron-nuclear interaction integral is given as:

.. math::
   V_{ijklmn}^{efg}=\left(-1\right)^{e+f+g}\frac{2\pi}{p}\sum_{t}^{i+j}E_{t}^{ij}\sum_{u}^{k+l}E_{u}^{kl}\sum_{v}^{m+n}E_{v}^{mn}R_{t+e,u+f,v+g}

Here e, f and g detones the order of derivate with respect to x, y and z

FUNCTION:

- MolecularIntegrals.electricfield(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, derivative, atomidx)
- return VijA

Input:

- a, Gaussian exponent factor
- b, Gaussian exponent factor
- Ax, position along the x-axis
- Ay, position along the y-axis
- Az, position along the z-axis
- Bx, position along the x-axis
- By, position along the y-axis
- Bz, position along the z-axis
- l1, angular momentum quantum number
- l2, angular momentum quantum number
- m1, angular momentum quantum number
- n1, angular momentum quantum number
- n2, angular momentum quantum number
- N1, normalization constant
- N2, normalization constant
- c1, Gaussian prefactor
- c2, Gaussian prefactor
- input, inputfile object
- derivative, axis of derivative (dx,dy or dz)
- atomidx, index of atom for which the electricfield is calculated

Output:

- VijA, normalized electron-nuclei field of nuclei A matrix element

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory

Electron-electron repulsion
---------------------------

The electron-electron repulsion integral is calculated as:

.. math::
   g_{abcd}=\sum_{t}^{l1+l2}E_{t}^{ab}\sum_{u}^{m1+m2}E_{u}^{ab}\sum_{v}^{n1+n2}E_{v}^{ab}\sum_{\tau}^{l3+l4}E_{\tau}^{cd}\sum_{\nu}^{m3+m4}E_{\nu}^{cd}\sum_{\phi}^{n3+n4}E_{\phi}^{cd}\left(-1\right)^{\tau+\nu+\phi}\frac{2\pi^{5/2}}{pq\sqrt{p+q}}R_{t+\tau,u+\nu,v+\phi}\left(\alpha,R_{\mathrm{PQ}}\right)

FUNCTION:

- MolecularIntegrals.elelrep(a, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4)
- return Veeijkl

Input:

- a, Gaussian exponent factor
- b, Gaussian exponent factor
- c, Gaussian exponent factor
- d, Gaussian exponent factor
- Ax, position along the x-axis
- Ay, position along the y-axis
- Az, position along the z-axis
- Bx, position along the x-axis
- By, position along the y-axis
- Bz, position along the z-axis
- Cx, position along the x-axis
- Cy, position along the y-axis
- Cz, position along the z-axis
- Dx, position along the x-axis
- Dy, position along the y-axis
- Dz, position along the z-axis
- l1, angular momentum quantum number
- l2, angular momentum quantum number
- l3, angular momentum quantum number
- l4, angular momentum quantum number
- m1, angular momentum quantum number
- m2, angular momentum quantum number
- m3, angular momentum quantum number
- m4, angular momentum quantum number
- n1, angular momentum quantum number
- n2, angular momentum quantum number
- n3, angular momentum quantum number
- n4, angular momentum quantum number
- N1, normalization constant
- N2, normalization constant
- N3, normalization constant
- N4, normalization constant
- c1, Gaussian prefactor
- c2, Gaussian prefactor
- c3, Gaussian prefactor
- c4, Gaussian prefactor

Output:

- Veeijkl, normalized electron-electron repulsion matrix element

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory
- http://joshuagoings.com/assets/integrals.pdf

Nuclear-nuclear repulsion
-------------------------

The nucleus-nucleus repulsion term is calculated with classical nuclie as follows: 

.. math::
   V_{NN}=\sum_{A}\sum_{B<A}\frac{Z_{A}Z_{B}}{r_{AB}}

FUNCTION:

- MolecularIntegrals.nucrep(input)
- return Vnn

Input:

- input, inputfile object

Output:

- Vnn, nuclear repulsion energy

References:

- None

Nuclear-nuclear field
---------------------

The nucleus-nucleus field term is calculated with classical nuclie as follows: 

.. math::
   \frac{\partial V_{NN}}{\partial X_{A}}=-Z_{A}\sum_{B\neq A}\frac{Z_{B}\left(X_{B}-X_{A}\right)}{r_{AB}^{3}}

FUNCTION:

- MolecularIntegrals.nucdiff(input, atomidx, direction)
- return Vnn

Input:

- input, inputfile object
- atomidx, atom which is differentiated with respect to
- direction, axis of differentiation (1 = dx, 2 = dy, 3 = dz)

Output:

- Vnn, nucleus-nucleus field

References:

- None

Dipole moment integral
----------------------

The dipole moment integral is calculated by using the following relations:

.. math::
   S_{i+1,j}^{e}=X_{\mathrm{PA}}S_{i,j}^{e}+\frac{1}{2p}\left(iS_{i-1,j}^{e}+jS_{i,j-1}^{e}+eS_{ij}^{e-1}\right)

   S_{i,j+1}^{e}=X_{\mathrm{PB}}S_{i,j}^{e}+\frac{1}{2p}\left(iS_{i-1,j}^{e}+jS_{i,j-1}^{e}+eS_{ij}^{e-1}\right)

   S_{i,j}^{e+1}=X_{\mathrm{PC}}S_{i,j}^{e}+\frac{1}{2p}\left(iS_{i-1,j}^{e}+jS_{i,j-1}^{e}+eS_{ij}^{e-1}\right)

Here e is the order of multipole moment, e=1 is dipole moment.

FUNCTION:

- MolecularIntegrals.u_ObaraSaika(a1, a2, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb, N1, N2, c1, c2, input)
- return muxij, muyij, muzij

Input:

- a1, Gaussian exponent factor
- a2, Gaussian exponent factor
- Ax, position along x axis
- Ay, position along y axis
- Az, position along z axis
- Bx, position along x axis
- By, position along y axis
- Bz, position along z axis
- la, angular momentum quantum number
- lb, angular momentum quantum number
- ma, angular momentum quantum number
- mb, angular momentum quantum number
- na, angular momentum quantum number
- nb, angular momentum quantum number
- N1, normalization constant
- N2, normalization constant
- c1, Gaussian prefactor
- c2, Gaussian prefactor
- input, inputfile object

Output:

- muxij, dipolemoment integral matrix element for x axis
- muyij, dipolemoment integral matrix element for y axis
- muzij, dipolemoment integral matrix element for z axis

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory

Cauchy-Schwarz inequality
-------------------------

The electron-electron repulsion integrals is screened by the use of the Cauchy-Schwarz inequality given as:

.. math::
   \left|g_{abcd}\right|\leq\sqrt{g_{abab}}\sqrt{g_{cdcd}}

FUNCTION:

- None

References:

- Trygve Helgaker, Poul Jorgensen and Jeppe Olsen, Molecular Electronic-Structure Theory
