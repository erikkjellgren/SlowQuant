

Geometry optimization
=====================

In this section the geometry optimization are described. In general the derivative of the Gaussian basisset functions is given as:

.. math::
   \frac{\partial\phi_{l,m,n}}{\partial X_{A}}=\left[\left(2l+1\right)a\right]^{1/2}\phi_{l+1,m,n}-2l\left(\frac{a}{2l-1}\right)^{1/2}\phi_{l-1,m,n}

Gradient Descent
----------------

The molecular structure is propagated using the Gradient Descent method:

.. math::
   X_{A,i+1}=X_{A,i}-\zeta\frac{\partial E_{\mathrm{HF}}}{\partial X_{A}}

FUNCTION:

- None

References:

- None 

Analytic Hartree-Fock
----------------------

The analytic Hartree-Fock derivative is given as:

.. math::
   \frac{\partial E_{\mathrm{HF}}}{\partial X_{A}}=\sum_{i,j}D_{i,j}\frac{H_{i,j}^{\mathrm{core}}}{\partial X_{A}}+\frac{1}{2}\sum_{i,j,k,l}D_{j,i}D_{k,l}\frac{\partial\left(ij\left|\right|kl\right)}{\partial X_{A}}-2\sum_{i,j}\sum_{a}^{\mathrm{occ}}\varepsilon_{a}C_{j,a}C_{i,a}\frac{\partial S_{i,j}}{\partial X_{A}}+\frac{\partial V_{NN}}{\partial X_{A}}

FUNCTION:

- GeometryOptimization.run_analytic(input, set, results)

Input:

- input, inputfile object
- set, settingsfile object
- results, results object

Output:

- input, inputfile object with opdated coordinates

References:

- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory

Finite difference
------------------

The numerical gradient is calculated by the use of finite difference, and is found as:

.. math::
   \frac{\partial E_{\mathrm{HF}}}{\partial X_{A}}=\frac{E_{\mathrm{HF}}\left(X_{A}+\epsilon\right)-E_{\mathrm{HF}}\left(X_{A}-\epsilon\right)}{2\epsilon}

FUNCTION:

- GeometryOptimization.run_numeric(input, set, results)
- return input

Input:

- input, inputfile object
- set, settingsfile object
- results, results object

Output:

- input, inputfile object with opdated coordinates

References:

- None
