
Integral Transformations
========================

In this section the integral transformations will be described.

AO to MO basis 2 electron integrals
-----------------------------------

The integral transformation of the AO two electron integrals to the MO two electron integrals is given as:

.. math::
   \left(ij\left.\right|kl\right)=\sum_{\mu}\sum_{\nu}\sum_{\lambda}\sum_{\sigma}C_{\mu}^{i}C_{\nu}^{j}\left(\mu\nu\left.\right|\lambda\sigma\right)C_{\lambda}^{k}C_{\sigma}^{l}

FUNCTION:

- IntegralTransform.Transform2eMO(C, Vee)
- return VeeMO

Input:

- C, MO coefficients from SCF calculation
- Vee, two electron integrals in AO basis

Output:

- VeeMO, two electron integrals in MO basis

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project4

MO spatial to MO spin basis 2 electron integrals
------------------------------------------------

The integral transformation from MO spatial to MO spin orbitals is given by the following equation:

.. math::
   \left\langle \left.ij\right|kl\right\rangle =\left(\left.\sigma_{1}i\sigma_{2}k\right|\sigma_{3}j\sigma_{4}l\right)\delta_{\sigma_{1}\sigma_{2}}\delta_{\sigma_{3}\sigma_{4}}
   
FUNCTION:

- IntegralTransform.Transform2eSPIN(Vee)
- return VeeSpin

Input:

- Vee, two electron integrals

Output:

- VeeSpin, two electron integrals in spinbasis

References:

- None