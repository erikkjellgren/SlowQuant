
Integral Transformations
========================

In this section the integral transformations will be described.

AO to MO basis 2 electron integrals
-----------------------------------

The integral transformation of the AO two electron integrals to the MO two electron integrals is given as:

.. math::
   \left(ij\left|\right|kl\right)=\sum_{\mu}\sum_{\nu}\sum_{\lambda}\sum_{\sigma}C_{\mu}^{i}C_{\nu}^{j}\left(\mu\nu\left|\right|\lambda\sigma\right)C_{\lambda}^{k}C_{\sigma}^{l}

FUNCTION:

- IntegralTransform.TransformMO(C, basis, set, Vee)

Input:

- C, MO coefficients from SCF calculation
- basis, basisset obejct
- set, settingsfile object
- Vee, ERI matrix

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project4
