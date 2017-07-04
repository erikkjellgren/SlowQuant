Propeties
=========

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


