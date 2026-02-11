[![Documentation Status](https://readthedocs.org/projects/slowquant/badge/?version=latest)](http://slowquant.readthedocs.io/en/latest/?badge=latest)

# SlowQuant

![SlowQuant logo](https://cloud.githubusercontent.com/assets/11976167/26658726/5e125b02-466c-11e7-8790-8412789fc9fb.jpg)

SlowQuant is a molecular quantum chemistry program written in Python for classic and quantum computing.
Its specialty is unitary parameterized wave functions and (time-dependent) linear response in various novel parametrization schemes.
Even the computational demanding parts are written in Python, so it lacks speed, thus the name SlowQuant.

Documentation can be found at:

http://slowquant.readthedocs.io/en/latest/

## Quantum Computing targeting hardware through Qiskit

Suitable for ideal simulator, shot noise simulator, or quantum hardware via IBM Quantum Hub (Interface via Qiskit)

### Variational quantum eigensolver

- Various fermionic ansätze such as; factorized UCC, tUPS, QNP, etc.
- Active-space approximation and orbital-optimzation for all ansätze.
- Analytical electronic gradients with parameter-shift rule.
- Both single reference wave functions, and state-averaged wave fuctions.

### Linear response

All linear response is up to SDTQ56 with SD being singlet spin-adapted operators.

- Naive linear response with contribution from orbital response.
- Projected linear response with contribution from orbital response.
- Self-consistent linear response in the active-space.
- State-transfer linear response in the active-space.

### Error mitigation techniques

- Post-selection of Pauli-strings in the computational basis.
- Mansatz0 error mitigation technique.

### Other

- Qubit-wise commutatitive to save on measurements.
- Pauli saving to save measurements.
- Efficient implementation of fermionic circuits.

## State-vector simulator for unitary wave functions

### Wave functions

All unitary product state wave functions that are in the quantum computing part of SlowQuant are parameter compatible with the state-vector versions.

- Unitary product state wave functions such as; factorized UCC, tUPS, QNP etc.
- Unitary coupled-cluster without factorization or trotterization.
- Active-space approximation and orbital-optimzation for all wave functions.
- Both single reference wave functions, and state-averaged wave fuctions.

### Linear response

All linear response is up to SDTQ56 with SD being singlet spin-adapted operators.

- Naive linear response with contribution from orbital response.
- Projected linear response with contribution from orbital response.
- Self-consistent linear response with contribution from orbital response.
- State-transfer linear response with contribution from orbital response.

## Usual features

SlowQuant also got some conventional methods, such as Hartree-Fock and molecular integrals.
Just use [PySCF](https://github.com/pyscf/pyscf) instead.

## Cited in

- Kjellgren, E.R., Reinholdt, P., Ziems, K.M., Sauer, S., Coriani, S. and Kongsted, J., 2025. Redundant parameter dependencies in conventional and quantum linear response and equation of motion theory for unitary parameterized wave functions. The Journal of Chemical Physics, 163(13).
- Kjellgren, E. R., Ziems, K. M., Reinholdt, P., Sauer, S., Coriani, S., & Kongsted, J. (2025). Exact closed-form expression for unitary spin-adapted fermionic singlet double excitation operators. The Journal of Chemical Physics, 163, 134115 (2025)
- Jensen, P.W., Hedemark, G.S., Ziems, K.M., Kjellgren, E.R., Reinholdt, P., Knecht, S., Coriani, S., Kongsted, J. and Sauer, S.P., 2025. Hyperfine coupling constants on quantum computers: Performance, errors, and future prospects. Journal of Chemical Theory and Computation, 21(16), pp.7878-7889.
- Ziems, K. M., Kjellgren, E. R., Sauer, S. P., Kongsted, J., & Coriani, S. (2025). Understanding and mitigating noise in molecular quantum linear response for spectroscopic properties on quantum computers. Chemical Science.
- Kjellgren, E. R., Reinholdt, P., Ziems, K. M., Sauer, S., Coriani, S., & Kongsted, J. (2024). Divergences in classical and quantum linear response and equation of motion formulations. The Journal of Chemical Physics, 161(12).
- von Buchwald, T. J., Ziems, K. M., Kjellgren, E. R., Sauer, S. P., Kongsted, J., & Coriani, S. (2024). Reduced density matrix formulation of quantum linear response. Journal of Chemical Theory and Computation, 20(16), 7093-7101.
- Chan, M., Verstraelen, T., Tehrani, A., Richer, M., Yang, X. D., Kim, T. D., ... & Ayers, P. W. (2024). The tale of HORTON: Lessons learned in a decade of scientific software development. The Journal of Chemical Physics, 160(16).
- Ziems, K. M., Kjellgren, E. R., Reinholdt, P., Jensen, P. W., Sauer, S. P., Kongsted, J., & Coriani, S. (2024). Which options exist for NISQ-friendly linear response formulations?. Journal of Chemical Theory and Computation, 20(9), 3551-3565.
- Chaves, B. D. P. G. (2023). Desenvolvimentos em python aplicados ao ensino da química quântica.
- Lehtola, S., & Karttunen, A. J. (2022). Free and open source software for computational chemistry education. Wiley Interdisciplinary Reviews: Computational Molecular Science, 12(5), e1610.

## Feature Graveyard

| Feature                                        | Last living commit                       |
|------------------------------------------------|------------------------------------------|
| Qiskit Estimator                               | 1fe8c4cac7ff5a620b760ee18ff1a8179cf40898 |
| RDM trace correction quantum wave function     | e26074fc8aae8dc0f6528308022ad265c5ca18bc |
| No submatrix saving in proj and all-proj LR    | 3f5df6818c4dbbb2b54606d0a1a4e00badfb766d |
| Approxmiate Hermitification in linear response | 3f5df6818c4dbbb2b54606d0a1a4e00badfb766d |
| Approxmiate linear response formalism          | 3f5df6818c4dbbb2b54606d0a1a4e00badfb766d |
| KS-DFT                                         | 1b9c5669ab72dfceee0a69c8dca1c67dd4b31bfd |
| MP2                                            | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| RPA                                            | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| Geometry Optimization                          | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CIS                                            | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CCSD                                           | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CCSD(T)                                        | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| BOMD                                           | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
