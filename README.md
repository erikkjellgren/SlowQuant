[![Documentation Status](https://readthedocs.org/projects/slowquant/badge/?version=latest)](http://slowquant.readthedocs.io/en/latest/?badge=latest)

# SlowQuant

![SlowQuant logo](https://cloud.githubusercontent.com/assets/11976167/26658726/5e125b02-466c-11e7-8790-8412789fc9fb.jpg)

SlowQuant is a molecular quantum chemistry program written in Python for classic and quantum computing.
Its specialty is unitary coupled cluster and (time-dependent) linear response in various novel parametrization schemes.
Even the computational demanding parts are written in Python, so it lacks speed, thus the name SlowQuant.

Documentation can be found at:

http://slowquant.readthedocs.io/en/latest/

## Quantum Computing, Variational Quantum Eigensolver

- tUCCSD (trotterized UCCSD through Qiskit)
- fUCCSD (factorized UCCSD)
- tUPS (tiled Unitary Product State)
- Naive Linear Response SD, with singlet spin-adapted operators
- Projected Linear Response SD, with singlet spin-adapted operators

These features are also implemented with the active-space approximation and orbital-optimization.
Suitable for ideal simulator, shot noise simulator, or quantum hardware via IBM Quantum Hub (Interface via Qiskit)

## Conventional Computing, Unitary Coupled Cluster

Current implementation supports:

- UCCSD, spin-adapted operators
- UCCSDTQ56
- Linear Response SD, spin-adapted operators
- Linear Response SDTQ56

These features are also implemented with the active-space approximation and orbital-optimization.

## Usual features

SlowQuant also got some conventional methods, such as Hartree-Fock and molecular integrals.
Just use [PySCF](https://github.com/pyscf/pyscf) instead.

## Cited in

- Kjellgren, Erik Rosendahl, et al. "Divergences in classical and quantum linear response and equation of motion formulations." arXiv preprint arXiv:2406.17141 (2024).
- von Buchwald, Theo Juncker, et al. "Reduced density matrix formulation of quantum linear response." arXiv preprint arXiv:2404.16586 (2024).
- Chan, Matthew, et al. "The tale of HORTON: Lessons learned in a decade of scientific software development." The Journal of Chemical Physics 160.16 (2024).
- Ziems, Karl Michael, et al. "Which options exist for NISQ-friendly linear response formulations?." Journal of Chemical Theory and Computation (2024).
- Chaves, B. D. P. G. (2023). Desenvolvimentos em python aplicados ao ensino da química quântica.
- Lehtola, S., & Karttunen, A. J. (2022). Free and open source software for computational chemistry education. Wiley Interdisciplinary Reviews: Computational Molecular Science, 12(5), e1610.

## Feature Graveyard

| Feature               | Last living commit                       |
|-----------------------|------------------------------------------|
| KS-DFT                | 1b9c5669ab72dfceee0a69c8dca1c67dd4b31bfd |
| MP2                   | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| RPA                   | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| Geometry Optimization | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CIS                   | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CCSD                  | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CCSD(T)               | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| BOMD                  | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
