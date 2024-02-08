[![Documentation Status](https://readthedocs.org/projects/slowquant/badge/?version=latest)](http://slowquant.readthedocs.io/en/latest/?badge=latest)

# SlowQuant

![SlowQuant logo](https://cloud.githubusercontent.com/assets/11976167/26658726/5e125b02-466c-11e7-8790-8412789fc9fb.jpg)

SlowQuant is a molecular quantum chemistry program written in Python.
Even the computational demanding parts are written in Python, so it lacks speed, thus the name SlowQuant.

Documentation can be found at:

http://slowquant.readthedocs.io/en/latest/

## Quantum Computing, VQE

- UCCSD (through qiskit)
- Linear Response SD, spin-adapted operators

## Conventional Computing, Unitary Coupled Cluster

Current implementation supports:

- UCCSD, spin-adapted operators
- UCCSDTQ56
- Linear Response SD, spin-adapted operators
- Linear Respinse SDTQ56

These features are also implemented with the active-space approximation and orbital-optimization.

## Usual features

SlowQuant also got some conventional methods, such as Hartree-Fock and molecular integrals.
Just use [PySCF](https://github.com/pyscf/pyscf) instead.

## Cited in

- Ziems, K. M., Kjellgren, E. R., Reinholdt, P., Jensen, P. W., Sauer, S., Kongsted, J., & Coriani, S. (2023). Which options exist for NISQ-friendly linear response formulations?. arXiv preprint arXiv:2312.13937.
- Lehtola, S., & Karttunen, A. J. (2022). Free and open source software for computational chemistry education. Wiley Interdisciplinary Reviews: Computational Molecular Science, 12(5), e1610.
- Chaves, B. D. P. G. (2023). Desenvolvimentos em python aplicados ao ensino da química quântica.

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
