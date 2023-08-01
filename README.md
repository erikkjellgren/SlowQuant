[![Build Status](https://travis-ci.com/erikkjellgren/SlowQuant.svg?branch=master)](https://travis-ci.com/erikkjellgren/SlowQuant)
[![Coverage Status](https://coveralls.io/repos/github/erikkjellgren/SlowQuant/badge.svg?branch=master)](https://coveralls.io/github/erikkjellgren/SlowQuant?branch=master)
[![Documentation Status](https://readthedocs.org/projects/slowquant/badge/?version=latest)](http://slowquant.readthedocs.io/en/latest/?badge=latest)

# SlowQuant

![SlowQuant logo](https://cloud.githubusercontent.com/assets/11976167/26658726/5e125b02-466c-11e7-8790-8412789fc9fb.jpg)

SlowQuant is a molecular quantum chemistry program written in Python.
Even the computational demanding parts are written in Python, so it lacks speed, thus the name SlowQuant.

Documentation can be found at:

http://slowquant.readthedocs.io/en/latest/

## Unique features -- Unitary Coupled Cluster

Current implementation supports:

- UCCSD, spin-adapted operators
- UCCSDTQ
- Linear Response SD, spin-adapted operators
- Linear Respinse SDTQ

These features are also implemented with the active-space approximation and orbital-optimization.

## Usual features

SlowQuant also got some conventional methods, such as Hartree-Fock and Density Functional Theory.
Just use [PySCF](https://github.com/pyscf/pyscf) instead.

## Feature Graveyard

| Feature               | Last living commit                       |
|-----------------------|------------------------------------------|
| MP2                   | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| RPA                   | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| Geometry Optimization | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CIS                   | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CCSD                  | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| CCSD(T)               | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
| BOMD                  | 46bf811dfcf217ce0c37ddec77d34ef00da769c3 |
