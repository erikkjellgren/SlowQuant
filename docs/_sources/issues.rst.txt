
Issues
======

In this section known problems with the code can be found

General
-------

- MolecularIntegrals.py is a mess
- Code in general is a gigantic mess
- Numerical methods makes code none module like
- No checking of input files at all at any stages of the code

Need testing
------------

- No tests for normalization of contracted basisfunctions of D or higher angular momentum
- Geometry optimization is not properly tested
- DCPT2 not tested, no case in paper, where cartesian integrals was used (all had D orbitals)

Broken
------

- Self overlap integrals is non-one for contracted  basisfunctions
