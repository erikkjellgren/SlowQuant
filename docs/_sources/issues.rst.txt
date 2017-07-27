
Issues
======

In this section known problems with the code can be found

General
-------

- Code in general is a gigantic mess
- Numerical methods makes code none module like
- No checking of input files at all at any stages of the code
- Basisset object has a bad structure
- Loops should be replaced with np.einsum() in many cases

Performance
-----------

- Integral derivatives dosen't fully utilize precalculation of expansion coefficients, E

Need testing
------------

- No tests for normalization of contracted basisfunctions of D or higher angular momentum
- Geometry optimization is not properly tested
- DCPT2 not tested, no case in paper, where cartesian integrals was used (all had D orbitals)

Broken
------

- Self overlap integrals is non-one for contracted  basisfunctions
- No check for singularity in DIIS, H2/STO3G breaks the code if used with DIIS
