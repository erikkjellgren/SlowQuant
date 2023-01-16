
Issues
======

In this section known problems with the code can be found

General
-------

- No checking of input files at all at any stages of the code
- Basisset object has a bad structure
- Loops should be replaced with np.einsum() in many cases
- Documentation of integral code is out-of-date

Performance
-----------

- Alot of recalculations in R, when calculating higher order angular momentum integrals
- Not sure variables are passed around in integral code the right way with respect to Cython
- The SCF seem to have trouble converging if BOMD time steps are above 1.0 a.u.
- All information is stored directly in memory; Integrals scale as N^4 for memory

Need testing
------------

- DCPT2 not tested, no case in paper, where cartesian integrals was used (all had D orbitals)
- Numerical Forces are not tested yet

Broken
------

- Self overlap integrals is non-one for contracted  basisfunctions
- No check for singularity in DIIS, H2/STO3G breaks the code if used with DIIS
- Changing Cython directive_defaults seem to make the code break sometimes, not very reproduceable
