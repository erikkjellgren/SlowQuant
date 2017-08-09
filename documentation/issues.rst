
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

- Alot of recalculations in R. No efficient way to cache Cython compiled functions found
- Not sure variables are passed around in integral code the right way with respect to Cython
- The SCF seem to have trouble converging if BOMD time steps are above 1.0 a.u.

Need testing
------------

- BOMD not tested at all
- DCPT2 not tested, no case in paper, where cartesian integrals was used (all had D orbitals)

Broken
------

- Self overlap integrals is non-one for contracted  basisfunctions
- No check for singularity in DIIS, H2/STO3G breaks the code if used with DIIS
- Changing Cython directive_defaults seem to make the code break sometimes, not very reproduceable
