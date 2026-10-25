# 4c update v12 (includes v7-v11)

One-time CI setup compiled with numba (pure-Python fallback kept; results unchanged, 19 tests pass):
- `ci_sparse_DHF.py`, `ActiveCI._hamiltonian_pattern`: Slater-Condon sparsity pattern generated row by row in a
  parallel numba kernel (`_pattern`, uses all cores). ((4,4),16), 12870 determinants: 4.0 s -> 1.1 s on 2 cores.
  Used when the determinant space is the full active space (always for generalized fUCCSD); otherwise numpy path.
- `UPSEngine`: excitation pair lists from a numba kernel (`_string_pairs`); dense-free, sparse T_k matrices only
  built on demand (`_mat(k)`) for non-pair fallbacks. Build 2.5 s -> 0.3 s at ((4,4),16).
- `generalized_ups_wavefunction_DHF.py`: uses `eng._mat(k)` in the non-pair diagonal-Hessian fallback.
Checks: H vs v7 <= 3e-13; UPS state 1e-16, theta gradient 3e-13.
First call compiles the kernels (~5 s, cached afterwards in __pycache__).
