# 4c update v10 (includes v7-v9.1)

Changed files: `ci_sparse_DHF.py`, `DHF_integral_direct.py`. Results unchanged; 19 tests pass.

- UPS state and theta-gradient sweeps compiled with numba (`pip install numba`); pure-Python fallback without numba.
  ~3x faster UPS part of the OO loop.
- CI Hamiltonian value update: singles vectorized (one dense O @ W product per determinant chunk).
- Cholesky build: pivots selected on the small candidate block (no rank-1 updates of the full column block);
  earlier vectors subtracted only from the selected columns. HBr v2z/cc-pVDZ (718 vectors): 67 s -> 41 s,
  same pivots. Remaining cost: integrals + the O(n_pair k^2) BLAS step.
