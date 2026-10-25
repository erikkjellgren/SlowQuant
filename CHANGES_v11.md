# 4c update v11 (includes v7-v10)

More numba kernels (all with pure-Python fallback; results unchanged, 19 tests pass):
- `ci_sparse_DHF.py`: E_tu application (phi, apply, transition RDMs) in one compiled pass; CI Hamiltonian value
  update compiled (rebuild 0.64 s -> 0.27 s at 12870 determinants); `quad_forms` = <v|H|v> for sparse v on CSR H.
- `generalized_ups_wavefunction_DHF.py`: exact theta diagonal Hessian via quad_forms (46 s -> 13 s at ((4,4),16));
  "auto" uses it while nnz(H) * n_param <= 1e9, else the diag(H) estimate (now vectorized).
- `linear_response/generalized_direct_DHF.py`: exact LR diagonal via quad_forms.
Remaining hot spots are BLAS (dense n^4 x n_det products in the LR sigma), libcint integrals and PySCF ao2mo/J-K;
numba does not help there.
