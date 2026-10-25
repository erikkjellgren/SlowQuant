# 4c update v8: faster CI part and LR (includes v7 GIAO speed-up)

Changed files (drop-in, results unchanged):
- `slowquant/unitary_coupled_cluster/ci_sparse_DHF.py`
  - CI Hamiltonian built with vectorized Slater-Condon rules; sparsity pattern and CSR layout built once per
    active space, only the values are recomputed when orbitals change (memory-lean int32 indices).
  - UPS engine: every fUCC exponential (T_k^2 = 0) applied as 2x2 rotations on the connected determinant pairs,
    in place; theta gradient from pair dot products (no full-length sparse products).
- `slowquant/unitary_coupled_cluster/generalized_ups_wavefunction_DHF.py`
  - `_theta_diag_hessian` (L-BFGS preconditioner, recomputed at every point) without the dense ndet x n_param
    matrix. `wf.theta_hessian = "exact" | "diag" | "auto"` (default auto: diag(H)-based estimate when
    nnz(H) * n_param > 2e8; same energies, similar iteration counts in tests).
- `slowquant/unitary_coupled_cluster/linear_response/generalized_direct_DHF.py`
  - G operators as sparse matrices; H G|0> no longer precomputed/stored (H applied to the combined vector per
    sigma call); conjugates precomputed; exact diagonal from H restricted to each operator's support.
- `slowquant/unitary_coupled_cluster/DHF_integral_direct.py`
  - v7 GIAO speed-up; einsums in transformed_operator / transition_fock use BLAS (optimize=True).

Timings (2 cores, N2 6-31G DHF, 1-step OO, fUCCSD generalized):
| | before | v8 |
|---|---|---|
| ((4,4),16) 12870 dets: 10 OO iterations | 536 s | 38 s |
| ((5,5),18) 43758 dets: CI Hamiltonian build | 254 s | 20 s first, 2.3 s per orbital update |
| ((4,4),12) LR shieldings+SSCC | 25 s | 20 s |
| ((4,4),16) LR setup | 51 s | 8.8 s |

Tests: tests/test_DHF_integral_direct.py + tests/test_qiskit_DHF_circuit.py: 18 passed.
Memory note: the explicit CI Hamiltonian has nnz ~ ndet * (1 + n_s + n_d); for ~1e5 determinants and more it needs
several GB.
