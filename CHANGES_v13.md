# 4c update v13 (includes v7-v12)

Results unchanged (differences <= 1e-15 where deterministic); 19 tests pass.

1. Parallel numba kernels (all cores; numba threads = NUMBA_NUM_THREADS):
   `quad_forms` (exact theta/LR diagonals; 1.9x on 2 cores), `_h_fill`, `_e_scatter`, `_e_gather` (row-sorted,
   race-free), UPS rotations/gradient dots for parameters with > 20000 determinant pairs.
2. LR solver (`generalized_direct_DHF.py`):
   - Cost = number of sigma calls (one J/K integral pass each; direct J/K for 2 vs 18 densities: 3.6 vs 7.3 s).
     All right-hand sides (3 B + 3 per nucleus) are solved as one block, as before.
   - Trial vectors per J/K pass now from mol.max_memory (16..128, `max_batch()`; was fixed 16): fewer passes
     when there are many nuclei.
   - `get_properties(..., b_only=None)`: B-only solve (3 RHS, tol/100, linear expression); automatic only when
     3 + 3 natm > max_batch() (shieldings only).
   - Better preconditioners tested (exact diagonal, exact active-space block): 0-20 % fewer iterations, not added.
   - Note: the second-order shielding expression makes errors quadratic in the residual; lr_tol = 1e-4 changes
     shieldings by < 1e-4 ppm and saves ~25 % iterations (default left at 1e-5).
3. CD-GIAO contractions as GEMM (`giao_integrals_cd`): K terms from one contraction (second block = Hermitian
   conjugate), V_b reused from stored U_k, one GEMM for both g terms. HBr dyall-v2z 88.8 -> 72.9 s,
   HCl cc-pVTZ 31.2 -> 28.2 s. The exact GIAO path already used GEMM only.

4. (v13.1) DirectState reuses the (N a|a a) MO integrals when the orbitals are unchanged (the reference
   printout and the first OO point no longer compute them twice; saves one out-of-core transformation with DHFDirect).
