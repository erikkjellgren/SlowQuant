# 4c update v15 (includes v7-v14.3)

1. CD-GIAO from Cholesky pivots (`DHFCholesky(store=False)`, blocks "LS"): the W = S^-1 G[P,:] contractions
   (U_k, w_d, wP, omP) from one pass over the pivot columns, J from a second pass; no stored vectors. Same result as
   stored CD-GIAO (1e-9 rel.; HCl cc-pVDZ shieldings identical). Used automatically in the LR for pivot backends.
2. Two-level LR preconditioner: `LinearResponseDirect(..., two_level=True)` (run script "lr_two_level"): diagonal for
   orbital rotations, exact active-space block (E2_GG - omega S_GG, CI part only, LU-factorized) for G.
   N2 6-31G (4,4): 38 -> 32 trial vectors. Off by default (small gain for small active spaces; building the block
   costs ~n_G applications of the CI Hamiltonian).
3. Exact J/K for many densities by GEMM (`_jk_gemm`, `DHFDirect.jk_gemm_min`): correct but slower than PySCF here
   (HBr 24 densities: 151 s vs 80 s; no integral symmetry used) -> off by default (jk_gemm_min = 10**9).
4. run_4c_calculation.py: "nuclei" accepts atom indices or element symbols (e.g. ["O"]); "lr_b_only", "lr_two_level".
