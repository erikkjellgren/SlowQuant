# 4c update v16 (includes v7-v15)

1. b_only is no longer the default (get_properties(..., b_only=False); run script "lr_b_only": False).
2. Two-level LR preconditioner with a cheap exact Hessian block: `LinearResponseDirect(..., two_level=n)`
   (run script "lr_two_level": n, e.g. 200). Exact blocks for (a) the active-space parameters (CI part only) and
   (b) the n orbital rotations with the smallest diagonal. Block (b) is built with the normal sigma code but with J/K
   replaced by a projection onto the orbitals O involved (+ active): v = S C_O [(OO|OO) contraction] C_O^+ S, which
   is exact on the O block (one (OO|OO) MO-integral call, no J/K pass). Diagonal elsewhere; blocks LU-factorized.
   - Block vs exact Hessian: 2e-10 (HF 6-31G), 5e-10 (N2 6-31G).
   - HF 6-31G (2,2), lr_tol 1e-6: 14 -> 5 iterations (106 -> 43 trial vectors with J/K); shieldings equal to 1e-9 ppm.
   - N2 6-31G (4,4), lr_tol 1e-6: 14 -> 11 iterations (117 -> 93); shieldings equal to 6e-7 ppm.
   - Cost of the block: sigma without J/K for 2n unit vectors (25-135 s here); pays off when J/K dominates
     (large heavy-element molecules). Off by default.
3. Measured: CD-GIAO from Cholesky pivots vs exact GIAO, HBr dyall-v2z: 246 s vs 1552 s (6.3x), rel. diff 8e-8 (W),
   1e-6 (g).
