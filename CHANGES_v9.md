# 4c update v9: Cholesky-based GIAO derivative integrals (CD-GIAO)

Changed files: `DHF_integral_direct.py`, `linear_response/generalized_direct_DHF.py`, `run_4c_calculation.py`,
`tests/test_DHF_integral_direct.py` (+1 test). Everything from v7/v8 included.

- `DHFCholesky` records its pivots (`direct.piv`).
- `giao_integrals_cd`: GIAO two-electron derivatives from the derivative of the RI-equivalent pivoted CD
  (Nottoli, Burger, Stopkowicz, Gauss, Lipparini 2022):
  G^B ~ A W + W^+ A^+ - W^+ A_PP W, A = derivative integrals at the pivot columns only, W = G_PP^-1 G_P,: = L^-+ V^*.
  Only n^2 x (pivot shell pairs) derivative integrals; time-reversal symmetry used for the row pairs; badc columns
  from the id slices; J^B, K^B (low-rank densities) and (N u|v w)^B from vector contractions.
- Used automatically in the LR (shieldings) when the backend is `DHFCholesky` with its vectors still in memory
  (`direct.cd_giao = False` switches it off). With `cd_jk=False` W overwrites the vectors in place (no extra
  memory) and they are released afterwards. `run_4c_calculation.py` therefore keeps the vectors until the GIAO step
  when shieldings are requested (peak memory = LR-setup peak, as before).
- Accuracy: error decreases with the CD threshold for realistic basis sets (rel. ~1e-8 at tol 1e-8);
  HCl cc-pVTZ ((2,2),6) shieldings identical to 4 decimals (H 32.2032, Cl 1113.836 ppm).
  Not for minimal basis sets (derivative pair functions outside the pivot space: ~1e-4 errors for STO-3G).
- GIAO step (2 cores): HCl cc-pVTZ 61 s -> 26 s; HBr v2z ~3x faster than v8 exact. Gain grows with basis size.

If your own scripts call `direct.release()` right after the LR setup, move it after `get_properties`
(otherwise the exact GIAO path is used).

## v9.1 (memory trim of CD-GIAO)
- K-intermediates X^+ W_P built per pivot group / per block of Q instead of stored for all pivots.
- Pivot-column block A built for the L and S row blocks one after the other (half the size); no conjugated copies.
- Extra memory of the CD-GIAO step ~ 32 k n2c (r_inactive + n_active) + 3 n2c^2 64 x 16 B + ~2 GB
  (HAt dyall-ae3z/aug-cc-pVTZ: ~12 GB on top of the 42 GB vectors; set direct.cd_giao = False for the old peak).
