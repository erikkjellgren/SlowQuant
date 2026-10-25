# 4c update v14 (includes v7-v13.1)

New integral backends (DHF_integral_direct.py), all exact J/K unless stated, GIAO integrals exact:

1. Cholesky pivots: `DHFCholesky(mol, tol=1e-8, store=False, blocks="LS"|"S")`
   - Pivot search keeps the partial vectors only on the significant pairs (diag >= tol); afterwards only the pivots
     and the pivot metric S = G[P, P] are kept. MO integrals: (12|34) = (12|P) S^-1 (P|34) from recomputed pivot
     columns (one integral pass per eri call if both half-transformed sides fit in max_memory/4, else two).
   - Same integrals as the stored vectors (1e-9). J/K exact (cd_jk is ignored), CD-GIAO not used.
2. Density fitting: `DHFDensityFit(mol, auxbasis=None, beta=2.0, blocks="LS"|"S", df_jk=False)`
   - One scalar auxiliary basis for LL and SS densities (int3c2e_spinor, int3c2e_spsp1_spinor); default even-tempered
     aug_etb(beta) (smaller beta = larger set). blocks="S": only (SS|SS) fitted, the rest exact.
   - df_jk=False: exact J/K, fitted MO integrals integral-direct in aux-shell batches (no 3-index storage).
   - df_jk=True: fitted J/K (stores n_aux x n_blocks x n2c^2 x 16 B).
3. run_4c_calculation.py: "backend": "cholesky" | "cd_pivots" | "df" | "direct", plus "cd_blocks", "df_auxbasis",
   "df_beta", "df_jk". example_4c_backends.py: the example script with a BACKEND switch.
4. Test: test_cholesky_pivots_and_density_fitting (20 tests pass).

Measured (HBr dyall-v2z, (2,2) in 6 spin orbitals, one-step OO tol 1e-7, LR tol 1e-6; reference = stored CD, cd_jk=False):
- DF all blocks, beta 1.6 (565 aux): dE = -6.8e-7 Eh, d sigma(H) = -8e-5 ppm, d sigma(Br) = -3.0e-3 ppm.
- DF (SS|SS) only: MO integral error 2e-10 (beta 2.0) / 1e-11 (beta 1.6); DHF energy error from fitted J/K 2e-10 Eh.
- Cholesky pivots: MO integrals equal to stored CD (1e-9).

5. (v14.1) DHFHybrid(jk_backend, eri_backend): J/K from one backend, MO integrals/GIAO from the other, e.g. DF (SS|SS) J/K
   + Cholesky-pivot MO integrals. run script: "jk_backend": "df_ssss". HCl 6-31G: identical to pivots-only (1e-10 Eh, 1e-5 ppm).

6. (v14.2) Cholesky/DF J/K temporaries limited to max_memory/8 (were up to ~max_memory).
