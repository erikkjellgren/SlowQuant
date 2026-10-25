# 4c update v7: faster GIAO two-electron step

Changed file: `slowquant/unitary_coupled_cluster/DHF_integral_direct.py` (`giao_integrals`; everything else as v6).

- Time-reversal (Kramers) symmetry of the GIAO derivative integrals:
  all classes (ij|kl) = s_k s_l (ij|p(l) p(k)) -> only k-block >= l-block computed, rest by permutation;
  derivative classes (g1, spgsp1, g1spsp2, spgsp1spsp2) also (ij|kl) = -s_i s_j (p(j) p(i)|kl) -> only j < i computed.
  This is a property of the AO integrals only; densities, RDMs and MO coefficients are used in full (no symmetry assumed).
- Contractions fused: J/K and (N u|v w)^B terms grouped into 4-6 passes over each integral block, no full transposed copy.
- Results identical to v6 (max rel. difference 1e-15 for W and (N u|v w)^B); tests/test_DHF_integral_direct.py: 11 passed.

Timings of the GIAO step (2 cores): HCl cc-pVTZ (n2c 96) 103 s -> 59 s; HBr dyall-v2z/cc-pVDZ (n2c 176) 512 s -> 357 s.

Install: copy the file over the existing one (or unzip the package over the branch root).
