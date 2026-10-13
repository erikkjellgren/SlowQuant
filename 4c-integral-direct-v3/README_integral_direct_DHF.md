# Integral-direct 4c OO-UPS: RMB-GIAO shieldings and SSCC (v3)

Base: `4-component` branch at commit fd7616e ("Response + RMB GIAO shieldings corrected").
Upstream changes since 423a12d do not touch any file here, so the files can also be copied onto a newer checkout.

## Install
```bash
cd SlowQuant                      # your checkout of the 4-component branch
git apply --check integral_direct_DHF_v3.patch && git apply integral_direct_DHF_v3.patch
# or: unzip and copy the files in slowquant/, tests/ and the examples over the checkout
python -m pytest tests/test_DHF_integral_direct.py               # 8 fast tests
python -m pytest tests/test_DHF_integral_direct.py -m slow       # H2 dyall-v2z regression
```

## Files
| File | What |
|---|---|
| `unitary_coupled_cluster/DHF_integral_direct.py` | `DHFDirect` (exact direct J/K, out-of-core MO integrals; options `screen_tol`, `ssss`), `DHFCholesky` (pivoted Cholesky of the Dirac-Coulomb operator), `giao_integrals`, `DirectState` |
| `unitary_coupled_cluster/ci_sparse_DHF.py` | `ActiveCI`: sparse CI Hamiltonian, operator application, (transition) RDMs |
| `unitary_coupled_cluster/linear_response/generalized_direct_DHF.py` | `LinearResponseDirect`: matrix-free E2/S, RMB-GIAO operator, `get_properties(..., nuclei=)` |
| `unitary_coupled_cluster/linear_response/solvers_DHF.py` | Paired Davidson / weighted subspace linear solver |
| `unitary_coupled_cluster/generalized_ups_wavefunction_DHF.py` (patched) | `run_wf_optimization_direct_DHF`, `set_integral_backend` |
| `tests/test_DHF_integral_direct.py` | brute-force checks (LiH) + H2 regression |
| `example_H2_cholesky.py` | H2 dyall-v2z ((1,1),6) RMB-GIAO shieldings, Schwarz 1e-10 + CD-MO 1e-8 |
| `example_TlH_integral_direct.py` | TlH dyall-cv4z, Schwarz 1e-10, exact MO integrals |

## Integral backends
```python
from slowquant.unitary_coupled_cluster.DHF_integral_direct import DHFDirect, DHFCholesky
WF.set_integral_backend(DHFDirect(mol, screen_tol=1e-10))                          # exact, Schwarz 1e-10
WF.set_integral_backend(DHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10))  # + Cholesky MO integrals
WF.set_integral_backend(DHFCholesky(mol, tol=1e-4, blocks="S", eri_tol=1e-6, screen_tol=1e-9))  # + (SS|SS) J/K by CD
```
Call before `run_wf_optimization_direct_DHF`; `LinearResponseDirect` uses the same backend.

| Option | Meaning |
|---|---|
| `screen_tol` | Schwarz/density screening of the direct J/K (PySCF `direct_scf_tol`, default 1e-13) |
| `ssss` | where (SS\|SS) is kept: `full`; `mf` (not in active MO integrals / (N u\|v w)^B); `ground` (only in ground-state F^I, F^A); `nogiao`, `noresp` (diagnostics); `none` |
| `DHFCholesky(tol, blocks="LS")` | J/K and MO integrals from CD vectors of the LL+SS pair metric |
| `cd_jk=False` | exact J/K, CD only for MO integrals (active integrals, response tensors) |
| `blocks="S"` | only (SS\|SS) from CD vectors, LLLL/SSLL exact |
| `eri_tol` | separate LL+SS decomposition for the MO integrals |

Memory of CD vectors: n_K x 2 n2c^2 x 16 B (`blocks="LS"`), n_K x n2c^2 x 16 B (`blocks="S"`).
HBr dyall-cv2z (n2c = 212): n_K = 556 / 729 / 931 / 1125 for tol 1e-4 / 1e-6 / 1e-8 / 1e-10 (0.8 - 1.6 GB).
TlH dyall-cv4z (n2c = 810), estimate: LS ~4500 vectors at 1e-6 (~95 GB, too large for 64 GB);
SS only ~1500 vectors at 1e-6 (~16 GB).

## Response right-hand sides
`get_properties(shieldings, sscc, tol, nuclei=None)`: B and the nuclear moments of `nuclei` (default all) are solved together.
Where both response vectors exist the error-cancelling expression -(V_A^H X_B + X_A^H V_B - X_A^H E2 X_B) is used
(error quadratic in the residual), otherwise -V_A^H X_B (linear). Shieldings of all nuclei need only X_B,
an SSCC K-L needs K or L in `nuclei`. With fewer right-hand sides use a tighter `tol` (LiH: linear formula at tol 1e-5
gave 9e-3 ppm error, error-cancelling 5e-6 ppm).

## Accuracy and timings (HBr dyall-cv2z, fUCCSD ((1,1),4), 2 cores)
Fixed (exactly optimized) wave function; errors vs fully exact integrals. Reference: sigma(Br) = 2961.00137 ppm,
sigma(H) = 36.65235 ppm, reduced SSCC = -4.29485 Hz. Times in s (GIAO pass 844 s counted for all runs with exact GIAO).

| Run | n_K | dE (Eh) | dsigma Br | dsigma H | dJ (Hz) | J/K per call | Total LR |
|---|---|---|---|---|---|---|---|
| exact | - | 0 | 0 | 0 | 0 | 73 | 2729 |
| Schwarz 1e-9 | - | 8.8e-8 | -2.4e-4 | 8e-7 | 1.4e-5 | 51 | 2226 |
| Schwarz 1e-10 + CD-MO 1e-10 | 1125 | 2.0e-9 | -5.9e-5 | -4e-7 | 2e-8 | 52 | 2161 |
| Schwarz 1e-10 + CD-MO 1e-8 | 931 | 9e-10 | -5.7e-5 | -4e-7 | -2e-8 | 53 | 2150 |
| Schwarz 1e-9 + CD-MO 1e-6 | 729 | 7.3e-8 | -2.7e-4 | -2e-8 | 1.1e-5 | 49 | 2044 |
| Schwarz 1e-9 + CD-MO 1e-4 | 556 | -1.5e-7 | 3.0e-3 | 7.5e-5 | 2.2e-4 | 50 | 2031 |
| exact J/K + CD-MO 1e-4 | 556 | -2.4e-7 | 3.3e-3 | 7.4e-5 | 2.1e-4 | 66 | 2383 |
| full CD 1e-4 | 556 | 1.3e-4 | 9.6e-2 | -1.6e-4 | -7.7e-3 | 57 | 2191 |
| SS-CD 1e-4 | 107 | -1.4e-4 | -6.5e-4 | 9e-6 | -5e-6 | 51 | 2156 |
| Schwarz 1e-9 + SS-CD 1e-4 + CD-MO 1e-6 | 107+729 | -1.4e-4 | -6.0e-4 | 8.7e-6 | -6.5e-7 | 49 | 2034 |
| SSSS mf | - | -8.8e-9 | -1.3e-5 | 3e-8 | 7e-7 | 64 | 2380 |
| SSSS ground | - | -8.8e-9 | -5.7e-2 | -4.7e-5 | 1.0e-3 | 48 | 1615 |

MO integrals (3 response tensors): exact 272 s, CD 15-22 s. OO iteration: exact ~160 s (64 % in the MO transformation), CD-MO ~40 s.
Timing noise on this VM ~10 %.
