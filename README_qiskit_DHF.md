# Hardware-compatible 4c circuit wave function + integral-direct LR (v4)

Base: `4-component` branch (fd7616e) + v3 integral-direct files + `slowquant/qiskit_interface` from `generalized_orbitals`.

## Install
Copy the files below into the branch (or `git apply qiskit_DHF_v4.patch` on a tree that already has the v3 files and the
`generalized_orbitals` qiskit_interface). Requires qiskit 1.x, qiskit-nature, qiskit-aer, numpy >= 2.0 (`np.bitwise_count`).
```bash
python -m pytest tests/test_qiskit_DHF_circuit.py             # 3 fast tests
python -m pytest tests/test_qiskit_DHF_circuit.py -m slow     # circuit OO + LR vs classical integral-direct code
```

## Files
| File | What |
|---|---|
| `qiskit_interface/fast_pauli.py` (new) | Pauli layer: vectorized Jordan-Wigner (qiskit convention), RDM1/RDM2 and Hamiltonian as sparse linear maps of Pauli expectation values, bitmask QWC grouping, `PauliEstimator` (exact statevector + Walsh-Hadamard, or QI sampler with one batched device call and QI mitigation), commutator / double-commutator Pauli algebra |
| `qiskit_interface/generalized_circuit_wavefunction_DHF.py` (new) | `GeneralizedWaveFunctionCircuitDHF`: DHF orbital spaces incl. positronic, measured RDMs, integral-direct energy/orbital gradient, parameter-shift gradient and Hessian, `run_wf_optimization_2step_DHF` |
| `qiskit_interface/linear_response/generalized_direct_DHF_circuit.py` (new) | `LinearResponseCircuitDirect`: integral-direct naive LR (RMB-GIAO shieldings, SSCC, excitation energies) for the circuit wave function |
| `qiskit_interface/generalized_interface.py` (patched) | fast JW `op_to_qbit`, vectorized Pauli evaluation, `sampler_distributions` (cached measurement circuits, parameter sweeps), postselection call fixed |
| `qiskit_interface/util.py` (patched) | `Clique` with integer bitmasks (same API), `labels_to_masks`, `pauli_values_from_distr` |
| `tests/test_qiskit_DHF_circuit.py`, `example_qiskit_DHF_H2.py` | tests, example |

## Usage
```python
QI = QuantumInterface(Sampler(), "fUCCSD", JordanWignerMapper(),
                      ansatz_options={"n_layers": 1, "is_spin_conserving": False}, shots=None)
WF = GeneralizedWaveFunctionCircuitDHF(((1, 1), 4), C_dhf, mol, QI, include_active_kappa=True)
WF.set_integral_backend(DHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10))   # optional
WF.set_thetas_initial(theta_real, theta_imag)        # Cartesian -> (r, phi); or WF.set_thetas(r, phi)
WF.run_wf_optimization_2step_DHF(tol=1e-6, theta_method="l-bfgs-b")   # "trust-exact": Newton with parameter-shift Hessian
res = LinearResponseCircuitDirect(WF, "SD").get_properties(shieldings=True, sscc=True, tol=1e-5)
```
`shots=None` with a local sampler: exact statevector path (ideal simulation). With `shots` (or `exact=False`): sampler path, as on hardware.
Theta parameters are the QI circuit parameters (norm r and phase phi per excitation, `QI.param_names` order).

## What is measured (everything else is classical)
| Quantity | Measured | Rank |
|---|---|---|
| RDM1, RDM2 (energy, orbital gradient, orbital step, qq block of E2) | once per macro iteration | 2-body |
| Energy at shifted parameters (gradient, Hessian) | sum_k 2R_k points (gradient), sum_k 2R_k + sum_{k<l} 4 R_k R_l (Hessian) | 2-body |
| K1 = <[E_tu, G_j]>, K2 = <[e_tuvw, G_j]> (qG blocks, active part of property gradients) | once | <= 3-body |
| A_GG (symmetrized), B_GG, S_GG | once | <= 4-body |
Each set is one union of Paulis grouped into qubit-wise commuting cliques and sent in one sampler call (shot budget 2e7 per call).
G^+ quantities follow from <[O, G^+]> = -<[O^+, G]>^*.

## Parameter-shift rules (Wierichs et al., Quantum 6, 677 (2022))
For E(x) with integer frequencies 1..R (R from `QI.grad_param_R_r` / `grad_param_R_phi`):
E'(0) = sum_{mu=1}^{2R} E(x_mu) (-1)^{mu-1} / (4R sin^2(x_mu/2)), x_mu = (2mu-1)pi/(2R);
E''(0) = -(2R^2+1)/6 E(0) + sum_{mu=1}^{2R-1} E(x~_mu) (-1)^{mu-1} / (2 sin^2(x~_mu/2)), x~_mu = mu pi/R;
off-diagonal: product of the first-derivative rules in both parameters.

## Optimization (`run_wf_optimization_2step_DHF`)
Per macro iteration: (1) theta optimization at fixed orbitals on the device (any scipy method; L-BFGS-B with parameter-shift
gradient, or trust-exact/trust-krylov with the parameter-shift Hessian); (2) RDMs measured once; (3) orbital optimization
at fixed RDMs, purely classical and integral-direct (L-BFGS e-e minimized, diagonal Newton e-p maximized). Converged when the
orbital gradient after (1) is below tol.

## Validation (LiH STO-3G DHF, ((1,1),4))
Parameter-shift gradient/Hessian vs finite differences 1e-8; sampler vs exact path 1e-14; circuit OO vs classical OO 3e-12 Eh;
circuit LR vs classical LR (same orbitals): E2 2e-5 rel (from theta convergence), shieldings <= 1e-6 ppm, SSCC 1e-6 Hz.

## Cost (ideal simulation, 2 cores, LiH 6-31G ((2,2),8): 104 parameters, 1941 RDM Paulis)
RDMs + energy 0.4 s; parameter-shift gradient (560 circuits) 79 s; LR active-space quantities 135 s (52 G operators, 22816 Paulis), 0.7 GB.
The GG block scales as 256 n_G^2 n_Pauli(H) classical work and needs 4-body Paulis on the device: practical up to about 8-10 active spin orbitals.

## Notes
- `qiskit_interface/generalized_circuit_wavefunction.py` and the qiskit `generalized_naive` LR from `generalized_orbitals` import
  `generalized_hamiltonian_full_space`, which does not exist in the 4-component `generalized_operators.py`. They are not used by the new files.
- The generalized excitation circuits in `generalized_custom_ansatz` are written for Jordan-Wigner; other mappers give wrong states
  (particle number not conserved). The Pauli layer itself supports any mapper.

## v5 additions

- **Batched statevector simulator** (`fast_pauli.CircuitSimulator`), used automatically by `PauliEstimator` on the exact path (`exact=True`).
  - It is checked against qiskit `Statevector` when the circuit is built, and falls back to qiskit if they disagree.
  - Energies of many parameter sets (parameter-shift gradients and Hessians) are evaluated in one numpy pass.
  - About 25–50× faster than v4 on 4 qubits.
- **`symmetrize_qG` option** in `LinearResponseDirect` and `LinearResponseCircuitDirect`. The default is `False`, which is the v3/v4 behaviour.
  - `False`: plain qG blocks ⟨[G†,[H,q]]⟩. This is the exact Hessian for exp(−κ)exp(S)|0⟩, with the orbital rotation applied last.
  - `True`: symmetrized ½⟨[G†,[H,q]] + [[G†,H],q]⟩, identical to `generalized_naive_DHF` (E2 to 1e-11, including e–p rotations).
  - With `True`, the circuit LR measures the extra Pauli expectation values ⟨a†_u [a_t,G†]⟩, ⟨a†_u a†_w a_x [a_t,G†]⟩ (and the analogues with a_u, a†_t). This costs extra measurements on hardware.
- **Davidson** returns its current roots, with a warning, when the subspace cannot be extended (exactly degenerate Kramers roots at very tight tolerance).
- **Example** `example_qiskit_DHF_H2_paramshift_hessian.py`: ideal simulator with the parameter-shift θ Hessian (`theta_method="trust-exact"`).

## v6 (speed update)

- **Classical UPS** (`ci_sparse_DHF.UPSEngine`): each UPS factor is applied in closed form, U = I + (sin r/r)A + ((1−cos r)/r²)A², on sparse CI vectors (no expm). The θ gradient uses one state build and adjoint differentiation; L-BFGS-B gets the energy and gradient together (`jac=True`). F^A is built lazily.
- **One-step OO** `run_wf_optimization_onestep_direct_DHF(tol, maxiter, ...)`:
  - One L-BFGS run over the e–e κ and θ together; e–p κ take a diagonal Newton step (maximized), as in the 2-step driver.
  - Each point costs one J/K pass with 2 densities (inactive + active) plus one (N a|a a). The 2-step needs a separate J/K for F^A after every θ relaxation.
  - θ start: optimized at the initial orbitals (no J/K). θ preconditioner: h_k = 2(⟨V_k0|H|V_k0⟩ − E⟨V_k0|V_k0⟩).
  - Same energy as the 2-step (≤1.3e-9 at tol 1e-6). `run_4c_calculation.py` uses it by default (`oo_method="1step"`).
- **J/K**: `hermi=1` for density-matrix builds. Orbital rotations use `expm_antihermitian` (one `eigh`).
- **Circuit, exact path**:
  - `fast_pauli.FermionSimulator` simulates the fUCC circuit directly in the determinant space with JW signs. It is checked against qiskit at 2 random points.
  - `energy_gradient` returns E and the exact θ gradient by adjoint differentiation. The Hessian applies the first-order shift rule to these gradients.
  - Set `WF.parameter_shift = True` to force the parameter-shift rules (hardware-style), e.g. for testing.
- **Sampler**: counts are vectorized (`BitArray` → `np.unique`).
- **Bug fix** in `operators_circuits._double_excitation_efficient_generalized`. When `l<j` or `k<i`, creation and annihilation indices were swapped, which gave a different operator. Those orderings now use an exact Pauli-rotation circuit, validated against the fermionic exp(θT − θ*T†) to 1e-13. **Circuit results for affected excitations change.**
- **`run_4c_calculation.py`** runs the full workflow with all settings printed, timings and an npz with `c_mo` + θ. It runs:
  - PySCF DHF, NMR (RMB-GIAO), SSC (RKB) and TDA;
  - this code's OO-UPS, LR excitation energies, RMB-GIAO shieldings and RKB SSCC;
  - a final comparison table.

| Timing (2 cores) | v5 | v6 |
|---|---|---|
| HF (4c) OO | 68 s | 44 s |
| LiH 6-31G ((2,2),8) OO, 2-step → 1-step | 22.5 s, 71 J/K | 2.9 s, 39 J/K |
| H4 6-31G ((2,2),8) OO, 2-step → 1-step | 33.5 s, 138 J/K | 4.6 s, 84 J/K |
| H2 dyall-v2z circuit OO (exact) | 6.7 s | 2.4 s |
| Exact θ gradient, 10 qubits, 168 parameters | 29 s | 0.013 s |

**Output format (v6).** All iterative steps print aligned, fixed-width tables with a title, the settings, columns with units, and a summary. This covers the OO drivers (1-step, 2-step, circuit), the Davidson and response solvers, excitation energies, shieldings, SSCC and the run-script comparisons and timings. The formatting lives in `slowquant/unitary_coupled_cluster/printing.py` (`IterationTable`, `value_table`, `roots_table`, `info_block`).

**Fix: variational collapse in the 4c OO (e-p rotations).**
- e-p rotations are maximized with a diagonal Newton step −g/h. The diagonal estimate h ignores the coupling to θ: with θ relaxed, the true curvature is H_κκ − H_κθ H_θθ⁻¹ H_θκ, which is more negative than h.
- For weakly occupied active orbitals (active–positronic "ap" rotations) |h| can be more than 2× too small. Each step then overshoots the maximum, the gradient grows geometrically, E drops (overshooting a maximum lowers E, so the line search accepts it), and the run ends in collapse.
- Both 4c drivers now use h_i = min(h_diag, Δg_i/Δκ_i) per e-p rotation, the measured secant curvature (`_ep_secant`).
- Verified on HF (F dyall-cv3z, H aug-cc-pVTZ uncontracted, CAS((2,2),8), random θ ±0.05): the old 2-step diverged from iteration 119; the fixed 2-step and 1-step stay stable.
