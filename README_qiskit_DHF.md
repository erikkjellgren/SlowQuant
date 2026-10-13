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
