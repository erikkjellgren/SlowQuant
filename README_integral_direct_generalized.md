# Integral-direct, hardware-compatible generalized (GHF) OO-UPS and linear response

Base: `generalized_orbitals` @ b132e1f. Same structure, approximations and options as the 4c v3/v4 code.

## Files

New
- `slowquant/unitary_coupled_cluster/integral_direct.py`: `GHFDirect` and `GHFCholesky` integral backends, plus `DirectState`, `orbital_gradient`
- `slowquant/unitary_coupled_cluster/ci_sparse.py`: `ActiveCI` (sparse active-space CI) and `ladder` (N±1 spaces)
- `slowquant/unitary_coupled_cluster/linear_response/solvers.py`: paired Davidson eigenvalue and linear-equation solvers
- `slowquant/unitary_coupled_cluster/linear_response/generalized_direct.py`: `LinearResponseDirect`
- `slowquant/qiskit_interface/fast_pauli.py`: Pauli layer, QWC grouping, estimator and batched statevector simulator
- `slowquant/qiskit_interface/generalized_circuit_wavefunction_direct.py`: `GeneralizedWaveFunctionCircuitDirect`
- `slowquant/qiskit_interface/linear_response/generalized_direct_circuit.py`: `LinearResponseCircuitDirect`
- tests: `tests/test_generalized_integral_direct.py`, `tests/test_qiskit_generalized_direct.py`
- examples: `example_generalized_integral_direct.py`, `example_qiskit_generalized_direct.py`

Modified
- `generalized_ups_wavefunction.py`
  - Energy, RDMs and gradients are integral-direct.
  - New methods: `set_integral_backend`, `run_wf_optimization_direct`.
  - The old methods keep their names and signatures. `h_mo`/`g_mo` still exist, but are only built if called.
- `generalized_integral_manager.py`
  - `orig == None` → `orig is None` (crashed for numpy arrays).
  - `magnetic_field_H` is now cached.
- `qiskit_interface/generalized_interface.py`, `qiskit_interface/util.py`: fast operator mapping (same as 4c v4).

The old `GeneralizedWaveFunctionCircuit` and the naive LR classes are unchanged.

## Wave function (classical)

```python
WF = GeneralizedWaveFunctionUPS(cas, C, mol, "fUCCSD", do_x2c=False, ansatz_options=..., include_active_kappa=True,
                                ecp=False, B=None, orig=None)
WF.set_integral_backend(GHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10))   # optional
WF.run_wf_optimization_direct(tol=1e-6)        # or the old run_wf_optimization_1step / 2step (now integral-direct)
```

**One-electron Hamiltonian.** It is taken from the wave function: T+V (+ECP), the X2C h_core, or the finite-field H(**B**).
`set_integral_backend` puts it on the backend. You do not need to pass `h_ao` yourself.

**Backend options**

| Option | Meaning |
|---|---|
| `GHFDirect(mol, screen_tol=1e-13, incore_memory=256)` | Exact. J/K by direct SCF with Schwarz/density screening at `screen_tol`. MO integrals from the in-memory 8-fold AO integrals if they need less than `incore_memory` MB, otherwise out-of-core. |
| `GHFCholesky(mol, tol, span=1e-2, cd_jk=True, screen_tol=...)` | Pivoted Cholesky decomposition of the Coulomb operator. CD is used for the MO integrals (N a\|a a), (N x\|v w), ... |
| `cd_jk=True` | CD also for J/K. |
| `cd_jk=False` | Exact screened J/K, with CD only for the MO integrals ("CD MO"). |

## Linear response

```python
LR = LinearResponseDirect(WF, "SD", symmetrize_qG=True)
LR.calc_excitation_energies(n_roots=6, tol=1e-6)        # Davidson
LR.get_oscillator_strength()                             # default int1e_r; or pass (3, nao, nao) / (3, 2nao, 2nao)
LR.get_polarizability(omega=0.0)                         # 3x3, error-cancelling expression
LR.explicit()                                            # full E2, S (testing)
LR.solve_response(V, omega)                              # (E2 - omega S) X = V
```

**Operators**
- q: `kappa_no_activeactive_spin_idx`, screened.
- G: active S, D, T, ... excitations.

**`symmetrize_qG`**
- `True` (default): the qG blocks are ½⟨[G†,[H,q]] + [[G†,H],q]⟩. This matches `generalized_naive` (E2 agrees to 1e-14).
- `False`: plain ⟨[G†,[H,q]]⟩. This is the exact Hessian for exp(−κ)exp(S)|0⟩, as in the 4c code.
- The difference is ½⟨0|H[q,G†]|0⟩. It is computed once, using (N−1)/(N+1)-electron active-space intermediates.

## Quantum-hardware compatible circuit version

```python
QI = QuantumInterface(sampler, "fUCCSD", JordanWignerMapper(), ansatz_options=..., shots=None or N, ISA=...)
WF = GeneralizedWaveFunctionCircuitDirect(cas, C, mol, QI, do_x2c=False, include_active_kappa=True, ecp=False,
                                          B=None, orig=None, direct=GHFCholesky(...), exact=None)
WF.run_wf_optimization_2step_direct(tol=1e-6, theta_method="l-bfgs-b")
LR = LinearResponseCircuitDirect(WF, "SD", symmetrize_qG=True)
```

**What comes from the device.** Only Pauli expectation values of the ansatz circuit:
- energies
- parameter-shift gradients: `theta_gradient`
- the parameter-shift Hessian: `theta_hessian`, used by `theta_method="trust-exact"`, `"trust-krylov"`, `"newton-cg"`
- RDMs
- the LR active-space tensors (and, with `symmetrize_qG=True`, ⟨a†_u [a_t,G†]⟩ and ⟨a†_u a†_w a_x [a_t,G†]⟩)

**What is classical.** Everything else, integral-direct. The orbital step uses the measured RDMs, kept fixed, and costs no quantum time.

**`exact`**
- `None`: statevector if `QI.shots is None` and not ISA; sampler otherwise.
- `True`: ideal. Uses the batched numpy statevector simulator, checked against qiskit when the circuit is built.
- `False`: sampler path (QWC cliques, batched device calls, mitigation flags of the QI).

**Settings that minimize quantum time**
1. Warm start from an ideal-simulator run. Pass `WF_sim.c_mo`, then `set_thetas(WF_sim._thetas_real, WF_sim._thetas_imag)`.
2. Use tolerances at the shot-noise level.
3. Use `l-bfgs-b` (or `cobyla` when the noise is large). Avoid Hessian-based methods on hardware.
4. Use `symmetrize_qG=False` and/or `"S"` if the extra measurements are too expensive.

## Validation

The test systems are H3 doublet, H4, LiH and H2O.

| Quantity | Compared with | Agreement |
|---|---|---|
| Energy, RDMs, orbital and θ gradients (non-rel., X2C, B) | original einsum code | 1e-14 |
| `run_wf_optimization_direct` (H2/6-31G) | 1-step result | 1e-12 |
| `LinearResponseDirect` (symmetrized): E2/S, excitation energies, oscillator strengths, polarizability | `generalized_naive` | 1e-14 (38 s → 3.4 s) |
| Circuit OO | classical OO | 1e-13 |
| Circuit LR, E2 | classical direct LR | 1e-8, limited by θ convergence |
