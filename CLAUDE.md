# SlowQuant

Quantum chemistry package for unitary parameterized wave functions and linear response.
Two parallel implementations of the same physics, kept parameter-compatible:

- `slowquant/unitary_coupled_cluster/` — state-vector (exact classical simulation)
- `slowquant/qiskit_interface/` — circuit/sampler mirror

## Commands

```bash
pytest tests/                      # full suite, 89 tests
pytest tests/test_unitary_product_state.py -x    # state-vector only
pre-commit run --all-files         # ruff check + ruff format + mypy
```

**Known pre-existing failure.** `tests/test_properties.py::test_properties_ch3_sto3g` fails on
`master` (UHF spin contamination 0.015082 vs an expected 0.0152 ± 1e-4 — off by 1.2e-4, i.e. a
marginal tolerance rather than a wrong number). Treat a 1-failed/97-passed run as green unless
the failure is a different one.

**Import path warning.** A second checkout lives at `~/gitreps/SlowQuant` and is on `PYTHONPATH`.
Running from this directory picks up the local package (cwd wins over `PYTHONPATH`, verified for
`python -c`, heredocs and pytest), but a script run from any other directory silently imports the
*other* checkout. For standalone scripts, prefix explicitly:

```bash
PYTHONPATH=/home/kjellgren/gitreps/SlowQuant_claude python script.py
```

## Invariants — violating these gives wrong numbers, not exceptions

- **Never multiply folded operators.** `FermionicOperator.get_folded_operator` is valid only as
  the last step. Build the product first, fold last.
- **Spin-orbital index ordering is interleaved:** spin orbital of (spatial `p`, spin σ) is `2p + σ`.
  Determinants are ints with `2N` bits, index 0 at the **most significant** bit.
  (A migration to spin-blocked ordering is in progress — see `docs/spin_blocked_migration.md`
  if present, and update this section when it lands.)
- **`Epq`, `epqrs`, `G1_sa`, `G2_sa` take spatial indices; `G1`–`G6` take spin-orbital indices.**
  Mixing them up is silent.
- **Orbital-count names are fixed and meaningful** — the name states which space an index lives in:
  `num_orbs` (all spatial), `num_active_orbs` (active spatial), `num_spin_orbs` (all spin),
  `num_active_spin_orbs` (active spin). Never use a generic name for these.
- **`util.py` is shared with the Qiskit implementation.** `UpsStructure.excitation_indices` is
  consumed by `qiskit_interface/operators_circuits.py`, and several tests assert
  `qWF.thetas = WF.thetas` round-trips. Changing excitation ordering there is never a local edit.
- **Relabeling spin orbitals carries a sign.** Determinant basis vectors pick up a permutation
  sign (`get_reordering_sign`); relative signs in multi-determinant states are physical.
- Numba is set to **1 thread at import** (`operator_state_algebra._init`). `propagate_state`
  picks its serial/threaded kernel from `nb.get_num_threads()`.
- `do_unsafe=True` tolerates determinants falling outside the CI space; leaving it `False`
  turns that into a hard error, which is usually what you want.

## Conventions

- Ruff: line length 110, double quotes, Google-style docstrings (`D` rules are on — every public
  function needs Args/Returns). isort via ruff.
- mypy runs in pre-commit; keep annotations on new functions.
- Numerical thresholds in use: `1e-28` skips near-zero thetas/amplitudes, `1e-14` screens integrals.
- Wave function classes (`ucc_`, `ups_`, `sa_ups_wavefunction.py`) are three near-copies.
  A change to index bookkeeping in one almost always belongs in all three.

## Working preferences

- Plan first for anything touching more than ~2 files.
- Small, reviewable commits — one migration step per commit.
- The test suite asserts only invariants (energies, excitation energies, oscillator strengths),
  so it is a reliable oracle for refactors that should not change results.
- Keep the name convention and coding style as close as possible to the original, but only when it does not sacrifice   performance.
