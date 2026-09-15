# SlowQuant

Quantum chemistry package for unitary parameterized wave functions and linear response.
Two parallel implementations of the same physics, kept parameter-compatible:

- `slowquant/unitary_coupled_cluster/` — state-vector (exact classical simulation)
- `slowquant/qiskit_interface/` — circuit/sampler mirror

## Commands

```bash
pytest tests/                      # full suite, 146 tests, ~13 min
pytest tests/test_unitary_product_state.py -x    # state-vector only
pre-commit run --all-files         # ruff check + ruff format + mypy
```

**Known pre-existing failure.** `tests/test_properties.py::test_properties_ch3_sto3g` fails on
`master` (UHF spin contamination 0.015082 vs an expected 0.0152 ± 1e-4 — off by 1.2e-4, i.e. a
marginal tolerance rather than a wrong number). Treat a run whose only failure is this one as
green.

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
- **A fermionic operator is stored normal ordered and sorted.** A key of
  `FermionicOperator.operators` is `(creation_indices, annihilation_indices)`, each a tuple of
  spin-orbital indices sorted **descending**. Two spellings of the same string must produce the
  same key or they stop combining in the dict and the operator silently gains terms, so anything
  building a key by hand has to sort it the same way. `get_folded_operator` also reads the
  descending order as application order (rightmost operator first).
- **Spin-orbital ordering is α/β-blocked, matching Qiskit Nature:** the spin orbital of
  (spatial `p`, spin σ) is `p` for α and `p + N` for β, where `N` is the number of spatial
  orbitals *of the space the index lives in*. Determinants are ints with `2N` bits, index 0 at
  the **most significant** bit, so the α string occupies the high half of the integer and the β
  string the low half: `det >> N` and `det & ((1 << N) - 1)`. `spin_ordering.py` is the single
  source of truth; use `alpha_idx`/`beta_idx`/`spatial_idx`/`is_alpha` rather than open-coding.
- **Interleaved ordering survives only as a human-readable input format** — reference
  determinants, state-averaged state specifications. Convert it at the boundary with
  `det_interleaved_to_blocked`, and remember the sign (below).
- **The CI space is a product of an α and a β string space.** `get_indexing` enumerates α in the
  outer loop and β in the inner one, so a determinant's index is
  `idx_alpha*num_beta_strings + idx_beta`, and `CI_Info` carries the per-spin maps. This is load
  bearing for any future factorized operator-state algebra — **do not reorder those loops.**
  It does not hold for `get_indexing_extended`, which is not a spin product.
- **`Epq`, `epqrs`, `G1_sa`, `G2_sa` take spatial indices; `G1`–`G6` take spin-orbital indices.**
  Mixing them up is silent.
- **Orbital-count names are fixed and meaningful** — the name states which space an index lives in:
  `num_orbs` (all spatial), `num_active_orbs` (active spatial), `num_spin_orbs` (all spin),
  `num_active_spin_orbs` (active spin). Never use a generic name for these.
- **`util.py` is shared with the Qiskit implementation.** `UpsStructure.excitation_indices` is
  consumed by `qiskit_interface/operators_circuits.py`, and several tests assert
  `qWF.thetas = WF.thetas` round-trips. Changing excitation ordering there is never a local edit.
- **Rewriting a determinant in the other ordering carries a sign.** A determinant is an ordered
  product of creation operators, so the relabeling permutes them (`get_reordering_sign`). For a
  single determinant that is a global phase, but for a user-supplied *superposition* the relative
  signs are physical — get this wrong and an open-shell singlet silently becomes a triplet.
- **A factorized ansatz is order dependent.** fUCC and friends are products of exponentials of
  non-commuting generators, so the order the excitations are generated in is part of the ansatz,
  not a detail. Energies from a factorized ansatz are therefore *not* invariant under a change to
  the iteration order, unlike everything else. A single `expm(T)`, as in `WaveFunctionUCC`, is.
- **`_double_excitation_efficient` only supports some index topologies.** Writing the sorted
  creation/annihilation indices as `c`/`a`, it is correct for `aacc`, `acac` and `ccaa`, and
  wrong for `acca`, `caac` and `caca`. It raises on the unsupported ones; do not loosen that
  check without re-running `tests/test_excitation_circuits.py`.
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
- The test suite mostly asserts invariants (energies, excitation energies, oscillator strengths),
  so it is a good oracle for refactors that should not change results. Two exceptions: the
  variational minimum of a *factorized* ansatz depends on the factor order, and the noisy
  `FakeTorino` tests in `test_qiskit_interface.py` depend on the qubit-to-orbital assignment.
- Keep the name convention and coding style as close as possible to the original, but only when it does not sacrifice performance.
- When merging into "forwarded_master" always do a squash merge.
