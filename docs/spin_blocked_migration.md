# Migration: interleaved → α/β-blocked spin-orbital ordering

**Branch:** `alpha_beta_blocking`
**Scope:** option (A) — the codebase becomes natively blocked; interleaved survives only as a
human-facing input format (reference determinants, SA state specifications).

**Primary goal:** intuitive Qiskit interop — Qiskit Nature is blocked by default, so the
conversion layer (`f2q`, `get_qiskit_form`'s remap, `get_reordering_sign`) collapses to identity.

**Secondary goal (future work, not this migration):** unlock α/β factorization in the
operator–state algebra. Structural decisions below are chosen to keep that door open.

---

## 0. The convention

| | index of (spatial `p`, spin σ) | determinant string |
|---|---|---|
| before | `2p + σ` | `α₀β₀α₁β₁…` |
| after | α → `p`, β → `p + N` | `α₀α₁…α_{N-1}β₀β₁…β_{N-1}` |

Bit layout is unchanged: `2N` bits, index `i` at bit `1 << (2N-1-i)`.

---

## 1. Design decisions

### D1 — orbital count threaded explicitly, named for its space

`a_op(p, spin, dagger)` → `a_op(p, spin, dagger, num_orbs)`, rippling through `Epq`, `epqrs`,
`Eminuspq`, `G1_sa`, `G2_sa`, `hamiltonian_*`, `one_elec_op_*`. No new abstraction class —
an added argument is the change that stays closest to the existing style.

**The argument name states which space the index lives in**, using the names the wave function
classes already use:

| name | meaning |
|---|---|
| `num_orbs` | all spatial orbitals |
| `num_active_orbs` | active spatial orbitals |
| `num_spin_orbs` | all spin orbitals |
| `num_active_spin_orbs` | active spin orbitals |

So `Epq(p, q, num_orbs)` is a full-space operator and `Epq(p, q, num_active_orbs)` an active-space
one, visible at the call site. Two spaces are in play and must not be mixed:

- **full space** — Hamiltonians, property operators, LR q-operators, RDM `Epq`
- **active space** — anything built post-fold, or built active-local
  (`construct_ups_state`, `get_ucc_T`, `get_grad_action`, …)

Interleaved ordering hid this distinction. Blocked ordering forces every construction site to
declare which space it is in — an improvement, but the main source of new bugs, which is why the
naming is mandatory rather than stylistic.

### D2 — α is the high half of the determinant integer

Index 0 sits at the MSB, so α (indices `0..N-1`) occupies bits `[2N-1 : N]` and β the low `N` bits:

```python
alpha_string = det >> N
beta_string  = det & ((1 << N) - 1)
```

This is the structural payoff and the foundation of all future performance work.

### D3 — the α-outer/β-inner CI enumeration is load-bearing

`get_indexing` already enumerates α in the outer loop and β in the inner one, so

```
idx = i_alpha * n_beta + i_beta
```

holds today (verified). Blocking makes it *arithmetically exploitable*, because the α and β halves
of `det` become separable per D2. **Do not reorder those loops.**

Note this applies to `get_indexing` (the CAS space used by every wave function). It does *not*
hold for `get_indexing_extended`, which appends de-duplicated blocks and is only used by
linear response.

### D4 — add factorization metadata to `CI_Info` now, use it later

Populate `num_alpha_strings`, `num_beta_strings`, and per-spin string→index maps during Phase 1.
Roughly ten extra lines, and it turns the future kernel rewrite into a local change in
`operator_state_algebra.py` instead of another cross-cutting migration.

Keep `det2idx` as-is for now — the kernels stay untouched in this migration.

**Why it matters.** The innermost line of both kernels is a numba typed-dict hash lookup per
determinant per operator string:

```python
tmp_state[det2idx[det]] += sign * factor * state_i
```

After blocking this can become two small array lookups and a multiply–add:

```python
idx = alpha_str2idx[det >> N] * n_beta + beta_str2idx[det & mask]
```

And for operators touching only one spin — which, after folding, is most of them, since `Epq`
splits into an α term and a β term — the action factorizes over the other spin entirely: reshape
the state to `(n_alpha, n_beta)` and apply to rows or columns uniformly. That is the direct-CI
sigma-vector structure, and it is only available with blocked ordering.

### D5 — preserve two more perf-relevant properties

- **Parity simplification.** An α operator's Jordan–Wigner string never crosses β. A β operator's
  crosses the entire α block, contributing the constant `(-1)^{N_α}` (fixed within a
  spin-conserving space). So `parity_check` can later become per-spin (`N` bits) plus a constant —
  don't bake the 2N-wide mask in anywhere that would block this.
- **Kernel bit convention stays untouched** this migration, so performance work later lands on a
  green suite rather than on top of a convention change.

### D6 — replace `space_extension_offset` arithmetic, don't patch it

84 sites in `operator_state_algebra.py` (42 × `+ offset`, 42 × `+ 2 * offset`). Under blocking the
shift is spin-dependent: α shifts by `offset`, β by `offset + (N_ext − N_act)`. Introduce one
helper that embeds an active-space index into the extended space and call it once per branch.

### D7 — `util.py` goes blocked too, so the Qiskit side moves with it

`UpsStructure.excitation_indices` is consumed by `qiskit_interface/operators_circuits.py`, and
several tests assert `qWF.thetas = WF.thetas` round-trips. Phases 5 and 6 therefore land as one
commit, gated on those tests.

---

## 2. Phases

Suite-green checkpoints are **Phase 0, 4, 5+6, 7**. The 89-test suite cannot pass between Phase 1
and Phase 4; per-phase unit tests and the Phase 0 oracle cover that window.

### Phase 0 — scaffolding and oracle *(no behavior change)* — **done**
- [x] `slowquant/unitary_coupled_cluster/spin_ordering.py`: `alpha_idx`, `beta_idx`,
      `spin_orb_idx`, `spatial_idx`, `is_alpha`, `interleaved_to_blocked`,
      `blocked_to_interleaved`, `det_interleaved_to_blocked`, `det_blocked_to_interleaved`,
      `get_reordering_sign`.
- [x] Moved `f2q` and `get_reordering_sign` down from `qiskit_interface/util.py` (the dependency
      ran the wrong way). `util.py` re-exports `interleaved_to_blocked as f2q` so the twelve
      existing call sites are untouched until Phase 5+6; `interface.py` imports the sign directly.
- [x] `tests/test_spin_ordering.py` — round trips, bijectivity, determinant-integer splitting
      (D2), reordering sign against a brute-force inversion count **and** against a frozen copy
      of the pre-migration implementation.
- [x] `tests/test_migration_oracle.py` + `tests/reference_data/migration_oracle.json` — see below.

**The oracle.** Four systems spanning the folding paths: `h2_sto3g_cas22` (no inactive, no
virtual), `lih_sto3g_cas22` (both), `lih_sto3g_cas44` (virtual only), `h2o_sto3g_cas44`
(inactive only). Pinned quantities, all invariant under relabelling: the **active-space CI
Hamiltonian spectrum**, ⟨HF|H|HF⟩, and the reference-determinant RDM1. The spectrum is the
valuable one — it exercises `get_indexing`, the operator construction and `get_folded_operator`
together, without going through the wave function classes, so it stays usable while those are
mid-migration.

The reference JSON must never be regenerated. To recreate the generating checkout:

```bash
git worktree add /tmp/sq_master master
PYTHONPATH=/tmp/sq_master python <script>   # note the two-checkout PATH hazard in CLAUDE.md
```

### Phase 1 — convention core: `ci_spaces.py` + `operators.py` — **done**
- [x] `get_indexing`: builds blocked determinants via `det_from_spin_strings`, which *is* D2 made
      executable. Loop order preserved and now documented as load-bearing (D3).
- [x] `get_indexing_extended`: four interleave sites replaced; they did simplify.
- [x] `CI_Info`: `alpha_str2idx`, `beta_str2idx`, `num_alpha_strings`, `num_beta_strings` (D4),
      left empty/0 for the extended space, which is not a spin product. Stored as plain dicts
      rather than `| None`, so future consumers do not each need an Optional check.
- [x] `a_op` + signature ripple (D1). `num_orbs` is positional for `a_op`, `Epq`, `epqrs` and
      `Eminuspq`, so a missed call site raises `TypeError`. It is **keyword-only** for `G1_sa`
      and `G2_sa`: both already end in a bool/int with a default, so a positional addition could
      silently bind to the wrong parameter, and `bool` is a subtype of `int` so mypy would not
      catch it. `hamiltonian_0i_0a` and `one_elec_op_0i_0a` gained `num_virtual_orbs`, matching
      their `1i_1a` siblings, so they can compute the full-space size. `G1`–`G6` are untouched:
      they already take spin-orbital indices.
- [x] **Test:** the `h2_sto3g_cas22` oracle case (no inactive, no virtual, so folding is trivial)
      reproduces the interleaved CI spectrum, ⟨HF|H|HF⟩ and RDM1 exactly. Product-structure and
      occupation tests added to `test_spin_ordering.py`.

The other three oracle cases fail on `ci_eigenvalues`, as expected — they have inactive and/or
virtual orbitals and so need Phase 2.

### Phase 2 — folding — **done**
- [x] `get_folded_operator`: space classification by `orb_idx % num_orbs`, index remap
      (α `n_i+k → k`, β `N+n_i+k → n_a+k`).
- [x] The sign is no longer a closed form. It is obtained by **simulating the string on the
      inactive subsystem**, applying operators right to left from a filled inactive space and
      accumulating the number of occupied inactive orbitals each operator has to be moved past.
      Terms are dropped when a virtual index appears, when an inactive creation hits an occupied
      orbital (or an annihilation an empty one), or when the inactive occupations are not
      restored. This is ordering-agnostic and does not assume a normal-ordered input.
- [x] **The subtlety that broke the first attempt.** Every α orbital lies below every inactive β
      one, so an inactive β operator must also move past the *active* α electrons. Their count at
      the start of the string is a constant of the CI space and cancels — the inactive operators
      pair up, so an even number of them are β — but the *running change* caused by active α
      operators earlier in the string does not. Tracking that change is what `active_alpha_change`
      does. Without it, `epqrs(0,0,1,1)` on a space with one inactive orbital comes out with the
      wrong sign on the α–β cross terms.
- [x] **Test:** `tests/test_operator_folding.py` builds the operator matrix over the whole
      orbital space, restricts it to the determinants with inactive filled and virtual empty, and
      compares that block to the folded operator's matrix. The reference needs no sign reasoning
      at all, so it is a real check rather than a restatement of the implementation. Six orbital
      spaces, covering no inactive/no virtual, inactive only, virtual only, both, more inactive
      than active, and unequal α/β occupation.
- [x] **All four oracle cases now pass.**

### Phase 3 — wave function classes — **done**
`ups_`, `ucc_`, `sa_ups_wavefunction.py`, changed together.
- [x] Spin-space construction rewritten. The old code derived the spaces by walking interleaved
      spin-orbital indices and halving; it now derives the spatial partition directly from
      `cas` and builds spin indices with `spin_ordering.spin_indices`. Shorter and no longer
      ordering-dependent. The three blocks were identical apart from an even-electron check
      that `ucc_wavefunction` never had, and that asymmetry is preserved rather than "fixed".
- [x] `_shifted` lists are built per spin from the shifted spatial indices, not by subtracting a
      single minimum, which is invalid when the active space is two separate blocks.
- [x] `hf_det`, and the perfect-pairing determinant, stay in the human-readable interleaved form
      (including the `"1100"` blocks and the MO-column swap) and are converted with
      `det_interleaved_to_blocked` where they are looked up.
- [x] **SA-UPS `states`:** determinants converted *and coefficients multiplied by the reordering
      sign*, before the orthonormality check. Verified on the open-shell singlet used in the
      tests: `["10010000", "01100000"]` get signs `+1` and `−1`, so the user's
      `[+1/√2, −1/√2]` is stored as `[+1/√2, +1/√2]`. Same physical state — without the sign it
      would silently have become the triplet.
- [x] Operator call sites updated. `sa_ups` builds its RDM operators on **active-local** indices
      with `do_folding=False`, so those take `num_active_orbs` while the other two classes take
      `num_orbs` — the first real instance of the full-vs-active hazard D1 predicted, made
      visible by the argument naming.

**Phase 4 boundary, as observed.** `WaveFunctionUCC` now raises a `TypeError` when the state is
built (`get_ucc_T` calls `G1_sa` without `num_orbs`), which is the keyword-only guard working.
`WaveFunctionUPS` does **not** raise: `construct_ups_state` only calls `G1`–`G6`, which take raw
spin indices, so it happily applies `G1(i*2, a*2)` — interleaved arithmetic against a blocked
basis. For tUPS on CAS(2,2) that builds a spin-flip operator which annihilates the reference, so
the energy comes back unchanged and *looks* fine. Phase 4 must not be judged by whether things
run.

### Phase 4 — `operator_state_algebra.py` ansatz branches **and `util.py`**

**Scope correction.** The plan had `util.py` in Phase 5, but its `iterate_t*` iterators are fed
the index lists the wave function classes build, and Phase 3 made those blocked. A spin test of
`a % 2 == 0` against a blocked index is simply wrong, so `util.py` had to move here. Phase 5 is
now the Qiskit interface alone.

- [x] `embed_spatial_indices` and `embed_spin_indices` in `operator_state_algebra.py` replace all
      84 `+ offset` / `+ 2 * offset` sites (D6). The spin variant is where blocking actually bites:
      α shifts by `offset`, β by `offset + (N_ext − N_act)`.
- [x] `get_ucc_T` takes `ci_info` instead of a bare `offset`, since embedding now needs the width
      of the target space as well as the offset.
- [x] `G1(i*2, a*2)` / `G1(i*2+1, a*2+1)` in the spin-adapted singles branches become
      `alpha_idx` / `beta_idx`; the 36 `G1_sa`/`G2_sa` calls gained `num_orbs`.
- [x] `get_determinant_expansion_from_operator_on_HF` builds a blocked HF string.
- [x] `util.py`: the 48 `% 2` spin tests become half-of-the-register tests; `iterate_t1`–`t6` take
      `num_orbs`; the generalized variants derive it from `num_spin_orbs`; `iterate_pair_t2` and
      `create_tiled`'s doubles emit blocked indices; `create_SDSfUCC`'s same-spin test and spatial
      extraction converted.
- [x] `UpsStructure` and `UccStructure` carry `num_active_orbs`. The excitation indices are only
      interpretable together with the size of the space they were built over, which the structures
      previously did not record.

**Gate result: 20 passed, 25 failed.** All but two failures are `TypeError: G1_sa` from
linear response (Phase 7) plus the known CH3 UHF flake. The exception is
`test_ups_n2_fuccsdtq56`, which is discussed under "the invariance caveat" below.

**Noted in passing, not fixed:** `create_SDSfUCC`'s `do_pD` branch labels a two-index excitation
`"double"` where the analogous `do_GpD` branch uses `"sa_single"`; `"double"` unpacks four indices,
so that path would raise. It is unreachable from the named ansätze. Filed separately.

### Phase 5 — the Qiskit interface *(`util.py` moved to Phase 4)*
- [x] `circuit_wavefunction.py` and `sa_circuit_wavefunction.py` index construction converted the
      same way as Phase 3 did for the state-vector classes.
- [x] **`f2q` is gone from the codebase.** A blocked fermionic index *is* the Jordan-Wigner qubit
      index, so `operators_circuits.py` uses the indices as they come and
      `get_determinant_reference` is now `qc.x(i)`.
- [x] **`get_qiskit_form` no longer remaps.** `operator_to_qiskit_key` lost its `remapping`
      argument and `get_qiskit_form` lost `num_orbs` — both sides use the same ordering, so only
      the key format differs.
- [x] **`get_reordering_sign` dropped out of `interface.py`.** The determinant strings reaching
      `quantum_expectation_value_csfs` are blocked already. In exchange, `sa_circuit_wavefunction`
      converts the user's interleaved states at construction, sign and all, mirroring
      `WaveFunctionSAUPS`.
- [x] Spin tests in `_double_excitation_efficient` and the spin-adapted single builders converted.
- [x] `linear_response/{selfconsistent,statetransfer}.py` build blocked reference determinants, to
      match what `get_determinant_expansion_from_operator_on_HF` now returns.
- [x] `post_selection` **unchanged**, as predicted — endianness is a readout concern and was
      already in Qiskit's frame.
- [x] `circuit_wavefunction`'s perfect-pairing determinants stay interleaved: they only decide
      which MO columns to swap and never reach a circuit. Commented to that effect.
- [x] **The double-excitation circuit needed a corrected precondition, not a deleted one.**
      Details below.
- [ ] **Gate: `test_qiskit_unitary_product_state.py` and `test_qiskit_interface.py` green.**

**The index topology of `_double_excitation_efficient`.** The hand-derived circuit rejected the
first blocked tUPS double outright: it required both creation indices to sit above both
annihilation indices, which held automatically under interleaved ordering but does not under
blocked. For a double that shares spatial orbitals between the spins, `(α_i, β_i) → (α_a, β_a)`,
the sorted indices now *interleave*, because every α index is below every β one.

Rather than guess, the circuit unitary was compared against `expm` of the Jordan-Wigner mapped
generator built with Qiskit's own mapper, over every spin-conserving quadruple. Writing the sorted
indices as `c`/`a`:

| sorted pattern | correct | where it comes from |
|---|---|---|
| `aacc` | yes | same-spin double; the only pattern interleaved ordering ever produced |
| `acac` | yes | double sharing spatial orbitals between spins; **new under blocked** |
| `ccaa` | yes | reversed excitation, normalized by the swaps |
| `acca`, `caac`, `caca` | **no** | nested/crossed; not generated by any ansatz |

So the circuit was already correct for the new topology; only the guard was wrong. It now checks
the real contract and still raises for the three unsupported orderings, rather than silently
emitting a wrong circuit. Every builder was enumerated and emits only `aacc` and `acac` —
`tests/test_excitation_circuits.py` pins all of this, including that the unsupported orderings
raise.

### Phase 7 — linear response
- [ ] `lr_baseclass.py` in both packages; the eight state-vector variants.
- [ ] `density_matrix.py` needs **no changes** (purely spatial/RDM).
- [ ] **Gate: full suite green.**

### Phase 8 — cleanup
- [ ] Delete now-identity conversions; update docstrings that describe the interleaved convention.
- [ ] Update the index-ordering section of `CLAUDE.md`.
- [ ] Delete this document.

### Phase 9 — performance *(separate project)*
D2/D3/D4 kernel rewrite, per-spin parity, factorized single-spin operator application.

---

## 3. Risk register

| Risk | Phase | Symptom | Mitigation |
|---|---|---|---|
| Folding sign mis-derived | 2 | plausible but wrong energies | derive from scratch; oracle test with inactive orbitals |
| Wrong `num_orbs` (full vs active) | 1, 4 | wrong operator, often still Hermitian | name the argument consistently; assert index bounds in debug |
| `space_extension_offset` spin-dependence | 4 | LR-only breakage | D6 helper, no raw arithmetic |
| One of three wave function classes missed | 3 | one class silently wrong | change all three in the same commit |
| θ parameter ordering drifts | 5+6 | Qiskit/state-vector mismatch | compare parameter name lists element-wise |
| SA superposition relative signs | 3 | wrong SA energies only | apply reordering sign to coefficients |

## 4. What must not change, and the invariance caveat

Energies, excitation energies, oscillator strengths, RDMs and θ **values** are invariant under a
relabeling of spin orbitals. CI coefficient vectors are **not** — they get permuted *and*
individually sign-flipped.

**The caveat, found in Phase 4.** That invariance covers the *relabeling*. It does not cover the
change in **iteration order** that comes with it. `iterate_t1`–`t6` walk the spin-index lists, and
those lists went from α₀β₀α₁β₁… to α₀α₁…β₀β₁…, so the excitations come out in a different order.

For a single exponential that is harmless — `WaveFunctionUCC` builds one `expm(T)`, T is a sum, and
order does not matter. Every UCC energy test passes unchanged, through quadruples.

For a **factorized** ansatz it is not harmless. fUCC is a product of exponentials of
non-commuting generators, so reordering the factors changes the variational manifold itself.
`test_ups_n2_fuccsdtq56` (N2/STO-3G, CAS(6,6), fUCC through sextuples, 399 parameters) converges
to −131.1964192800081 instead of the recorded −131.1965135680605.

Evidence that this is the reordering and not a bug:

- The excitation **set** is unchanged. Dumping the fUCC S+D generators as (spatial, spin) pairs
  from `master` and from this branch gives the same 26 excitations, same set, different order —
  nothing lost, nothing spurious.
- Repeated BFGS restarts converge to exactly −131.1964192800081, so it is a true stationary point,
  not an unconverged one.
- Both numbers sit **above** the CASCI limit for that space, −131.1966323482769, so neither
  ordering spans the full CAS and an order-dependent result is expected.
- tUPS, whose factor order is generated by `create_tiled` and is *unchanged* by the migration,
  matches exactly.

So the suite is a reliable oracle for everything except the variational minimum of a factorized
ansatz whose factor order the migration changed. **Decided:** keep the natural ascending blocked
order, which is also the α-major layout the future factorization work wants, and re-record that
one reference with the reasoning in the test's docstring. fUCC numbers produced before the
migration therefore do not reproduce bit-for-bit; the difference here is 9.4e-5 Hartree, well
inside the gap to the CAS limit.
