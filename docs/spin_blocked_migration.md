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

### Phase 3 — wave function classes *(three near-copies)*
`ups_`, `ucc_`, `sa_ups_wavefunction.py` — a change in one almost always belongs in all three.
- [ ] Spin-space construction (occupied α = `[0, n_α)`, occupied β = `[N, N+n_β)`).
- [ ] Spatial extraction `idx // 2` → `idx % N`.
- [ ] `_shifted` index lists — per-spin shift, not one contiguous shift.
- [ ] `hf_det` → `"1"*n_α + "0"*(N_act−n_α) + "1"*n_β + "0"*(N_act−n_β)`.
- [ ] Perfect-pairing determinant: keep the builder interleaved (human-readable), convert at the end.
- [ ] **SA-UPS `states`:** convert each user determinant *and multiply its coefficient by the
      reordering sign* — relative signs in a superposition are physical. Do this before the
      orthonormality check.

### Phase 4 — `operator_state_algebra.py` ansatz branches
- [ ] One helper converting a `UpsStructure`/`UccStructure` entry to blocked spin indices; call it
      once per `exc_type` branch (collapses ~90 edits into ~14).
- [ ] Apply D6 to the offset sites.
- [ ] `get_determinant_expansion_from_operator_on_HF`: blocked HF string.
- [ ] **Gate: state-vector test files green.**

### Phase 5+6 — `util.py` iterators and the Qiskit interface *(one commit, per D7)*
- [ ] `iterate_t1`–`t6` spin tests `a % 2 == 0` → `a < N`; `iterate_pair_t2*`; `create_tiled`
      doubles; `create_SDSfUCC`.
- [ ] `f2q` → identity; `get_qiskit_form` remap → delete; `get_reordering_sign` at
      `interface.py:821` → drops out; `get_determinant_reference` → `qc.x(i)`.
- [ ] `post_selection` is **unaffected** (endianness is a readout concern, already in Qiskit's frame).
- [ ] **Gate: `qWF.thetas = WF.thetas` tests green, and parameter ordering byte-identical.**

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

## 4. What must not change

Energies, excitation energies, oscillator strengths, RDMs and θ **values** are invariant under a
relabeling of spin orbitals. CI coefficient vectors are **not** — they get permuted *and*
individually sign-flipped. The existing suite asserts only invariants, which is what makes it a
usable oracle.
