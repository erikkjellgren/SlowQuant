# α/β factorization of the operator–state algebra

Notes for a future performance project. Nothing here is implemented; this records the structure
the spin-blocked ordering makes available, so the groundwork laid for it is not accidentally
removed.

## What blocking bought

**The determinant integer splits.** Spin-orbital index `i` sits at bit `2N-1-i`, and α occupies
indices `0..N-1`, so the α string is the high half of the integer and the β string the low half:

```python
alpha_string = det >> num_orbs
beta_string = det & ((1 << num_orbs) - 1)
```

**The CI space is a product.** `get_indexing` enumerates α in the outer loop and β in the inner
one, so

```
idx = idx_alpha * num_beta_strings + idx_beta
```

`CI_Info` already carries `alpha_str2idx`, `beta_str2idx`, `num_alpha_strings` and
`num_beta_strings`. Both facts are pinned by `tests/test_spin_ordering.py`; the loop order in
`get_indexing` is load bearing and must not be changed.

This does **not** hold for `get_indexing_extended`, which appends de-duplicated blocks and is
only used by linear response.

## The three opportunities

### 1. Replace the hash lookup in the kernels

The innermost line of `apply_operator_serial` and `apply_operator_threaded` is a Numba typed-dict
probe, once per determinant per operator string:

```python
tmp_state[det2idx[det]] += sign * factor * state_i
```

With the product structure that becomes two small array lookups and a multiply–add:

```python
idx = alpha_str2idx[det >> num_orbs] * num_beta_strings + beta_str2idx[det & mask]
```

For the active-space sizes in use, `num_orbs` is small enough that the per-spin maps can be
direct-address arrays of length `2**num_orbs` rather than dicts, which is what makes this worth
doing at all.

### 2. Factorize single-spin operators over the other spin

After folding, most operators touch one spin only — `Epq` splits into an α term and a β term.
Such an operator acts identically on every string of the other spin. Reshaping the state to
`(num_alpha_strings, num_beta_strings)` turns its application into a uniform operation on rows or
on columns, which is the direct-CI sigma-vector structure. This is the large win, and it is only
available with blocked ordering.

### 3. Shrink the parity computation

`parity_check` is currently a `2N`-wide mask table. Under blocking:

- an α operator's Jordan–Wigner string never reaches a β orbital, so its parity is intra-α;
- a β operator's string crosses the *entire* α block, contributing `(-1)^{N_α}`, and `N_α` is
  constant within a spin-conserving CI space, plus an intra-β part.

So the parity work per operator drops from `2N` bits to `N` bits plus a constant. Note that the
same cancellation already appears in `get_folded_operator`, where the constant drops out because
the inactive operators pair up — see the comment there, and note that the *running* change from
active α operators earlier in a string does **not** cancel.

## Constraints to respect

- The kernels' bit convention (index 0 at the most significant bit) is assumed throughout
  `spin_ordering.py` and `ci_spaces.py`. Changing it is a separate project.
- `do_unsafe` exists because some callers can produce determinants outside the CI space. A
  factorized addressing scheme still has to handle that, or explicitly refuse those callers.
- `get_indexing_extended` is not a product space, so any factorized path needs a fallback for it.
