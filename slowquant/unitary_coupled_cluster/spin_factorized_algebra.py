r"""Spin-factorized application of a fermionic operator to a state vector.

Every operator that is :math:`S_z` conserving has, in every one of its normal-ordered strings,
as many :math:`\alpha` creation as :math:`\alpha` annihilation operators, and likewise for
:math:`\beta`. Such a string factorizes into a product of a purely :math:`\alpha` and a purely
:math:`\beta` string,

.. math::
    \hat{O} = \pm\hat{O}_\alpha\otimes\hat{O}_\beta

Three properties of the alpha/beta-blocked ordering make this cheap, see spin_ordering,

#. An :math:`\alpha` operator only has :math:`\alpha` spin orbitals below it, so its phase is
   computed entirely inside the :math:`\alpha` string.

#. A :math:`\beta` operator has every :math:`\alpha` spin orbital below it, so it picks up the
   number of active :math:`\alpha` electrons as a phase. A string holds an even number of
   :math:`\beta` operators, so that contribution always cancels and no correction is needed.

#. The only phase left is the one from moving the two blocks past each other, which is
   :math:`\left(-1\right)^{k_\alpha k_\beta}` with :math:`k_\sigma` the number of creation
   operators of spin :math:`\sigma`.

Because each spin block conserves its own particle number, the resulting spin string always has
the same number of electrons and is therefore always inside the string space. The determinant
index is :math:`I = I_\alpha N_\beta + I_\beta`, so no determinant lookup is needed at all.

This only applies to a determinant expansion that is a product of an alpha and a beta string
space, i.e. CI_Info.is_spin_product, which excludes get_indexing_extended.
"""

from __future__ import annotations

import functools

import numba as nb
import numpy as np

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info, bitcount
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator

# A dense array over all 2**num_active_orbs spin strings is the fastest way to get the index of
# a spin string. It stays small for any active space a state-vector expansion can hold, but is
# guarded so an unreasonable request falls back to the general algebra instead of allocating.
MAX_DENSE_LOOKUP_ORBS = 24

# Spin sub-string of an operator that acts as the identity on that spin.
IDENTITY_SUB_STRING: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())


class SpinFactorizedOperator:
    __slots__ = (
        "alpha_dst",
        "alpha_sign",
        "alpha_src",
        "beta_dst",
        "beta_sign",
        "beta_src",
        "mixed_alpha_start",
        "mixed_alpha_stop",
        "mixed_beta_start",
        "mixed_beta_stop",
        "mixed_factor",
        "pure_alpha_factor",
        "pure_alpha_start",
        "pure_alpha_stop",
        "pure_beta_factor",
        "pure_beta_start",
        "pure_beta_stop",
    )

    def __init__(
        self,
        alpha_src: np.ndarray,
        alpha_dst: np.ndarray,
        alpha_sign: np.ndarray,
        beta_src: np.ndarray,
        beta_dst: np.ndarray,
        beta_sign: np.ndarray,
        pure_alpha_start: np.ndarray,
        pure_alpha_stop: np.ndarray,
        pure_alpha_factor: np.ndarray,
        pure_beta_start: np.ndarray,
        pure_beta_stop: np.ndarray,
        pure_beta_factor: np.ndarray,
        mixed_alpha_start: np.ndarray,
        mixed_alpha_stop: np.ndarray,
        mixed_beta_start: np.ndarray,
        mixed_beta_stop: np.ndarray,
        mixed_factor: np.ndarray,
    ) -> None:
        """Initialize the spin-factorized form of a fermionic operator.

        The alpha and beta arrays are the shared per-spin arenas of the CI space, holding the
        excitation map of every spin sub-string built so far. Each term of the operator is a
        slice into each of them, so no per-operator copy of a map is ever made.

        The terms are split by which spins they act on, because the three cases are applied by
        different algorithms. A term acting on one spin only is a matrix acting on one side of
        the state, while a term acting on both couples the two.

        Args:
            alpha_src: Alpha string indices an alpha sub-string maps from.
            alpha_dst: Alpha string indices an alpha sub-string maps to.
            alpha_sign: Phase of the alpha sub-string application.
            beta_src: Beta string indices a beta sub-string maps from.
            beta_dst: Beta string indices a beta sub-string maps to.
            beta_sign: Phase of the beta sub-string application.
            pure_alpha_start: Start of an alpha only term's slice into the alpha arena.
            pure_alpha_stop: End of an alpha only term's slice into the alpha arena.
            pure_alpha_factor: Factor in front of an alpha only term.
            pure_beta_start: Start of a beta only term's slice into the beta arena.
            pure_beta_stop: End of a beta only term's slice into the beta arena.
            pure_beta_factor: Factor in front of a beta only term.
            mixed_alpha_start: Start of a mixed term's slice into the alpha arena.
            mixed_alpha_stop: End of a mixed term's slice into the alpha arena.
            mixed_beta_start: Start of a mixed term's slice into the beta arena.
            mixed_beta_stop: End of a mixed term's slice into the beta arena.
            mixed_factor: Factor in front of a mixed term.
        """
        self.alpha_src = alpha_src
        self.alpha_dst = alpha_dst
        self.alpha_sign = alpha_sign
        self.beta_src = beta_src
        self.beta_dst = beta_dst
        self.beta_sign = beta_sign
        self.pure_alpha_start = pure_alpha_start
        self.pure_alpha_stop = pure_alpha_stop
        self.pure_alpha_factor = pure_alpha_factor
        self.pure_beta_start = pure_beta_start
        self.pure_beta_stop = pure_beta_stop
        self.pure_beta_factor = pure_beta_factor
        self.mixed_alpha_start = mixed_alpha_start
        self.mixed_alpha_stop = mixed_alpha_stop
        self.mixed_beta_start = mixed_beta_start
        self.mixed_beta_stop = mixed_beta_stop
        self.mixed_factor = mixed_factor


@functools.lru_cache(maxsize=2**18)
def split_spin_string(
    op_key: tuple[tuple[int, ...], tuple[int, ...]], num_active_orbs: int
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]], int] | None:
    r"""Split a fermionic string into its alpha and beta sub-strings.

    The blocked ordering puts every beta index above every alpha index, and both blocks of the
    string are sorted descending, so filtering on spin keeps each sub-string sorted descending.
    The returned sub-string indices are spatial indices inside the spin block.

    The phase is the one from reordering

    .. math::
        \hat{c}_\beta\hat{c}_\alpha\hat{a}_\beta\hat{a}_\alpha
        \rightarrow \left(\hat{c}_\alpha\hat{a}_\alpha\right)\left(\hat{c}_\beta\hat{a}_\beta\right)

    which for a spin conserving string is :math:`\left(-1\right)^{k_\alpha k_\beta}`.

    Args:
        op_key: Fermionic string, tuple of creation and annihilation spin-orbital indices.
        num_active_orbs: Number of active spatial orbitals.

    Returns:
        Alpha sub-string, beta sub-string, and phase, or None if the string is not spin conserving.
    """
    creation, annihilation = op_key
    creation_alpha = tuple(idx for idx in creation if idx < num_active_orbs)
    creation_beta = tuple(idx - num_active_orbs for idx in creation if idx >= num_active_orbs)
    annihilation_alpha = tuple(idx for idx in annihilation if idx < num_active_orbs)
    annihilation_beta = tuple(idx - num_active_orbs for idx in annihilation if idx >= num_active_orbs)
    if len(creation_alpha) != len(annihilation_alpha) or len(creation_beta) != len(annihilation_beta):
        # The string changes the number of electrons of a spin, so it does not map the spin
        # product onto itself and cannot be factorized.
        return None
    sign = 1 - 2 * ((len(creation_alpha) * len(creation_beta)) & 1)
    return (creation_alpha, annihilation_alpha), (creation_beta, annihilation_beta), sign


@nb.jit(nopython=True)
def build_spin_excitation_map(
    idx2spin_str: np.ndarray,
    spin_str_lookup: np.ndarray,
    a_string: np.ndarray,
    create_screen: np.ndarray,
    anni_idx: np.ndarray,
    num_active_orbs: int,
    parity_check: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the excitation map of one spin sub-string over a spin string space.

    Follows the same two-step algorithm as apply_operator_serial, with the determinant replaced
    by a single spin string, see the module docstring for why that is exact. Screening and
    traversal order are the same, so the phase matches the unfactorized kernel term by term.

    Args:
        idx2spin_str: Maps spin string index to spin string.
        spin_str_lookup: Maps spin string to spin string index.
        a_string: Creation and annihilation operator indices.
        create_screen: Creation operator indices without indices in anni_idx.
        anni_idx: Indices for annihilation operators.
        num_active_orbs: Number of active spatial orbitals.
        parity_check: Array used to check the parity when an operator is applied.

    Returns:
        Spin string indices mapped from, spin string indices mapped to, and the phases.
    """
    num_spin_strings = len(idx2spin_str)
    num_orbs_m1 = num_active_orbs - 1
    anni_mask = 0
    for orb_idx in anni_idx:
        anni_mask |= 1 << (num_orbs_m1 - orb_idx)
    create_mask = 0
    for orb_idx in create_screen:
        create_mask |= 1 << (num_orbs_m1 - orb_idx)
    src = np.empty(num_spin_strings, dtype=np.int64)
    dst = np.empty(num_spin_strings, dtype=np.int64)
    sign = np.empty(num_spin_strings, dtype=np.float64)
    num_survivors = 0
    for i in range(num_spin_strings):
        spin_str = idx2spin_str[i]
        if (spin_str & anni_mask) != anni_mask:
            continue
        if (spin_str & create_mask) != 0:
            continue
        phase_changes = 0
        for orb_idx in a_string:
            spin_str = spin_str ^ (1 << (num_orbs_m1 - orb_idx))
            phase_changes += bitcount(spin_str & parity_check[orb_idx])
        src[num_survivors] = i
        dst[num_survivors] = spin_str_lookup[spin_str]
        sign[num_survivors] = 1.0 - 2.0 * (phase_changes & 1)
        num_survivors += 1
    return src[:num_survivors], dst[:num_survivors], sign[:num_survivors]


def get_spin_string_lookup(ci_info: CI_Info, is_alpha: bool) -> np.ndarray:
    """Get the dense spin string to spin string index map, building it on first use.

    Args:
        ci_info: Information about the CI space.
        is_alpha: Get the alpha map, otherwise the beta one.

    Returns:
        Maps spin string to spin string index, with -1 for a string outside the space.
    """
    lookup = ci_info.alpha_str_lookup if is_alpha else ci_info.beta_str_lookup
    if lookup is None:
        idx2spin_str = ci_info.idx2alpha_str if is_alpha else ci_info.idx2beta_str
        lookup = np.full(1 << ci_info.num_active_orbs, -1, dtype=np.int64)
        lookup[idx2spin_str] = np.arange(len(idx2spin_str), dtype=np.int64)
        if is_alpha:
            ci_info.alpha_str_lookup = lookup
        else:
            ci_info.beta_str_lookup = lookup
    return lookup


def get_spin_sub_string_slice(
    ci_info: CI_Info, sub_string: tuple[tuple[int, ...], tuple[int, ...]], is_alpha: bool
) -> tuple[int, int]:
    """Get the slice of the spin arena holding the excitation map of one spin sub-string.

    The map depends only on the spin string space and the sub-string, never on the factor in
    front of it, so it is built once and appended to the arena of that spin. Every later
    operator over the same CI space reuses it.

    Args:
        ci_info: Information about the CI space.
        sub_string: Spin sub-string, tuple of creation and annihilation spatial indices.
        is_alpha: The sub-string acts on the alpha strings, otherwise on the beta ones.

    Returns:
        Start and end of the sub-string's slice of the spin arena.
    """
    cache_key = (is_alpha, sub_string[0], sub_string[1])
    if cache_key in ci_info.spin_op_cache:
        return ci_info.spin_op_cache[cache_key]
    num_active_orbs = ci_info.num_active_orbs
    creation, annihilation = sub_string
    # Create bitstrings for parity check. Contains occupied spin string up to orbital index.
    parity_check = np.zeros(num_active_orbs + 1, dtype=int)
    num = 0
    for i in range(num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[num_active_orbs - i] = num
    # When screening spin strings, no need to consider the annihilation index of an operator
    # that has the same creation index.
    create_screen = np.array([idx for idx in creation if idx not in annihilation], dtype=np.int64)
    anni_idx = np.array(annihilation, dtype=np.int64)
    a_string = np.array([*annihilation, *creation], dtype=np.int64)
    spin_map = build_spin_excitation_map(
        ci_info.idx2alpha_str if is_alpha else ci_info.idx2beta_str,
        get_spin_string_lookup(ci_info, is_alpha),
        a_string,
        create_screen,
        anni_idx,
        num_active_orbs,
        parity_check,
    )
    start = ci_info.spin_arena_length[is_alpha]
    ci_info.spin_arena[is_alpha].append(spin_map)
    ci_info.spin_arena_length[is_alpha] = start + len(spin_map[0])
    # The arena grew, so the packed form is stale.
    ci_info.spin_arena_packed[is_alpha] = None
    arena_slice = (start, start + len(spin_map[0]))
    ci_info.spin_op_cache[cache_key] = arena_slice
    return arena_slice


def get_spin_arena(ci_info: CI_Info, is_alpha: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the excitation maps of every spin sub-string built so far, laid out back to back.

    Args:
        ci_info: Information about the CI space.
        is_alpha: Get the alpha arena, otherwise the beta one.

    Returns:
        Spin string indices mapped from, spin string indices mapped to, and the phases.
    """
    packed = ci_info.spin_arena_packed[is_alpha]
    if packed is None:
        arena = ci_info.spin_arena[is_alpha]
        if len(arena) == 0:
            packed = (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64),
            )
        else:
            packed = (
                np.concatenate([entry[0] for entry in arena]),
                np.concatenate([entry[1] for entry in arena]),
                np.concatenate([entry[2] for entry in arena]),
            )
        ci_info.spin_arena_packed[is_alpha] = packed
    return packed


def factorize_operator(op: FermionicOperator, ci_info: CI_Info) -> SpinFactorizedOperator | None:
    """Split every string of an operator into its alpha and beta sub-strings.

    Args:
        op: Folded fermionic operator.
        ci_info: Information about the CI space.

    Returns:
        Spin-factorized operator, or None if it cannot be factorized over this CI space.
    """
    if not ci_info.is_spin_product or ci_info.num_active_orbs > MAX_DENSE_LOOKUP_ORBS:
        return None
    num_active_orbs = ci_info.num_active_orbs
    num_terms = len(op.operators)
    alpha_start = np.empty(num_terms, dtype=np.int64)
    alpha_stop = np.empty(num_terms, dtype=np.int64)
    beta_start = np.empty(num_terms, dtype=np.int64)
    beta_stop = np.empty(num_terms, dtype=np.int64)
    factor = np.empty(num_terms, dtype=np.float64)
    is_pure_alpha = np.empty(num_terms, dtype=np.bool_)
    is_pure_beta = np.empty(num_terms, dtype=np.bool_)
    # This loop runs once per string of the operator, so the two cache lookups are done
    # against the dict directly rather than through get_spin_sub_string_slice, which would
    # otherwise be two Python calls per string.
    sub_string_cache = ci_info.spin_op_cache
    for term, (op_key, fac) in enumerate(op.operators.items()):
        split = split_spin_string(op_key, num_active_orbs)
        if split is None:
            # A single non spin conserving string makes the whole operator fall back.
            return None
        alpha_sub, beta_sub, sign = split
        alpha_slice = sub_string_cache.get((True, alpha_sub[0], alpha_sub[1]))
        if alpha_slice is None:
            alpha_slice = get_spin_sub_string_slice(ci_info, alpha_sub, True)
        beta_slice = sub_string_cache.get((False, beta_sub[0], beta_sub[1]))
        if beta_slice is None:
            beta_slice = get_spin_sub_string_slice(ci_info, beta_sub, False)
        alpha_start[term] = alpha_slice[0]
        alpha_stop[term] = alpha_slice[1]
        beta_start[term] = beta_slice[0]
        beta_stop[term] = beta_slice[1]
        factor[term] = fac * sign
        is_pure_alpha[term] = beta_sub == IDENTITY_SUB_STRING
        is_pure_beta[term] = alpha_sub == IDENTITY_SUB_STRING
    # A term acting on one spin only is applied as a matrix on that side of the state, a term
    # acting on both couples the two, so the three groups are kept apart.
    pure_alpha = np.flatnonzero(is_pure_alpha & ~is_pure_beta)
    pure_beta = np.flatnonzero(is_pure_beta & ~is_pure_alpha)
    mixed = np.flatnonzero(~is_pure_alpha & ~is_pure_beta)
    # A term that is the identity on both spins is a plain scaling of the state. It is put with
    # the alpha only terms, whose map is then the identity map, which handles it correctly.
    identity = np.flatnonzero(is_pure_alpha & is_pure_beta)
    pure_alpha = np.concatenate((pure_alpha, identity))
    alpha_src, alpha_dst, alpha_sign = get_spin_arena(ci_info, True)
    beta_src, beta_dst, beta_sign = get_spin_arena(ci_info, False)
    return SpinFactorizedOperator(
        alpha_src,
        alpha_dst,
        alpha_sign,
        beta_src,
        beta_dst,
        beta_sign,
        alpha_start[pure_alpha],
        alpha_stop[pure_alpha],
        factor[pure_alpha],
        beta_start[pure_beta],
        beta_stop[pure_beta],
        factor[pure_beta],
        alpha_start[mixed],
        alpha_stop[mixed],
        beta_start[mixed],
        beta_stop[mixed],
        factor[mixed],
    )


@nb.jit(nopython=True)
def build_pure_spin_matrix(
    matrix: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    sign: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    r"""Sum every term acting on one spin only into a single matrix over that spin's strings.

    All of those terms act on the same side of the state, so their sum is one matrix and the
    whole group is applied with one matrix multiplication. The sum is also a compression: the
    terms typically hold far more excitations than the matrix has entries.

    Args:
        matrix: Matrix to accumulate into, over the strings of one spin.
        src: Spin string indices a sub-string maps from.
        dst: Spin string indices a sub-string maps to.
        sign: Phase of the sub-string application.
        starts: Start of each term's slice into the arena.
        stops: End of each term's slice into the arena.
        factors: Factor in front of each term.

    Returns:
        Matrix over the strings of one spin.
    """
    for term in range(len(factors)):
        fac = factors[term]
        for k in range(starts[term], stops[term]):
            matrix[dst[k], src[k]] += fac * sign[k]
    return matrix


@nb.jit(nopython=True)
def apply_mixed_terms(
    state: np.ndarray,
    tmp_state: np.ndarray,
    num_beta_strings: int,
    alpha_src: np.ndarray,
    alpha_dst: np.ndarray,
    alpha_sign: np.ndarray,
    beta_src: np.ndarray,
    beta_dst: np.ndarray,
    beta_sign: np.ndarray,
    alpha_start: np.ndarray,
    alpha_stop: np.ndarray,
    beta_start: np.ndarray,
    beta_stop: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    r"""Apply the terms that act on both spins, pairing the surviving strings of each.

    The determinant index is :math:`I = I_\alpha N_\beta + I_\beta`, so a term only has to walk
    the surviving alpha and beta strings and can combine them by arithmetic.

    Args:
        state: Original state.
        tmp_state: New state.
        num_beta_strings: Number of beta strings.
        alpha_src: Alpha string indices an alpha sub-string maps from.
        alpha_dst: Alpha string indices an alpha sub-string maps to.
        alpha_sign: Phase of the alpha sub-string application.
        beta_src: Beta string indices a beta sub-string maps from.
        beta_dst: Beta string indices a beta sub-string maps to.
        beta_sign: Phase of the beta sub-string application.
        alpha_start: Start of each term's slice into the alpha arena.
        alpha_stop: End of each term's slice into the alpha arena.
        beta_start: Start of each term's slice into the beta arena.
        beta_stop: End of each term's slice into the beta arena.
        factors: Factor in front of each term.

    Returns:
        New state.
    """
    for term in range(len(factors)):
        factor = factors[term]
        for idx_a in range(alpha_start[term], alpha_stop[term]):
            src = alpha_src[idx_a] * num_beta_strings
            dst = alpha_dst[idx_a] * num_beta_strings
            fac = factor * alpha_sign[idx_a]
            for idx_b in range(beta_start[term], beta_stop[term]):
                tmp_state[dst + beta_dst[idx_b]] += fac * beta_sign[idx_b] * state[src + beta_src[idx_b]]
    return tmp_state


@nb.jit(nopython=True)
def apply_pure_spin_terms(
    state: np.ndarray,
    tmp_state: np.ndarray,
    num_alpha_strings: int,
    num_beta_strings: int,
    is_alpha: bool,
    src: np.ndarray,
    dst: np.ndarray,
    sign: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """Apply the terms that act on one spin only, as a sparse scatter.

    Used when the dense matrix of that spin would be too wasteful, see use_dense_spin_matrix.
    An alpha only term moves a whole beta block of the state at once, while a beta only term
    touches every alpha block at one offset, with a stride.

    Args:
        state: Original state.
        tmp_state: New state.
        num_alpha_strings: Number of alpha strings.
        num_beta_strings: Number of beta strings.
        is_alpha: The terms act on the alpha strings, otherwise on the beta ones.
        src: Spin string indices a sub-string maps from.
        dst: Spin string indices a sub-string maps to.
        sign: Phase of the sub-string application.
        starts: Start of each term's slice into the arena.
        stops: End of each term's slice into the arena.
        factors: Factor in front of each term.

    Returns:
        New state.
    """
    for term in range(len(factors)):
        factor = factors[term]
        for k in range(starts[term], stops[term]):
            fac = factor * sign[k]
            if is_alpha:
                src_offset = src[k] * num_beta_strings
                dst_offset = dst[k] * num_beta_strings
                for i in range(num_beta_strings):
                    tmp_state[dst_offset + i] += fac * state[src_offset + i]
            else:
                src_offset = src[k]
                dst_offset = dst[k]
                for i in range(num_alpha_strings):
                    tmp_state[i * num_beta_strings + dst_offset] += (
                        fac * state[i * num_beta_strings + src_offset]
                    )
    return tmp_state


def build_sigma3_layout(
    factorized: SpinFactorizedOperator, num_alpha_strings: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Rewrite the terms acting on both spins as a contraction over their sub-strings.

    Those terms are a sum of products, and only a few distinct sub-strings appear in them, so
    the whole group is a coefficient matrix over (alpha sub-string, beta sub-string) pairs,

    .. math::
        \hat{O}_\text{mixed} = \sum_{ab}g_{ab}\hat{A}_a\otimes\hat{B}_b

    That is what turns the group into a dense contraction, see apply_sigma3_terms. The start of
    a sub-string's slice of the arena identifies it uniquely, so it is used as its name here.

    The alpha excitations are regrouped by the alpha string they produce rather than by the
    sub-string they belong to, because the contraction is done one output alpha string at a
    time. The groups are padded to a common width with a zero phase, which contributes nothing.

    Args:
        factorized: Spin-factorized operator.
        num_alpha_strings: Number of alpha strings.

    Returns:
        Coefficient matrix, alpha sub-string index, alpha source string, alpha phase, and the
        beta offsets, sources, targets and phases.
    """
    # A sub-string whose every excitation is killed owns an empty slice, and an empty slice
    # starts where the next one does, so a sub-string is named by both ends of its slice. Empty
    # sub-strings then share one name, which is harmless because they contribute nothing.
    alpha_key = factorized.mixed_alpha_start * (len(factorized.alpha_src) + 1) + (factorized.mixed_alpha_stop)
    beta_key = factorized.mixed_beta_start * (len(factorized.beta_src) + 1) + (factorized.mixed_beta_stop)
    _, alpha_first, alpha_id = np.unique(alpha_key, return_index=True, return_inverse=True)
    _, beta_first, beta_id = np.unique(beta_key, return_index=True, return_inverse=True)
    alpha_name = factorized.mixed_alpha_start[alpha_first]
    beta_name = factorized.mixed_beta_start[beta_first]
    num_alpha_subs = len(alpha_name)
    num_beta_subs = len(beta_name)
    # Terms sharing a pair of sub-strings differ only by their factor, so they add up.
    coefficients = np.bincount(
        alpha_id * num_beta_subs + beta_id,
        weights=factorized.mixed_factor,
        minlength=num_alpha_subs * num_beta_subs,
    ).reshape(num_alpha_subs, num_beta_subs)
    alpha_stop = factorized.mixed_alpha_stop[alpha_first]
    beta_stop = factorized.mixed_beta_stop[beta_first]

    alpha_entries = np.concatenate(
        [np.arange(lo, hi) for lo, hi in zip(alpha_name, alpha_stop)]
        if num_alpha_subs > 0
        else [np.zeros(0, dtype=np.int64)]
    )
    owner = np.repeat(np.arange(num_alpha_subs), alpha_stop - alpha_name)
    target = factorized.alpha_dst[alpha_entries]
    order = np.argsort(target, kind="stable")
    target = target[order]
    counts = np.bincount(target, minlength=num_alpha_strings)
    width = int(counts.max()) if len(counts) > 0 else 0
    group_start = np.zeros(num_alpha_strings + 1, dtype=np.int64)
    group_start[1:] = np.cumsum(counts)
    slot = np.arange(len(target), dtype=np.int64) - np.repeat(group_start[:-1], counts)
    coupling_sub = np.zeros((num_alpha_strings, max(width, 1)), dtype=np.int64)
    coupling_src = np.zeros((num_alpha_strings, max(width, 1)), dtype=np.int64)
    coupling_sign = np.zeros((num_alpha_strings, max(width, 1)))
    coupling_sub[target, slot] = owner[order]
    coupling_src[target, slot] = factorized.alpha_src[alpha_entries][order]
    coupling_sign[target, slot] = factorized.alpha_sign[alpha_entries][order]

    beta_entries = np.concatenate(
        [np.arange(lo, hi) for lo, hi in zip(beta_name, beta_stop)]
        if num_beta_subs > 0
        else [np.zeros(0, dtype=np.int64)]
    )
    beta_offsets = np.zeros(num_beta_subs + 1, dtype=np.int64)
    beta_offsets[1:] = np.cumsum(beta_stop - beta_name)
    return (
        coefficients,
        coupling_sub,
        coupling_src,
        coupling_sign,
        beta_offsets,
        factorized.beta_src[beta_entries],
        factorized.beta_dst[beta_entries],
        factorized.beta_sign[beta_entries],
    )


@nb.jit(nopython=True)
def apply_sigma3_terms(
    state: np.ndarray,
    tmp_state: np.ndarray,
    num_alpha_strings: int,
    num_beta_strings: int,
    coefficients: np.ndarray,
    coupling_sub: np.ndarray,
    coupling_src: np.ndarray,
    coupling_sign: np.ndarray,
    beta_offsets: np.ndarray,
    beta_src: np.ndarray,
    beta_dst: np.ndarray,
    beta_sign: np.ndarray,
) -> np.ndarray:
    r"""Apply the terms acting on both spins as a dense contraction, one alpha string at a time.

    For a single output alpha string the sum over terms collapses to three steps. Gather the
    rows of the state that feed it, contract them against the coefficient matrix over the
    sub-strings, and scatter the result through the beta excitations.

    The gather and the scatter are cheap, and the contraction between them is a matrix
    multiplication, which is what makes this faster than pairing the surviving strings of every
    term one by one. Only the alpha sub-strings that actually reach this alpha string take part,
    so the contraction stays narrow.

    Args:
        state: Original state.
        tmp_state: New state.
        num_alpha_strings: Number of alpha strings.
        num_beta_strings: Number of beta strings.
        coefficients: Factor of each (alpha sub-string, beta sub-string) pair.
        coupling_sub: Alpha sub-string index of each excitation into an alpha string.
        coupling_src: Alpha string each excitation into an alpha string comes from.
        coupling_sign: Phase of each excitation into an alpha string.
        beta_offsets: Start of each beta sub-string's excitations.
        beta_src: Beta string indices a beta sub-string maps from.
        beta_dst: Beta string indices a beta sub-string maps to.
        beta_sign: Phase of the beta sub-string application.

    Returns:
        New state.
    """
    num_beta_subs = coefficients.shape[1]
    width = coupling_sub.shape[1]
    gathered = np.zeros((width, num_beta_strings))
    coefficients_t = np.zeros((num_beta_subs, width))
    for dst_alpha in range(num_alpha_strings):
        for slot in range(width):
            sign = coupling_sign[dst_alpha, slot]
            src = coupling_src[dst_alpha, slot] * num_beta_strings
            for i in range(num_beta_strings):
                gathered[slot, i] = sign * state[src + i]
            sub = coupling_sub[dst_alpha, slot]
            for idx_b in range(num_beta_subs):
                coefficients_t[idx_b, slot] = coefficients[sub, idx_b]
        combined = np.dot(coefficients_t, gathered)
        base = dst_alpha * num_beta_strings
        for idx_b in range(num_beta_subs):
            for k in range(beta_offsets[idx_b], beta_offsets[idx_b + 1]):
                tmp_state[base + beta_dst[k]] += beta_sign[k] * combined[idx_b, beta_src[k]]
    return tmp_state


def use_sigma3(
    num_alpha_strings: int,
    num_beta_strings: int,
    num_beta_subs: int,
    width: int,
    num_pair_products: int,
) -> bool:
    """Decide whether to apply the terms acting on both spins as a dense contraction.

    The contraction is dense over the sub-strings, so it touches more elements than pairing the
    surviving strings of each term does. It is still much faster per element, so it is used
    while the excess stays bounded.

    Args:
        num_alpha_strings: Number of alpha strings.
        num_beta_strings: Number of beta strings.
        num_beta_subs: Number of distinct beta sub-strings.
        width: Number of alpha excitations reaching one alpha string, padded.
        num_pair_products: Number of string pairs the term by term algorithm would visit.

    Returns:
        True if the group should be applied as a dense contraction.
    """
    dense_work = num_alpha_strings * num_beta_subs * width * num_beta_strings
    return dense_work <= 8 * max(num_pair_products, 1)


def use_dense_spin_matrix(num_strings: int, num_other_strings: int, num_excitations: int) -> bool:
    """Decide whether to apply a one-spin group as a dense matrix instead of a sparse scatter.

    The matrix is dense over the spin's strings, so it can hold more entries than the terms have
    excitations. That is still a win, because the matrix multiplication runs an order of
    magnitude faster per entry than the scatter, but only while the excess stays bounded. The
    second test keeps the matrix from dwarfing the state vector when the two spin spaces are
    very different in size.

    Args:
        num_strings: Number of strings of the spin the matrix is over.
        num_other_strings: Number of strings of the other spin.
        num_excitations: Number of excitations the terms of this spin hold in total.

    Returns:
        True if the group should be applied as a dense matrix.
    """
    return num_strings * num_strings <= 8 * max(num_excitations, 1) and num_strings <= 8 * max(
        num_other_strings, 1
    )


class PreparedOperator:
    __slots__ = ("alpha_matrix", "beta_matrix", "ci_info", "factorized", "sigma3_layout")

    def __init__(
        self,
        factorized: SpinFactorizedOperator,
        ci_info: CI_Info,
        alpha_matrix: np.ndarray | None,
        beta_matrix: np.ndarray | None,
        sigma3_layout: tuple[np.ndarray, ...] | None,
    ) -> None:
        """Hold everything about an operator that does not depend on the state it acts on.

        A state-averaged wave function applies the same operator to every one of its states, so
        the dense matrices and the contraction layout are built once here and reused.

        Args:
            factorized: Spin-factorized operator.
            ci_info: Information about the CI space.
            alpha_matrix: Matrix of the alpha only terms, or None to scatter them instead.
            beta_matrix: Matrix of the beta only terms, or None to scatter them instead.
            sigma3_layout: Layout of the mixed terms, or None to pair their strings instead.
        """
        self.factorized = factorized
        self.ci_info = ci_info
        self.alpha_matrix = alpha_matrix
        self.beta_matrix = beta_matrix
        self.sigma3_layout = sigma3_layout


def prepare_factorized_operator(op: FermionicOperator, ci_info: CI_Info) -> PreparedOperator | None:
    """Do all the work on an operator that does not depend on the state it will act on.

    Args:
        op: Folded fermionic operator.
        ci_info: Information about the CI space.

    Returns:
        Prepared operator, or None if it cannot be factorized over this CI space.
    """
    factorized = factorize_operator(op, ci_info)
    if factorized is None:
        return None
    num_alpha_strings = ci_info.num_alpha_strings
    num_beta_strings = ci_info.num_beta_strings
    matrices: list[np.ndarray | None] = []
    for is_alpha, starts, stops, factors in (
        (
            True,
            factorized.pure_alpha_start,
            factorized.pure_alpha_stop,
            factorized.pure_alpha_factor,
        ),
        (False, factorized.pure_beta_start, factorized.pure_beta_stop, factorized.pure_beta_factor),
    ):
        num_strings = num_alpha_strings if is_alpha else num_beta_strings
        num_other_strings = num_beta_strings if is_alpha else num_alpha_strings
        if len(factors) == 0 or not use_dense_spin_matrix(
            num_strings, num_other_strings, int(np.sum(stops - starts))
        ):
            matrices.append(None)
            continue
        src, dst, sign = (
            (factorized.alpha_src, factorized.alpha_dst, factorized.alpha_sign)
            if is_alpha
            else (factorized.beta_src, factorized.beta_dst, factorized.beta_sign)
        )
        matrices.append(
            build_pure_spin_matrix(
                np.zeros((num_strings, num_strings)), src, dst, sign, starts, stops, factors
            )
        )
    sigma3_layout = None
    if len(factorized.mixed_factor) != 0:
        layout = build_sigma3_layout(factorized, num_alpha_strings)
        num_pair_products = int(
            np.sum(
                (factorized.mixed_alpha_stop - factorized.mixed_alpha_start)
                * (factorized.mixed_beta_stop - factorized.mixed_beta_start)
            )
        )
        if use_sigma3(
            num_alpha_strings,
            num_beta_strings,
            layout[0].shape[1],
            layout[1].shape[1],
            num_pair_products,
        ):
            sigma3_layout = layout
    return PreparedOperator(factorized, ci_info, matrices[0], matrices[1], sigma3_layout)


def apply_prepared_operator(
    prepared: PreparedOperator, state: np.ndarray, tmp_state: np.ndarray
) -> np.ndarray:
    """Apply a prepared operator to one state.

    Args:
        prepared: Prepared operator.
        state: Original state.
        tmp_state: New state, assumed to be zeroed.

    Returns:
        New state.
    """
    factorized = prepared.factorized
    num_alpha_strings = prepared.ci_info.num_alpha_strings
    num_beta_strings = prepared.ci_info.num_beta_strings
    # The state is stored as I = I_alpha*N_beta + I_beta, so it is already a matrix over the two
    # spin string spaces and needs no copy to be seen as one.
    state_matrix = np.ascontiguousarray(state).reshape(num_alpha_strings, num_beta_strings)
    tmp_matrix = tmp_state.reshape(num_alpha_strings, num_beta_strings)
    for is_alpha, matrix, starts, stops, factors in (
        (
            True,
            prepared.alpha_matrix,
            factorized.pure_alpha_start,
            factorized.pure_alpha_stop,
            factorized.pure_alpha_factor,
        ),
        (
            False,
            prepared.beta_matrix,
            factorized.pure_beta_start,
            factorized.pure_beta_stop,
            factorized.pure_beta_factor,
        ),
    ):
        if matrix is not None:
            if is_alpha:
                tmp_matrix += matrix @ state_matrix
            else:
                tmp_matrix += state_matrix @ matrix.T
        elif len(factors) != 0:
            src, dst, sign = (
                (factorized.alpha_src, factorized.alpha_dst, factorized.alpha_sign)
                if is_alpha
                else (factorized.beta_src, factorized.beta_dst, factorized.beta_sign)
            )
            apply_pure_spin_terms(
                state,
                tmp_state,
                num_alpha_strings,
                num_beta_strings,
                is_alpha,
                src,
                dst,
                sign,
                starts,
                stops,
                factors,
            )
    if len(factorized.mixed_factor) == 0:
        return tmp_state
    if prepared.sigma3_layout is not None:
        apply_sigma3_terms(state, tmp_state, num_alpha_strings, num_beta_strings, *prepared.sigma3_layout)
        return tmp_state
    apply_mixed_terms(
        state,
        tmp_state,
        num_beta_strings,
        factorized.alpha_src,
        factorized.alpha_dst,
        factorized.alpha_sign,
        factorized.beta_src,
        factorized.beta_dst,
        factorized.beta_sign,
        factorized.mixed_alpha_start,
        factorized.mixed_alpha_stop,
        factorized.mixed_beta_start,
        factorized.mixed_beta_stop,
        factorized.mixed_factor,
    )
    return tmp_state


def propagate_state_factorized(
    op: FermionicOperator, state: np.ndarray, ci_info: CI_Info, tmp_state: np.ndarray
) -> np.ndarray | None:
    """Apply a folded operator to a state using the spin-factorized algebra.

    Args:
        op: Folded fermionic operator.
        state: Original state.
        ci_info: Information about the CI space.
        tmp_state: New state, assumed to be zeroed.

    Returns:
        New state, or None if the operator cannot be factorized over this CI space.
    """
    prepared = prepare_factorized_operator(op, ci_info)
    if prepared is None:
        return None
    return apply_prepared_operator(prepared, state, tmp_state)


def propagate_state_SA_factorized(
    op: FermionicOperator, states: np.ndarray, ci_info: CI_Info, tmp_states: np.ndarray
) -> np.ndarray | None:
    """Apply a folded operator to every state of a state-averaged wave function.

    The operator is the same for every state, so everything that does not depend on the state
    is done once and only the application is repeated.

    Args:
        op: Folded fermionic operator.
        states: Original states, one per row.
        ci_info: Information about the CI space.
        tmp_states: New states, assumed to be zeroed.

    Returns:
        New states, or None if the operator cannot be factorized over this CI space.
    """
    prepared = prepare_factorized_operator(op, ci_info)
    if prepared is None:
        return None
    for state, tmp_state in zip(states, tmp_states):
        apply_prepared_operator(prepared, state, tmp_state)
    return tmp_states
