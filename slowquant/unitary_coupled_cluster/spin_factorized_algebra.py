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
        "alpha_is_identity",
        "alpha_sign",
        "alpha_src",
        "beta_dst",
        "beta_is_identity",
        "beta_sign",
        "beta_src",
        "term_alpha_start",
        "term_alpha_stop",
        "term_beta_start",
        "term_beta_stop",
        "term_factor",
        "term_is_pure_alpha",
        "term_is_pure_beta",
    )

    def __init__(
        self,
        alpha_src: np.ndarray,
        alpha_dst: np.ndarray,
        alpha_sign: np.ndarray,
        beta_src: np.ndarray,
        beta_dst: np.ndarray,
        beta_sign: np.ndarray,
        term_alpha_start: np.ndarray,
        term_alpha_stop: np.ndarray,
        term_beta_start: np.ndarray,
        term_beta_stop: np.ndarray,
        term_factor: np.ndarray,
        term_is_pure_alpha: np.ndarray,
        term_is_pure_beta: np.ndarray,
    ) -> None:
        """Initialize the spin-factorized form of a fermionic operator.

        The alpha and beta arrays hold the excitation maps of every spin sub-string the operator
        uses, laid out back to back. Each term of the operator is a slice into each of them.

        Args:
            alpha_src: Alpha string indices an alpha sub-string maps from.
            alpha_dst: Alpha string indices an alpha sub-string maps to.
            alpha_sign: Phase of the alpha sub-string application.
            beta_src: Beta string indices a beta sub-string maps from.
            beta_dst: Beta string indices a beta sub-string maps to.
            beta_sign: Phase of the beta sub-string application.
            term_alpha_start: Start of the term's slice into the alpha arrays.
            term_alpha_stop: End of the term's slice into the alpha arrays.
            term_beta_start: Start of the term's slice into the beta arrays.
            term_beta_stop: End of the term's slice into the beta arrays.
            term_factor: Factor in front of the term, including the block reordering phase.
            term_is_pure_alpha: True if the term acts as the identity on the beta string.
            term_is_pure_beta: True if the term acts as the identity on the alpha string.
        """
        self.alpha_src = alpha_src
        self.alpha_dst = alpha_dst
        self.alpha_sign = alpha_sign
        self.beta_src = beta_src
        self.beta_dst = beta_dst
        self.beta_sign = beta_sign
        self.term_alpha_start = term_alpha_start
        self.term_alpha_stop = term_alpha_stop
        self.term_beta_start = term_beta_start
        self.term_beta_stop = term_beta_stop
        self.term_factor = term_factor
        self.term_is_pure_alpha = term_is_pure_alpha
        self.term_is_pure_beta = term_is_pure_beta


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


def get_spin_excitation_map(
    ci_info: CI_Info, sub_string: tuple[tuple[int, ...], tuple[int, ...]], is_alpha: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the excitation map of one spin sub-string, building it on first use.

    The map depends only on the spin string space and the sub-string, never on the factor in
    front of it, so it is cached on the CI space and reused by every later operator.

    Args:
        ci_info: Information about the CI space.
        sub_string: Spin sub-string, tuple of creation and annihilation spatial indices.
        is_alpha: The sub-string acts on the alpha strings, otherwise on the beta ones.

    Returns:
        Spin string indices mapped from, spin string indices mapped to, and the phases.
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
    ci_info.spin_op_cache[cache_key] = spin_map
    return spin_map


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
    alpha_maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    beta_maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    alpha_slice: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, int]] = {}
    beta_slice: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, int]] = {}
    alpha_len = 0
    beta_len = 0
    term_alpha_start = []
    term_alpha_stop = []
    term_beta_start = []
    term_beta_stop = []
    term_factor = []
    term_is_pure_alpha = []
    term_is_pure_beta = []
    for op_key, fac in op.operators.items():
        split = split_spin_string(op_key, num_active_orbs)
        if split is None:
            # A single non spin conserving string makes the whole operator fall back.
            return None
        alpha_sub, beta_sub, sign = split
        if alpha_sub not in alpha_slice:
            spin_map = get_spin_excitation_map(ci_info, alpha_sub, True)
            alpha_maps.append(spin_map)
            alpha_slice[alpha_sub] = (alpha_len, alpha_len + len(spin_map[0]))
            alpha_len += len(spin_map[0])
        if beta_sub not in beta_slice:
            spin_map = get_spin_excitation_map(ci_info, beta_sub, False)
            beta_maps.append(spin_map)
            beta_slice[beta_sub] = (beta_len, beta_len + len(spin_map[0]))
            beta_len += len(spin_map[0])
        term_alpha_start.append(alpha_slice[alpha_sub][0])
        term_alpha_stop.append(alpha_slice[alpha_sub][1])
        term_beta_start.append(beta_slice[beta_sub][0])
        term_beta_stop.append(beta_slice[beta_sub][1])
        term_factor.append(fac * sign)
        term_is_pure_alpha.append(beta_sub == IDENTITY_SUB_STRING)
        term_is_pure_beta.append(alpha_sub == IDENTITY_SUB_STRING)

    def concatenate(maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]], entry: int) -> np.ndarray:
        if len(maps) == 0:
            return np.zeros(0, dtype=np.int64 if entry < 2 else np.float64)
        return np.concatenate([spin_map[entry] for spin_map in maps])

    return SpinFactorizedOperator(
        concatenate(alpha_maps, 0),
        concatenate(alpha_maps, 1),
        concatenate(alpha_maps, 2),
        concatenate(beta_maps, 0),
        concatenate(beta_maps, 1),
        concatenate(beta_maps, 2),
        np.array(term_alpha_start, dtype=np.int64),
        np.array(term_alpha_stop, dtype=np.int64),
        np.array(term_beta_start, dtype=np.int64),
        np.array(term_beta_stop, dtype=np.int64),
        np.array(term_factor, dtype=np.float64),
        np.array(term_is_pure_alpha, dtype=np.bool_),
        np.array(term_is_pure_beta, dtype=np.bool_),
    )


@nb.jit(nopython=True)
def apply_factorized_operator(
    state: np.ndarray,
    tmp_state: np.ndarray,
    num_alpha_strings: int,
    num_beta_strings: int,
    alpha_src: np.ndarray,
    alpha_dst: np.ndarray,
    alpha_sign: np.ndarray,
    beta_src: np.ndarray,
    beta_dst: np.ndarray,
    beta_sign: np.ndarray,
    term_alpha_start: np.ndarray,
    term_alpha_stop: np.ndarray,
    term_beta_start: np.ndarray,
    term_beta_stop: np.ndarray,
    term_factor: np.ndarray,
    term_is_pure_alpha: np.ndarray,
    term_is_pure_beta: np.ndarray,
) -> np.ndarray:
    r"""Apply a spin-factorized operator to a state for a single state wave function.

    The determinant index is :math:`I = I_\alpha N_\beta + I_\beta`, so a term only has to walk
    the surviving alpha and beta strings and can combine them by arithmetic. There are three
    cases, depending on which spin the term acts on,

    #. Only alpha, so a whole beta block of the state moves at once.

    #. Only beta, so every alpha block is touched at one offset, with a stride.

    #. Both, so the surviving alpha and beta strings are combined pairwise.

    Args:
        state: Original state.
        tmp_state: New state.
        num_alpha_strings: Number of alpha strings.
        num_beta_strings: Number of beta strings.
        alpha_src: Alpha string indices an alpha sub-string maps from.
        alpha_dst: Alpha string indices an alpha sub-string maps to.
        alpha_sign: Phase of the alpha sub-string application.
        beta_src: Beta string indices a beta sub-string maps from.
        beta_dst: Beta string indices a beta sub-string maps to.
        beta_sign: Phase of the beta sub-string application.
        term_alpha_start: Start of the term's slice into the alpha arrays.
        term_alpha_stop: End of the term's slice into the alpha arrays.
        term_beta_start: Start of the term's slice into the beta arrays.
        term_beta_stop: End of the term's slice into the beta arrays.
        term_factor: Factor in front of the term.
        term_is_pure_alpha: True if the term acts as the identity on the beta string.
        term_is_pure_beta: True if the term acts as the identity on the alpha string.

    Returns:
        New state.
    """
    for term in range(len(term_factor)):
        factor = term_factor[term]
        if term_is_pure_alpha[term]:
            for idx_a in range(term_alpha_start[term], term_alpha_stop[term]):
                src = alpha_src[idx_a] * num_beta_strings
                dst = alpha_dst[idx_a] * num_beta_strings
                fac = factor * alpha_sign[idx_a]
                for i in range(num_beta_strings):
                    tmp_state[dst + i] += fac * state[src + i]
        elif term_is_pure_beta[term]:
            for idx_b in range(term_beta_start[term], term_beta_stop[term]):
                src = beta_src[idx_b]
                dst = beta_dst[idx_b]
                fac = factor * beta_sign[idx_b]
                for i in range(num_alpha_strings):
                    tmp_state[i * num_beta_strings + dst] += fac * state[i * num_beta_strings + src]
        else:
            for idx_a in range(term_alpha_start[term], term_alpha_stop[term]):
                src = alpha_src[idx_a] * num_beta_strings
                dst = alpha_dst[idx_a] * num_beta_strings
                fac = factor * alpha_sign[idx_a]
                for idx_b in range(term_beta_start[term], term_beta_stop[term]):
                    tmp_state[dst + beta_dst[idx_b]] += fac * beta_sign[idx_b] * state[src + beta_src[idx_b]]
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
    factorized = factorize_operator(op, ci_info)
    if factorized is None:
        return None
    return apply_factorized_operator(
        state,
        tmp_state,
        ci_info.num_alpha_strings,
        ci_info.num_beta_strings,
        factorized.alpha_src,
        factorized.alpha_dst,
        factorized.alpha_sign,
        factorized.beta_src,
        factorized.beta_dst,
        factorized.beta_sign,
        factorized.term_alpha_start,
        factorized.term_alpha_stop,
        factorized.term_beta_start,
        factorized.term_beta_stop,
        factorized.term_factor,
        factorized.term_is_pure_alpha,
        factorized.term_is_pure_beta,
    )
