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

With the CI vector held as a matrix over the two string spaces, :math:`C_{I_\alpha I_\beta}`, an
operator then splits into the same three contributions a determinant-based full CI program makes
its sigma vector from,

.. math::
    \boldsymbol{\sigma} = \underbrace{\boldsymbol{A}\boldsymbol{C}}_{\sigma_2}
        + \underbrace{\boldsymbol{C}\boldsymbol{B}^T}_{\sigma_1}
        + \underbrace{\sum_t c_t\boldsymbol{A}_t\boldsymbol{C}\boldsymbol{B}_t^T}_{\sigma_3}

The two one-spin terms are a single matrix multiplication each and the alpha-beta coupling term
is the expensive one, see apply_sigma3_terms. Nothing here is specific to the Hamiltonian: the
excitation generators of an ansatz and the one-electron operators of a property calculation go
through the same machinery, and a generator that is one excitation and its adjoint degenerates
into a sweep of Givens rotations over pairs of spin strings, see build_string_rotation_layout.

#. 10.1016/0009-2614(84)85513-X (determinant full CI over alpha and beta strings)
#. 10.1016/0010-4655(89)90033-7 (determinant full CI over alpha and beta strings)
#. 10.1063/1.455063 (the sigma1, sigma2 and sigma3 split)
#. Molecular Electronic-Structure Theory, Ch. 11,
   https://onlinelibrary.wiley.com/doi/book/10.1002/9781119019572
"""

from __future__ import annotations

import numba as nb
import numpy as np

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info, bitcount
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator

# Spin sub-string of an operator that acts as the identity on that spin.
IDENTITY_SUB_STRING: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
# Smallest inner size n_beta*w at which the sigma3 contraction reaches matrix multiplication
# speed. Below it the contraction is a strided vector update and pairing the surviving
# strings of each term is cheaper. Measured on Hamiltonians and excitation generators from
# CAS(4,4) to CAS(12,12), which sit two orders of magnitude apart on this scale.
CONTRACTION_MIN_MATMUL_SIZE = 128
# How much larger the one-spin matrix may be than the number of excitations it holds before it
# is cheaper to walk the excitations instead. A matrix product runs faster per element than a
# scattered walk, so the matrix pays off well before it is full. The two cases this separates sit
# two to three orders of magnitude apart, see use_dense_spin_matrix, so the exact value is not
# delicate.
DENSE_SPIN_MATRIX_FILL = 64


class SpinFactorizedOperator:
    r"""Operator split into an alpha and a beta part over a spin-product CI space.

    An :math:`S_z` conserving operator is a sum of products of a purely alpha and a purely beta
    string, see the module docstring,

    .. math::
        \hat{O} = \sum_t c_t\,\hat{A}_t\otimes\hat{B}_t

    With the CI vector held as a matrix over the two string spaces,
    :math:`C_{I_\alpha I_\beta}`, the three kinds of term act in three different ways, which is
    why they are stored apart,

    .. math::
        \begin{align}
        \hat{A}_t\otimes 1 &\rightarrow \boldsymbol{A}_t\boldsymbol{C}\\
        1\otimes\hat{B}_t &\rightarrow \boldsymbol{C}\boldsymbol{B}_t^T\\
        \hat{A}_t\otimes\hat{B}_t &\rightarrow \boldsymbol{A}_t\boldsymbol{C}\boldsymbol{B}_t^T
        \end{align}

    The first two are the one-spin contributions that the sigma-vector literature calls
    :math:`\sigma_1` and :math:`\sigma_2`, and the third is the alpha-beta coupling term
    :math:`\sigma_3`, which is the expensive one. This is the same split a determinant-based
    full CI program makes, applied here to an arbitrary folded operator rather than only to the
    Hamiltonian.

    The excitation maps themselves are not stored here. They live in the per-spin arenas of the
    CI space, shared by every operator over that space, and each term only records which slice
    of them it uses, see get_spin_sub_string_slice.

    #. 10.1016/0009-2614(84)85513-X (determinant full CI over alpha and beta strings)
    #. 10.1063/1.455063 (the sigma1, sigma2 and sigma3 split)
    """

    __slots__ = (
        "alpha_dst",
        "alpha_matrix",
        "alpha_sign",
        "alpha_src",
        "beta_dst",
        "beta_matrix",
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
        "sigma3_layout",
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
        # Filled in once the slices are known, see factorize_operator. They are the same for
        # every state the operator is applied to, which is what a state-averaged wave function
        # needs, and None means the corresponding group falls back to a scatter.
        self.alpha_matrix: np.ndarray | None = None
        self.beta_matrix: np.ndarray | None = None
        self.sigma3_layout: tuple[np.ndarray, ...] | None = None


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

    which for a spin conserving string is :math:`\left(-1\right)^{k_\alpha k_\beta}`, with
    :math:`k_\sigma` the number of creation operators of spin :math:`\sigma`. Moving the
    :math:`k_\beta` beta creation operators past the :math:`k_\alpha` alpha creation operators
    costs :math:`k_\alpha k_\beta` transpositions, and moving the annihilation operators back
    costs the same again, but the beta annihilation operators are moved past the alpha
    annihilation operators rather than the alpha creation ones, so the two do not cancel. The
    remaining two phases of the factorization are handled inside the spin blocks, see the module
    docstring.

    A string that is not :math:`S_z` conserving, one that moves an electron between the spins,
    does not map the spin product onto itself at all and has no such splitting.

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
    spin_str2idx: dict[int, int],
    a_string: np.ndarray,
    create_screen: np.ndarray,
    anni_idx: np.ndarray,
    num_active_orbs: int,
    parity_check: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Build the excitation map of one spin sub-string over a spin string space.

    A spin sub-string acts on the strings of its own spin as a signed one-to-one map,

    .. math::
        \hat{A}\left|J_\sigma\right> = \Gamma_{J_\sigma}\left|K_\sigma\right>
        \quad\text{or}\quad 0

    so all it needs is, for every string that survives the annihilation and creation screening,
    the string it is taken to and the phase :math:`\Gamma_{J_\sigma}=\pm 1` it picks up. That is
    the classic string excitation list of a determinant-based full CI program, here for one
    normal-ordered string of an arbitrary operator rather than for the single excitations of a
    Hamiltonian.

    The map depends only on the spin string space, never on the factor the operator puts in
    front of the string, and there are only as many entries as there are spin strings. That is
    what makes it worth building once and sharing, see get_spin_sub_string_slice.

    Follows the same two-step algorithm as apply_operator_serial, with the determinant replaced
    by a single spin string, see the module docstring for why that is exact. Screening and
    traversal order are the same, so the phase matches the unfactorized kernel term by term.

    #. 10.1016/0009-2614(84)85513-X (string excitation lists)

    Args:
        idx2spin_str: Maps spin string index to spin string.
        spin_str2idx: Maps spin string to spin string index.
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
        dst[num_survivors] = spin_str2idx[spin_str]
        sign[num_survivors] = 1.0 - 2.0 * (phase_changes & 1)
        num_survivors += 1
    return src[:num_survivors], dst[:num_survivors], sign[:num_survivors]


def get_spin_sub_string_slice(
    ci_info: CI_Info, sub_string: tuple[tuple[int, ...], tuple[int, ...]], is_alpha: bool
) -> tuple[int, int]:
    r"""Get the slice of the spin arena holding the excitation map of one spin sub-string.

    The map depends only on the spin string space and the sub-string, never on the factor in
    front of it, so it is built once and appended to the arena of that spin. Every later
    operator over the same CI space reuses it.

    That sharing is what keeps the cost of an operator away from the determinant count. A
    Hamiltonian over :math:`N` active orbitals has :math:`O\left(N^4\right)` strings but only
    :math:`O\left(N^2\right)` distinct sub-strings per spin, the one-electron excitations
    :math:`\hat{E}^\sigma_{pq}`, and each of their maps holds at most one entry per spin string.

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
        ci_info.alpha_str2idx_nb if is_alpha else ci_info.beta_str2idx_nb,
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

    The arena is the single flat copy of every string excitation list over this CI space, and a
    term of an operator is a slice of it rather than a map of its own. It only grows, so the
    concatenated form is rebuilt when a new sub-string has been added and reused otherwise.

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
    r"""Split every string of an operator into its alpha and beta sub-strings.

    The determinant basis is a product of an alpha and a beta string space,
    :math:`\left|I\right> = \left|I_\alpha\right>\otimes\left|I_\beta\right>`, and an
    :math:`S_z` conserving string factorizes the same way, so the whole operator becomes

    .. math::
        \hat{O} = \sum_t c_t\hat{A}_t\otimes\hat{B}_t

    For the active space Hamiltonian this is the familiar form behind a string-driven full CI
    sigma vector,

    .. math::
        \hat{H} = \sum_{pq}h_{pq}\hat{E}_{pq}
                + \frac{1}{2}\sum_{pqrs}g_{pqrs}\left(\hat{E}_{pq}\hat{E}_{rs}
                  - \delta_{qr}\hat{E}_{ps}\right)

    whose one-spin part gives :math:`\sigma_1` and :math:`\sigma_2` and whose
    :math:`\hat{E}^\alpha_{pq}\hat{E}^\beta_{rs}` part gives :math:`\sigma_3`. Nothing here is
    specific to the Hamiltonian though: any folded, :math:`S_z` conserving operator splits the
    same way, which is what lets the excitation generators of an ansatz and the one-electron
    operators of a property calculation go through the same machinery.

    The terms are split into those acting on one spin only, which are a matrix on one side of
    the CI vector, and those acting on both, which couple the two. The sub-string excitation
    maps live on the CI space and are shared, so this only records which slice of them each
    term uses, and then build_derived_forms decides how each group will be applied.

    #. 10.1016/0009-2614(84)85513-X (determinant full CI over alpha and beta strings)
    #. 10.1063/1.455063 (the sigma1, sigma2 and sigma3 split)

    Args:
        op: Folded fermionic operator.
        ci_info: Information about the CI space.

    Returns:
        Spin-factorized operator, or None if it cannot be factorized over this CI space.
    """
    if not ci_info.is_spin_product:
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
    factorized = SpinFactorizedOperator(
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
    build_derived_forms(factorized, ci_info)
    return factorized


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

    An operator that touches one spin only leaves the other string untouched, so with the CI
    vector seen as a matrix :math:`C_{I_\alpha I_\beta}` it acts on one side,

    .. math::
        \boldsymbol{\sigma} = \boldsymbol{A}\boldsymbol{C}\quad\text{or}\quad
        \boldsymbol{\sigma} = \boldsymbol{C}\boldsymbol{B}^T

    with the one-spin matrix collected from the excitation maps of the individual terms,

    .. math::
        A_{K_\alpha J_\alpha} = \sum_t c_t \Gamma^{(t)}_{J_\alpha}
        \delta_{K_\alpha, \hat{A}_t J_\alpha}

    These are the :math:`\sigma_1` and :math:`\sigma_2` contributions of a string-driven CI
    sigma vector, and writing them as one matrix product per spin is what that literature does
    too. All such terms share that side, so their sum is a single matrix and the whole group
    costs one matrix multiplication. Summing them first is also a compression, since a
    Hamiltonian holds far more excitations than the matrix has entries.

    #. 10.1063/1.455063 (the one-spin sigma1 and sigma2 contributions)

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

    These are the :math:`\sigma_3` terms, taken one term at a time rather than as a contraction,

    .. math::
        \sigma\left(K_\alpha,K_\beta\right) \mathrel{+}= \sum_t c_t\,
            \Gamma^{(t)}_{J_\alpha}\Gamma^{(t)}_{J_\beta}\,C\left(J_\alpha,J_\beta\right)

    where for term :math:`t` the pair :math:`\left(J_\alpha,J_\beta\right)` runs over the
    product of the alpha strings and the beta strings that survive it. The determinant index is
    :math:`I = I_\alpha N_\beta + I_\beta`, so a term only has to walk the surviving alpha and
    beta strings and can combine them by arithmetic, with no determinant lookup.

    This touches exactly the elements that contribute and nothing else, which is why it wins for
    a sparse operator such as a single excitation generator. A dense operator such as a
    Hamiltonian is better served by apply_sigma3_terms, see prefer_contraction_over_pairs.

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
    r"""Apply the terms that act on one spin only, as a sparse scatter.

    The same one-sided action as build_pure_spin_matrix,

    .. math::
        \sigma\left(K_\alpha,I_\beta\right) \mathrel{+}= \sum_t c_t\Gamma^{(t)}_{J_\alpha}
            C\left(J_\alpha,I_\beta\right),\qquad
        \left|K_\alpha\right> \propto \hat{A}_t\left|J_\alpha\right>

    but walked excitation by excitation instead of formed into a matrix, for an operator too
    sparse to fill one. Because the CI vector is stored as
    :math:`I = I_\alpha N_\beta + I_\beta`, an alpha excitation moves a whole contiguous beta
    block of it, while a beta excitation touches one entry of every alpha block.

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
    if is_alpha:
        # An alpha excitation moves a whole beta block, which is contiguous.
        for term in range(len(factors)):
            factor = factors[term]
            for k in range(starts[term], stops[term]):
                fac = factor * sign[k]
                src_offset = src[k] * num_beta_strings
                dst_offset = dst[k] * num_beta_strings
                for i in range(num_beta_strings):
                    tmp_state[dst_offset + i] += fac * state[src_offset + i]
    else:
        # A beta excitation touches one entry of every alpha block, so walking the excitations
        # on the outside strides through the whole state once per excitation and uses eight
        # bytes of every cache line it touches. Walking the alpha blocks on the outside instead
        # keeps one block, which is a contiguous row, in cache while all of the excitations are
        # applied to it. The contributions to an entry still arrive in the same order.
        for i in range(num_alpha_strings):
            base = i * num_beta_strings
            for term in range(len(factors)):
                factor = factors[term]
                for k in range(starts[term], stops[term]):
                    tmp_state[base + dst[k]] += factor * sign[k] * state[base + src[k]]
    return tmp_state


def build_sigma3_layout(
    factorized: SpinFactorizedOperator, num_alpha_strings: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Rewrite the terms acting on both spins as a contraction over their sub-strings.

    Those terms are a sum of products, and only a few distinct sub-strings appear in them, so
    the whole group is a coefficient matrix over (alpha sub-string, beta sub-string) pairs,

    .. math::
        \hat{O}_\text{mixed} = \sum_{ab}g_{ab}\hat{A}_a\otimes\hat{B}_b

    For a Hamiltonian the sub-strings are the one-electron excitations of each spin,
    :math:`\hat{A}_a = \hat{E}^\alpha_{pq}` and :math:`\hat{B}_b = \hat{E}^\beta_{rs}`, and
    :math:`g_{ab}` is the two-electron integral matrix :math:`g_{pqrs}`, which is the form the
    sigma vector literature writes this in. The number of sub-strings is then at most
    :math:`N^2` per spin however many strings the operator holds, so a Hamiltonian with
    :math:`O\left(N^4\right)` terms collapses onto an :math:`N^2\times N^2` matrix.

    That is what turns the group into a dense contraction, see apply_sigma3_terms. The start of
    a sub-string's slice of the arena identifies it uniquely, so it is used as its name here.

    The alpha excitations are regrouped by the alpha string they produce rather than by the
    sub-string they belong to, because the contraction is done one output alpha string at a
    time. The groups are padded to a common width with a zero phase, which contributes nothing.

    #. 10.1063/1.455063 (sigma3 as a contraction over one-electron excitations)

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

    This is the mixed spin part of a CI sigma vector, the term usually written

    .. math::
        \sigma_3\left(I_\alpha I_\beta\right) = \sum_{pqrs}g_{pqrs}
            \sum_{J_\alpha J_\beta}
            \left<I_\alpha\left|\hat{E}^\alpha_{pq}\right|J_\alpha\right>
            \left<I_\beta\left|\hat{E}^\beta_{rs}\right|J_\beta\right>
            C\left(J_\alpha J_\beta\right)

    Written per output alpha string it is three steps, and only the middle one costs anything,

    .. math::
        \begin{align}
        D_{a I_\beta} &= \Gamma_{J_\alpha}C\left(J_\alpha,I_\beta\right)
            &&\text{gather the rows reaching } I_\alpha\\
        F_{b I_\beta} &= \sum_a g_{ab} D_{a I_\beta}
            &&\text{contract over the alpha sub-strings}\\
        \sigma\left(I_\alpha,K_\beta\right) &\mathrel{+}= \Gamma_{J_\beta}F_{b J_\beta}
            &&\text{scatter through the beta excitations}
        \end{align}

    The gather and the scatter are cheap and the contraction between them is a matrix
    multiplication, which is what makes this faster than pairing the surviving strings of every
    term. Only the alpha sub-strings that actually reach this alpha string take part, so the
    contraction stays narrow rather than running over all of them, and the CI vector is read
    once per output alpha string rather than once per term.

    #. 10.1016/0009-2614(84)85513-X (the gather, contract and scatter structure)
    #. 10.1063/1.455063 (sigma3)

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


def prefer_contraction_over_pairs(factorized: SpinFactorizedOperator, ci_info: CI_Info) -> bool:
    r"""Decide whether the terms acting on both spins are dense enough to contract.

    The contraction walks the output alpha strings and, for each, multiplies the beta
    sub-string blocks of the CI matrix by the alpha excitations that reach that string,

    .. math::
        \sigma\left(I_\alpha,I_\beta\right) = \sum_{k_\alpha}\sum_{k_\beta}
            A_{k_\alpha k_\beta}\,C\left(k_\alpha,k_\beta\left(I_\beta\right)\right)

    a matrix multiplication whose inner dimensions are the number of distinct beta sub-strings
    :math:`n_\beta` and the number of alpha excitations reaching one alpha string :math:`w`.
    Pairing the surviving strings of each term instead touches only the elements that survive,
    but pays a scattered update for every one of them.

    The contraction always touches more elements than the pairing, so it pays off only when
    :math:`n_\beta w` is large enough for the multiplication to run at matrix multiplication
    speed. Below that it degenerates into a strided vector update and the extra elements are
    paid in full. A Hamiltonian fills its sub-string matrix and clears the bar at every active
    space size past the smallest, while a single excitation generator reaches :math:`n_\beta w`
    of a handful at any size. So this asks how dense the operator is, not how large the CI space
    is.

    The width is estimated from the average rather than built, so that the layout is only
    constructed when it is going to be used. Both branches are correct, so an occasional
    misjudgement costs a little time and nothing else.

    Args:
        factorized: Spin-factorized operator.
        ci_info: Information about the CI space.

    Returns:
        True if the group should be applied as a contraction.
    """
    # Many terms share a sub-string, and the excitations of a shared one are walked once, so
    # this counts distinct sub-strings rather than terms.
    alpha_first = np.unique(
        factorized.mixed_alpha_start * (len(factorized.alpha_src) + 1) + factorized.mixed_alpha_stop,
        return_index=True,
    )[1]
    num_alpha_entries = int(
        np.sum(factorized.mixed_alpha_stop[alpha_first] - factorized.mixed_alpha_start[alpha_first])
    )
    num_beta_subs = len(
        np.unique(factorized.mixed_beta_start * (len(factorized.beta_src) + 1) + factorized.mixed_beta_stop)
    )
    width = max(1, -(-num_alpha_entries // max(ci_info.num_alpha_strings, 1)))
    return num_beta_subs * width >= CONTRACTION_MIN_MATMUL_SIZE


def use_dense_spin_matrix(num_strings: int, num_other_strings: int, num_excitations: int) -> bool:
    r"""Decide whether a one-spin group is dense enough to be worth forming as a matrix.

    The matrix has :math:`N^2` entries while the terms hold only as many excitations as they
    hold, so forming one is worth it when the operator fills a reasonable fraction of it. A
    Hamiltonian does; a single excitation generator fills a couple of diagonals at any size.

    The matrix is worth forming well before it is full, because a matrix product runs faster per
    element than a walk over the excitations. Measured on hydrogen chains, a Hamiltonian needs
    N^2 of about 2, 4 and 8 times its excitation count at CAS(10,10), CAS(12,12) and CAS(14,14)
    and is 2.2 to 2.4 times faster dense at all three, while a single excitation needs 450 to
    6400 times and is 17 to 200 times slower dense. The bar sits between those, far from both.

    The second test keeps the matrix from dwarfing the CI vector itself when the two spin spaces
    are very different in size, as they are for a high spin state.

    Args:
        num_strings: Number of strings of the spin the matrix is over.
        num_other_strings: Number of strings of the other spin.
        num_excitations: Number of excitations the terms of this spin hold in total.

    Returns:
        True if the group should be applied as a dense matrix.
    """
    return num_strings * num_strings <= DENSE_SPIN_MATRIX_FILL * max(
        num_excitations, 1
    ) and num_strings <= 8 * max(num_other_strings, 1)


def build_derived_forms(factorized: SpinFactorizedOperator, ci_info: CI_Info) -> None:
    """Fill in the parts of an operator that do not depend on the state it will act on.

    The one-spin groups become a matrix each where that is worth it, see use_dense_spin_matrix,
    and the terms acting on both spins get their contraction layout where that is worth it, see
    prefer_contraction_over_pairs. Both choices only reorder the summation, so either branch
    gives the same numbers and a misjudgement costs time and nothing else.

    A state-averaged wave function applies the same operator to every one of its states, and an
    optimizer applies the same Hamiltonian at every iteration, so this is done once and reused.

    Args:
        factorized: Spin-factorized operator, updated in place.
        ci_info: Information about the CI space.
    """
    num_alpha_strings = ci_info.num_alpha_strings
    num_beta_strings = ci_info.num_beta_strings
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
            continue
        src, dst, sign = (
            (factorized.alpha_src, factorized.alpha_dst, factorized.alpha_sign)
            if is_alpha
            else (factorized.beta_src, factorized.beta_dst, factorized.beta_sign)
        )
        matrix = build_pure_spin_matrix(
            np.zeros((num_strings, num_strings)), src, dst, sign, starts, stops, factors
        )
        if is_alpha:
            factorized.alpha_matrix = matrix
        else:
            factorized.beta_matrix = matrix
    if len(factorized.mixed_factor) == 0:
        return
    if prefer_contraction_over_pairs(factorized, ci_info):
        factorized.sigma3_layout = build_sigma3_layout(factorized, num_alpha_strings)


def apply_factorized_operator(
    factorized: SpinFactorizedOperator, state: np.ndarray, ci_info: CI_Info, tmp_state: np.ndarray
) -> np.ndarray:
    r"""Apply a factorized operator to one state.

    Assembles the three contributions of the sigma vector,

    .. math::
        \boldsymbol{\sigma} = \underbrace{\boldsymbol{A}\boldsymbol{C}}_{\sigma_2}
            + \underbrace{\boldsymbol{C}\boldsymbol{B}^T}_{\sigma_1}
            + \underbrace{\sum_t c_t\boldsymbol{A}_t\boldsymbol{C}\boldsymbol{B}_t^T}_{\sigma_3}

    where the CI vector is reshaped into the matrix :math:`C_{I_\alpha I_\beta}` at no cost,
    because the determinant index is already :math:`I = I_\alpha N_\beta + I_\beta`. Each of the
    three has two kernels, a dense one and a sparse one, and which of them runs was decided once
    by build_derived_forms.

    Args:
        factorized: Spin-factorized operator.
        state: Original state.
        ci_info: Information about the CI space.
        tmp_state: New state, assumed to be zeroed.

    Returns:
        New state.
    """
    num_alpha_strings = ci_info.num_alpha_strings
    num_beta_strings = ci_info.num_beta_strings
    # The state is stored as I = I_alpha*N_beta + I_beta, so it is already a matrix over the two
    # spin string spaces and needs no copy to be seen as one.
    state_matrix = np.ascontiguousarray(state).reshape(num_alpha_strings, num_beta_strings)
    tmp_matrix = tmp_state.reshape(num_alpha_strings, num_beta_strings)
    for is_alpha, matrix, starts, stops, factors in (
        (
            True,
            factorized.alpha_matrix,
            factorized.pure_alpha_start,
            factorized.pure_alpha_stop,
            factorized.pure_alpha_factor,
        ),
        (
            False,
            factorized.beta_matrix,
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
    if factorized.sigma3_layout is not None:
        apply_sigma3_terms(state, tmp_state, num_alpha_strings, num_beta_strings, *factorized.sigma3_layout)
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


# Which spins a generator's rotation touches, which decides the kernel that applies it.
ROTATION_ALPHA = 0
ROTATION_BETA = 1
ROTATION_MIXED = 2


@nb.jit(nopython=True, cache=True)
def rotate_alpha_string_pairs(
    states: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    sign: np.ndarray,
    cos_theta: float,
    sin_theta: float,
) -> None:
    r"""Rotate paired alpha strings of the CI matrix in place, a row operation.

    An alpha-only generator leaves the beta string alone, so a single pair of alpha strings
    rotates two whole rows of :math:`C_{I_\alpha I_\beta}` against each other,

    .. math::
        \begin{pmatrix}C_{p,:}\\C_{q,:}\end{pmatrix} \leftarrow
        \begin{pmatrix}\cos\theta & -\Gamma\sin\theta\\
                       \Gamma\sin\theta & \cos\theta\end{pmatrix}
        \begin{pmatrix}C_{p,:}\\C_{q,:}\end{pmatrix}

    The pairs are disjoint and the rows are contiguous, so this needs no output vector and
    touches each element of the state at most once.

    Args:
        states: States as (number of states, alpha strings, beta strings), updated in place.
        src: First alpha string of each pair.
        dst: Second alpha string of each pair.
        sign: Phase of each pair.
        cos_theta: Cosine of the rotation angle.
        sin_theta: Sine of the rotation angle.
    """
    for pair in range(len(src)):
        p = src[pair]
        q = dst[pair]
        signed_sin = sign[pair] * sin_theta
        for state_idx in range(states.shape[0]):
            for col in range(states.shape[2]):
                amplitude_p = states[state_idx, p, col]
                amplitude_q = states[state_idx, q, col]
                states[state_idx, p, col] = cos_theta * amplitude_p - signed_sin * amplitude_q
                states[state_idx, q, col] = signed_sin * amplitude_p + cos_theta * amplitude_q


@nb.jit(nopython=True, cache=True)
def rotate_beta_string_pairs(
    states: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    sign: np.ndarray,
    cos_theta: float,
    sin_theta: float,
) -> None:
    r"""Rotate paired beta strings of the CI matrix in place, a column operation.

    The transpose of rotate_alpha_string_pairs,

    .. math::
        \begin{pmatrix}C_{:,p} & C_{:,q}\end{pmatrix} \leftarrow
        \begin{pmatrix}C_{:,p} & C_{:,q}\end{pmatrix}
        \begin{pmatrix}\cos\theta & \Gamma\sin\theta\\
                       -\Gamma\sin\theta & \cos\theta\end{pmatrix}

    A column is strided rather than contiguous, so the rows are walked on the outside and every
    pair is applied to a row while it is in cache.

    Args:
        states: States as (number of states, alpha strings, beta strings), updated in place.
        src: First beta string of each pair.
        dst: Second beta string of each pair.
        sign: Phase of each pair.
        cos_theta: Cosine of the rotation angle.
        sin_theta: Sine of the rotation angle.
    """
    for state_idx in range(states.shape[0]):
        for row in range(states.shape[1]):
            for pair in range(len(src)):
                p = src[pair]
                q = dst[pair]
                signed_sin = sign[pair] * sin_theta
                amplitude_p = states[state_idx, row, p]
                amplitude_q = states[state_idx, row, q]
                states[state_idx, row, p] = cos_theta * amplitude_p - signed_sin * amplitude_q
                states[state_idx, row, q] = signed_sin * amplitude_p + cos_theta * amplitude_q


@nb.jit(nopython=True, cache=True)
def rotate_string_grid(
    states: np.ndarray,
    alpha_src: np.ndarray,
    alpha_dst: np.ndarray,
    alpha_sign: np.ndarray,
    beta_src: np.ndarray,
    beta_dst: np.ndarray,
    beta_sign: np.ndarray,
    cos_theta: float,
    sin_theta: float,
) -> None:
    r"""Rotate the CI matrix in place for a generator that moves both spins.

    The generator takes :math:`\left(p_\alpha,p_\beta\right)` to
    :math:`\left(q_\alpha,q_\beta\right)`, so the determinant pairs are the product of the alpha
    pairs and the beta pairs and the phase is the product of the two phases,

    .. math::
        \begin{pmatrix}C_{p_\alpha p_\beta}\\C_{q_\alpha q_\beta}\end{pmatrix} \leftarrow
        \begin{pmatrix}\cos\theta & -\Gamma_\alpha\Gamma_\beta\sin\theta\\
                       \Gamma_\alpha\Gamma_\beta\sin\theta & \cos\theta\end{pmatrix}
        \begin{pmatrix}C_{p_\alpha p_\beta}\\C_{q_\alpha q_\beta}\end{pmatrix}

    Unlike the one-spin cases this reaches single elements rather than whole rows, so the
    number of pairs it walks is the product of the two string counts.

    Args:
        states: States as (number of states, alpha strings, beta strings), updated in place.
        alpha_src: First alpha string of each alpha pair.
        alpha_dst: Second alpha string of each alpha pair.
        alpha_sign: Phase of each alpha pair.
        beta_src: First beta string of each beta pair.
        beta_dst: Second beta string of each beta pair.
        beta_sign: Phase of each beta pair.
        cos_theta: Cosine of the rotation angle.
        sin_theta: Sine of the rotation angle.
    """
    for alpha_pair in range(len(alpha_src)):
        pa = alpha_src[alpha_pair]
        qa = alpha_dst[alpha_pair]
        for beta_pair in range(len(beta_src)):
            pb = beta_src[beta_pair]
            qb = beta_dst[beta_pair]
            signed_sin = alpha_sign[alpha_pair] * beta_sign[beta_pair] * sin_theta
            for state_idx in range(states.shape[0]):
                amplitude_p = states[state_idx, pa, pb]
                amplitude_q = states[state_idx, qa, qb]
                states[state_idx, pa, pb] = cos_theta * amplitude_p - signed_sin * amplitude_q
                states[state_idx, qa, qb] = signed_sin * amplitude_p + cos_theta * amplitude_q


def build_string_rotation_layout(
    op: FermionicOperator, ci_info: CI_Info
) -> tuple[int, tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None:
    r"""Find the spin strings that the exponential of a generator rotates.

    An excitation generator :math:`\hat{T} = \hat{\tau} - \hat{\tau}^\dagger` obeys
    :math:`\hat{T}^3 = -\hat{T}`, because :math:`\hat{\tau}^\dagger\hat{\tau}` is a projector
    onto the determinants the excitation does not annihilate. Its exponential therefore closes
    after two terms,

    .. math::
        \exp\left(\theta\hat{T}\right) = \hat{I} + \hat{T}\sin\theta
                                       + \hat{T}^2\left(1-\cos\theta\right)

    and since :math:`-\hat{T}^2` is that projector, the determinant basis splits into the
    determinants the generator annihilates, which the unitary leaves alone, and pairs
    :math:`\left\{\left|p\right>,\left|q\right>\right\}` on which the three terms above sum to a
    single Givens rotation by :math:`\Gamma\theta`.

    Over a spin product the pairing need not be held per determinant at all. The factorized form
    of such a generator holds exactly two terms with opposite factors, one carrying the
    excitation and the other carrying it back, so reading the first one off gives the pairing
    directly in the space of alpha and beta strings,

    .. math::
        \hat{T}\left|p_\alpha p_\beta\right> = \Gamma\left|q_\alpha q_\beta\right>,\qquad
        \Gamma = \Gamma_\alpha\Gamma_\beta

    which is the same rotation the determinant form describes, held in a form that does not grow
    with the CI space. A generator touching one spin pairs strings of that spin and leaves the
    other alone, so the rotation is a row or column operation on the CI matrix and a single pair
    covers a whole row; one touching both spins pairs the products of its two sets of strings.

    That is the whole memory argument for large active spaces. A single excitation at CAS(16,16)
    pairs about 3,400 alpha strings, against 44.7 million determinants, and the state itself is
    the only thing left that grows with the determinant count.

    #. 10.48550/arXiv.2303.10825, Eq. 29-32 (v1)
    #. 10.48550/arXiv.2505.00883, Eq. 6 and 7

    Args:
        op: Excitation generator, already folded into the active space.
        ci_info: Information about the CI space, which must be a spin product.

    Returns:
        Which spins the rotation touches and the alpha and beta string pairs with their phases,
        or None if the operator is not one excitation and its adjoint over this space.
    """
    if not ci_info.is_spin_product:
        return None
    factorized = factorize_operator(op, ci_info)
    if factorized is None:
        return None
    num_pure_alpha = len(factorized.pure_alpha_factor)
    num_pure_beta = len(factorized.pure_beta_factor)
    num_mixed = len(factorized.mixed_factor)
    # One excitation and its adjoint, and nothing else, or this is not a pairing.
    if num_pure_alpha + num_pure_beta + num_mixed != 2:
        return None
    # The arena is only final once every sub-string of the operator has asked for its slice.
    alpha_arena = get_spin_arena(ci_info, True)
    beta_arena = get_spin_arena(ci_info, False)
    empty = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=float))

    def take(arena, start, stop, factor):
        """Copy one term's slice out of the shared arena, with its factor folded into the phase."""
        return (
            np.array(arena[0][start:stop], dtype=np.int32),
            np.array(arena[1][start:stop], dtype=np.int32),
            np.array(arena[2][start:stop], dtype=float) * factor,
        )

    if num_pure_alpha == 2:
        if factorized.pure_alpha_factor[0] * factorized.pure_alpha_factor[1] >= 0:
            return None
        pairs = take(
            alpha_arena,
            factorized.pure_alpha_start[0],
            factorized.pure_alpha_stop[0],
            factorized.pure_alpha_factor[0],
        )
        return (ROTATION_ALPHA, pairs, empty) if len(pairs[0]) else None
    if num_pure_beta == 2:
        if factorized.pure_beta_factor[0] * factorized.pure_beta_factor[1] >= 0:
            return None
        pairs = take(
            beta_arena,
            factorized.pure_beta_start[0],
            factorized.pure_beta_stop[0],
            factorized.pure_beta_factor[0],
        )
        return (ROTATION_BETA, empty, pairs) if len(pairs[0]) else None
    if num_mixed == 2:
        if factorized.mixed_factor[0] * factorized.mixed_factor[1] >= 0:
            return None
        alpha_pairs = take(
            alpha_arena,
            factorized.mixed_alpha_start[0],
            factorized.mixed_alpha_stop[0],
            factorized.mixed_factor[0],
        )
        beta_pairs = take(beta_arena, factorized.mixed_beta_start[0], factorized.mixed_beta_stop[0], 1.0)
        if not len(alpha_pairs[0]) or not len(beta_pairs[0]):
            return None
        return ROTATION_MIXED, alpha_pairs, beta_pairs
    return None


def apply_string_rotation(
    states: np.ndarray,
    layout: tuple[int, tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
    theta: float,
    ci_info: CI_Info,
) -> None:
    r"""Apply :math:`\exp(\theta\hat{T})` through its string pairs, in place.

    .. math::
        \left|\tilde{\nu}\right> = \exp\left(\theta\hat{T}\right)\left|\nu\right>

    One sweep of Givens rotations over the pairs found by build_string_rotation_layout, which
    replaces the three-term closed form and the two extra state vectors it would need.

    Args:
        states: States as (number of states, number of determinants), updated in place.
        layout: Which spins the rotation touches and the string pairs, see
                build_string_rotation_layout.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.
    """
    kind, alpha_pairs, beta_pairs = layout
    matrix = states.reshape(states.shape[0], ci_info.num_alpha_strings, ci_info.num_beta_strings)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    if kind == ROTATION_ALPHA:
        rotate_alpha_string_pairs(matrix, *alpha_pairs, cos_theta, sin_theta)
    elif kind == ROTATION_BETA:
        rotate_beta_string_pairs(matrix, *beta_pairs, cos_theta, sin_theta)
    else:
        rotate_string_grid(matrix, *alpha_pairs, *beta_pairs, cos_theta, sin_theta)


@nb.jit(nopython=True, cache=True)
def accumulate_alpha_string_pairs(
    out: np.ndarray, states: np.ndarray, src: np.ndarray, dst: np.ndarray, sign: np.ndarray
) -> None:
    r"""Add :math:`\hat{T}\left|\nu\right>` for an alpha-only generator into out.

    .. math::
        \begin{align}
        C_{q,:} &\mathrel{+}= \Gamma\,C_{p,:}\\
        C_{p,:} &\mathrel{-}= \Gamma\,C_{q,:}
        \end{align}

    the bare generator rather than its exponential, which is the antisymmetric part of the
    Givens rotation in rotate_alpha_string_pairs without the cosine and sine.

    Args:
        out: States to add into, as (number of states, alpha strings, beta strings).
        states: States the generator acts on, same shape.
        src: First alpha string of each pair.
        dst: Second alpha string of each pair.
        sign: Phase of each pair.
    """
    for pair in range(len(src)):
        p = src[pair]
        q = dst[pair]
        phase = sign[pair]
        for state_idx in range(states.shape[0]):
            for col in range(states.shape[2]):
                out[state_idx, q, col] += phase * states[state_idx, p, col]
                out[state_idx, p, col] -= phase * states[state_idx, q, col]


@nb.jit(nopython=True, cache=True)
def accumulate_beta_string_pairs(
    out: np.ndarray, states: np.ndarray, src: np.ndarray, dst: np.ndarray, sign: np.ndarray
) -> None:
    r"""Add :math:`\hat{T}\left|\nu\right>` for a beta-only generator into out.

    The transpose of accumulate_alpha_string_pairs,

    .. math::
        \begin{align}
        C_{:,q} &\mathrel{+}= \Gamma\,C_{:,p}\\
        C_{:,p} &\mathrel{-}= \Gamma\,C_{:,q}
        \end{align}

    with the rows walked on the outside so each contiguous row is read once while every pair is
    applied to it.

    Args:
        out: States to add into, as (number of states, alpha strings, beta strings).
        states: States the generator acts on, same shape.
        src: First beta string of each pair.
        dst: Second beta string of each pair.
        sign: Phase of each pair.
    """
    for state_idx in range(states.shape[0]):
        for row in range(states.shape[1]):
            for pair in range(len(src)):
                p = src[pair]
                q = dst[pair]
                phase = sign[pair]
                out[state_idx, row, q] += phase * states[state_idx, row, p]
                out[state_idx, row, p] -= phase * states[state_idx, row, q]


@nb.jit(nopython=True, cache=True)
def accumulate_string_grid(
    out: np.ndarray,
    states: np.ndarray,
    alpha_src: np.ndarray,
    alpha_dst: np.ndarray,
    alpha_sign: np.ndarray,
    beta_src: np.ndarray,
    beta_dst: np.ndarray,
    beta_sign: np.ndarray,
) -> None:
    r"""Add :math:`\hat{T}\left|\nu\right>` for a generator moving both spins into out.

    .. math::
        \begin{align}
        C_{q_\alpha q_\beta} &\mathrel{+}= \Gamma_\alpha\Gamma_\beta\,C_{p_\alpha p_\beta}\\
        C_{p_\alpha p_\beta} &\mathrel{-}= \Gamma_\alpha\Gamma_\beta\,C_{q_\alpha q_\beta}
        \end{align}

    over the product of the alpha pairs and the beta pairs.

    Args:
        out: States to add into, as (number of states, alpha strings, beta strings).
        states: States the generator acts on, same shape.
        alpha_src: First alpha string of each alpha pair.
        alpha_dst: Second alpha string of each alpha pair.
        alpha_sign: Phase of each alpha pair.
        beta_src: First beta string of each beta pair.
        beta_dst: Second beta string of each beta pair.
        beta_sign: Phase of each beta pair.
    """
    for alpha_pair in range(len(alpha_src)):
        pa = alpha_src[alpha_pair]
        qa = alpha_dst[alpha_pair]
        for beta_pair in range(len(beta_src)):
            pb = beta_src[beta_pair]
            qb = beta_dst[beta_pair]
            phase = alpha_sign[alpha_pair] * beta_sign[beta_pair]
            for state_idx in range(states.shape[0]):
                out[state_idx, qa, qb] += phase * states[state_idx, pa, pb]
                out[state_idx, pa, pb] -= phase * states[state_idx, qa, qb]


def accumulate_string_pairing(
    out: np.ndarray,
    states: np.ndarray,
    layout: tuple[int, tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
    ci_info: CI_Info,
) -> None:
    r"""Add :math:`\hat{T}\left|\nu\right>` into out, through the generator's string pairs.

    .. math::
        \hat{T}\left|p\right> = \Gamma\left|q\right>,\qquad
        \hat{T}\left|q\right> = -\Gamma\left|p\right>

    The gradient of a unitary product state needs the bare generator applied once, and the
    pairing that its exponential uses describes that too: it is the same map without the cosine
    and sine. Going through it costs one sweep rather than a walk over the operator's strings.

    #. 10.48550/arXiv.2303.10825, Eq. 36 (v1)

    Args:
        out: States to add into, as (number of states, number of determinants).
        states: States the generator acts on, same shape.
        layout: Which spins the generator touches and its string pairs.
        ci_info: Information about the CI space.
    """
    kind, alpha_pairs, beta_pairs = layout
    shape = (states.shape[0], ci_info.num_alpha_strings, ci_info.num_beta_strings)
    out_matrix = out.reshape(shape)
    state_matrix = states.reshape(shape)
    if kind == ROTATION_ALPHA:
        accumulate_alpha_string_pairs(out_matrix, state_matrix, *alpha_pairs)
    elif kind == ROTATION_BETA:
        accumulate_beta_string_pairs(out_matrix, state_matrix, *beta_pairs)
    else:
        accumulate_string_grid(out_matrix, state_matrix, *alpha_pairs, *beta_pairs)


@nb.jit(nopython=True, cache=True)
def label_connected_spin_strings(src: np.ndarray, dst: np.ndarray, num_strings: int) -> np.ndarray:
    """Label each spin string with the group of strings the operator connects it to.

    Reads the operator's excitation maps as the edges of a graph on the strings of one spin and
    returns its connected components, by union-find. A string the operator never touches is a
    component on its own.

    The components are what build_spin_cells calls cells: the operator cannot take a string out
    of the component it lies in, so it cannot take a determinant out of the pair of components
    its two strings lie in.

    Args:
        src: Spin string an operator term acts on.
        dst: Spin string it is taken to.
        num_strings: Number of spin strings.

    Returns:
        Group label of every spin string.
    """
    parent = np.arange(num_strings)
    for entry in range(len(src)):
        root_a = src[entry]
        root_b = dst[entry]
        while parent[root_a] != root_a:
            parent[root_a] = parent[parent[root_a]]
            root_a = parent[root_a]
        while parent[root_b] != root_b:
            parent[root_b] = parent[parent[root_b]]
            root_b = parent[root_b]
        if root_a != root_b:
            parent[root_a] = root_b
    for string_idx in range(num_strings):
        root = string_idx
        while parent[root] != root:
            root = parent[root]
        parent[string_idx] = root
    return parent


@nb.jit(nopython=True, cache=True)
def rotate_spin_cell_pairs(
    states: np.ndarray,
    alpha_start: np.ndarray,
    alpha_members: np.ndarray,
    alpha_sig: np.ndarray,
    beta_start: np.ndarray,
    beta_members: np.ndarray,
    beta_sig: np.ndarray,
    pair_offset: np.ndarray,
    pair_count: np.ndarray,
    group_start: np.ndarray,
    group_size: np.ndarray,
    group_rows: np.ndarray,
    group_cols: np.ndarray,
    group_shape: np.ndarray,
    rotations: np.ndarray,
    buffer: np.ndarray,
    rows_buffer: np.ndarray,
) -> None:
    r"""Apply the exponential to each pair of spin-string cells, in place.

    .. math::
        c_{d_r} \leftarrow \sum_s \left[\exp\left(\theta T^{(A,B)}_g\right)\right]_{rs}c_{d_s}

    The operator cannot move a determinant out of the cell pair its two strings belong to, and
    inside a pair it splits further into groups it cannot mix. Both structures depend only on the
    pair of cell signatures, of which there are a handful however large the CI space is, so the
    rotations are shared and only the strings of each cell are looked up per pair.

    The groups are disjoint, so no output vector is needed. The rows of one alpha cell are
    streamed into a contiguous buffer first, because they are scattered through the CI matrix
    but are read once per beta cell; every alpha string belongs to exactly one cell, so the
    whole state is still read and written once in total.

    Args:
        states: States as (number of states, alpha strings, beta strings), updated in place.
        alpha_start: Where each alpha cell begins in alpha_members, with the end appended.
        alpha_members: Alpha strings of every cell, one cell after another.
        alpha_sig: Signature of each alpha cell.
        beta_start: Where each beta cell begins in beta_members, with the end appended.
        beta_members: Beta strings of every cell, one cell after another.
        beta_sig: Signature of each beta cell.
        pair_offset: First group of each signature pair.
        pair_count: Number of groups of each signature pair.
        group_start: Where each group begins in group_rows and group_cols.
        group_size: Number of determinants in each group.
        group_rows: Position within its alpha cell of every group member.
        group_cols: Position within its beta cell of every group member.
        group_shape: Which distinct rotation each group uses.
        rotations: Rotation of each distinct group, as (number of them, n, n).
        buffer: Scratch of at least the largest group.
        rows_buffer: Scratch of (largest alpha cell) by (number of beta strings).
    """
    alpha_here = np.empty(64, dtype=np.int64)
    beta_here = np.empty(64, dtype=np.int64)
    num_beta_strings = states.shape[2]
    for alpha_cell in range(len(alpha_sig)):
        first_alpha = alpha_start[alpha_cell]
        size_alpha = alpha_start[alpha_cell + 1] - first_alpha
        signature_alpha = alpha_sig[alpha_cell]
        for row in range(size_alpha):
            alpha_here[row] = alpha_members[first_alpha + row]
        for state_idx in range(states.shape[0]):
            state = states[state_idx]
            # The cell's rows are scattered through the CI matrix but its work touches all of
            # them many times over, so they are streamed into one small contiguous block first.
            # Every alpha string belongs to exactly one cell, so this reads and writes the whole
            # state once in total.
            for row in range(size_alpha):
                source = state[alpha_here[row]]
                for col in range(num_beta_strings):
                    rows_buffer[row, col] = source[col]
            for beta_cell in range(len(beta_sig)):
                first_beta = beta_start[beta_cell]
                size_beta = beta_start[beta_cell + 1] - first_beta
                for col in range(size_beta):
                    beta_here[col] = beta_members[first_beta + col]
                offset = pair_offset[signature_alpha, beta_sig[beta_cell]]
                for group_idx in range(pair_count[signature_alpha, beta_sig[beta_cell]]):
                    group = offset + group_idx
                    size = group_size[group]
                    begin = group_start[group]
                    rotation = rotations[group_shape[group]]
                    for member in range(size):
                        buffer[member] = rows_buffer[
                            group_rows[begin + member], beta_here[group_cols[begin + member]]
                        ]
                    for member in range(size):
                        total = 0.0
                        for other in range(size):
                            total += rotation[member, other] * buffer[other]
                        rows_buffer[group_rows[begin + member], beta_here[group_cols[begin + member]]] = total
            for row in range(size_alpha):
                target = state[alpha_here[row]]
                for col in range(num_beta_strings):
                    target[col] = rows_buffer[row, col]


def build_spin_cells(
    num_strings: int, terms: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group the strings of one spin into the sets the operator's terms connect.

    The cells are the connected components of label_connected_spin_strings, sorted so that the
    members of a cell are contiguous. A cell is small: a generator touches a handful of
    orbitals, so a cell holds at most a few strings whatever the active space size, and the
    number of cells grows with the number of strings rather than with the number of
    determinants.

    Args:
        num_strings: Number of spin strings.
        terms: Each term's map over the strings of this spin.

    Returns:
        Where each cell begins, the strings of every cell, and each string's position in its cell.
    """
    if terms:
        src = np.concatenate([term[0] for term in terms]).astype(np.int64)
        dst = np.concatenate([term[1] for term in terms]).astype(np.int64)
    else:
        src = dst = np.zeros(0, dtype=np.int64)
    label = label_connected_spin_strings(src, dst, num_strings)
    order = np.argsort(label, kind="stable")
    boundaries = np.flatnonzero(np.concatenate(([True], label[order][1:] != label[order][:-1])))
    starts = np.concatenate((boundaries, [num_strings]))
    position = np.empty(num_strings, dtype=np.int64)
    position[order] = np.arange(num_strings) - np.repeat(starts[:-1], np.diff(starts))
    return starts.astype(np.int32), order.astype(np.int32), position


def spin_cell_signatures(
    starts: np.ndarray,
    members: np.ndarray,
    position: np.ndarray,
    label_of: np.ndarray,
    terms: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    pure_terms: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list]:
    r"""Describe each cell by how the operator's terms move its members, with their phases.

    A signature is the cell size together with, for every term of the operator, the list of
    moves :math:`\left(i \rightarrow j, \Gamma\right)` it makes inside the cell, written in
    positions within the cell rather than absolute string indices. Two cells with the same
    signature therefore carry literally the same small matrix, so the exponential is built once
    per signature pair and not once per cell.

    Only a handful of signatures exist however large the CI space is, because a cell is fixed by
    the occupation of the few orbitals the operator touches, and there are only so many ways to
    occupy them. That is what makes this layout independent of the determinant count.

    Args:
        starts: Where each cell begins in members.
        members: Spin strings of every cell.
        position: Each string's position within its cell.
        label_of: Which cell each string belongs to.
        terms: Each term acting on both spins, as a map over this spin's strings.
        pure_terms: Each term acting on this spin alone.

    Returns:
        Signature of each cell, and the distinct signatures.
    """
    identifiers = np.empty(len(starts) - 1, dtype=np.int32)
    seen: dict = {}
    distinct: list = []
    moves_by_cell: list = [[[] for _ in range(len(terms) + len(pure_terms))] for _ in range(len(starts) - 1)]
    for term_idx, (src, dst, sign) in enumerate(list(terms) + list(pure_terms)):
        for from_string, to_string, phase in zip(src, dst, sign):
            cell = label_of[int(from_string)]
            moves_by_cell[cell][term_idx].append(
                (int(position[int(from_string)]), int(position[int(to_string)]), float(phase))
            )
    for cell in range(len(starts) - 1):
        key = (
            int(starts[cell + 1] - starts[cell]),
            tuple(tuple(sorted(moves)) for moves in moves_by_cell[cell]),
        )
        found = seen.get(key)
        if found is None:
            found = len(distinct)
            seen[key] = found
            distinct.append(key)
        identifiers[cell] = found
    return identifiers, distinct


# A cell pair spans this many determinants at most before the layout is refused as too dense.
MAX_SPIN_CELL_PAIR = 64


def local_generator_matrix(
    alpha_signature: tuple,
    beta_signature: tuple,
    mixed_factors: np.ndarray,
    pure_alpha_factors: np.ndarray,
    pure_beta_factors: np.ndarray,
) -> np.ndarray:
    r"""Build the generator restricted to one pair of spin-string cells.

    The generator is a sum of factorized terms,

    .. math::
        \hat{T} = \sum_t c_t\,\hat{A}_t\otimes\hat{B}_t
                + \sum_u c_u\,\hat{A}_u\otimes 1
                + \sum_v c_v\,1\otimes\hat{B}_v

    so on the :math:`\left|A\right|\left|B\right|` determinants that the cell pair
    :math:`\left(A,B\right)` spans it is a small dense matrix,

    .. math::
        T^{(A,B)}_{\left(a^\prime b^\prime\right),\left(ab\right)}
            = \sum_t c_t\Gamma^{(t)}_{a}\Gamma^{(t)}_{b}
            + \delta_{b^\prime b}\sum_u c_u\Gamma^{(u)}_{a}
            + \delta_{a^\prime a}\sum_v c_v\Gamma^{(v)}_{b}

    with :math:`a,b` positions within their cell. It is real and antisymmetric, being the
    restriction of an anti-Hermitian real generator, and the exponential of that is what the
    cell pair needs.

    Args:
        alpha_signature: Size of the alpha cell and how each term moves its members.
        beta_signature: Size of the beta cell and how each term moves its members.
        mixed_factors: Factor in front of each term acting on both spins.
        pure_alpha_factors: Factor in front of each term acting on alpha alone.
        pure_beta_factors: Factor in front of each term acting on beta alone.

    Returns:
        The generator on the cell pair.
    """
    size_alpha, alpha_moves = alpha_signature
    size_beta, beta_moves = beta_signature
    num_mixed = len(mixed_factors)
    matrix = np.zeros((size_alpha * size_beta, size_alpha * size_beta))
    for term, factor in enumerate(mixed_factors):
        for from_alpha, to_alpha, alpha_phase in alpha_moves[term]:
            for from_beta, to_beta, beta_phase in beta_moves[term]:
                matrix[to_alpha * size_beta + to_beta, from_alpha * size_beta + from_beta] += (
                    factor * alpha_phase * beta_phase
                )
    # A term on one spin alone leaves the other spin's string where it is.
    for term, factor in enumerate(pure_alpha_factors):
        for from_alpha, to_alpha, alpha_phase in alpha_moves[num_mixed + term]:
            for beta in range(size_beta):
                matrix[to_alpha * size_beta + beta, from_alpha * size_beta + beta] += factor * alpha_phase
    for term, factor in enumerate(pure_beta_factors):
        for from_beta, to_beta, beta_phase in beta_moves[num_mixed + term]:
            for alpha in range(size_alpha):
                matrix[alpha * size_beta + to_beta, alpha * size_beta + from_beta] += factor * beta_phase
    return matrix


def build_spin_block_layout(op: FermionicOperator, ci_info: CI_Info) -> tuple | None:
    r"""Block diagonalize a generator over pairs of spin-string cells.

    A generator that is not one excitation and its adjoint, a spin-adapted double above all, is
    not a rotation of pairs. Those operators carry several frequencies, so their closed form is
    a sum over powers rather than a single Givens rotation,

    .. math::
        \exp\left(\theta\,{}^\text{SA}\hat{G}\right) = \hat{I}
            + \sum_n\sum_{m\,\text{odd}} k^{(m)}_n\,{}^\text{SA}\hat{G}^m\sin\left(S_n\theta\right)
            + \sum_n\sum_{m\,\text{even}} k^{(m)}_n\,{}^\text{SA}\hat{G}^m
              \left(\cos\left(S_n\theta\right)-1\right)

    which costs one application of the generator per power, up to ten of them, see
    operator_state_algebra.SPIN_ADAPTED_DOUBLE_SERIES.

    Taking the exponential structurally instead costs one sweep. The generator is still block
    diagonal, and over a spin product the blocks are visible per spin: the strings of each spin
    fall into cells that the operator's terms connect, the operator cannot take a determinant
    out of the cell pair its two strings lie in, and inside a pair it splits further into groups,
    so

    .. math::
        \exp\left(\theta\hat{T}\right)
            = \bigoplus_{A,B}\bigoplus_{g}\exp\left(\theta T^{(A,B)}_g\right)

    Each small block is exponentiated by diagonalizing it once, see apply_spin_block_layout.

    Cells are small and the number of distinct shapes does not grow with the active space,
    because a cell is fixed by the occupation of the few orbitals the generator touches. So the
    whole layout is two arrays over the spin strings and a handful of small matrices, rather
    than the determinant-indexed groups this replaces, which cost one entry per determinant.

    #. 10.48550/arXiv.2505.00883, Eq. 45, 47, and, 49 (SA doubles)
    #. 10.48550/arXiv.2505.02984, Eq. 35, D1, and, D2 (SA doubles)

    Args:
        op: Excitation generator, already folded into the active space.
        ci_info: Information about the CI space, which must be a spin product.

    Returns:
        The cells of each spin, the signature of each cell, and the groups of each signature pair
        with their diagonalized generators. None if the generator does not block this way.
    """
    if not ci_info.is_spin_product:
        return None
    factorized = factorize_operator(op, ci_info)
    if (
        factorized is None
        or len(factorized.mixed_factor) + len(factorized.pure_alpha_factor) + len(factorized.pure_beta_factor)
        == 0
    ):
        return None
    alpha_arena = get_spin_arena(ci_info, True)
    beta_arena = get_spin_arena(ci_info, False)

    def slices(arena, starts, stops):
        """Each term's map over the strings of one spin."""
        return [(arena[0][s:e], arena[1][s:e], arena[2][s:e]) for s, e in zip(starts, stops)]

    alpha_terms = slices(alpha_arena, factorized.mixed_alpha_start, factorized.mixed_alpha_stop)
    beta_terms = slices(beta_arena, factorized.mixed_beta_start, factorized.mixed_beta_stop)
    alpha_pure = slices(alpha_arena, factorized.pure_alpha_start, factorized.pure_alpha_stop)
    beta_pure = slices(beta_arena, factorized.pure_beta_start, factorized.pure_beta_stop)

    alpha_start, alpha_members, alpha_position = build_spin_cells(
        ci_info.num_alpha_strings, alpha_terms + alpha_pure
    )
    beta_start, beta_members, beta_position = build_spin_cells(
        ci_info.num_beta_strings, beta_terms + beta_pure
    )
    alpha_label = np.empty(ci_info.num_alpha_strings, dtype=np.int64)
    alpha_label[alpha_members] = np.repeat(np.arange(len(alpha_start) - 1), np.diff(alpha_start))
    beta_label = np.empty(ci_info.num_beta_strings, dtype=np.int64)
    beta_label[beta_members] = np.repeat(np.arange(len(beta_start) - 1), np.diff(beta_start))
    alpha_sig, alpha_distinct = spin_cell_signatures(
        alpha_start, alpha_members, alpha_position, alpha_label, alpha_terms, alpha_pure
    )
    beta_sig, beta_distinct = spin_cell_signatures(
        beta_start, beta_members, beta_position, beta_label, beta_terms, beta_pure
    )
    if max(np.diff(alpha_start).max(), 1) * max(np.diff(beta_start).max(), 1) > MAX_SPIN_CELL_PAIR:
        return None

    pair_offset = np.zeros((len(alpha_distinct), len(beta_distinct)), dtype=np.int32)
    pair_count = np.zeros((len(alpha_distinct), len(beta_distinct)), dtype=np.int32)
    group_start: list[int] = []
    group_size: list[int] = []
    group_rows: list[int] = []
    group_cols: list[int] = []
    group_shape: list[int] = []
    generators: list[np.ndarray] = []
    seen_blocks: dict[bytes, int] = {}
    members_so_far = 0
    for first, alpha_signature in enumerate(alpha_distinct):
        for second, beta_signature in enumerate(beta_distinct):
            matrix = local_generator_matrix(
                alpha_signature,
                beta_signature,
                factorized.mixed_factor,
                factorized.pure_alpha_factor,
                factorized.pure_beta_factor,
            )
            if not np.allclose(matrix, -matrix.T):
                # An anti-Hermitian generator gives antisymmetric blocks; if it did not, the
                # maps above did not describe it.
                return None
            size_beta = beta_signature[0]
            reach = np.abs(matrix) + np.abs(matrix.T) + np.eye(len(matrix))
            label = label_connected_spin_strings(*np.nonzero(reach), len(matrix))
            pair_offset[first, second] = len(group_start)
            for root in np.unique(label):
                local = np.flatnonzero(label == root)
                group_start.append(members_so_far)
                group_size.append(len(local))
                group_rows.extend(int(x) // size_beta for x in local)
                group_cols.extend(int(x) % size_beta for x in local)
                members_so_far += len(local)
                block = matrix[np.ix_(local, local)]
                # Most groups repeat: the same few small generators appear over and over, so
                # only the distinct ones are diagonalized and kept.
                key = block.tobytes()
                found = seen_blocks.get(key)
                if found is None:
                    found = len(generators)
                    seen_blocks[key] = found
                    generators.append(block)
                group_shape.append(found)
            pair_count[first, second] = len(group_start) - pair_offset[first, second]
    largest = max(group_size)
    vectors = np.zeros((len(generators), largest, largest), dtype=complex)
    values = np.zeros((len(generators), largest))
    for shape, matrix in enumerate(generators):
        dim = len(matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(1j * matrix)
        vectors[shape, :dim, :dim] = eigenvectors
        values[shape, :dim] = eigenvalues
        for pad in range(dim, largest):
            vectors[shape, pad, pad] = 1.0
    return (
        alpha_start,
        alpha_members,
        alpha_sig,
        beta_start,
        beta_members,
        beta_sig,
        pair_offset,
        pair_count,
        np.array(group_start, dtype=np.int32),
        np.array(group_size, dtype=np.int32),
        np.array(group_rows, dtype=np.int32),
        np.array(group_cols, dtype=np.int32),
        np.array(group_shape, dtype=np.int32),
        vectors,
        values,
        largest,
    )


def apply_spin_block_layout(states: np.ndarray, layout: tuple, theta: float, ci_info: CI_Info) -> None:
    r"""Apply :math:`\exp(\theta\hat{T})` group by group within each cell pair, in place.

    Each group's generator was diagonalized once when the layout was built. A real antisymmetric
    :math:`T` makes :math:`iT` Hermitian, so :math:`iT = V\lambda V^\dagger` and the exponential
    at any angle is two small matrix products,

    .. math::
        \exp\left(\theta T\right) = V e^{-i\theta\lambda}V^\dagger

    which is real. Only the distinct group shapes are exponentiated, a handful of matrices no
    larger than the cell pair they live in, and the sweep over the state then costs one pass.

    Args:
        states: States as (number of states, number of determinants), updated in place.
        layout: The cells, signature pairs and their groups, see build_spin_block_layout.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.
    """
    (
        alpha_start,
        alpha_members,
        alpha_sig,
        beta_start,
        beta_members,
        beta_sig,
        pair_offset,
        pair_count,
        group_start,
        group_size,
        group_rows,
        group_cols,
        group_shape,
        vectors,
        values,
        largest,
    ) = layout
    phases = np.exp(-1j * theta * values)[:, np.newaxis, :]
    rotations = np.ascontiguousarray(
        np.real((vectors * phases) @ np.conjugate(np.transpose(vectors, (0, 2, 1))))
    )
    matrix = states.reshape(states.shape[0], ci_info.num_alpha_strings, ci_info.num_beta_strings)
    rotate_spin_cell_pairs(
        matrix,
        alpha_start,
        alpha_members,
        alpha_sig,
        beta_start,
        beta_members,
        beta_sig,
        pair_offset,
        pair_count,
        group_start,
        group_size,
        group_rows,
        group_cols,
        group_shape,
        rotations,
        np.empty(largest),
        np.empty((int(np.diff(alpha_start).max()), ci_info.num_beta_strings)),
    )


@nb.jit(nopython=True, cache=True)
def overlap_alpha_string_pairs(
    bra: np.ndarray,
    ket: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    sign: np.ndarray,
    out: np.ndarray,
) -> None:
    r"""Add :math:`\left<\text{bra}\right|\hat{T}\left|\text{ket}\right>` for an alpha generator.

    .. math::
        \left<\text{bra}\right|\hat{T}\left|\text{ket}\right> = \sum_{\left\{p,q\right\}}\Gamma
            \sum_{I_\beta}\left(b_{q I_\beta}k_{p I_\beta} - b_{p I_\beta}k_{q I_\beta}\right)

    Args:
        bra: Bra states as (number of states, alpha strings, beta strings).
        ket: Ket states, same shape.
        src: First alpha string of each pair.
        dst: Second alpha string of each pair.
        sign: Phase of each pair.
        out: One overlap per state, added into.
    """
    for pair in range(len(src)):
        p = src[pair]
        q = dst[pair]
        phase = sign[pair]
        for state_idx in range(bra.shape[0]):
            partial = 0.0
            for col in range(bra.shape[2]):
                partial += (
                    bra[state_idx, q, col] * ket[state_idx, p, col]
                    - bra[state_idx, p, col] * ket[state_idx, q, col]
                )
            out[state_idx] += phase * partial


@nb.jit(nopython=True, cache=True)
def overlap_beta_string_pairs(
    bra: np.ndarray,
    ket: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    sign: np.ndarray,
    out: np.ndarray,
) -> None:
    r"""Add :math:`\left<\text{bra}\right|\hat{T}\left|\text{ket}\right>` for a beta generator.

    .. math::
        \left<\text{bra}\right|\hat{T}\left|\text{ket}\right> = \sum_{\left\{p,q\right\}}\Gamma
            \sum_{I_\alpha}\left(b_{I_\alpha q}k_{I_\alpha p} - b_{I_\alpha p}k_{I_\alpha q}\right)

    The alpha blocks are walked on the outside so that each one, which is contiguous, is read
    once while every pair is applied to it.

    Args:
        bra: Bra states as (number of states, alpha strings, beta strings).
        ket: Ket states, same shape.
        src: First beta string of each pair.
        dst: Second beta string of each pair.
        sign: Phase of each pair.
        out: One overlap per state, added into.
    """
    for state_idx in range(bra.shape[0]):
        total = 0.0
        for row in range(bra.shape[1]):
            for pair in range(len(src)):
                p = src[pair]
                q = dst[pair]
                total += sign[pair] * (
                    bra[state_idx, row, q] * ket[state_idx, row, p]
                    - bra[state_idx, row, p] * ket[state_idx, row, q]
                )
        out[state_idx] += total


@nb.jit(nopython=True, cache=True)
def overlap_string_grid(
    bra: np.ndarray,
    ket: np.ndarray,
    alpha_src: np.ndarray,
    alpha_dst: np.ndarray,
    alpha_sign: np.ndarray,
    beta_src: np.ndarray,
    beta_dst: np.ndarray,
    beta_sign: np.ndarray,
    out: np.ndarray,
) -> None:
    r"""Add :math:`\left<\text{bra}\right|\hat{T}\left|\text{ket}\right>` for a mixed generator.

    .. math::
        \left<\text{bra}\right|\hat{T}\left|\text{ket}\right>
            = \sum_{\left\{p_\alpha,q_\alpha\right\}}\sum_{\left\{p_\beta,q_\beta\right\}}
              \Gamma_\alpha\Gamma_\beta\left(b_{q_\alpha q_\beta}k_{p_\alpha p_\beta}
              - b_{p_\alpha p_\beta}k_{q_\alpha q_\beta}\right)

    Args:
        bra: Bra states as (number of states, alpha strings, beta strings).
        ket: Ket states, same shape.
        alpha_src: First alpha string of each alpha pair.
        alpha_dst: Second alpha string of each alpha pair.
        alpha_sign: Phase of each alpha pair.
        beta_src: First beta string of each beta pair.
        beta_dst: Second beta string of each beta pair.
        beta_sign: Phase of each beta pair.
        out: One overlap per state, added into.
    """
    for alpha_pair in range(len(alpha_src)):
        pa = alpha_src[alpha_pair]
        qa = alpha_dst[alpha_pair]
        for beta_pair in range(len(beta_src)):
            pb = beta_src[beta_pair]
            qb = beta_dst[beta_pair]
            phase = alpha_sign[alpha_pair] * beta_sign[beta_pair]
            for state_idx in range(bra.shape[0]):
                out[state_idx] += phase * (
                    bra[state_idx, qa, qb] * ket[state_idx, pa, pb]
                    - bra[state_idx, pa, pb] * ket[state_idx, qa, qb]
                )


def string_pairing_overlap(
    bra: np.ndarray,
    ket: np.ndarray,
    layout: tuple[int, tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
    ci_info: CI_Info,
    out: np.ndarray,
) -> None:
    r"""Add :math:`\left<\text{bra}\right|\hat{T}\left|\text{ket}\right>` through the pairing.

    The gradient of a unitary product state is this number, not the state the generator
    produces,

    .. math::
        \frac{\partial\left<E\right>}{\partial\theta_j}
            = 2\left<\psi^\prime_{j+1}\right|\hat{T}_j\left|\psi_j\right>

    and the pairing gives it directly: a generator moves each determinant of a pair to the
    other, so the overlap is a sum over the pairs,

    .. math::
        \left<\text{bra}\right|\hat{T}\left|\text{ket}\right>
            = \sum_{\left\{p,q\right\}}\Gamma\left(b_q k_p - b_p k_q\right)

    Forming the state first costs a pass to zero it, a pass to fill it and a pass to read it
    back; this reads only the paired entries.

    #. 10.48550/arXiv.2303.10825, Eq. 36 and 37 (v1)

    Args:
        bra: Bra states as (number of states, number of determinants).
        ket: Ket states, same shape.
        layout: Which spins the generator touches and its string pairs.
        ci_info: Information about the CI space.
        out: One overlap per state, added into.
    """
    kind, alpha_pairs, beta_pairs = layout
    shape = (bra.shape[0], ci_info.num_alpha_strings, ci_info.num_beta_strings)
    bra_matrix = bra.reshape(shape)
    ket_matrix = ket.reshape(shape)
    if kind == ROTATION_ALPHA:
        overlap_alpha_string_pairs(bra_matrix, ket_matrix, *alpha_pairs, out)
    elif kind == ROTATION_BETA:
        overlap_beta_string_pairs(bra_matrix, ket_matrix, *beta_pairs, out)
    else:
        overlap_string_grid(bra_matrix, ket_matrix, *alpha_pairs, *beta_pairs, out)


def propagate_state_factorized(
    op: FermionicOperator, state: np.ndarray, ci_info: CI_Info, tmp_state: np.ndarray
) -> np.ndarray | None:
    r"""Apply a folded operator to a state using the spin-factorized algebra.

    .. math::
        \left|\tilde{0}\right> = \hat{O}\left|0\right>

    The spin-factorized route is the fast one and is tried first by propagate_state. It needs a
    CI space that is a product of an alpha and a beta string space and an operator all of whose
    strings are :math:`S_z` conserving; anything else returns None and falls back to the general
    determinant kernel.

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
    return apply_factorized_operator(factorized, state, ci_info, tmp_state)


def propagate_state_SA_factorized(
    op: FermionicOperator, states: np.ndarray, ci_info: CI_Info, tmp_states: np.ndarray
) -> np.ndarray | None:
    r"""Apply a folded operator to every state of a state-averaged wave function.

    .. math::
        \left|\tilde{\nu}\right> = \hat{O}\left|\nu\right>

    The operator is the same for every state, so everything that does not depend on the state,
    the one-spin matrices and the contraction layout, is built once by build_derived_forms and
    only the application is repeated.

    Args:
        op: Folded fermionic operator.
        states: Original states, one per row.
        ci_info: Information about the CI space.
        tmp_states: New states, assumed to be zeroed.

    Returns:
        New states, or None if the operator cannot be factorized over this CI space.
    """
    factorized = factorize_operator(op, ci_info)
    if factorized is None:
        return None
    for state, tmp_state in zip(states, tmp_states):
        apply_factorized_operator(factorized, state, ci_info, tmp_state)
    return tmp_states
