import functools
from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss
from sympy.utilities.iterables import multiset_permutations

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    G3,
    G4,
    G5,
    G6,
    Epq,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
)
from slowquant.unitary_coupled_cluster.util import UccStructure, UpsStructure


def get_indexing(
    num_active_orbs: int, num_active_elec_alpha: int, num_active_elec_beta: int
) -> tuple[list[int], dict[int, int]]:
    """Get indexing between index and determiant.

    Args:
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.

    Returns:
        List to map index to determiant and dictionary to map determiant to index.
    """
    idx = 0
    idx2det = []
    det2idx = {}
    for alpha_string in multiset_permutations(
        [1] * num_active_elec_alpha + [0] * (num_active_orbs - num_active_elec_alpha)
    ):
        for beta_string in multiset_permutations(
            [1] * num_active_elec_beta + [0] * (num_active_orbs - num_active_elec_beta)
        ):
            det_str = ""
            for a, b in zip(alpha_string, beta_string):
                det_str += str(a) + str(b)
            det = int(det_str, 2)
            idx2det.append(det)
            det2idx[det] = idx
            idx += 1
    return idx2det, det2idx


def build_operator_matrix(
    op: FermionicOperator,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_active_orbs: int,
) -> np.ndarray:
    """Build matrix representation of operator.

    Args:
        op: Fermionic number and spin conserving operator.
        idx2det: Index to determinant.
        det2idx: Determinant to index.
        num_active_orbs: Number of active spatial orbitals.

    Returns:
        Matrix representation of operator.
    """
    num_dets = len(idx2det)
    op_mat = np.zeros((num_dets, num_dets))
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    for i in range(num_dets):
        det_ = idx2det[i]
        for fermi_label in op.factors:
            det = det_
            phase_changes = 0
            for fermi_op in op.operators[fermi_label][::-1]:
                orb_idx = fermi_op.idx
                nth_bit = (det >> 2 * num_active_orbs - 1 - orb_idx) & 1
                if nth_bit == 0 and fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
                elif nth_bit == 1 and fermi_op.dagger:
                    break
                elif nth_bit == 0 and not fermi_op.dagger:
                    break
                elif nth_bit == 1 and not fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
            else:  # nobreak
                val = op.factors[fermi_label] * (-1) ** phase_changes
                if abs(val) > 10**-14:
                    op_mat[det2idx[det], i] += val
    return op_mat


def propagate_state(
    operators: list[FermionicOperator | str],
    state: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure | UccStructure,
) -> np.ndarray:
    r"""Propagate state by applying operator.

    The operator will be folded to only work on the active orbitals.

    .. math::
        \left|\tilde{0}\right> = \hat{O}\left|0\right>

    Args:
        operators: Operator.
        state: State.

    Returns:
        New state.
    """
    if len(operators) == 0:
        return np.copy(state)
    num_dets = len(idx2det)
    new_state = np.copy(state)
    tmp_state = np.zeros(num_dets)
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    for op in operators[::-1]:
        tmp_state[:] = 0.0
        if isinstance(op, str):
            if op not in ("U", "Ud"):
                raise ValueError(f"Unknown str operator, expected ('U', 'Ud') got {op}")
            dagger = False
            if op == "Ud":
                dagger = True
            if isinstance(wf_struct, UpsStructure):
                new_state = construct_ups_state(
                    new_state,
                    num_active_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    wf_struct,
                    dagger=dagger,
                )
            elif isinstance(wf_struct, UccStructure):
                new_state = construct_ucc_state(
                    new_state,
                    num_active_orbs,
                    num_active_elec_beta,
                    num_active_elec_alpha,
                    thetas,
                    wf_struct,
                    dagger=dagger,
                )
            else:
                raise TypeError(f"Got unknown wave function structure type, {type(wf_struct)}")
        else:
            op_folded = op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
            for i in range(num_dets):
                if abs(new_state[i]) < 10**-14:
                    continue
                det_ = idx2det[i]
                for fermi_label in op_folded.factors:
                    det = det_
                    phase_changes = 0
                    for fermi_op in op_folded.operators[fermi_label][::-1]:
                        orb_idx = fermi_op.idx
                        nth_bit = (det >> 2 * num_active_orbs - 1 - orb_idx) & 1
                        if nth_bit == 0 and fermi_op.dagger:
                            det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                            phase_changes += (det & parity_check[orb_idx]).bit_count()
                        elif nth_bit == 1 and fermi_op.dagger:
                            break
                        elif nth_bit == 0 and not fermi_op.dagger:
                            break
                        elif nth_bit == 1 and not fermi_op.dagger:
                            det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                            phase_changes += (det & parity_check[orb_idx]).bit_count()
                    else:  # nobreak
                        val = op_folded.factors[fermi_label] * (-1) ** phase_changes
                        if abs(val) > 10**-14:
                            tmp_state[det2idx[det]] += val * new_state[i]
            new_state = np.copy(tmp_state)
    return new_state


def expectation_value(
    bra: np.ndarray,
    operators: list[FermionicOperator | str],
    ket: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure | UccStructure,
) -> float:
    """Calculate expectation value of operator.

    Args:
        bra: Bra state.
        op: Operator.
        ket: Ket state.
        idx2det: Index to determinant mapping.
        det2idx: Determinant to index mapping.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual orbitals.

    Returns:
        Expectation value.
    """
    op_ket = propagate_state(
        operators,
        ket,
        idx2det,
        det2idx,
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_active_elec_alpha,
        num_active_elec_beta,
        thetas,
        wf_struct,
    )
    val = bra @ op_ket
    if not isinstance(val, float):
        raise ValueError(f"Calculated expectation value is not a float, got type {type(val)}")
    return val


def expectation_value_mat(bra: np.ndarray, op: np.ndarray, ket: np.ndarray) -> float:
    """Calculate expectation value of operator in matrix form.

    Args:
        bra: Bra state.
        op: Operator.
        ket: Ket state.

    Returns:
        Expectation value.
    """
    return np.matmul(bra, np.matmul(op, ket))


@functools.cache
def Epq_matrix(
    p: int, q: int, num_active_orbs: int, num_active_elec_alpha: int, num_active_elec_beta: int
) -> ss.lil_array:
    """Get matrix representation of Epq operator.

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.
        num_active_orbs: Number of active sptial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of Epq operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_active_elec_alpha, num_active_elec_beta)
    return ss.lil_array(build_operator_matrix(Epq(p, q), idx2det, det2idx, num_active_orbs))


@functools.cache
def T1_sa_matrix(
    i: int,
    a: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T1 spin-adapted cluster operator.

    Args:
        i: Strongly occupied spatial orbital index.
        a: Weakly occupied spatial orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G1_sa(i, a), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_1_sa_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T2 spin-adapted cluster operator.

    Args:
        i: Strongly occupied spatial orbital index.
        j: Strongly occupied spatial orbital index.
        a: Weakly occupied spatial orbital index.
        b: Weakly occupied spatial orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G2_1_sa(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_2_sa_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T2 spin-adapted cluster operator.

    Args:
        i: Strongly occupied spatial orbital index.
        j: Strongly occupied spatial orbital index.
        a: Weakly occupied spatial orbital index.
        b: Weakly occupied spatial orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G2_2_sa(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T1_matrix(i: int, a: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T1 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G1(i, a), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T2 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G2(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T3_matrix(
    i: int,
    j: int,
    k: int,
    a: int,
    b: int,
    c: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T3 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G3(i, j, k, a, b, c), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T4_matrix(
    i: int,
    j: int,
    k: int,
    l: int,
    a: int,
    b: int,
    c: int,
    d: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T4 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G4(i, j, k, l, a, b, c, d), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T5_matrix(
    i: int,
    j: int,
    k: int,
    l: int,
    m: int,
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T5 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        m: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        e: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G5(i, j, k, l, m, a, b, c, d, e), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T6_matrix(
    i: int,
    j: int,
    k: int,
    l: int,
    m: int,
    n: int,
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    f: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T6 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        m: Strongly occupied spin orbital index.
        n: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        e: Weakly occupied spin orbital index.
        f: Weakly occupied spin orbital index.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G6(i, j, k, l, m, n, a, b, c, d, e, f), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


def construct_ucc_state(
    state: np.ndarray,
    num_active_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    ucc_struct: UccStructure,
    dagger: bool = False,
) -> np.ndarray:
    """Contruct unitary transformation matrix.

    Args:
        num_det: Number of determinants.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).

    Returns:
        Unitary transformation matrix.
    """
    T = np.zeros((len(state), len(state)))
    for exc_type, exc_indices, theta in zip(
        ucc_struct.excitation_operator_type, ucc_struct.excitation_indicies, thetas
    ):
        if abs(theta) < 10**-14:
            continue
        if exc_type == "sa_single":
            (i, a) = exc_indices
            T += (
                theta
                * T1_sa_matrix(i, a, num_active_orbs, num_active_elec_alpha, num_active_elec_beta).todense()
            )
        elif exc_type == "sa_double_1":
            (i, j, a, b) = exc_indices
            T += (
                theta
                * T2_1_sa_matrix(
                    i, j, a, b, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            )
        elif exc_type == "sa_double_2":
            (i, j, a, b) = exc_indices
            T += (
                theta
                * T2_2_sa_matrix(
                    i, j, a, b, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            )
        elif exc_type == "triple":
            (i, j, k, a, b, c) = exc_indices
            T += (
                theta
                * T3_matrix(
                    i, j, k, a, b, c, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            )
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = exc_indices
            T += (
                theta
                * T4_matrix(
                    i,
                    j,
                    k,
                    l,
                    a,
                    b,
                    c,
                    d,
                    num_active_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                ).todense()
            )
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = exc_indices
            T += (
                theta
                * T5_matrix(
                    i,
                    j,
                    k,
                    l,
                    m,
                    a,
                    b,
                    c,
                    d,
                    e,
                    num_active_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                ).todense()
            )
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = exc_indices
            T += (
                theta
                * T6_matrix(
                    i,
                    j,
                    k,
                    l,
                    m,
                    n,
                    a,
                    b,
                    c,
                    d,
                    e,
                    f,
                    num_active_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                ).todense()
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    if dagger:
        return ss.linalg.expm_multiply(-T, state)
    return ss.linalg.expm_multiply(T, state)


def construct_ups_state(
    state: np.ndarray,
    num_active_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
    dagger: bool = False,
) -> np.ndarray:
    r"""Construct unitary product state.

    .. math::
        \boldsymbol{U}_N...\boldsymbol{U}_0\left|\nu\right> = \left|\nu\right>

    #. 10.48550/arXiv.2303.10825, Eq. 15

    Args:
        state: Reference state vector.
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_betaa: Number of active beta electrons.
        thetas: Ansatz parameters values.
        ups_struct: Unitary product state structure.
        dagger: If do dagger unitaries.

    Returns:
        New state vector with unitaries applied.
    """
    tmp = state.copy()
    order = 1
    if dagger:
        order = -1
    for exc_type, exc_indices, theta in zip(
        ups_struct.excitation_operator_type[::order], ups_struct.excitation_indicies[::order], thetas[::order]
    ):
        if abs(theta) < 10**-14:
            continue
        if dagger:
            theta = -theta
        if exc_type in ("tups_single", "sa_single"):
            if exc_type == "tups_single":
                (p,) = exc_indices
                Ta = T1_matrix(
                    p * 2, (p + 1) * 2, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
                Tb = T1_matrix(
                    p * 2 + 1, (p + 1) * 2 + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            elif exc_type == "sa_single":
                (i, a) = exc_indices
                Ta = T1_matrix(
                    i * 2, a * 2, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
                Tb = T1_matrix(
                    i * 2 + 1, a * 2 + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            tmp = (
                tmp
                + np.sin(2 ** (-1 / 2) * theta) * np.matmul(Ta, tmp)
                + (1 - np.cos(2 ** (-1 / 2) * theta)) * np.matmul(Ta, np.matmul(Ta, tmp))
            )
            tmp = (
                tmp
                + np.sin(2 ** (-1 / 2) * theta) * np.matmul(Tb, tmp)
                + (1 - np.cos(2 ** (-1 / 2) * theta)) * np.matmul(Tb, np.matmul(Tb, tmp))
            )
        elif exc_type in ("tups_double", "single", "double"):
            if exc_type == "tups_double":
                (p,) = exc_indices
                T = T2_1_sa_matrix(
                    p, p, p + 1, p + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            elif exc_type == "single":
                (i, a) = exc_indices
                T = T1_matrix(i, a, num_active_orbs, num_active_elec_alpha, num_active_elec_beta).todense()
            elif exc_type == "double":
                (i, j, a, b) = exc_indices
                T = T2_matrix(
                    i, j, a, b, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
                ).todense()
            tmp = (
                tmp
                + np.sin(theta) * np.matmul(T, tmp)
                + (1 - np.cos(theta)) * np.matmul(T, np.matmul(T, tmp))
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def propagate_unitary(
    state: np.ndarray,
    idx: int,
    num_active_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
) -> np.ndarray:
    """Apply unitary from operator number 'idx' to state.

    Args:
        state: State vector.
        idx: Index of operator in the ups_struct.
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        thetas: Values for ansatz parameters.
        ups_struct: UPS structure object.

    Returns:
        State with unitary applied.
    """
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indicies[idx]
    theta = thetas[idx]
    if abs(theta) < 10**-14:
        return np.copy(state)
    if exc_type in ("tups_single", "sa_single"):
        if exc_type == "tups_single":
            (p,) = exc_indices
            Ta = T1_matrix(
                p * 2, (p + 1) * 2, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
            Tb = T1_matrix(
                p * 2 + 1, (p + 1) * 2 + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
        elif exc_type == "sa_single":
            (i, a) = exc_indices
            Ta = T1_matrix(
                i * 2, a * 2, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
            Tb = T1_matrix(
                i * 2 + 1, a * 2 + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
        A = 2 ** (-1 / 2)
        tmp = (
            state
            + np.sin(A * theta) * np.matmul(Ta, state)
            + (1 - np.cos(A * theta)) * np.matmul(Ta, np.matmul(Ta, state))
        )
        tmp = (
            tmp
            + np.sin(A * theta) * np.matmul(Tb, tmp)
            + (1 - np.cos(A * theta)) * np.matmul(Tb, np.matmul(Tb, tmp))
        )
    elif exc_type in ("tups_double", "single", "double"):
        if exc_type == "tups_double":
            (p,) = exc_indices
            T = T2_1_sa_matrix(
                p, p, p + 1, p + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
        elif exc_type == "single":
            (i, a) = exc_indices
            T = T1_matrix(i, a, num_active_orbs, num_active_elec_alpha, num_active_elec_beta).todense()
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T = T2_matrix(i, j, a, b, num_active_orbs, num_active_elec_alpha, num_active_elec_beta).todense()
        tmp = (
            state
            + np.sin(theta) * np.matmul(T, state)
            + (1 - np.cos(theta)) * np.matmul(T, np.matmul(T, state))
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def get_grad_action(
    state: np.ndarray,
    idx: int,
    num_active_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    ups_struct: UpsStructure,
) -> np.ndarray:
    r"""Get effect of differentiation with respect to "idx" operator in the UPS expansion.

    .. math::
        \frac{\partial}{\partial \theta_i}\left(\left<\text{CSF}\right|\boldsymbol{U}(\theta_{i-1})\boldsymbol{U}(\theta_i)\right) =
        \left<\text{CSF}\right|\boldsymbol{U}(\theta_{i-1})\frac{\partial \boldsymbol{U}(\theta_i)}{\partial \theta_i}

    With,

    .. math::
        \begin{align}
        \frac{\partial \boldsymbol{U}(\theta_i)}{\partial \theta_i} &= \frac{\partial}{\partial \theta_i}\exp\left(\theta_i \hat{T}_i\right)\\
                &= \exp\left(\theta_i \hat{T}_i\right)\hat{T}_i
        \end{align}

    This function only applies the $\hat{T}_i$ part to the state.

    #. 10.48550/arXiv.2303.10825, Eq. 20

    Args:
        state: State vector.
        idx: Index of operator in the ups_struct.
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        ups_struct: UPS structure object.

    Returns:
        State with derivative of the idx'th unitary applied.
    """
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indicies[idx]
    if exc_type in ("tups_single", "sa_single"):
        if exc_type == "tups_single":
            (p,) = exc_indices
            Ta = T1_matrix(
                p * 2, (p + 1) * 2, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
            Tb = T1_matrix(
                p * 2 + 1, (p + 1) * 2 + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
        elif exc_type == "sa_single":
            (i, a) = exc_indices
            Ta = T1_matrix(
                i * 2, a * 2, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
            Tb = T1_matrix(
                i * 2 + 1, a * 2 + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
        A = 2 ** (-1 / 2)
        tmp = np.matmul(A * (Ta + Tb), state)
    elif exc_type in ("tups_double", "single", "double"):
        if exc_type == "tups_double":
            (p,) = exc_indices
            T = T2_1_sa_matrix(
                p, p, p + 1, p + 1, num_active_orbs, num_active_elec_alpha, num_active_elec_beta
            ).todense()
        elif exc_type == "single":
            (i, a) = exc_indices
            T = T1_matrix(i, a, num_active_orbs, num_active_elec_alpha, num_active_elec_beta).todense()
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T = T2_matrix(i, j, a, b, num_active_orbs, num_active_elec_alpha, num_active_elec_beta).todense()
        tmp = np.matmul(T, state)
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp
