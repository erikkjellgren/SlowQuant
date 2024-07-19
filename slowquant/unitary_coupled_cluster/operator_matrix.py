import functools
from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss
from sympy.utilities.iterables import multiset_permutations

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    Epq,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
)


def get_indexing(num_orbs: int, num_elec_alpha: int, num_elec_beta: int) -> tuple[list[int], dict[int, int]]:
    """Get indexing between index and determiant.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec_alpha: Number of alpha electrons.
        num_elec_beta: Number of beta electrons.

    Returns:
        List to map index to determiant and dictionary to map determiant to index.
    """
    idx = 0
    idx2det = []
    det2idx = {}
    for alpha_string in multiset_permutations([1] * num_elec_alpha + [0] * (num_orbs - num_elec_alpha)):
        for beta_string in multiset_permutations([1] * num_elec_beta + [0] * (num_orbs - num_elec_beta)):
            det = ""
            for a, b in zip(alpha_string, beta_string):
                det += str(a) + str(b)
            det = int(det, 2)
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


def expectation_value(
    bra: np.ndarray,
    op: FermionicOperator,
    ket: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
) -> float:
    op_mat = build_operator_matrix(
        op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs),
        idx2det,
        det2idx,
        num_active_orbs,
    )
    return np.matmul(bra, np.matmul(op_mat, ket))


def expectation_value_commutator(
    bra: np.ndarray,
    A: FermionicOperator,
    B: FermionicOperator,
    ket: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
) -> float:
    op = A * B - B * A
    op_mat = build_operator_matrix(
        op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs),
        idx2det,
        det2idx,
        num_active_orbs,
    )
    return np.matmul(bra, np.matmul(op_mat, ket))


def expectation_value_double_commutator(
    bra: np.ndarray,
    A: FermionicOperator,
    B: FermionicOperator,
    C: FermionicOperator,
    ket: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
) -> float:
    op = A * B * C - A * C * B - B * C * A + C * B * A
    op_mat = build_operator_matrix(
        op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs),
        idx2det,
        det2idx,
        num_active_orbs,
    )
    return np.matmul(bra, np.matmul(op_mat, ket))


def expectation_value_mat(bra: np.ndarray, op: np.ndarray, ket: np.ndarray) -> float:
    return np.matmul(bra, np.matmul(op, ket))


@functools.cache
def Epq_matrix(p: int, q: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    return ss.lil_array(build_operator_matrix(Epq(p, q), idx2det, det2idx, num_active_orbs))


@functools.cache
def G1_sa_matrix(
    i: int, a: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    return ss.lil_array(build_operator_matrix(G1_sa(i, a), idx2det, det2idx, num_active_orbs))


@functools.cache
def G2_1_sa_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    return ss.lil_array(build_operator_matrix(G2_1_sa(i, j, a, b), idx2det, det2idx, num_active_orbs))


@functools.cache
def G2_2_sa_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    return ss.lil_array(build_operator_matrix(G2_2_sa(i, j, a, b), idx2det, det2idx, num_active_orbs))


@functools.cache
def G1_matrix(i: int, a: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    return ss.lil_array(build_operator_matrix(G1(i, a), idx2det, det2idx, num_active_orbs))


@functools.cache
def G2_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    return ss.lil_array(build_operator_matrix(G2(i, j, a, b), idx2det, det2idx, num_active_orbs))


@functools.cache
def T1_sa_matrix(
    i: int, a: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op_mat = build_operator_matrix(G1_sa(i, a), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op_mat - op_mat.conjugate().transpose())


@functools.cache
def T2_1_sa_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op_mat = build_operator_matrix(G2_1_sa(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op_mat - op_mat.conjugate().transpose())


@functools.cache
def T2_2_sa_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op_mat = build_operator_matrix(G2_2_sa(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op_mat - op_mat.conjugate().transpose())


@functools.cache
def T1_matrix(i: int, a: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op_mat = build_operator_matrix(G1(i, a), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op_mat - op_mat.conjugate().transpose())


@functools.cache
def T2_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op_mat = build_operator_matrix(G2(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op_mat - op_mat.conjugate().transpose())
