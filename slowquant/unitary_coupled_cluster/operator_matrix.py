import functools
from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss
from sympy.utilities.iterables import multiset_permutations

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import G1, G2, G1_sa, G2_1_sa, G2_2_sa


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
            det = ""
            for a, b in zip(alpha_string, beta_string):
                det += str(a) + str(b)
            det = int(det, 2)
            idx2det.append(det)
            det2idx[det] = idx
            idx += 1
    return idx2det, det2idx


def get_indexing_extended(
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs,
    num_active_elec_alpha: int,
    num_active_elec_beta,
    order: int,
) -> tuple[list[int], dict[int, int]]:
    inactive_singles = []
    virtual_singles = []
    for inactive, virtual in generate_singles(num_inactive_orbs, num_virtual_orbs):
        inactive_singles.append(inactive)
        virtual_singles.append(virtual)
    inactive_doubles = []
    virtual_doubles = []
    if order >= 2:
        for inactive, virtual in generate_doubles(num_inactive_orbs, num_virtual_orbs):
            inactive_doubles.append(inactive)
            virtual_doubles.append(virtual)
    idx = 0
    idx2det = []
    det2idx = {}
    # Generate 0th space
    for alpha_string in multiset_permutations(
        [1] * num_active_elec_alpha + [0] * (num_active_orbs - num_active_elec_alpha)
    ):
        for beta_string in multiset_permutations(
            [1] * num_active_elec_beta + [0] * (num_active_orbs - num_active_elec_beta)
        ):
            print(alpha_string, beta_string)
            det = ""
            for a, b in zip(
                [1] * num_inactive_orbs + alpha_string + [0] * num_virtual_orbs,
                [1] * num_inactive_orbs + beta_string + [0] * num_virtual_orbs,
            ):
                det += str(a) + str(b)
            det = int(det, 2)
            if det in idx2det:
                continue
            idx2det.append(det)
            det2idx[det] = idx
            idx += 1
    # Generate 1,2 exc alpha space
    for alpha_inactive, alpha_virtual in zip(
        inactive_singles + inactive_doubles, virtual_singles + virtual_doubles
    ):
        active_alpha_elec = (
            num_active_elec_alpha - np.sum(alpha_virtual) + num_inactive_orbs - np.sum(alpha_inactive)
        )
        for alpha_string in multiset_permutations(
            [1] * active_alpha_elec + [0] * (num_active_orbs - active_alpha_elec)
        ):
            for beta_string in multiset_permutations(
                [1] * num_active_elec_beta + [0] * (num_active_orbs - num_active_elec_beta)
            ):
                det = ""
                for a, b in zip(
                    alpha_inactive + alpha_string + alpha_virtual,
                    [1] * num_inactive_orbs + beta_string + [0] * num_virtual_orbs,
                ):
                    det += str(a) + str(b)
                det = int(det, 2)
                if det in idx2det:
                    continue
                idx2det.append(det)
                det2idx[det] = idx
                idx += 1
    # Generate 1,2 exc beta space
    for beta_inactive, beta_virtual in zip(
        inactive_singles + inactive_doubles, virtual_singles + virtual_doubles
    ):
        active_beta_elec = (
            num_active_elec_beta - np.sum(beta_virtual) + num_inactive_orbs - np.sum(beta_inactive)
        )
        for alpha_string in multiset_permutations(
            [1] * num_active_elec_alpha + [0] * (num_active_orbs - num_active_elec_alpha)
        ):
            for beta_string in multiset_permutations(
                [1] * active_beta_elec + [0] * (num_active_orbs - active_beta_elec)
            ):
                det = ""
                for a, b in zip(
                    [1] * num_inactive_orbs + alpha_string + [0] * num_virtual_orbs,
                    beta_inactive + beta_string + beta_virtual,
                ):
                    det += str(a) + str(b)
                det = int(det, 2)
                if det in idx2det:
                    continue
                idx2det.append(det)
                det2idx[det] = idx
                idx += 1
    # Generate 1 exc alpha 1 exc beta space
    if order >= 2:
        for alpha_inactive, alpha_virtual in zip(inactive_singles, virtual_singles):
            active_alpha_elec = (
                num_active_elec_alpha - np.sum(alpha_virtual) + num_inactive_orbs - np.sum(alpha_inactive)
            )
            for beta_inactive, beta_virtual in zip(inactive_singles, virtual_singles):
                active_beta_elec = (
                    num_active_elec_beta - np.sum(beta_virtual) + num_inactive_orbs - np.sum(beta_inactive)
                )
                for alpha_string in multiset_permutations(
                    [1] * active_alpha_elec + [0] * (num_active_orbs - active_alpha_elec)
                ):
                    for beta_string in multiset_permutations(
                        [1] * active_beta_elec + [0] * (num_active_orbs - active_beta_elec)
                    ):
                        det = ""
                        for a, b in zip(
                            alpha_inactive + alpha_string + alpha_virtual,
                            beta_inactive + beta_string + beta_virtual,
                        ):
                            det += str(a) + str(b)
                        det = int(det, 2)
                        if det in idx2det:
                            continue
                        idx2det.append(det)
                        det2idx[det] = idx
                        idx += 1
    return idx2det, det2idx


def generate_singles(num_inactive_orbs: int, num_virtual_orbs: int):
    inactive = [1] * num_inactive_orbs
    virtual = [0] * num_virtual_orbs
    for i in range(num_inactive_orbs + 1):
        if i != num_inactive_orbs:
            inactive[i] = 0
        for j in range(num_virtual_orbs + 1):
            if j != num_virtual_orbs:
                virtual[j] = 1
            yield inactive.copy(), virtual.copy()
            if j != num_virtual_orbs:
                virtual[j] = 0
        if i != num_inactive_orbs:
            inactive[i] = 1


def generate_doubles(num_inactive_orbs: int, num_virtual_orbs: int):
    inactive = [1] * num_inactive_orbs
    virtual = [0] * num_virtual_orbs
    for i in range(num_inactive_orbs + 1):
        if i != num_inactive_orbs:
            inactive[i] = 0
        for i2 in range(min(i + 1, num_inactive_orbs), num_inactive_orbs + 1):
            if i2 != num_inactive_orbs:
                inactive[i2] = 0
            for j in range(num_virtual_orbs + 1):
                if j != num_virtual_orbs:
                    virtual[j] = 1
                for j2 in range(min(j + 1, num_virtual_orbs), num_virtual_orbs + 1):
                    if j2 != num_virtual_orbs:
                        virtual[j2] = 1
                    yield inactive.copy(), virtual.copy()
                    if j2 != num_virtual_orbs:
                        virtual[j2] = 0
                if j != num_virtual_orbs:
                    virtual[j] = 0
            if i2 != num_inactive_orbs:
                inactive[i2] = 1
        if i != num_inactive_orbs:
            inactive[i] = 1


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


def apply_op_to_vec(
    op: FermionicOperator,
    state: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_active_orbs: int,
) -> np.ndarray:
    num_dets = len(idx2det)
    new_state = np.zeros(num_dets)
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    for i in range(num_dets):
        if abs(state[i]) < 10**-14:
            continue
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
                    new_state[det2idx[det]] += val * state[i]
    return new_state


def propagate_state(
    op: FermionicOperator,
    state: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
) -> np.ndarray:
    return apply_op_to_vec(
        op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs),
        state,
        idx2det,
        det2idx,
        num_active_orbs,
    )


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
def T1_sa_matrix(
    i: int, a: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G1_sa(i, a), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_1_sa_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G2_1_sa(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_2_sa_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G2_2_sa(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T1_matrix(i: int, a: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G1(i, a), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_matrix(
    i: int, j: int, a: int, b: int, num_active_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> ss.lil_array:
    idx2det, det2idx = get_indexing(num_active_orbs, num_elec_alpha, num_elec_beta)
    op = build_operator_matrix(G2(i, j, a, b), idx2det, det2idx, num_active_orbs)
    return ss.lil_array(op - op.conjugate().transpose())
