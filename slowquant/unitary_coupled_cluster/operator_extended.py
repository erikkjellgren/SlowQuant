import functools
from collections.abc import Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as ss
from sympy.utilities.iterables import multiset_permutations

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G3,
    G4,
    G5,
    G6,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
)
from slowquant.unitary_coupled_cluster.util import ThetaPicker


def get_indexing_extended(
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
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
            det_str = ""
            for a, b in zip(
                [1] * num_inactive_orbs + alpha_string + [0] * num_virtual_orbs,
                [1] * num_inactive_orbs + beta_string + [0] * num_virtual_orbs,
            ):
                det_str += str(a) + str(b)
            det = int(det_str, 2)
            if det in idx2det:
                continue
            idx2det.append(det)
            det2idx[det] = idx
            idx += 1
    # Generate 1,2 exc alpha space
    for alpha_inactive, alpha_virtual in zip(
        inactive_singles + inactive_doubles, virtual_singles + virtual_doubles
    ):
        active_alpha_elec = int(
            num_active_elec_alpha - np.sum(alpha_virtual) + num_inactive_orbs - np.sum(alpha_inactive)
        )
        for alpha_string in multiset_permutations(
            [1] * active_alpha_elec + [0] * (num_active_orbs - active_alpha_elec)
        ):
            for beta_string in multiset_permutations(
                [1] * num_active_elec_beta + [0] * (num_active_orbs - num_active_elec_beta)
            ):
                det_str = ""
                for a, b in zip(
                    alpha_inactive + alpha_string + alpha_virtual,
                    [1] * num_inactive_orbs + beta_string + [0] * num_virtual_orbs,
                ):
                    det_str += str(a) + str(b)
                det = int(det_str, 2)
                if det in idx2det:
                    continue
                idx2det.append(det)
                det2idx[det] = idx
                idx += 1
    # Generate 1,2 exc beta space
    for beta_inactive, beta_virtual in zip(
        inactive_singles + inactive_doubles, virtual_singles + virtual_doubles
    ):
        active_beta_elec = int(
            num_active_elec_beta - np.sum(beta_virtual) + num_inactive_orbs - np.sum(beta_inactive)
        )
        for alpha_string in multiset_permutations(
            [1] * num_active_elec_alpha + [0] * (num_active_orbs - num_active_elec_alpha)
        ):
            for beta_string in multiset_permutations(
                [1] * active_beta_elec + [0] * (num_active_orbs - active_beta_elec)
            ):
                det_str = ""
                for a, b in zip(
                    [1] * num_inactive_orbs + alpha_string + [0] * num_virtual_orbs,
                    beta_inactive + beta_string + beta_virtual,
                ):
                    det_str += str(a) + str(b)
                det = int(det_str, 2)
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
                        det_str = ""
                        for a, b in zip(
                            alpha_inactive + alpha_string + alpha_virtual,
                            beta_inactive + beta_string + beta_virtual,
                        ):
                            det_str += str(a) + str(b)
                        det = int(det_str, 2)
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


def build_operator_matrix_extended(
    op: FermionicOperator,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_orbs: int,
) -> np.ndarray:
    """Build matrix representation of operator.

    Args:
        op: Fermionic number and spin conserving operator.
        idx2det: Index to determinant.
        det2idx: Determinant to index.
        num_orbs: Number of spatial orbitals.

    Returns:
        Matrix representation of operator.
    """
    num_dets = len(idx2det)
    op_mat = np.zeros((num_dets, num_dets))
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_orbs - i] = num
    for i in range(num_dets):
        det_ = idx2det[i]
        for fermi_label in op.factors:
            det = det_
            phase_changes = 0
            for fermi_op in op.operators[fermi_label][::-1]:
                orb_idx = fermi_op.idx
                nth_bit = (det >> 2 * num_orbs - 1 - orb_idx) & 1
                if nth_bit == 0 and fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
                elif nth_bit == 1 and fermi_op.dagger:
                    break
                elif nth_bit == 0 and not fermi_op.dagger:
                    break
                elif nth_bit == 1 and not fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
            else:  # nobreak
                if det not in det2idx:
                    continue
                val = op.factors[fermi_label] * (-1) ** phase_changes
                if abs(val) > 10**-14:
                    op_mat[det2idx[det], i] += val
    return op_mat


def apply_op_to_vec_extended(
    op: FermionicOperator,
    state: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_orbs: int,
) -> np.ndarray:
    num_dets = len(idx2det)
    new_state = np.zeros(num_dets)
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_orbs - i] = num
    for i in range(num_dets):
        if abs(state[i]) < 10**-14:
            continue
        det_ = idx2det[i]
        for fermi_label in op.factors:
            det = det_
            phase_changes = 0
            for fermi_op in op.operators[fermi_label][::-1]:
                orb_idx = fermi_op.idx
                nth_bit = (det >> 2 * num_orbs - 1 - orb_idx) & 1
                if nth_bit == 0 and fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
                elif nth_bit == 1 and fermi_op.dagger:
                    break
                elif nth_bit == 0 and not fermi_op.dagger:
                    break
                elif nth_bit == 1 and not fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
            else:  # nobreak
                if det not in det2idx:
                    continue
                val = op.factors[fermi_label] * (-1) ** phase_changes
                if abs(val) > 10**-14:
                    new_state[det2idx[det]] += val * state[i]
    return new_state


def propagate_state_extended(
    op: FermionicOperator,
    state: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_orbs: int,
) -> np.ndarray:
    return apply_op_to_vec_extended(
        op,
        state,
        idx2det,
        det2idx,
        num_orbs,
    )


def expectation_value_extended(
    bra: np.ndarray,
    op: FermionicOperator,
    ket: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_orbs: int,
) -> float:
    op_mat = build_operator_matrix_extended(
        op,
        idx2det,
        det2idx,
        num_orbs,
    )
    return np.matmul(bra, np.matmul(op_mat, ket))


@functools.cache
def T1_sa_extended_matrix(
    i: int,
    a: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
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
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs, num_active_orbs, num_virtual_orbs, num_elec_alpha, num_elec_beta, 1
    )
    op = build_operator_matrix_extended(
        G1_sa(i, a), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_1_sa_extended_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
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
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs, num_active_orbs, num_virtual_orbs, num_elec_alpha, num_elec_beta, 1
    )
    op = build_operator_matrix_extended(
        G2_1_sa(i, j, a, b), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_2_sa_extended_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
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
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs, num_active_orbs, num_virtual_orbs, num_elec_alpha, num_elec_beta, 1
    )
    op = build_operator_matrix_extended(
        G2_2_sa(i, j, a, b), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


def construct_ucc_u_extended(
    num_det: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    theta: Sequence[float],
    theta_picker: ThetaPicker,
    excitations: str,
) -> np.ndarray:
    """Contruct unitary transformation matrix.

    Args:
        num_det: Number of determinants.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        theta: Active-space parameters.
               Ordered as (S, D, T, ...).
        theta_picker: Helper class to pick the parameters in the right order.
        excitations: Excitation orders to include.

    Returns:
        Unitary transformation matrix.
    """
    T = np.zeros((num_det, num_det))
    counter = 0
    if "s" in excitations:
        for a, i, _ in theta_picker.get_t1_generator_sa():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T1_sa_extended_matrix(
                        i + num_inactive_orbs,
                        a + num_inactive_orbs,
                        num_inactive_orbs,
                        num_active_orbs,
                        num_virtual_orbs,
                        num_elec_alpha,
                        num_elec_beta,
                    ).todense()
                )
            counter += 1
    if "d" in excitations:
        for a, i, b, j, _, type_idx in theta_picker.get_t2_generator_sa():
            if theta[counter] != 0.0:
                if type_idx == 1:
                    T += (
                        theta[counter]
                        * T2_1_sa_extended_matrix(
                            i + num_inactive_orbs,
                            j + num_inactive_orbs,
                            a + num_inactive_orbs,
                            b + num_inactive_orbs,
                            num_inactive_orbs,
                            num_active_orbs,
                            num_virtual_orbs,
                            num_elec_alpha,
                            num_elec_beta,
                        ).todense()
                    )
                elif type_idx == 2:
                    T += (
                        theta[counter]
                        * T2_2_sa_extended_matrix(
                            i + num_inactive_orbs,
                            j + num_inactive_orbs,
                            a + num_inactive_orbs,
                            b + num_inactive_orbs,
                            num_inactive_orbs,
                            num_active_orbs,
                            num_virtual_orbs,
                            num_elec_alpha,
                            num_elec_beta,
                        ).todense()
                    )
                else:
                    raise ValueError(f"Expected type_idx to be in (1,2) got {type_idx}")
            counter += 1
    if "t" in excitations:
        for a, i, b, j, c, k in theta_picker.get_t3_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T3_matrix(i, j, k, a, b, c, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                )
            counter += 1
    if "q" in excitations:
        for a, i, b, j, c, k, d, l in theta_picker.get_t4_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
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
                        num_elec_alpha,
                        num_elec_beta,
                    ).todense()
                )
            counter += 1
    if "5" in excitations:
        for a, i, b, j, c, k, d, l, e, m in theta_picker.get_t5_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
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
                        num_elec_alpha,
                        num_elec_beta,
                    ).todense()
                )
            counter += 1
    if "6" in excitations:
        for a, i, b, j, c, k, d, l, e, m, f, n in theta_picker.get_t6_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
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
                        num_elec_alpha,
                        num_elec_beta,
                    ).todense()
                )
            counter += 1
    assert counter == len(theta)
    A = scipy.linalg.expm(T)
    return A
