import numpy as np
import scipy.sparse as ss
from sympy.utilities.iterables import multiset_permutations

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.qiskit_interface.base import FermionicOperator


def get_indexing(
    num_orbs: int, num_elec_alpha: int, num_elec_beta: int
) -> tuple[dict[int, int], dict[int, int]]:
    idx = 0
    idx2det = {}
    det2idx = {}
    for alpha_string in multiset_permutations([1] * num_elec_alpha + [0] * (num_orbs - num_elec_alpha)):
        for beta_string in multiset_permutations([1] * num_elec_beta + [0] * (num_orbs - num_elec_beta)):
            det = ""
            for a, b in zip(alpha_string, beta_string):
                det += str(a) + str(b)
            det = int(det, 2)
            idx2det[idx] = det
            det2idx[det] = idx
            idx += 1
    return idx2det, det2idx


def build_operator(
    op: FermionicOperator, idx2det: dict[int, int], det2idx: dict[int, int], num_orbs: int
) -> np.ndarray:
    num_dets = len(idx2det)
    op_mat = np.zeros((num_dets, num_dets))
    for i in range(num_dets):
        op_state_vec = np.zeros(num_dets)
        for fermi_label in op.factors:
            det = np.array([int(x) for x in format(idx2det[i], f"#0{2*num_orbs+2}b")[2:]], dtype=int)
            phase_changes = 0
            for fermi_op in op.operators[fermi_label][::-1]:
                orb_idx = fermi_op.idx
                if det[orb_idx] == 0 and fermi_op.dagger:
                    det[orb_idx] = 1
                    phase_changes += np.sum(det[0:orb_idx])
                elif det[orb_idx] == 1 and fermi_op.dagger:
                    break
                elif det[orb_idx] == 0 and not fermi_op.dagger:
                    break
                elif det[orb_idx] == 1 and not fermi_op.dagger:
                    det[orb_idx] = 0
                    phase_changes += np.sum(det[0:orb_idx])
            else:  # nobreak
                det_idx = int("".join([str(x) for x in det]), 2)
                op_state_vec[det2idx[det_idx]] += op.factors[fermi_label] * (-1) ** phase_changes
        op_mat[i, :] = op_state_vec
    return op_mat


def expectation_value(
    bra: np.ndarray,
    op: FermionicOperator,
    ket: np.ndarray,
    idx2det: dict[int, int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
) -> float:
    op_mat = build_operator(
        op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs),
        idx2det,
        det2idx,
        num_active_orbs,
    )
    return np.matmul(bra, np.matmul(op_mat, ket))


def expectation_value_mat(bra: np.ndarray, op: np.ndarray, ket: np.ndarray) -> float:
    return np.matmul(bra, np.matmul(op, ket))
