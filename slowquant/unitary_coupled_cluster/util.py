import numpy as np
import scipy.linalg

from slowquant.unitary_coupled_cluster.base import a_op_spin_matrix


def construct_integral_trans_mat(
    c_orthonormal: np.ndarray, kappa: list[float], kappa_idx: list[list[int]]
) -> np.ndarray:
    kappa_mat = np.zeros_like(c_orthonormal)
    for kappa_val, (p, q) in zip(kappa, kappa_idx):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    return c_trans


def iterate_T1(
    active_occ: list[int], active_unocc: list[int], is_spin_conserving: bool = False
) -> tuple[int]:
    theta_idx = -1
    # Force false, not sure it is correct, but keeping the code for now
    is_spin_conserving = False
    for a in active_unocc:
        for i in active_occ:
            theta_idx += 1
            num_alpha = 0
            num_beta = 0
            if a % 2 == 0:
                num_alpha += 1
            else:
                num_beta += 1
            if i % 2 == 0:
                num_alpha += 1
            else:
                num_beta += 1
            if (num_alpha % 2 != 0 or num_beta % 2 != 0) and is_spin_conserving:
                continue
            yield theta_idx, a, i


def iterate_T2(
    active_occ: list[int], active_unocc: list[int], is_spin_conserving: bool = False
) -> tuple[int]:
    theta_idx = -1
    # Force false, not sure it is correct, but keeping the code for now
    is_spin_conserving = False
    for a in active_unocc:
        for b in active_unocc:
            if a >= b:
                continue
            for i in active_occ:
                for j in active_occ:
                    if i >= j:
                        continue
                    theta_idx += 1
                    num_alpha = 0
                    num_beta = 0
                    if a % 2 == 0:
                        num_alpha += 1
                    else:
                        num_beta += 1
                    if b % 2 == 0:
                        num_alpha += 1
                    else:
                        num_beta += 1
                    if i % 2 == 0:
                        num_alpha += 1
                    else:
                        num_beta += 1
                    if j % 2 == 0:
                        num_alpha += 1
                    else:
                        num_beta += 1
                    if (num_alpha % 2 != 0 or num_beta % 2 != 0) and is_spin_conserving:
                        continue
                    yield theta_idx, a, i, b, j


import time


def construct_UCC_U(
    num_spin_orbs: int,
    num_elec: int,
    theta: list[float],
    excitations: str,
    active_occ: list[int],
    active_unocc: list[int],
    use_csr: int = 8,
) -> np.ndarray:
    t = np.zeros((2**num_spin_orbs, 2**num_spin_orbs))
    counter = 0
    start = time.time()
    if "s" in excitations:
        for (_, a, i) in iterate_T1(active_occ, active_unocc, is_spin_conserving=True):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, num_elec, use_csr=use_csr).dot(
                    a_op_spin_matrix(i, False, num_spin_orbs, num_elec, use_csr=use_csr)
                )
                t += theta[counter] * tmp
            counter += 1

    if "d" in excitations:
        for (_, a, i, b, j) in iterate_T2(active_occ, active_unocc, is_spin_conserving=True):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, num_elec, use_csr=use_csr).dot(
                    a_op_spin_matrix(b, True, num_spin_orbs, num_elec, use_csr=use_csr)
                )
                tmp = tmp.dot(a_op_spin_matrix(j, False, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(i, False, num_spin_orbs, num_elec, use_csr=use_csr))
                t += theta[counter] * tmp
            counter += 1
    
    if num_spin_orbs >= use_csr:
        T = t - t.conjugate().transpose()
        A = scipy.sparse.linalg.expm(T)
    else:
        T = t - np.conj(t).transpose()
        A = scipy.linalg.expm(T)
    #print(f"expm: {time.time() - start}")
    return A
