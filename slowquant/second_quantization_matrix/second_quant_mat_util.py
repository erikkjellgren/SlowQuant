import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from slowquant.second_quantization_matrix.second_quant_mat_base import a_op_spin

def construct_integral_trans_mat(c_orthonormal: np.ndarray, kappa: list[float], kappa_idx: list[list[int]]) -> np.ndarray:
    kappa_mat = np.zeros_like(c_orthonormal)
    for kappa_val, (p, q) in zip(kappa, kappa_idx):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    return c_trans

def iterate_T1(active_occ: list[int], active_unocc: list[int]) -> tuple[int]:
    for a in active_unocc:
        for i in active_occ:
            yield a, i

def iterate_T2(active_occ: list[int], active_unocc: list[int]) -> tuple[int]:
    for a in active_unocc:
        for b in active_unocc:
            if a >= b:
                continue
            for i in active_occ:
                for j in active_occ:
                    if i >= j:
                        continue
                    yield a, i, b, j

def construct_UCC_ket(num_spin_orbs: int, num_elec: int, on_vector: csr_matrix, theta: list[float], excitations: str, active_occ: list[int], active_unocc: list[int]) -> csr_matrix:
    t = np.zeros((2**num_spin_orbs,2**num_spin_orbs))
    counter = 0
    if "s" in excitations:
        for (a, i) in iterate_T1(active_occ, active_unocc):
            if theta[counter] != 0.0:
                tmp = a_op_spin(a, True, num_spin_orbs, num_elec).matrix_form.dot(a_op_spin(i, False, num_spin_orbs, num_elec).matrix_form)
                t += theta[counter]*tmp
            counter += 1
    
    if "d" in excitations:
        for (a, i, b, j) in iterate_T2(active_occ, active_unocc):
            if theta[counter] != 0.0:
                tmp = a_op_spin(a, True, num_spin_orbs, num_elec).matrix_form.dot(a_op_spin(b, True, num_spin_orbs, num_elec).matrix_form)
                tmp = tmp.dot(a_op_spin(j, False, num_spin_orbs, num_elec).matrix_form)
                tmp = tmp.dot(a_op_spin(i, False, num_spin_orbs, num_elec).matrix_form)
                t += theta[counter]*tmp
            counter += 1

    T = t - np.conj(t).transpose()
    U = csr_matrix(scipy.linalg.expm(T))
    UCC_ket = U.dot(on_vector.transpose())
    return UCC_ket
