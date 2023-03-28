import numpy as np
import scipy.linalg

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
