import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix

def construct_integral_trans_mat(c_orthonormal: np.ndarray, kappa: list[float], kappa_idx: list[list[int]]) -> np.ndarray:
    kappa_mat = np.zeros_like(c_orthonormal)
    for kappa_val, (p, q) in zip(kappa, kappa_idx):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    return c_trans
