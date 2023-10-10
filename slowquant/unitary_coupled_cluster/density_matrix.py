import numpy as np


def get_orbital_gradient(
    rdm1: np.ndarray, rdm2: np.ndarray, h_int: np.ndarray, g_int: np.ndarray, kappa_idx: list[list[int]]
) -> np.ndarray:
    """ """
    gradient = np.zeros(len(kappa_idx))
    num_orbs = len(h_int)
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_orbs):
            gradient[idx] += h_int[n, p] * rdm1[m, p]
            gradient[idx] -= h_int[p, m] * rdm1[p, n]
        # 2e contribution
        for p in range(num_orbs):
            for q in range(num_orbs):
                for r in range(num_orbs):
                    gradient[idx] += g_int[n, p, q, r] * rdm2[m, p, q, r]
                    gradient[idx] -= g_int[p, m, q, r] * rdm2[p, n, q, r]
                    gradient[idx] += g_int[p, q, n, r] * rdm2[p, q, m, r]
                    gradient[idx] -= g_int[p, q, r, m] * rdm2[p, q, r, n]
    return gradient
