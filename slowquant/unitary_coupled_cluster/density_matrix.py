import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)


def get_orbital_gradient(
        rdm1: np.ndarray, rdm2: np.ndarray, h_ao: np.ndarray, g_ao: np.ndarray, c_mo: np.ndarray, kappa_idx: list[list[int]]
) -> np.ndarray:
    """ """
    gradient = np.zeros(len(kappa_idx))
    h_int = one_electron_integral_transform(c_mo, h_ao)
    g_int = two_electron_integral_transform(c_mo, g_ao)
    num_orbs = len(h_int)
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_orbs):
            gradient[idx] += 2*h_int[n, p] * rdm1[m, p]
            gradient[idx] -= 2*h_int[p, m] * rdm1[p, n]
        # 2e contribution
        for p in range(num_orbs):
            for q in range(num_orbs):
                for r in range(num_orbs):
                    gradient[idx] += g_int[n, p, q, r] * rdm2[m, p, q, r]
                    gradient[idx] -= g_int[p, m, q, r] * rdm2[p, n, q, r]
                    gradient[idx] -= g_int[m, p, q, r] * rdm2[n, p, q, r]
                    gradient[idx] += g_int[p, n, q, r] * rdm2[p, m, q, r]
    return gradient
