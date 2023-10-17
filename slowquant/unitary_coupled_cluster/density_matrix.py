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
    gradient = np.zeros(len(kappa_idx))
    for idx, (p, q) in enumerate(kappa_idx):
        # 1e contribution
        for r in range(num_orbs):
            gradient[idx] += h_int[p, r] * rdm1[r, q]
            gradient[idx] -= h_int[q, r] * rdm1[r, p]
            gradient[idx] -= h_int[r, q] * rdm1[p, r]
            gradient[idx] += h_int[r, p] * rdm1[q, r]
        # 2e contribution
        for r in range(num_orbs):
            for s in range(num_orbs):
                for t in range(num_orbs):
                    gradient[idx] -= g_int[q, r, s, t] * rdm2[p, r, s, t]
                    gradient[idx] += g_int[p, r, s, t] * rdm2[q, r, s, t]
                    gradient[idx] += g_int[r, s, p, t] * rdm2[r, s, q, t]
                    gradient[idx] -= g_int[r, s, q, t] * rdm2[r, s, p, t]
    return gradient
