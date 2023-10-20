import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)


class ReducedDenstiyMatrix:
    def __init__(self, num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int, rdm1: np.ndarray, rdm2: np.ndarray | None = None) -> None:
        self.inactive_idx = []
        self.actitve_idx = []
        self.virtual_idx = []
        for idx in range(num_inactive_orbs):
            self.inactive_idx.append(idx)
        for idx in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            self.actitve_idx.append(idx)
        for idx in range(num_inactive_orbs + num_active_orbs, num_inactive_orbs + num_active_orbs + num_virtual_orbs):
            self.virtual_idx.append(idx)
        self.idx_shift = num_inactive_orbs
        self.rdm1 = rdm1
        self.rdm2 = rdm2

    def RDM1(self, p: int, q: int) -> float:
        if p in self.actitve_idx and q in self.actitve_idx:
            return self.rdm1[p-self.idx_shift,q-self.idx_shift]
        if p in self.inactive_idx and q in self.inactive_idx:
            if p == q:
                return 2
            return 0
        return 0

    def RDM2(self, p: int, q: int, r: int, s: int) -> float:
        if self.rdm2 is None:
            raise ValueError('RDM2 is not given.')
        if p in self.actitve_idx and q in self.actitve_idx and r in self.actitve_idx and s in self.actitve_idx:
            return self.rdm2[p-self.idx_shift,q-self.idx_shift,r-self.idx_shift,s-self.idx_shift]
        if p in self.inactive_idx and q in self.actitve_idx and r in self.actitve_idx and s in self.inactive_idx:
            if p == s:
                return -self.rdm1[q-self.idx_shift,r-self.idx_shift]
            return 0
        if p in self.actitve_idx and q in self.inactive_idx and r in self.inactive_idx and s in self.actitve_idx:
            if q == r:
                return -self.rdm1[p-self.idx_shift,s-self.idx_shift]
            return 0
        if p in self.actitve_idx and q in self.actitve_idx and r in self.inactive_idx and s in self.inactive_idx:
            if r == s:
                return 2*self.rdm1[p-self.idx_shift,q-self.idx_shift]
            return 0
        if p in self.inactive_idx and q in self.inactive_idx and r in self.actitve_idx and s in self.actitve_idx:
            if p == q:
                return 2*self.rdm1[r-self.idx_shift,s-self.idx_shift]
            return 0
        if p in self.inactive_idx and q in self.inactive_idx and r in self.inactive_idx and s in self.inactive_idx:
            val = 0
            if p == q and r == s:
                val += 4
            if q == r and p == s:
                val -= 2
            return val
        return 0


def get_orbital_gradient(
        rdms: ReducedDenstiyMatrix, h_ao: np.ndarray, g_ao: np.ndarray, c_mo: np.ndarray, kappa_idx: list[list[int]], num_inactive_orbs: int, num_active_orbs: int
) -> np.ndarray:
    """ """
    gradient = np.zeros(len(kappa_idx))
    h_int = one_electron_integral_transform(c_mo, h_ao)
    g_int = two_electron_integral_transform(c_mo, g_ao)
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs+num_active_orbs):
            gradient[idx] += 2*h_int[n, p] * rdms.RDM1(m, p)
            gradient[idx] -= 2*h_int[p, m] * rdms.RDM1(p, n)
        # 2e contribution
        for p in range(num_inactive_orbs+num_active_orbs):
            for q in range(num_inactive_orbs+num_active_orbs):
                for r in range(num_inactive_orbs+num_active_orbs):
                    gradient[idx] += g_int[n, p, q, r] * rdms.RDM2(m, p, q, r)
                    gradient[idx] -= g_int[p, m, q, r] * rdms.RDM2(p, n, q, r)
                    gradient[idx] -= g_int[m, p, q, r] * rdms.RDM2(n, p, q, r)
                    gradient[idx] += g_int[p, n, q, r] * rdms.RDM2(p, m, q, r)
    return gradient
