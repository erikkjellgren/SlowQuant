import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)


class ReducedDenstiyMatrix:
    def __init__(
        self,
        num_inactive_orbs: int,
        num_active_orbs: int,
        num_virtual_orbs: int,
        rdm1: np.ndarray,
        rdm2: np.ndarray | None = None,
    ) -> None:
        self.inactive_idx = []
        self.actitve_idx = []
        self.virtual_idx = []
        for idx in range(num_inactive_orbs):
            self.inactive_idx.append(idx)
        for idx in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            self.actitve_idx.append(idx)
        for idx in range(
            num_inactive_orbs + num_active_orbs, num_inactive_orbs + num_active_orbs + num_virtual_orbs
        ):
            self.virtual_idx.append(idx)
        self.idx_shift = num_inactive_orbs
        self.rdm1 = rdm1
        self.rdm2 = rdm2

    def RDM1(self, p: int, q: int) -> float:
        if p in self.actitve_idx and q in self.actitve_idx:
            return self.rdm1[p - self.idx_shift, q - self.idx_shift]
        if p in self.inactive_idx and q in self.inactive_idx:
            if p == q:
                return 2
            return 0
        return 0

    def RDM2(self, p: int, q: int, r: int, s: int) -> float:
        if self.rdm2 is None:
            raise ValueError('RDM2 is not given.')
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            return self.rdm2[p - self.idx_shift, q - self.idx_shift, r - self.idx_shift, s - self.idx_shift]
        if (
            p in self.inactive_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.inactive_idx
        ):
            if p == s:
                return -self.rdm1[q - self.idx_shift, r - self.idx_shift]
            return 0
        if (
            p in self.actitve_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.actitve_idx
        ):
            if q == r:
                return -self.rdm1[p - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if r == s:
                return 2 * self.rdm1[p - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            if p == q:
                return 2 * self.rdm1[r - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            val = 0
            if p == q and r == s:
                val += 4
            if q == r and p == s:
                val -= 2
            return val
        return 0


def get_orbital_gradient(
    rdms: ReducedDenstiyMatrix,
    h_ao: np.ndarray,
    g_ao: np.ndarray,
    c_mo: np.ndarray,
    kappa_idx: list[list[int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    """ """
    gradient = np.zeros(len(kappa_idx))
    h_int = one_electron_integral_transform(c_mo, h_ao)
    g_int = two_electron_integral_transform(c_mo, g_ao)
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] += 2 * h_int[n, p] * rdms.RDM1(m, p)
            gradient[idx] -= 2 * h_int[p, m] * rdms.RDM1(p, n)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    gradient[idx] += g_int[n, p, q, r] * rdms.RDM2(m, p, q, r)
                    gradient[idx] -= g_int[p, m, q, r] * rdms.RDM2(p, n, q, r)
                    gradient[idx] -= g_int[m, p, q, r] * rdms.RDM2(n, p, q, r)
                    gradient[idx] += g_int[p, n, q, r] * rdms.RDM2(p, m, q, r)
    return gradient


def get_orbital_hessian_A(
    rdms: ReducedDenstiyMatrix,
    h_ao: np.ndarray,
    g_ao: np.ndarray,
    c_mo: np.ndarray,
    kappa_idx: list[list[int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    A1e = np.zeros((len(kappa_idx), len(kappa_idx)))
    A2e = np.zeros((len(kappa_idx), len(kappa_idx)))
    h = one_electron_integral_transform(c_mo, h_ao)
    g = two_electron_integral_transform(c_mo, g_ao)
    for idx1, (u, t) in enumerate(kappa_idx):
        for idx2, (m, n) in enumerate(kappa_idx):
            # 1e contribution
            A1e[idx1, idx2] += h[n, t] * rdms.RDM1(m, u)
            A1e[idx1, idx2] += h[u, m] * rdms.RDM1(t, n)
            for p in range(num_inactive_orbs + num_active_orbs):
                if m == u:
                    A1e[idx1, idx2] -= h[n, p] * rdms.RDM1(t, p)
                if t == n:
                    A1e[idx1, idx2] -= h[p, m] * rdms.RDM1(p, u)
            # 2e contribution
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    A2e[idx1, idx2] += g[n, t, p, q] * rdms.RDM2(m, u, p, q)
                    A2e[idx1, idx2] -= g[n, p, u, q] * rdms.RDM2(m, t, p, q)
                    A2e[idx1, idx2] += g[n, p, q, t] * rdms.RDM2(m, p, q, u)
                    A2e[idx1, idx2] += g[u, m, p, q] * rdms.RDM2(t, n, p, q)
                    A2e[idx1, idx2] += g[p, m, u, q] * rdms.RDM2(p, n, t, q)
                    A2e[idx1, idx2] -= g[p, m, q, t] * rdms.RDM2(p, n, q, u)
                    A2e[idx1, idx2] -= g[u, p, n, t] * rdms.RDM2(t, p, m, q)
                    A2e[idx1, idx2] += g[p, t, n, q] * rdms.RDM2(p, u, m, q)
                    A2e[idx1, idx2] += g[p, q, n, t] * rdms.RDM2(p, q, m, u)
                    A2e[idx1, idx2] += g[u, p, q, m] * rdms.RDM2(t, p, q, n)
                    A2e[idx1, idx2] -= g[p, t, q, m] * rdms.RDM2(p, u, q, n)
                    A2e[idx1, idx2] += g[p, q, u, m] * rdms.RDM2(p, q, t, n)
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    for r in range(num_inactive_orbs + num_active_orbs):
                        if m == u:
                            A2e[idx1, idx2] -= g[n, p, q, r] * rdms.RDM2(t, p, q, r)
                        if t == n:
                            A2e[idx1, idx2] -= g[p, m, q, r] * rdms.RDM2(p, u, q, r)
                        if m == u:
                            A2e[idx1, idx2] -= g[p, q, n, r] * rdms.RDM2(p, q, t, r)
                        if t == n:
                            A2e[idx1, idx2] -= g[p, q, r, m] * rdms.RDM2(p, q, r, u)
    return 1 / 2 * A1e + 1 / 4 * A2e
