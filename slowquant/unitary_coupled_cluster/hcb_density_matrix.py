import numba as nb
import numpy as np


@nb.jit(nopython=True)
def RDM1(p: int, q: int, num_inactive_orbs, num_active_orbs, rdm1: np.ndarray) -> float:
    virt_start = num_inactive_orbs + num_active_orbs
    if p >= virt_start or q >= virt_start:
        return 0
    elif p >= num_inactive_orbs and q >= num_inactive_orbs:
        return rdm1[p - num_inactive_orbs, q - num_inactive_orbs]
    elif p == q and p < num_inactive_orbs:
        return 1
    return 0


@nb.jit(nopython=True)
def RDM2(
    p: int, q: int, num_inactive_orbs: int, num_active_orbs: int, rdm1: np.ndarray, rdm2: np.ndarray
) -> float:
    virt_start = num_inactive_orbs + num_active_orbs
    if p >= virt_start or q >= virt_start:
        return 0
    elif p >= num_inactive_orbs and q >= num_inactive_orbs:
        return rdm2[p - num_inactive_orbs, q - num_inactive_orbs]
    elif p < num_inactive_orbs and q >= num_inactive_orbs:
        return rdm1[q - num_inactive_orbs, q - num_inactive_orbs]
    elif p >= num_inactive_orbs and q < num_inactive_orbs:
        return rdm1[p - num_inactive_orbs, p - num_inactive_orbs]
    return 1


@nb.jit(nopython=True)
def get_electronic_energy_hcb(
    h1: np.ndarray,
    h2: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
) -> float:
    energy = 0
    for p in range(num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs + num_active_orbs):
            energy += h1[p, q] * RDM1(p, q, num_inactive_orbs, num_active_orbs, rdm1)
            if p != q:
                energy += h2[p, q] * RDM2(p, q, num_inactive_orbs, num_active_orbs, rdm1, rdm2)
    return energy


@nb.jit(nopython=True)
def get_orbital_gradient_hcb(
    h_int: np.ndarray,
    g_int: np.ndarray,
    kappa_idx: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
) -> np.ndarray:
    r"""Get the restricted orbital gradient for the HCB model.

    .. math::

        \begin{aligned}
        \hat{g}^\text{o}_{tu} &= 4h_{tu}\hat{b}_{t}^\dagger\hat{b}_t - 4h_{ut}\hat{b}_{u}^\dagger\hat{b}_u\\
        &\quad + 4\sum_{p\neq u}\left(g_{ptpu} - 2g_{pptu}\right)\hat{b}_{p}^\dagger\hat{b}_p\hat{b}_{u}^\dagger\hat{b}_u\\
        &\quad + 4\sum_{p\neq t}\left(2g_{pptu} -g_{ptpu}\right)\hat{b}_{p}^\dagger\hat{b}_p\hat{b}_{t}^\dagger\hat{b}_t\\
        &\quad + 2\sum_{p}\left(g_{tppu}\left(\hat{b}^\dagger_t\hat{b}_p + \hat{b}^\dagger_p\hat{b}_t\right) - g_{pupt}\left(\hat{b}^\dagger_p\hat{b}_u + \hat{b}^\dagger_u\hat{b}_p\right)\right)
        \end{aligned}

    Args:

    Returns:
        Orbital gradient.
    """
    gradient = np.zeros(len(kappa_idx))
    for idx, (t, u) in enumerate(kappa_idx):
        gradient[idx] += 4 * (h_int[t, u]) * RDM1(t, t, num_inactive_orbs, num_active_orbs, rdm1)
        gradient[idx] -= 4 * (h_int[u, t]) * RDM1(u, u, num_inactive_orbs, num_active_orbs, rdm1)
        for p in range(num_inactive_orbs + num_active_orbs):
            if p != u:
                gradient[idx] += (
                    4
                    * (g_int[p, t, p, u] - 2 * g_int[p, p, t, u])
                    * RDM2(p, u, num_inactive_orbs, num_active_orbs, rdm1, rdm2)
                )
            if p != t:
                gradient[idx] += (
                    4
                    * (2 * g_int[p, p, t, u] - g_int[p, t, p, u])
                    * RDM2(p, t, num_inactive_orbs, num_active_orbs, rdm1, rdm2)
                )
            # Factor 4 instead of 2, because RDM1 is the symmetrized, i.e. Dpq = 1/2(b^dagger_p b_q + b^dagger_q b_p)
            gradient[idx] += 4 * g_int[t, p, p, u] * RDM1(p, t, num_inactive_orbs, num_active_orbs, rdm1)
            gradient[idx] -= 4 * g_int[p, u, p, t] * RDM1(p, u, num_inactive_orbs, num_active_orbs, rdm1)
    return gradient


@nb.jit(nopython=True)
def get_unrestricted_orbital_gradient_hcb(
    h_aa: np.ndarray,
    h_bb: np.ndarray,
    g_aaaa: np.ndarray,
    g_bbbb: np.ndarray,
    g_aabb: np.ndarray,
    kappa_idx: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
) -> np.ndarray:
    r"""Get the unrestricted orbital gradient for the HCB model"""
    gradient = np.zeros(2 * len(kappa_idx))
    shift = len(kappa_idx)
    for idx, (t, u) in enumerate(kappa_idx):
        # alpha part
        gradient[idx] += 2 * h_aa[t, u] * RDM1(t, t, num_inactive_orbs, num_active_orbs, rdm1)
        gradient[idx] -= 2 * h_aa[u, t] * RDM1(u, u, num_inactive_orbs, num_active_orbs, rdm1)
        # beta part
        gradient[idx + shift] += 2 * h_bb[t, u] * RDM1(t, t, num_inactive_orbs, num_active_orbs, rdm1)
        gradient[idx + shift] -= 2 * h_bb[u, t] * RDM1(u, u, num_inactive_orbs, num_active_orbs, rdm1)
        for p in range(num_inactive_orbs + num_active_orbs):
            if p != u:
                # alpha part
                gradient[idx] += (
                    2
                    * (g_aaaa[p, t, p, u] - g_aaaa[p, p, t, u] - g_aabb[t, u, p, p])
                    * RDM2(p, u, num_inactive_orbs, num_active_orbs, rdm1, rdm2)
                )
                # beta part
                gradient[idx + shift] += (
                    2
                    * (g_bbbb[p, t, p, u] - g_bbbb[p, p, t, u] - g_aabb[p, p, t, u])
                    * RDM2(p, u, num_inactive_orbs, num_active_orbs, rdm1, rdm2)
                )
            if p != t:
                gradient[idx] += (
                    2
                    * (g_aaaa[p, p, t, u] - g_aaaa[p, t, p, u] + g_aabb[t, u, p, p])
                    * RDM2(p, t, num_inactive_orbs, num_active_orbs, rdm1, rdm2)
                )
                # beta part
                gradient[idx + shift] += (
                    2
                    * (g_bbbb[p, p, t, u] - g_bbbb[p, t, p, u] + g_aabb[p, p, t, u])
                    * RDM2(p, t, num_inactive_orbs, num_active_orbs, rdm1, rdm2)
                )
            # Factor 2 instead of 1, because RDM1 is the symmetrized, i.e. Dpq = 1/2(b^dagger_p b_q + b^dagger_q b_p)
            # alpha part
            gradient[idx] += 2 * g_aabb[p, u, t, p] * RDM1(p, t, num_inactive_orbs, num_active_orbs, rdm1)
            gradient[idx] -= 2 * g_aabb[p, t, p, u] * RDM1(p, u, num_inactive_orbs, num_active_orbs, rdm1)
            # beta part
            gradient[idx + shift] += (
                2 * g_aabb[t, p, p, u] * RDM1(p, t, num_inactive_orbs, num_active_orbs, rdm1)
            )
            gradient[idx + shift] -= (
                2 * g_aabb[p, u, p, t] * RDM1(p, u, num_inactive_orbs, num_active_orbs, rdm1)
            )
    return gradient


@nb.jit(nopython=True)
def get_generalized_orbital_gradient_hcb(
    h_mo: np.ndarray,
    g_mo: np.ndarray,
    kappa_idx: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
) -> np.ndarray:
    r"""Get the generalized orbital gradient for the HCB model"""
    gradient = np.zeros(len(kappa_idx))
    for idx, (T, U) in enumerate(kappa_idx):
        t = T // 2
        u = U // 2
        # 1 for alpha spin and -1 for beta spin
        gradient[idx] += 2 * h_mo[T, U] * RDM1(t, t, num_inactive_orbs, num_active_orbs, rdm1)
        gradient[idx] -= 2 * h_mo[U, T] * RDM1(u, u, num_inactive_orbs, num_active_orbs, rdm1)
        for P in range(num_inactive_orbs + num_active_orbs):
            p = P // 2
            if P != U:
                gradient[idx] += (g_mo[P, T, P, U] - g_mo[P, P, T, U]) * RDM2(
                    p, u, num_inactive_orbs, num_active_orbs, rdm1, rdm2
                )
            if P != T:
                gradient[idx] += (
                    2
                    * (g_mo[P, P, T, U] - g_mo[P, T, P, U])
                    * RDM2(p, t, num_inactive_orbs, num_active_orbs, rdm1, rdm2)
                )
            # Factor 2 instead of 1, because RDM1 is the symmetrized, i.e. Dpq = 1/2(b^dagger_p b_q + b^dagger_q b_p)
            pa = 2 * p
            pb = 2 * p + 1
            if U % 2 == 0:  # alpha spin
                ub = 2 * u + 1
                gradient[idx] += (
                    2
                    * (g_mo[pa, ub, pb, T] - g_mo[pb, ub, pa, T])
                    * RDM1(p, u, num_inactive_orbs, num_active_orbs, rdm1)
                )
            else:  # U is beta
                ua = 2 * u
                gradient[idx] -= (
                    2
                    * (g_mo[pa, ua, pb, T] - g_mo[pb, ua, pb, T])
                    * RDM1(p, u, num_inactive_orbs, num_active_orbs, rdm1)
                )
            if T % 2 == 0:  # alpha spin
                tb = 2 * t + 1
                gradient[idx] += (
                    2
                    * (g_mo[tb, pb, pa, U] - g_mo[tb, pa, pb, U])
                    * RDM1(p, t, num_inactive_orbs, num_active_orbs, rdm1)
                )
            else:  # T is beta
                ta = 2 * t
                gradient[idx] -= (
                    2
                    * (g_mo[ta, pb, pa, U] - g_mo[ta, pa, pb, U])
                    * RDM1(p, t, num_inactive_orbs, num_active_orbs, rdm1)
                )
    return gradient
