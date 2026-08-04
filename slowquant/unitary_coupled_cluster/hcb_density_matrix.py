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
