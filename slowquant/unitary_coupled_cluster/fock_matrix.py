import numpy as np


def build_fock_matrix(
    rdm1,
    rdm2,
    num_inactive_orbs,
    num_active_orbs,
    num_virtual_orbs,
    h_pi,
    g_pijj,
    g_piij,
    g_pivw,
    g_pvwi,
    h_pv,
    g_pvii_Ci,
    g_piiv_Ci,
    g_pvwx,
):
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    F = np.zeros((num_orbs, num_orbs))
    # Inactive-general F
    F[:num_inactive_orbs, :] += 2 * h_pi.T
    F[:num_inactive_orbs, :] += 4 * np.einsum(
        "nij->in", g_pijj, optimize=["einsum_path", (0,)]
    ) - 2 * np.einsum("nji->in", g_piij, optimize=["einsum_path", (0,)])
    F[:num_inactive_orbs, :] += 2 * np.einsum(
        "vw,nivw->in", rdm1, g_pivw, optimize=["einsum_path", (0, 1)]
    ) - np.einsum("vw,nwvi->in", rdm1, g_pvwi, optimize=["einsum_path", (0, 1)])
    # Active-general F
    F[num_inactive_orbs : num_inactive_orbs + num_active_orbs, :] += np.einsum(
        "vw,nw->vn", rdm1, h_pv, optimize=["einsum_path", (0, 1)]
    )
    F[num_inactive_orbs : num_inactive_orbs + num_active_orbs, :] += 2 * np.einsum(
        "vw,nw->vn", rdm1, g_pvii_Ci
    ) - np.einsum("vw,nw->vn", rdm1, g_piiv_Ci)
    F[num_inactive_orbs : num_inactive_orbs + num_active_orbs, :] += np.einsum(
        "vwxy,nwxy->vn", rdm2, g_pvwx, optimize=["einsum_path", (0, 1)]
    )
    return F


def get_orbital_gradient(kappa_idx: list[tuple[int, int]], fock_mat: np.ndarray) -> np.ndarray:
    gradient = np.zeros(len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        gradient[idx] = 2 * (fock_mat[m, n] - fock_mat[n, m])
    return gradient


def get_electronic_energy(
    rdm1, rdm2, h_ii_Ci, g_iijj_Cij, g_ijji_Cij, h_vw, g_vwxy, g_iivw_Ci, g_ivwi_Ci
) -> float:
    energy = 0.0
    # Core contribution
    energy += 2 * h_ii_Ci + 2 * g_iijj_Cij - g_ijji_Cij
    # Active contribution
    energy += np.einsum("vw,vw->", h_vw, rdm1)
    energy += 0.5 * np.einsum("vwxy,vwxy->", g_vwxy, rdm2)
    energy += 2 * np.einsum("vw,vw->", g_iivw_Ci, rdm1)
    energy += -np.einsum("vw,vw->", g_ivwi_Ci, rdm1)
    return energy
