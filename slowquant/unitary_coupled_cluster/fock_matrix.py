import numpy as np
from slowquant.unitary_coupled_cluster.integrals import Integrals


def build_fock_inactive(
    num_inactive_orbs,
    num_active_orbs,
    num_virtual_orbs,
    ints: Integrals,
):
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    inact = slice(0,num_inactive_orbs)
    act = slice(num_inactive_orbs,num_inactive_orbs+num_active_orbs)
    virt = slice(num_inactive_orbs+num_active_orbs, num_orbs)
    F = np.zeros((num_orbs, num_orbs))
    # inactive-inactive
    F[inact, inact] = ints.h_ij + 2*ints.g_ijkk_Ck - ints.g_ikkj_Ck
    # inactive-active
    F[inact, act] = ints.h_iv + 2*ints.g_ivjj_Cj - ints.g_ijjv_Cj
    F[act, inact] = F[inact, act].T
    # inactive-virtual
    F[inact, virt] = ints.h_ia + 2*ints.g_iajj_Cj - ints.gijja_Cj
    F[virt, inact] = F[inact, virt].T
    # active-active
    F[act, act] = ints.h_vw + 2*ints.g_vwjj_Cj - ints.g_vjjw_Cj
    # active-virtual
    F[act, virt] = ints.h_va + 2*ints.g_vajj_Cj - ints.g_vjja_Cj
    F[virt, act] = F[act, virt].T
    # virtual-virtual
    F[virt, virt] = ints.h_ab + 2*ints.g_iiab_Ci - ints.g_iabi_Ci
    return F


def build_fock_active(
    num_inactive_orbs,
    num_active_orbs,
    num_virtual_orbs,
    rdm1,
    ints: Integrals,
    ):
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    inact = slice(0,num_inactive_orbs)
    act = slice(num_inactive_orbs,num_inactive_orbs+num_active_orbs)
    virt = slice(num_inactive_orbs+num_active_orbs, num_orbs)
    F = np.zeros((num_orbs, num_orbs))
    # inactive-inactive
    F[inact, inact] = np.einsum('xy,mnxy->mn',rdm1,ints.g_ijvw) - 0.5*np.einsum('xy,mxny->mn',rdm1,ints.g_iwjv)
    # inactive-active
    F[inact, act] = np.einsum('xy,mnxy->mn',rdm1,ints.g_ivwx) - 0.5*np.einsum('xy,myxn->mn',rdm,ints.g_ivwx)
    F[act, inact] = F[inact, act].T
    # inactive-virtual
    F[inact, virt] = np.einsum('xy,mnxy->xy',rdm1,ints.g_iavw) - 0.5*np.einsum('xy,myxn->mn',rdm1,ints.g_ivwa)
    F[virt, inact] = F[inact, virt].T
    # active-active
    F[act, act] = np.einsum('xy,mnxy->mn',rdm1,ints.g_vwxy) - 0.5*np.einsum('xy,myxn->mn',rdm1,ints.g_vwxy)
    # active-virtual
    F[act, virt] = np.einsum('xy,xymn->mn',rdm1,ints.g_vwxa) - 0.5*np.einsum('xy,myxn->mn',rdm1,ints.g_vwxa)
    F[virt, act] = F[act, virt].T
    # virtual-virtual
    F[virt, virt] = np.einsum('xy,xymn->mn',rdm1,ints.g_vwab) - 0.5*np.einsum('xy,ymxn->mn',rdm1,ints.g_vawb)
    return F


def build_fock_matrix(
    fock_inactive,
    fock_active,
    rdm2,
    num_inactive_orbs,
    num_active_orbs,
    num_virtual_orbs,
    ints,
):
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    inact = slice(0,num_inactive_orbs)
    act = slice(num_inactive_orbs,num_inactive_orbs+num_active_orbs)
    virt = slice(num_inactive_orbs+num_active_orbs, num_orbs)
    F = np.zeros((num_orbs, num_orbs))
    Q_vi = np.einsum('vwxy,mwxy->vm',rdm2,ints.g_ivwx)
    Q_vw = np.einsum('vwxy,mwxy->vm',rdm2,ints.g_vwxy)
    Q_va = np.einsum('vwxy,xywm->vm',rdm2,ints.g_vwxa)
    # inactive-inactive
    F[inact, inact] = 2*fock_inactive[inact, inact] + 2*fock_active[inact, inact]
    # inactive-active
    F[inact, act] = 2*fock_inactive[inact, act] + 2*fock_active[inact, act]
    # inactive-virtual
    F[inact, virt] = 2*fock_inactive[inact, virt] + 2*fock_active[inact, virt]
    # active-inactive
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
