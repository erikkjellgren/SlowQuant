import numpy as np
from slowquant.unitary_coupled_cluster.integrals import Integrals


def build_fock_inactive(
        h_ij,
        h_iv,
        h_ia,
        h_vw,
        h_va,
        h_ab,
        g_ijkk_Ck,
        g_ikkj_Ck,
        g_iijv_Ci,
        g_ijjv_Cj,
        g_iija_Ci,
        g_ijja_Cj,
        g_iivw_Ci,
        g_iviw_Ci,
        g_iiva_Ci,
        g_ivia_Ci,
        g_iiab_Ci,
        g_iaib_Ci,
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
):
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    inact = slice(0,num_inactive_orbs)
    act = slice(num_inactive_orbs,num_inactive_orbs+num_active_orbs)
    virt = slice(num_inactive_orbs+num_active_orbs, num_orbs)
    F = np.zeros((num_orbs, num_orbs))
    # inactive-inactive
    F[inact, inact] = h_ij + 2*g_ijkk_Ck - g_ikkj_Ck
    # inactive-active
    F[inact, act] = h_iv + 2*g_iijv_Ci - g_ijjv_Cj
    F[act, inact] = F[inact, act].T
    # inactive-virtual
    F[inact, virt] = h_ia + 2*g_iija_Ci - g_ijja_Cj
    F[virt, inact] = F[inact, virt].T
    # active-active
    F[act, act] = h_vw + 2*g_iivw_Ci - g_iviw_Ci
    # active-virtual
    F[act, virt] = h_va + 2*g_iiva_Ci - g_ivia_Ci
    F[virt, act] = F[act, virt].T
    # virtual-virtual
    F[virt, virt] = h_ab + 2*g_iiab_Ci - g_iaib_Ci
    return F


def build_fock_active(
    rdm1,
    g_ijvw,
    g_ivjw,
    g_ivwx,
    g_iavw,
    g_ivwa,
    g_vwxy,
    g_vwxa,
    g_vwab,
    g_vawb,
    num_inactive_orbs,
    num_active_orbs,
    num_virtual_orbs,
    ):
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    inact = slice(0,num_inactive_orbs)
    act = slice(num_inactive_orbs,num_inactive_orbs+num_active_orbs)
    virt = slice(num_inactive_orbs+num_active_orbs, num_orbs)
    F = np.zeros((num_orbs, num_orbs))
    # inactive-inactive
    F[inact, inact] = np.einsum('xy,mnxy->mn',rdm1,g_ijvw) - 0.5*np.einsum('xy,mxny->mn',rdm1,g_ivjw)
    # inactive-active
    F[inact, act] = np.einsum('xy,mnxy->mn',rdm1,g_ivwx) - 0.5*np.einsum('xy,myxn->mn',rdm1,g_ivwx)
    F[act, inact] = F[inact, act].T
    # inactive-virtual
    F[inact, virt] = np.einsum('xy,mnxy->mn',rdm1,g_iavw) - 0.5*np.einsum('xy,myxn->mn',rdm1,g_ivwa)
    F[virt, inact] = F[inact, virt].T
    # active-active
    F[act, act] = np.einsum('xy,mnxy->mn',rdm1,g_vwxy) - 0.5*np.einsum('xy,myxn->mn',rdm1,g_vwxy)
    # active-virtual
    F[act, virt] = np.einsum('xy,xymn->mn',rdm1,g_vwxa) - 0.5*np.einsum('xy,myxn->mn',rdm1,g_vwxa)
    F[virt, act] = F[act, virt].T
    # virtual-virtual
    F[virt, virt] = np.einsum('xy,xymn->mn',rdm1,g_vwab) - 0.5*np.einsum('xy,ymxn->mn',rdm1,g_vawb)
    return F


def build_fock_matrix(
    fock_inactive,
    fock_active,
    rdm1,
    rdm2,
    g_ivwx,
    g_vwxy,
    g_vwxa,
    num_inactive_orbs,
    num_active_orbs,
    num_virtual_orbs,
):
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    inact = slice(0,num_inactive_orbs)
    act = slice(num_inactive_orbs,num_inactive_orbs+num_active_orbs)
    virt = slice(num_inactive_orbs+num_active_orbs, num_orbs)
    F = np.zeros((num_orbs, num_orbs))
    Q_vi = np.einsum('vwxy,mwxy->vm',rdm2,g_ivwx)
    Q_vw = np.einsum('vwxy,mwxy->vm',rdm2,g_vwxy)
    Q_va = np.einsum('vwxy,xywm->vm',rdm2,g_vwxa)
    # inactive-inactive
    F[inact, inact] = 2*fock_inactive[inact, inact] + 2*fock_active[inact, inact]
    # inactive-active
    F[inact, act] = 2*fock_inactive[inact, act] + 2*fock_active[inact, act]
    # inactive-virtual
    F[inact, virt] = 2*fock_inactive[inact, virt] + 2*fock_active[inact, virt]
    # active-inactive
    F[act, inact] = np.einsum('iw,vw->vi',fock_inactive[inact,act],rdm1) + Q_vi
    # active-active
    F[act, act] = np.einsum('xw,vw->vx',fock_inactive[act,act],rdm1) + Q_vw
    # active-virtual
    F[act, virt] = np.einsum('aw,vw->va',fock_inactive[virt,act],rdm1) + Q_va
    # virtual-X is zero
    return F


def get_orbital_gradient(kappa_idx: list[tuple[int, int]], fock_mat: np.ndarray) -> np.ndarray:
    gradient = np.zeros(len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        gradient[idx] = 2 * (fock_mat[m, n] - fock_mat[n, m])
    return gradient


def get_electronic_energy(
    rdm1, rdm2, h_ii_Ci, g_iijj_Cij, g_ijji_Cij, h_vw, g_vwxy, g_iivw_Ci, g_iviw_Ci
) -> float:
    energy = 0.0
    # Core contribution
    energy += 2 * h_ii_Ci + 2 * g_iijj_Cij - g_ijji_Cij
    # Active contribution
    energy += np.einsum("vw,vw->", h_vw, rdm1)
    energy += 0.5 * np.einsum("vwxy,vwxy->", g_vwxy, rdm2)
    energy += 2 * np.einsum("vw,vw->", g_iivw_Ci, rdm1)
    energy += -np.einsum("vw,vw->", g_iviw_Ci, rdm1)
    return energy


def get_orbital_hessian():
    return None
