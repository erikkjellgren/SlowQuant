import numpy as np
from slowquant.unitary_coupled_cluster.integrals import Integrals


def build_fock_inactive(
        h_ao,
        g_ao,
        c_mo,
        num_inactive_orbs,
):
    DI_ao = 2*np.einsum('Pi,Qi->PQ',c_mo[:,:num_inactive_orbs],c_mo[:,:num_inactive_orbs])
    F_ao = h_ao + np.einsum('RS,PQRS->PQ', DI_ao, g_ao) - 0.5*np.einsum('RS,PRSQ->PQ', DI_ao, g_ao)
    F = c_mo.T @ F_ao @ c_mo
    return F


def build_fock_active(
        g_ao,
        c_mo,
        rdm1,
        num_inactive_orbs,
        num_active_orbs,
    ):
    DA_ao = np.einsum('vw,Pv,Qw->PQ',rdm1,c_mo[:,num_inactive_orbs:num_inactive_orbs+num_active_orbs],c_mo[:,num_inactive_orbs:num_inactive_orbs+num_active_orbs])
    F_ao = np.einsum('RS,PQRS->PQ', DA_ao, g_ao) - 0.5*np.einsum('RS,PRSQ->PQ', DA_ao, g_ao)
    F = c_mo.T @ F_ao @ c_mo
    return F


def build_fock_matrix(
    g_ao,
    c_mo,
    fock_inactive,
    fock_active,
    rdm1,
    rdm2,
    num_inactive_orbs,
    num_active_orbs,
    num_virtual_orbs,
):
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    inact = slice(0,num_inactive_orbs)
    act = slice(num_inactive_orbs,num_inactive_orbs+num_active_orbs)
    virt = slice(num_inactive_orbs+num_active_orbs, num_orbs)
    F = np.zeros((num_orbs, num_orbs))
    #Q_vi = np.einsum('vwxy,mwxy->vm',rdm2,g_ivwx)
    #Q_vw = np.einsum('vwxy,mwxy->vm',rdm2,g_vwxy)
    #Q_va = np.einsum('vwxy,xywm->vm',rdm2,g_vwxa)
    """
    Q_vn = np.einsum('vwxy,PQRS,Pn,Qw,Rx,Sy->vn',
            rdm2,
            g_ao,
            c_mo,
            c_mo[:,num_inactive_orbs:num_inactive_orbs+num_active_orbs],
            c_mo[:,num_inactive_orbs:num_inactive_orbs+num_active_orbs],
            c_mo[:,num_inactive_orbs:num_inactive_orbs+num_active_orbs],
            optimize=['einsum_path', (1, 3), (2, 4), (2, 3), (0, 2), (0, 1)],
    )
    """
    nao = g_ao.shape[0]
    nmo = c_mo.shape[1]
    n_in = num_inactive_orbs
    n_act = num_active_orbs
    C_act = c_mo[:, n_in:n_in+n_act]

    # Your einsum: PQRS, Qw, Rx, Sy -> Pwxy
    # We transform the LAST 3 indices of g_ao first
    
    # Transform S -> y (Index 3)
    tmp = g_ao.reshape(-1, nao) @ C_act        # (P*Q*R, y)
    tmp = tmp.reshape(nao, nao, nao, n_act)
    
    # Transform R -> x (Index 2)
    tmp = tmp.transpose(0, 1, 3, 2).reshape(-1, nao) @ C_act # (P*Q*y, x)
    tmp = tmp.reshape(nao, nao, n_act, n_act)
    
    # Transform Q -> w (Index 1)
    # Move Q to the end: (P, y, x, Q)
    tmp = tmp.transpose(0, 2, 3, 1).reshape(-1, nao) @ C_act # (P*x*y, w)
    # Resulting mixed integral: g_P_wxy
    g_mixed = tmp.reshape(nao, n_act, n_act, n_act).transpose(0, 3, 2, 1)
    
    # Contract with 2-RDM: d_vwxy * g_P_wxy -> Q_vP
    # Flatten wxy for a single large GEMM
    Q_vP = rdm2.reshape(n_act, -1) @ g_mixed.reshape(nao, -1).T
    
    # Final project P -> n
    # Q_vn = Q_vP * C_Pn
    Q_vn = Q_vP @ c_mo
    

    # inactive-inactive
    F[inact, inact] = 2*fock_inactive[inact, inact] + 2*fock_active[inact, inact]
    # inactive-active
    F[inact, act] = 2*fock_inactive[inact, act] + 2*fock_active[inact, act]
    # inactive-virtual
    F[inact, virt] = 2*fock_inactive[inact, virt] + 2*fock_active[inact, virt]
    # active-inactive
    F[act, inact] = np.einsum('iw,vw->vi',fock_inactive[inact,act],rdm1) + Q_vn[:,inact]
    # active-active
    F[act, act] = np.einsum('xw,vw->vx',fock_inactive[act,act],rdm1) + Q_vn[:,act]
    # active-virtual
    F[act, virt] = np.einsum('aw,vw->va',fock_inactive[virt,act],rdm1) + Q_vn[:,virt]
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
