import numpy as np


def build_fock_inactive(
    h_ao,
    g_ao,
    c_mo,
    DI_ao,
):
    F_ao = h_ao + np.einsum("RS,PQRS->PQ", DI_ao, g_ao, optimize=["einsum_path", (0, 1)])
    F_ao -= 0.5 * np.einsum("RS,PRSQ->PQ", DI_ao, g_ao, optimize=["einsum_path", (0, 1)])
    F = c_mo.T @ F_ao @ c_mo
    return F, F_ao


def build_fock_active(
    g_ao,
    c_mo,
    DA_ao,
):
    F_ao = np.einsum("RS,PQRS->PQ", DA_ao, g_ao, optimize=["einsum_path", (0, 1)])
    F_ao -= 0.5 * np.einsum("RS,PRSQ->PQ", DA_ao, g_ao, optimize=["einsum_path", (0, 1)])
    F = c_mo.T @ F_ao @ c_mo
    return F, F_ao


def build_fock_matrix(
    g_Pwxy,
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
    inact = slice(0, num_inactive_orbs)
    act = slice(num_inactive_orbs, num_inactive_orbs + num_active_orbs)
    virt = slice(num_inactive_orbs + num_active_orbs, num_orbs)
    F = np.zeros((num_orbs, num_orbs))
    # Contract with 2-RDM: d_vwxy * g_Pwxy -> Q_vP
    Q_vP = rdm2.reshape(num_active_orbs, -1) @ g_Pwxy.reshape(num_orbs, -1).T
    Q_vn = Q_vP @ c_mo
    # inactive-inactive
    F[inact, inact] = 2 * fock_inactive[inact, inact] + 2 * fock_active[inact, inact]
    # inactive-active
    F[inact, act] = 2 * fock_inactive[inact, act] + 2 * fock_active[inact, act]
    # inactive-virtual
    F[inact, virt] = 2 * fock_inactive[inact, virt] + 2 * fock_active[inact, virt]
    # active-inactive
    F[act, inact] = np.einsum("iw,vw->vi", fock_inactive[inact, act], rdm1) + Q_vn[:, inact]
    # active-active
    F[act, act] = np.einsum("xw,vw->vx", fock_inactive[act, act], rdm1) + Q_vn[:, act]
    # active-virtual
    F[act, virt] = np.einsum("aw,vw->va", fock_inactive[virt, act], rdm1) + Q_vn[:, virt]
    # virtual-X is zero
    return F


def get_orbital_gradient(kappa_idx: list[tuple[int, int]], fock_mat: np.ndarray) -> np.ndarray:
    gradient = np.zeros(len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        gradient[idx] = 2 * (fock_mat[m, n] - fock_mat[n, m])
    return gradient


def get_electronic_energy(
    h_ao, g_vwxy, f_inactive_ao, fock_inactive, DI_ao, rdm1, rdm2, num_inactive_orbs, num_active_orbs
) -> float:
    act = slice(num_inactive_orbs, num_inactive_orbs + num_active_orbs)
    energy = 0.5 * np.sum(DI_ao * (h_ao + f_inactive_ao))
    energy += np.sum(rdm1 * fock_inactive[act, act])
    energy += 0.5 * np.sum(rdm2 * g_vwxy)
    return energy


def get_orbital_hessian():
    return None
