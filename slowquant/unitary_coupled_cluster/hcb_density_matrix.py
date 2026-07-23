import numba as nb
import numpy as np


@nb.jit(nopython=True)
def RDM1(p: int, num_inactive_orbs, num_active_orbs, rdm1: np.ndarray) -> float:
    virt_start = num_inactive_orbs + num_active_orbs
    if p >= virt_start:
        return 0
    elif p >= num_inactive_orbs:
        return rdm1[p - num_inactive_orbs]
    return 2


@nb.jit(nopython=True)
def RDM2_diag(p: int, q: int, num_inactive_orbs: int, num_active_orbs: int, rdm1: np.ndarray, rdm2: np.ndarray) -> float:
    virt_start = num_inactive_orbs + num_active_orbs
    if p >= virt_start or q >= virt_start:
        return 0
    elif p >= num_inactive_orbs and q >= num_inactive_orbs:
        return rdm2[p - num_inactive_orbs, q - num_inactive_orbs]
    elif p < num_inactive_orbs and q >= num_inactive_orbs:
        return
    elif p >= num_inactive_orbs and q < num_inactive_orbs:
        return
    return


@nb.jit(nopython=True)
def RDM2_symmetrized() -> float:
    return


@nb.jit(nopython=True)
def get_electronic_energy_hcb() -> float:
    return


@nb.jit(nopython=True)
def get_orbital_gradient_hcb() -> np.ndarray:
    return
