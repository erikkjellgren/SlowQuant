import numpy as np

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)



def a_op_spin(spin_idx: int, dagger: bool) -> FermionicOperator:
    """Construct annihilation/creation operator.

    Args:
        spin_idx: Spin orbital index.
        dagger: If creation operator.

    Returns:
        Annihilation/creation operator.
    """
    return FermionicOperator({((spin_idx, dagger),): 1})

def generalized_hamiltonian_full_space(h_spin_mo: np.ndarray, g_spin_mo: np.ndarray, num_spin_orbs: int) -> FermionicOperator:
    r"""Construct full-space generalized electronic Hamiltonian.

    .. math::
        \hat{H} = ?

    Args:
        h_spin_mo: Core one-electron integrals in spin MO basis.
        g_spin_mo: Two-electron integrals in spin MO basis.
        num_spin_orbs: Number of spin orbitals.

    Returns:
        Generalized Hamiltonian operator in full-space.
    """
    H_operator = FermionicOperator({})
    # Build operator
    for p in range(num_spin_orbs):
        for q in range(num_spin_orbs):
            if abs(h_spin_mo[p, q]) < 10**-14:
                continue
            H_operator += h_spin_mo[p, q] * (a_op_spin(p, True)*a_op_spin(q, False))
    for p in range(num_spin_orbs):
        for q in range(num_spin_orbs):
            for r in range(num_spin_orbs):
                for s in range(num_spin_orbs):
                    if abs(g_spin_mo[p, q, r, s]) < 10**-14:
                        continue
                    H_operator += 1 / 2 * g_spin_mo[p, q, r, s] * (a_op_spin(p, True)*a_op_spin(r, True)*a_op_spin(s, False)*a_op_spin(q, False))
    return H_operator
  



def generalized_hamiltonian_0i_0a(
    h_mo: np.ndarray,
    g_mo: np.ndarray,
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
) -> FermionicOperator:
    """Get energy Hamiltonian operator.

    Args:
        h_mo: One-electron Hamiltonian integrals in MO.
        g_mo: Two-electron Hamiltonian integrals in MO.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Energy Hamiltonian fermionic operator.
    """
    hamiltonian_operator = FermionicOperator({})
    # Inactive one-electron
    for i in range(num_inactive_spin_orbs):
        if abs(h_mo[i, i]) > 10**-14:
            hamiltonian_operator += h_mo[i, i] * (a_op_spin(i, True)*a_op_spin(i, False))
    # # Active one-electron
    for p in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
        for q in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
            if abs(h_mo[p, q]) > 10**-14:
                hamiltonian_operator += h_mo[p, q] * (a_op_spin(p, True)*a_op_spin(q, False))
    # Inactive two-electron
    for i in range(num_inactive_spin_orbs):
        for j in range(num_inactive_spin_orbs):
            if i != j and abs(g_mo[i, i, j, j]) > 10**-14:
                hamiltonian_operator += 1 / 2 * g_mo[i, i, j, j] * (a_op_spin(i, True)*a_op_spin(j, True)*a_op_spin(j, False)*a_op_spin(i, False))
    # Inactive-Active two-electron
    for i in range(num_inactive_spin_orbs):
        for p in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
            for q in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
                if abs(g_mo[i, i, p, q]) > 10**-14:
                    hamiltonian_operator += 1 / 2 * g_mo[i, i, p, q] * (a_op_spin(i, True)*a_op_spin(p, True)*a_op_spin(q, False)*a_op_spin(i, False))
                if abs(g_mo[p, q, i, i]) > 10**-14:
                    hamiltonian_operator += 1 / 2 * g_mo[p, q, i, i] * (a_op_spin(p, True)*a_op_spin(i, True)*a_op_spin(i, False)*a_op_spin(q, False))
    # Active two-electron
    for p in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
        for q in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
            for r in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
                for s in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
                    if abs(g_mo[p, q, r, s]) > 10**-14:
                        hamiltonian_operator += 1 / 2 * g_mo[p, q, r, s] * (a_op_spin(p, True)*a_op_spin(r, True)*a_op_spin(s, False)*a_op_spin(q, False))
    return hamiltonian_operator



def generalized_hamiltonian_1i_1a(
    h_mo: np.ndarray,
    g_mo: np.ndarray,
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    num_virtual_spin_orbs: int,
) -> FermionicOperator:
    """Get Hamiltonian operator that works together with an extra inactive and an extra virtual index.

    Args:
        h_mo: One-electron Hamiltonian integrals in MO.
        g_mo: Two-electron Hamiltonian integrals in MO.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.
        num_virtual_orbs: Number of virtual orbitals in spatial basis.

    Returns:
        Modified Hamiltonian fermionic operator.
    """
    num_spin_orbs = num_inactive_spin_orbs + num_active_spin_orbs + num_virtual_spin_orbs
    hamiltonian_operator = FermionicOperator({})
    virtual_start = num_inactive_spin_orbs + num_active_spin_orbs
    for p in range(num_spin_orbs):
        for q in range(num_spin_orbs):
            if p >= virtual_start and q >= virtual_start:
                continue
            if p < num_inactive_spin_orbs and q < num_inactive_spin_orbs and p != q:
                continue
            if abs(h_mo[p, q]) > 10**-14:
                hamiltonian_operator += h_mo[p, q] * (a_op_spin(p, True)*a_op_spin(q, False))
    for p in range(num_spin_orbs):
        for q in range(num_spin_orbs):
            for r in range(num_spin_orbs):
                for s in range(num_spin_orbs):
                    num_virt = 0
                    if p >= virtual_start:
                        num_virt += 1
                    if q >= virtual_start:
                        num_virt += 1
                    if r >= virtual_start:
                        num_virt += 1
                    if s >= virtual_start:
                        num_virt += 1
                    if num_virt > 1:
                        continue
                    num_act = 0
                    if p < num_inactive_spin_orbs:
                        num_act += 1
                    if q < num_inactive_spin_orbs:
                        num_act += 1
                    if r < num_inactive_spin_orbs:
                        num_act += 1
                    if s < num_inactive_spin_orbs:
                        num_act += 1
                    if p < num_inactive_spin_orbs and q < num_inactive_spin_orbs and p == q:
                        num_act -= 2
                    if r < num_inactive_spin_orbs and s < num_inactive_spin_orbs and r == s:
                        num_act -= 2
                    if p < num_inactive_spin_orbs and s < num_inactive_spin_orbs and p == s:
                        num_act -= 2
                    if q < num_inactive_spin_orbs and r < num_inactive_spin_orbs and q == r:
                        num_act -= 2
                    if num_act > 1:
                        continue
                    if abs(g_mo[p, q, r, s]) > 10**-14:
                        hamiltonian_operator += 1 / 2 * g_mo[p, q, r, s] * (a_op_spin(p, True)*a_op_spin(r, True)*a_op_spin(s, False)*a_op_spin(q, False))
    return hamiltonian_operator
