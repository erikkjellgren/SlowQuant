import numpy as np

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)
from unitary_coupled_cluster.operators import a_op_spin


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
        num_inactive_orbs: Number of inactive orbitals in spin basis.
        num_active_orbs: Number of active orbitals in spin basis.
    Returns:
        Energy Hamiltonian fermionic operator.
    """
    hamiltonian_operator = FermionicOperator({})
    # Inactive one-electron
    for P in range(num_inactive_spin_orbs):
        if abs(h_mo[P, P]) > 10**-14:
            hamiltonian_operator += h_mo[P, P] * a_op_spin(P,dagger=True)*a_op_spin(P,dagger=False)
    # Active one-electron
    for P in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
        for Q in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
            if abs(h_mo[P, Q]) > 10**-14:
                hamiltonian_operator += h_mo[P, Q] * a_op_spin(P,dagger=True)*a_op_spin(Q,dagger=False)
    # Inactive two-electron
    for P in range(num_inactive_spin_orbs):
        for Q in range(num_inactive_spin_orbs):
            if P != Q and abs(g_mo[P, P, Q, Q]) > 10**-14:
                hamiltonian_operator += (1 / 2 * g_mo[P, P, Q, Q] 
                                         *a_op_spin(P,dagger=True)*a_op_spin(Q,dagger=True)
                                         *a_op_spin(Q,dagger=False)*a_op_spin(P,dagger=False))
    # Inactive-Active two-electron
    for I in range(num_inactive_spin_orbs):
        for P in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
            for Q in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
                if abs(g_mo[I, I, P, Q]) > 10**-14:
                    hamiltonian_operator += (1 / 2 * g_mo[I, I, P, Q] 
                                            *a_op_spin(I,dagger=True)*a_op_spin(P,dagger=True)
                                            *a_op_spin(Q,dagger=False)*a_op_spin(I,dagger=False))
                if abs(g_mo[P, Q, I, I]) > 10**-14:
                    hamiltonian_operator += (1 / 2 * g_mo[P, Q, I, I] 
                                            *a_op_spin(P,dagger=True)*a_op_spin(I,dagger=True)
                                            *a_op_spin(I,dagger=False)*a_op_spin(Q,dagger=False))
    # Active two-electron
    for P in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
        for Q in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
            for R in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
                for S in range(num_inactive_spin_orbs, num_inactive_spin_orbs + num_active_spin_orbs):
                    if abs(g_mo[P, Q, R, S]) > 10**-14:
                        hamiltonian_operator += 1 / 2 * (g_mo[P, Q, R, S] 
                                         *a_op_spin(P,dagger=True)*a_op_spin(R,dagger=True)
                                         *a_op_spin(S,dagger=False)*a_op_spin(Q,dagger=False))
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
        num_inactive_orbs: Number of inactive orbitals in spin basis.
        num_active_orbs: Number of active orbitals in spin basis.
        num_virtual_orbs: Number of virtual orbitals in spin basis.

    Returns:
        Modified Hamiltonian fermionic operator.
    """
    num_spin_orbs = num_inactive_spin_orbs + num_active_spin_orbs + num_virtual_spin_orbs
    hamiltonian_operator = FermionicOperator({})
    virtual_start = num_inactive_spin_orbs + num_active_spin_orbs
    for P in range(num_spin_orbs):
        for Q in range(num_spin_orbs):
            if P >= virtual_start and Q >= virtual_start:
                continue
            if P < num_inactive_spin_orbs and Q < num_inactive_spin_orbs and P != Q:
                continue
            if abs(h_mo[P, Q]) > 10**-14:
                hamiltonian_operator += h_mo[P, Q] * a_op_spin(P,dagger=True)*a_op_spin(Q,dagger=False)
    for P in range(num_spin_orbs):
        for Q in range(num_spin_orbs):
            for R in range(num_spin_orbs):
                for S in range(num_spin_orbs):
                    num_virt, num_act = (0,0)
                    for item in [P,Q,R,S]:
                        if item >= virtual_start:
                            num_virt += 1
                    if num_virt > 1:
                        continue
                    for item in [P,Q,R,S]:
                        if item < num_inactive_spin_orbs:
                            num_act += 1
                    if P < num_inactive_spin_orbs and Q < num_inactive_spin_orbs and P == Q:
                        num_act -= 2
                    if R < num_inactive_spin_orbs and S < num_inactive_spin_orbs and R == S:
                        num_act -= 2
                    if P < num_inactive_spin_orbs and S < num_inactive_spin_orbs and P == S:
                        num_act -= 2
                    if Q < num_inactive_spin_orbs and R < num_inactive_spin_orbs and Q == R:
                        num_act -= 2
                    if num_act > 1:
                        continue
                    if abs(g_mo[P, Q, R, S]) > 10**-14:
                        hamiltonian_operator += 1 / 2 * (g_mo[P, Q, R, S]
                                         *a_op_spin(P,dagger=True)*a_op_spin(R,dagger=True)
                                         *a_op_spin(S,dagger=False)*a_op_spin(Q,dagger=False))
    return hamiltonian_operator

