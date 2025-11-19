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
  


