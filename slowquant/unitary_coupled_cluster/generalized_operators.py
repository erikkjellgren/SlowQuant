import numpy as np

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator,
)


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
    return H_operator
