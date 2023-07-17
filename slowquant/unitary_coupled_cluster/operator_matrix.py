import functools

import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.unitary_coupled_cluster.base import kronecker_product


@functools.cache
def a_op_spin_matrix(
    idx: int, dagger: bool, number_spin_orbitals: int, number_electrons: int, use_csr: int = 10
) -> np.ndarray | ss.csr_matrix:
    r"""Get matrix representation of fermionic operator.
    This is the matrix form that depends on number of electrons i.e.:

    .. math::
        Z x Z x .. x a x .. I x I

    Args:
        idx: Spin orbital index.
        dagger: Creation or annihilation operator.
        number_spin_orbitals: Total number of spin orbitals.
        number_electrons: Total number of electrons.

    Returns:
        Matrix representation of ferminonic annihilation or creation operator.
    """
    Z_mat = np.array([[1, 0], [0, -1]], dtype=float)
    I_mat = np.array([[1, 0], [0, 1]], dtype=float)
    a_mat = np.array([[0, 1], [0, 0]], dtype=float)
    a_mat_dagger = np.array([[0, 0], [1, 0]], dtype=float)
    operators = []
    for i in range(number_spin_orbitals):
        if i == idx:
            if dagger:
                operators.append(a_mat_dagger)
            else:
                operators.append(a_mat)
        elif i < idx:
            operators.append(Z_mat)
        else:
            operators.append(I_mat)
    if number_spin_orbitals >= use_csr:
        return ss.csr_matrix(kronecker_product(operators))
    return kronecker_product(operators)


def Epq_matrix(
    p: int, q: int, num_spin_orbs: int, num_elec: int, use_csr: int = 10
) -> np.ndarray | ss.csr_matrix:
    E = lw.matmul(
        a_op_spin_matrix(p * 2, True, num_spin_orbs, num_elec, use_csr=use_csr),
        a_op_spin_matrix(q * 2, False, num_spin_orbs, num_elec, use_csr=use_csr),
    )
    E += lw.matmul(
        a_op_spin_matrix(p * 2 + 1, True, num_spin_orbs, num_elec, use_csr=use_csr),
        a_op_spin_matrix(q * 2 + 1, False, num_spin_orbs, num_elec, use_csr=use_csr),
    )
    return E
