import functools

import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.unitary_coupled_cluster.base import kronecker_product_cached


@functools.cache
def a_op_spin_matrix(
    idx: int, dagger: bool, number_spin_orbitals: int, use_csr: int = 10
) -> np.ndarray | ss.csr_matrix:
    r"""Get matrix representation of fermionic operator.

    This is the matrix form that depends on number of electrons i.e.:

    .. math::
        Z x Z x .. x a x .. I x I

    Args:
        idx: Spin orbital index.
        dagger: Creation or annihilation operator.
        number_spin_orbitals: Total number of spin orbitals.

    Returns:
        Matrix representation of ferminonic annihilation or creation operator.
    """
    prior = 0
    after = 0
    for i in range(number_spin_orbitals):
        if i == idx:
            continue
        elif i < idx:
            prior += 1
        else:
            after += 1
    if number_spin_orbitals >= use_csr:
        if dagger:
            return kronecker_product_cached(prior, after, "a_dagger", True, True)
        return kronecker_product_cached(prior, after, "a", True, True)
    if dagger:
        return kronecker_product_cached(prior, after, "a_dagger", False, True)
    return kronecker_product_cached(prior, after, "a", False, True)


def epq_matrix(p: int, q: int, num_spin_orbs: int, use_csr: int = 10) -> np.ndarray | ss.csr_matrix:
    """Contruct Epq operator.

    Args:
        p: Orbital index.
        q: Orbital index.
        num_spin_orbs: Number of spin orbitals.
        use_csr: Size when to use sparse matrices.

    Returns:
        Epq operator in matrix form.
    """
    E = lw.matmul(
        a_op_spin_matrix(p * 2, True, num_spin_orbs, use_csr=use_csr),
        a_op_spin_matrix(q * 2, False, num_spin_orbs, use_csr=use_csr),
    )
    E += lw.matmul(
        a_op_spin_matrix(p * 2 + 1, True, num_spin_orbs, use_csr=use_csr),
        a_op_spin_matrix(q * 2 + 1, False, num_spin_orbs, use_csr=use_csr),
    )
    return E
