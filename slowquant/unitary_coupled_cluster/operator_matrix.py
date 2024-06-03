import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.qiskit_interface.base import FermionicOperator, a_op
from slowquant.unitary_coupled_cluster.base import build_operator


def Epq_matrix(p: int, q: int, idx2det: dict[int, int], det2idx: dict[int, int], num_orbs: int) -> np.ndarray:
    r"""Construct the singlet one-electron excitation operator.

    .. math::
        \hat{E}_{pq} = \hat{a}^\dagger_{p,\alpha}\hat{a}_{q,\alpha} + \hat{a}^\dagger_{p,\beta}\hat{a}_{q,\beta}

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.

    Returns:
        Singlet one-electron excitation operator.
    """
    E = FermionicOperator(a_op(p, "alpha", dagger=True), 1) * FermionicOperator(
        a_op(q, "alpha", dagger=False), 1
    )
    E += FermionicOperator(a_op(p, "beta", dagger=True), 1) * FermionicOperator(
        a_op(q, "beta", dagger=False), 1
    )
    return build_operator(E, idx2det, det2idx, num_orbs)
