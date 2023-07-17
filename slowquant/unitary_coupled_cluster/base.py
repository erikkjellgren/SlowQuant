import functools
from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss


@functools.cache
def kronecker_product_cached(
    num_prior: int, num_after: int, pauli_mat_symbol: str, is_csr: bool
) -> np.ndarray | ss.csr_matrix:
    r"""Get operator in matrix form.
    The operator is returned in the form:

    .. math::
        I x I x .. o .. x I x I

    Args:
       num_prior: Number of left-hand side identity matrices.
       num_after: Number of right-hand side identity matrices.
       is_csr: If the resulting matrix representation should be a sparse matrix.

    Returns:
       Matrix representation of an operator.
    """
    mat = pauli_to_mat(pauli_mat_symbol)
    if is_csr:
        I1 = ss.identity(int(2**num_prior))
        I2 = ss.identity(int(2**num_after))
        mat = ss.csr_matrix(mat)
        return ss.csr_matrix(ss.kron(I1, ss.kron(mat, I2)))
    else:
        I1 = np.identity(int(2**num_prior))
        I2 = np.identity(int(2**num_after))
        return np.kron(I1, np.kron(mat, I2))


def kronecker_product(A: Sequence[np.ndarray]) -> np.ndarray:
    r"""Get Kronecker product of a list of matricies
    Does:

    .. math::
        P x P x P ...

    Args:
       A: List of matrices.

    Returns:
       Kronecker product of matrices.
    """
    if len(A) < 2:
        return A
    total = np.kron(A[0], A[1])
    for operator in A[2:]:
        total = np.kron(total, operator)
    return total


@functools.cache
def pauli_to_mat(pauli: str) -> np.ndarray:
    """Convert Pauli matrix symbol to matrix representation.

    Args:
        pauli: Pauli matrix symbol.

    Returns:
        Pauli matrix.
    """
    if pauli == "I":
        return np.array([[1, 0], [0, 1]], dtype=float)
    elif pauli == "Z":
        return np.array([[1, 0], [0, -1]], dtype=float)
    elif pauli == "X":
        return np.array([[0, 1], [1, 0]], dtype=float)
    elif pauli == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
