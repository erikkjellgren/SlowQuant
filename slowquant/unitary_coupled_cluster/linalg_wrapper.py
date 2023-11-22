import numpy as np
import scipy
import scipy.sparse as ss


def matmul(
    A: np.ndarray | ss.csr_matrix | ss.csc_matrix, B: np.ndarray | ss.csr_matrix | ss.csc_matrix
) -> np.ndarray | ss.csr_matrix | ss.csc_matrix:
    """Matrix multiplication that is agnostic to dense (numpy) and sparse (scipy) matrices.

    Args:
        A: Matrix.
        B: Matrix.

    Returns:
        Matrix product.
    """
    if isinstance(A, np.ndarray):
        if not isinstance(B, np.ndarray):
            raise TypeError(f"A and B are not same matrix type.\nA type: {type(A)}\nB type: {type(B)}")
    elif isinstance(A, (ss.csr_matrix, ss.csc_matrix)):
        if not isinstance(B, ss.csr_matrix) and not isinstance(B, ss.csc_matrix):
            raise TypeError(f"A and B are not same matrix type.\nA type: {type(A)}\nB type: {type(B)}")
    if isinstance(A, np.ndarray):
        return np.matmul(A, B)
    if isinstance(A, (ss.csr_matrix, ss.csc_matrix)):
        return A.dot(B)
    raise TypeError(f"A got unsupported type: {type(A)}")


def expm(A: np.ndarray | ss.csr_matrix | ss.csc_matrix) -> np.ndarray | ss.csr_matrix | ss.csc_matrix:
    """Matrix exponential that is agnostic to dense (numpy) and sparse (scipy) matrices.

    Args:
        A: Matrix.

    Returns:
        Matrix exponential of A.
    """
    if isinstance(A, np.ndarray):
        return scipy.linalg.expm(A)
    if isinstance(A, ss.csr_matrix):
        return ss.linalg.expm(A)
    raise TypeError(f"A got unsupported type: {type(A)}")


def zeros_like(A: np.ndarray | ss.csr_matrix | ss.csc_matrix) -> np.ndarray | ss.csr_matrix | ss.csc_matrix:
    """Create zero array of same shape as input array.

    Args:
        A: Array to take shape from.

    Returns:
        Zero array.
    """
    if isinstance(A, np.ndarray):
        return np.zeros_like(A)
    if isinstance(A, (ss.csr_matrix, ss.csc_matrix)):
        return ss.csr_matrix(A.shape)
    raise TypeError(f"A got unsupported type: {type(A)}")


def outer(
    A: np.ndarray | ss.csr_matrix | ss.csc_matrix, B: np.ndarray | ss.csr_matrix | ss.csc_matrix
) -> np.ndarray | ss.csr_matrix | ss.csc_matrix:
    """Outerp product between two vectors.

    Args:
        A: Vector.
        B: Vector.

    Returns:
        Outer product matrix.
    """
    if isinstance(A, np.ndarray):
        return np.outer(A, B)
    if isinstance(A, (ss.csr_matrix, ss.csc_matrix)):
        if A.transpose().get_shape() != B.get_shape():
            raise ValueError(
                "Shape mismatch between A and B, got A: {A.get_shape()}, and, B: {B.get_shape()}"
            )
        return A.dot(B)
    raise TypeError(f"A got unsupported type: {type(A)}")
