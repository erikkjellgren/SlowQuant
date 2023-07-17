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
    elif isinstance(A, ss.csr_matrix) or isinstance(A, ss.csc_matrix):
        if not isinstance(B, ss.csr_matrix) and not isinstance(B, ss.csc_matrix):
            raise TypeError(f"A and B are not same matrix type.\nA type: {type(A)}\nB type: {type(B)}")
    if isinstance(A, np.ndarray):
        return np.matmul(A, B)
    elif isinstance(A, ss.csr_matrix) or isinstance(A, ss.csc_matrix):
        return A.dot(B)
    else:
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
    elif isinstance(A, ss.csr_matrix):
        return ss.linalg.expm(A)
    else:
        raise TypeError(f"A got unsupported type: {type(A)}")
