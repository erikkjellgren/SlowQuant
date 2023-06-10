import numpy as np
import scipy
import scipy.sparse as ss


def matmul(A: np.ndarray | ss.csr_matrix, B: np.ndarray | ss.csr_matrix) -> np.ndarray | ss.csr_matrix:
    if not isinstance(A, type(B)):
        print("A and B are not same matrix type.")
        print(f"A type: {type(A)}")
        print(f"B type: {type(B)}")
        exit()
    elif isinstance(A, np.ndarray):
        return np.matmul(A, B)
    elif isinstance(A, ss.csr_matrix):
        return A.dot(B)
    else:
        print(f"A got unsupported type: {type(A)}")
        exit()


def expm(A: np.ndarray | ss.csr_matrix) -> np.ndarray | ss.csr_matrix:
    if isinstance(A, np.ndarray):
        return scipy.linalg.expm(A)
    elif isinstance(A, ss.csr_matrix):
        return ss.linalg.expm(A)
    else:
        print(f"A got unsupported type: {type(A)}")
        exit()
