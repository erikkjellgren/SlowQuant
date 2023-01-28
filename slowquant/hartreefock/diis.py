import numpy as np


class DIIS:
    def __init__(self, number_bf: int, number_fock_saved_: int = 6) -> None:
        """Initialize DIIS-solver.

        Args:
            number_bf: Number of basis functions.
            number_fock_saved_: Max number of previous Fock matrices saved.
        """
        self.fock_matrices: list[np.ndarray] = []
        self.density_matrices: list[np.ndarray] = []
        self.number_fock_saved = number_fock_saved_
        self.num_bf = number_bf

    def get_extrapolated_fock_matrix(
        self, fock_matrix: np.ndarray, density_matrix: np.ndarray, S: np.ndarray
    ):
        r"""Calculate F'.

        Reference: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2308

        Args:
            fock_matrix: Fock matrix.
            density_matrix: Density matrix.
            S: Overlap integral matrix.

        Returns:
            Extrapolated Fock matrix.
        """
        self.fock_matrices.append(fock_matrix)
        self.density_matrices.append(density_matrix)

        # Cannot extrapolate when only given a single Fock Matrix
        if len(self.fock_matrices) == 1:
            return fock_matrix

        if len(self.fock_matrices) > self.number_fock_saved:
            self.fock_matrices.pop(0)
            self.density_matrices.pop(0)

        # Calculate error matrix
        error_matrices = np.zeros(np.shape(self.fock_matrices))
        for i, (F, D) in enumerate(zip(self.fock_matrices, self.density_matrices)):
            error_matrices[i, :, :] = np.dot(np.dot(F, D), S) - np.dot(np.dot(S, D), F)

        # Build B matrix, and b0 vector
        B = np.zeros((len(self.fock_matrices) + 1, len(self.fock_matrices) + 1))
        B[-1, :] = B[:, -1] = -1
        B[-1, -1] = 0
        b0 = np.zeros(len(self.fock_matrices) + 1)
        b0[-1] = -1
        for i, error_matrix_i in enumerate(error_matrices):
            for j, error_matrix_j in enumerate(error_matrices):
                B[i, j] = np.trace(np.dot(error_matrix_i, error_matrix_j))

        # Solve B, b0 linear equation, and construct Fprime
        coefficients = np.linalg.solve(B, b0)
        F_extrapolated = np.zeros((self.num_bf, self.num_bf))
        for c, F in zip(coefficients[:-1], self.fock_matrices):
            F_extrapolated += c * F

        return F_extrapolated
