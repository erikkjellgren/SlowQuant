import time
import scipy
import numpy as np

from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

class Solvers(ABC):

    _start: float
    _iteration: int

    def __init__(self) -> None:
        ...

    @abstractmethod
    def solve(
        self,
        right_transform: Callable[..., Any],
        preconditioner: Sequence[np.ndarray],
        max_iteration: int,
        tolerance: float,
        n_roots: int,
        max_reduced_space: int,
        is_silent: bool,
    ) -> Any:
        """
        Solve the problem.

        Args:
            right_transform: Right transformation function.
            preconditioner: Preconditioner matrices.
            max_iteration: Maximum iterations.
            tolerance: Convergence tolerance.
            n_roots: Number of lowest eigenpairs to compute.
            max_reduced_space: Maximum dimension of the reduced space before a restart.
            is_silent: Suppress progress output.
        """

class Davidson(Solvers):

    _trial: np.typing.NDArray[np.complexfloating]
    """Subspace trial vectors"""
    _Ab: np.typing.NDArray[np.complexfloating]
    """Subspace matrix A @ b"""
    _Bb: np.typing.NDArray[np.complexfloating]
    """Subspace matrix B @ b*"""
    _Sb: np.typing.NDArray[np.complexfloating]
    """Subspace matrix Sigma @ b"""
    _Db: np.typing.NDArray[np.complexfloating]
    """Subspace matrix Delta @ b*"""

    def solve(
        self,
        right_transform: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        preconditioner: Sequence[np.ndarray],
        max_iteration: int,
        tolerance: float,
        n_roots: int,
        max_reduced_space: int | None = None,
        is_silent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the problem.

        Args:
            right_transform: Right transformation function.
            preconditioner: Preconditioner matrices (diagonal of A, diagonal of Sigma).
            max_iteration: Maximum iterations.
            tolerance: Convergence tolerance.
            n_roots: Number of lowest eigenpairs to compute.
            max_reduced_space: Maximum dimension of the reduced space before a restart (default 8*n_roots).
            is_silent: Suppress progress output.
        """

        self._start = time.time()
        self._iteration = 0
        self._trial = np.array(())
        self._Ab = np.array(())
        self._Bb = np.array(())
        self._Sb = np.array(())
        self._Db = np.array(())

        if not is_silent:
            print(f" Iteration | Time [s] | Max. residual norm | Subspace size | Roots ...")

        if max_reduced_space is None:
            max_reduced_space = 8 * n_roots

        diagonal_A, diagonal_Sigma = preconditioner
        dim = len(diagonal_A)

        if n_roots > dim:
            raise ValueError(f"Number of roots to compute (n_roots={n_roots}) cannot exceed the dimension of the problem (dim={dim}).")

        start_guess = np.zeros((dim, n_roots))
        start_guess[np.argsort(diagonal_A)[:n_roots], np.arange(n_roots)] = 1.0
        trial = self._orthonormalize(start_guess)

        for _ in range(max_iteration):
            self._iteration += 1

            Ab, Bb, Sb, Db = right_transform(trial)
            self._add_iteration_data(trial, Ab, Bb, Sb, Db)
            omega, X, R_plus, R_minus = self._compute_residual_vectors(n_roots)
            converged, res_norms = self._check_convergence(R_plus, tolerance)

            if converged:
                if not is_silent:
                    self._print_iteration_info(res_norms, omega, tolerance)
                return omega, X

            trial = self._update_trial_vectors(omega, R_plus, R_minus, diagonal_A, diagonal_Sigma)
            trial = self._remove_converged(trial, res_norms, tolerance)
            trial = self._project_trial_vectors(trial)
            trial = self._remove_linear_dependencies(trial)
            trial = self._orthonormalize(trial)
            # In case of no trial vector add a random one
            if trial.size == 0:
                trial = self._random_trial_vector()

            if self._trial.shape[1] + trial.shape[1] > max_reduced_space:
                if not is_silent:
                    print(f"Davidson iter {self._iteration+1:4d}: subspace dimension {self._trial.shape[1]+trial.shape[1]} exceeds max_red_space {max_reduced_space}, restarting with current Ritz vectors")
                self._trial = self._orthonormalize(X[:dim, :] + X[dim:, :])
                self._Ab, self._Bb, self._Sb, self._Db = right_transform(self._trial)

            if not is_silent:
                self._print_iteration_info(res_norms, omega, tolerance)

        raise RuntimeError(f"Davidson did not converge.")

    def _add_iteration_data(self, trial: np.ndarray, Ab: np.ndarray, Bb: np.ndarray, Sb: np.ndarray, Db: np.ndarray) -> None:
        """Add Z, Y, Ab, Bb, Sb, and Db matrices for the current iteration to the arrays."""
        if self._iteration == 1:
            self._trial = trial.copy()
            self._Ab = Ab.copy()
            self._Bb = Bb.copy()
            self._Sb = Sb.copy()
            self._Db = Db.copy()
        else:
            self._trial = np.hstack((self._trial, trial))
            self._Ab = np.hstack((self._Ab, Ab))
            self._Bb = np.hstack((self._Bb, Bb))
            self._Sb = np.hstack((self._Sb, Sb))
            self._Db = np.hstack((self._Db, Db))

    @staticmethod
    def _orthonormalize(trial: np.ndarray) -> np.ndarray:
        """Orthonormalize columns of trial_plus and trial_minus using QR and return Q with collapsed tiny columns removed."""
        Q, R = np.linalg.qr(np.vstack((trial, trial)))
        # remove near-zero columns (if any)
        diagR = np.abs(np.diag(R))
        keep = diagR > 1e-12
        new_trial = Q[:, keep]
        new_trial /= np.linalg.norm(new_trial, axis=0)
        new_trial = new_trial[:trial.shape[0], :]
        return new_trial

    def _compute_residual_vectors(self, n_roots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute residual vectors for the given Ritz vectors and values."""

        def _split_vector(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Split the vector into upper and lower halves."""
            n = vector.shape[0] // 2
            plus, minus = vector[:n, :], vector[n:, :]
            return plus, minus

        def _real_eigvals(
                w: np.typing.NDArray[np.complexfloating],
                v: np.typing.NDArray[np.complexfloating],
            ) -> tuple[np.ndarray, np.ndarray]:
            """Find the real eigenvalues or eigenvalues with small imaginary component."""
            imag_threshold = 1e-3
            abs_imag = abs(w.imag)

            # Determine the smallest imaginary components
            max_imag_tol = max(imag_threshold, np.sort(abs_imag)[min(w.size, n_roots) - 1])
            real_idx = np.where((abs_imag <= max_imag_tol))[0]

            idx = real_idx[w[real_idx].real.argsort()]
            w = w[idx]
            v = v[:, idx]

            degen_idx = np.where(w.imag != 0)[0]
            if degen_idx.size > 0:
                # Take the imaginary part of the "degenerated" eigenvectors as an
                # independent eigenvector then discard the imaginary part of v
                v[:, degen_idx[1::2]] = v[:, degen_idx[1::2]].imag
            return w.real, v.real

        sigma_plus = lambda: self._Ab + self._Bb
        sigma_minus = lambda: self._Ab - self._Bb
        tau_plus = lambda: self._Sb + self._Db
        tau_minus = lambda: self._Sb - self._Db

        E_plus = 2 * self._trial.conj().T @ sigma_plus()
        E_minus = 2 * self._trial.conj().T @ sigma_minus()
        # S_plus = S_minus.T
        S_minus = 2 * self._trial.conj().T @ tau_minus()

        E = np.block([
            [E_plus, np.zeros_like(E_minus)],
            [np.zeros_like(E_plus), E_minus]
        ])
        S = np.block([
            [np.zeros_like(S_minus), S_minus.T],
            [S_minus, np.zeros_like(S_minus)]
        ])

        # Solve the generalized eigenvalue problem E v = omega S v
        eigval, eigvec = scipy.linalg.eig(E, S)
        eigval, eigvec = _real_eigvals(eigval, eigvec)

        # Take positive eigenvalues and sort them
        sorting = eigval > 0
        eigval = eigval[sorting]
        eigvec = eigvec[:, sorting]
        sorting = np.argsort(eigval)
        eigval = eigval[sorting]
        eigvec = eigvec[:, sorting]

        # Extract the lowest n_roots eigenvalues and corresponding eigenvectors
        omega = np.real(eigval[:n_roots])
        x = eigvec[:, :n_roots]

        # Compute Ritz vectors (X) and residuals (R)
        x_plus, x_minus = _split_vector(x)
        norm = np.sqrt(np.abs(np.diag(
            x_minus.T @ S_minus @ x_plus + x_plus.T @ S_minus.T @ x_minus
        )))
        x_plus /= norm
        x_minus /= norm
        X = np.vstack((self._trial @ (x_plus + x_minus), self._trial @ (x_plus - x_minus)))

        R_plus = sigma_plus() @ x_plus - tau_plus() @ x_minus * omega
        R_minus = sigma_minus() @ x_minus - tau_minus() @ x_plus * omega

        return omega, X, R_plus, R_minus

    @staticmethod
    def _check_convergence(R_plus: np.ndarray, tolerance: float) -> tuple[bool, np.ndarray]:
        """Check if the maximum residual norm is below the tolerance."""
        res_norms = np.linalg.norm(R_plus, axis=0)
        return all(res_norms <= tolerance), res_norms

    def _print_iteration_info(self, res_norms_plus: np.ndarray, omega: np.ndarray, tolerance: float) -> None:
        """Print iteration information including the maximum residual norm."""
        roots = "  ".join(
            f"{f'{o:<.4e}' + '*' if r <= tolerance
               else f'{o:<.4e}' + ' '}"
               for o, r in zip(omega, res_norms_plus)
            )
        print(f" {self._iteration:^9} | {time.time() - self._start:^8.2f} | {max(res_norms_plus):^18.4e} | {self._trial.shape[1]:^13} | {roots}")
        self._start = time.time()

    @staticmethod
    def _update_trial_vectors(omega: np.ndarray, R_plus: np.ndarray, R_minus: np.ndarray, diagonal_A: np.ndarray, diagonal_Sigma: np.ndarray) -> np.ndarray:
        """Update trial vectors using the residuals and the diagonal preconditioner."""
        denominator = (
            -1
            / (diagonal_A.reshape(-1, 1) - diagonal_Sigma.reshape(-1, 1) @ omega.reshape(1, -1))
            / (diagonal_A.reshape(-1, 1) + diagonal_Sigma.reshape(-1, 1) @ omega.reshape(1, -1))
        )
        new_trial = denominator * (
            diagonal_A.reshape(-1, 1) * R_plus + diagonal_Sigma.reshape(-1, 1) * R_minus * omega.reshape(1, -1)
            + diagonal_A.reshape(-1, 1) * R_minus + diagonal_Sigma.reshape(-1, 1) * R_plus * omega.reshape(1, -1))
        return new_trial

    @staticmethod
    def _remove_converged(trial: np.ndarray, res_norms: np.ndarray, tol: float) -> np.ndarray:
        """Remove converged trial vectors based on residual norms."""
        keep = res_norms > tol
        if trial.ndim == 1:
            new_trial = trial[:, np.newaxis].copy()
        else:
            new_trial = trial.copy()
        return new_trial[:, keep]

    def _project_trial_vectors(self, trial: np.ndarray) -> np.ndarray:
        """Project trial vectors to be orthogonal to the current subspace."""
        new_trial = trial - self._trial @ (self._trial.conj().T @ trial)
        return new_trial

    def _remove_linear_dependencies(self, trial: np.ndarray) -> np.ndarray:
        """Remove linearly dependent trial vectors by checking the rank."""
        u, s, vh = np.linalg.svd(trial, full_matrices=False)
        tol = 1e-12
        keep = s > tol
        new_vector = (u[:, keep] * s[keep]) @ vh[keep, :]
        return new_vector

    def _random_trial_vector(self) -> np.ndarray:
        """Generate a orthogonal trial vector."""
        def gram_schmidt(vector: np.ndarray) -> np.ndarray:
            """Orthogonalize a vector using the Gram-Schmidt process."""
            w = vector.copy()
            for v in self._trial.T:
                w -= np.dot(w, v) / np.dot(v, v) * v
            return w

        print("No trial vectors left, adding a random one.")
        rng = np.random.default_rng()
        new_trial = rng.random(self._trial.shape[0])
        new_trial = gram_schmidt(new_trial)
        new_trial /= np.linalg.norm(new_trial)
        return new_trial[:, np.newaxis]
