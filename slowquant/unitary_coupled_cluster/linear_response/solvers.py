import time
import scipy
import numpy as np
import numba as nb

from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Callable

from slowquant.unitary_coupled_cluster.density_matrix import RDM1, RDM2

class Solvers(ABC):

    _start: float
    _iteration: int

    def __init__(self) -> None:
        ...

    @abstractmethod
    def solve(
        self,
        right_transform: Callable[..., Any],
        preconditioner: tuple[np.ndarray, ...],
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
    """Davidson solver for the standard eigenvalue problem.

    J. Comput. Phys. 17, 87-94 (1975).
    """

    _trial: np.typing.NDArray[np.complexfloating]
    """Subspace trial vectors"""

    def solve(
        self,
        right_transform: Callable[[np.ndarray], Any],
        preconditioner: tuple[np.ndarray, ...],
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

        Returns:
            omega: Eigenvalues.
            X: Eigenvectors.
        """

        self._start = time.time()
        self._iteration = 0
        self._trial = np.array(())
        self._sigma_plus = np.array(())
        self._sigma_minus = np.array(())
        self._tau_minus = np.array(())

        self._print_citation()

        if not is_silent:
            print(f" Iteration | Time [s] | Max. residual norm | Subspace size | Roots ...")

        if max_reduced_space is None:
            max_reduced_space = 8 * n_roots

        if isinstance(preconditioner, tuple):
            diag = preconditioner[0]
        else:
            raise ValueError("Preconditioner must be a tuple of numpy arrays.")
        dim = diag.shape[-1]

        if n_roots > dim:
            raise ValueError(f"Number of roots to compute (n_roots={n_roots}) cannot exceed the dimension of the problem (dim={dim}).")

        start_guess = np.zeros((dim, n_roots))
        start_guess[np.argsort(diag)[:n_roots], np.arange(n_roots)] = 1.0
        trial = self._orthonormalize(start_guess)
        for _ in range(max_iteration):
            self._iteration += 1

            right_transformed_vectors = right_transform(trial)
            self._add_iteration_data(trial, right_transformed_vectors)
            omega, X, R = self._compute_residual_vectors(n_roots)
            converged, res_norms = self._check_convergence(R, tolerance)

            if converged:
                if not is_silent:
                    self._print_iteration_info(res_norms, omega, tolerance)
                return omega, X

            trial = self._update_trial_vectors(omega, R, preconditioner)
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
                self._reset_reduced_space(right_transform)

            if not is_silent:
                self._print_iteration_info(res_norms, omega, tolerance)

        raise RuntimeError(f"Davidson did not converge.")

    @abstractmethod
    def _print_citation(self) -> None:
        """Print the citation for the Davidson method."""
        print()
        print("Davidson solver for eigenvalue problems (J. Comput. Phys. 17, 87-94 (1975))")

    @staticmethod
    def _orthonormalize(trial: np.ndarray) -> np.ndarray:
        """Orthonormalize columns of trial using QR and return Q with collapsed tiny columns removed."""
        Q, R = np.linalg.qr(trial)
        # remove near-zero columns (if any)
        diagR = np.abs(np.diag(R))
        keep = diagR > 1e-12
        new_trial = Q[:, keep]
        new_trial /= np.linalg.norm(new_trial, axis=0)
        return new_trial

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
        """Remove linearly dependent trial vectors."""
        tol = 1e-12

        nvecs = trial.shape[1]
        Sb = np.zeros((nvecs, nvecs))
        for i in range(nvecs):
            Sb[i, i] = np.dot(trial[:, i], trial[:, i])
            for j in range(i):
                Sb[i, j] = np.dot(trial[:, i], trial[:, j])
                Sb[j, i] = Sb[i, j]

        l, T = np.linalg.eigh(Sb)
        b_norm = np.sqrt(Sb.diagonal())
        keep = l > b_norm * tol

        new_vector = trial @ T[:, keep]
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

    @abstractmethod
    def _add_iteration_data(self, trial: np.ndarray, right_transformed_vectors: tuple[np.ndarray, ...]) -> None:
        """Add trial and right transformed matrices for the current iteration to the arrays."""

    @abstractmethod
    def _compute_residual_vectors(self, n_roots: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        """Compute residual vectors for the given Ritz vectors and values."""

    @staticmethod
    @abstractmethod
    def _check_convergence(R: tuple[np.ndarray, ...], tolerance: float) -> tuple[bool, np.ndarray]:
        """Check if the maximum residual norm is below the tolerance."""

    @staticmethod
    @abstractmethod
    def _update_trial_vectors(omega: np.ndarray, R: tuple[np.ndarray, ...], preconditioner: tuple[np.ndarray, ...]) -> np.ndarray:
        """Update trial vectors using the residuals and the diagonal preconditioner."""

    @abstractmethod
    def _reset_reduced_space(self, right_transform: Callable[[np.ndarray], Any]) -> None:
        """Reset the reduced space by keeping only the current Ritz vectors."""

class PairedDavidson(Davidson):
    """Davidson solver for the paired eigenvalue problem arising in the CASSCF linear response equations.

    J. Chem. Phys. 150, 174121 (2019)
    """

    _sigma_plus: np.typing.NDArray[np.complexfloating]
    """Subspace matrix A @ b + B @ b*"""
    _sigma_minus: np.typing.NDArray[np.complexfloating]
    """Subspace matrix A @ b - B @ b*"""
    _tau_minus: np.typing.NDArray[np.complexfloating]
    """Subspace matrix Sigma @ b"""

    def _print_citation(self) -> None:
        """Print the citation for the Davidson method."""
        super()._print_citation()
        print("Davidson solver for paired eigenvalue problems (J. Chem. Phys. 150, 174121 (2019))")

    def _add_iteration_data(self, trial: np.ndarray, right_transformed_vectors: tuple[np.ndarray, ...]) -> None:
        """Add trial and right transformed matrices for the current iteration to the arrays."""
        sigma_plus, sigma_minus, tau_minus = right_transformed_vectors
        if self._iteration == 1:
            self._trial = trial.copy()
            self._sigma_plus = sigma_plus.copy()
            self._sigma_minus = sigma_minus.copy()
            self._tau_minus = tau_minus.copy()
        else:
            self._trial = np.hstack((self._trial, trial))
            self._sigma_plus = np.hstack((self._sigma_plus, sigma_plus))
            self._sigma_minus = np.hstack((self._sigma_minus, sigma_minus))
            self._tau_minus = np.hstack((self._tau_minus, tau_minus))

    def _compute_residual_vectors(self, n_roots: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
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

        E_plus = 2 * self._trial.conj().T @ self._sigma_plus
        E_minus = 2 * self._trial.conj().T @ self._sigma_minus
        # S_plus = S_minus.T
        S_minus = 2 * self._trial.conj().T @ self._tau_minus

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

        # tau_plus = tau_minus
        R_plus = self._sigma_plus @ x_plus - self._tau_minus @ x_minus * omega
        R_minus = self._sigma_minus @ x_minus - self._tau_minus @ x_plus * omega

        return omega, X, (R_plus, R_minus)

    @staticmethod
    def _check_convergence(R: tuple[np.ndarray, ...], tolerance: float) -> tuple[bool, np.ndarray]:
        """Check if the maximum residual norm is below the tolerance."""
        R_plus, _ = R
        res_norms = np.linalg.norm(R_plus, axis=0)
        return all(res_norms <= tolerance), res_norms

    @staticmethod
    def _update_trial_vectors(omega: np.ndarray, R: tuple[np.ndarray, ...], preconditioner: tuple[np.ndarray, ...]) -> np.ndarray:
        """Update trial vectors using the residuals and the diagonal preconditioner."""
        diagonal_A, diagonal_Sigma = preconditioner
        R_plus, R_minus = R

        minus_contribution = diagonal_A.reshape(-1, 1) - diagonal_Sigma.reshape(-1, 1) @ omega.reshape(1, -1)
        plus_contribution = diagonal_A.reshape(-1, 1) + diagonal_Sigma.reshape(-1, 1) @ omega.reshape(1, -1)
        denominator = -1
        # Check if any of the contributions are close to zero to avoid division by zero, if so skip the division for that contribution
        if not np.any(np.isclose(minus_contribution, 0)):
            denominator /= minus_contribution
        if not np.any(np.isclose(plus_contribution, 0)):
            denominator /= plus_contribution
        new_trial = denominator * (
            diagonal_A.reshape(-1, 1) * R_plus + diagonal_Sigma.reshape(-1, 1) * R_minus * omega.reshape(1, -1)
            + diagonal_A.reshape(-1, 1) * R_minus + diagonal_Sigma.reshape(-1, 1) * R_plus * omega.reshape(1, -1)
        )
        return new_trial

    def _reset_reduced_space(self, right_transform: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        """Reset the reduced space by keeping only the current Ritz vectors."""
        self._trial = self._orthonormalize(self._trial)
        self._sigma_plus, self._sigma_minus, self._tau_minus = right_transform(self._trial)

def one_index_transform(K: np.ndarray, h_mo: np.ndarray, g_mo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""One index transformation of the Hamiltonian.

    .. math::
        \sum_\mu [\hat{H}, \kappa_\mu \hat{q}_\mu^\dagger + h \kappa_\mu^* \hat{q}_\mu] = \hat{\tilde{H}}_h, h \in {+1, -1}

    Args:
        K: Orbital rotation parameters (h=\pm 1).
        num_orbs: Number of spatial orbitals.
        h_mo: One-electron integrals in molecular orbital basis.
        g_mo: Two-electron integrals in molecular orbital basis.

    Returns:
        One index transformed h_{h} (h=\pm1)
        One index transformed g_{h} (h=\pm1)
    """
    inv_sqrt_2 = 1 / np.sqrt(2)
    h = np.einsum("pt,tq->pq", h_mo, K) - np.einsum("tq,pt->pq", h_mo, K)

    g = (
        np.einsum("ptrs,tq->pqrs", g_mo, K)
        - np.einsum("tqrs,pt->pqrs", g_mo, K)
    )
    g += np.einsum("pqrs->rspq", g)

    h *= inv_sqrt_2
    g *= inv_sqrt_2

    return h, g

@nb.jit(nopython=True)
def get_orbital_rotation_gradient(
    h_int: np.ndarray,
    g_int: np.ndarray,
    q_idx: list[tuple[int, int]] | np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
) -> np.ndarray:
    r"""Calculate the orbital gradient.

    .. math::
        g_{tu}^{\hat{q}} = \left<0\left|\left[\hat{q}_{tu},\hat{H}\right]\right|0\right>

    Args:
        h_int: One-electron integrals in MO in Hamiltonian.
        g_int: Two-electron integrals in MO in Hamiltonian.
        q_idx: Orbital rotation parameter indices in spatial basis.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.
        rdm1: Active part of 1-RDM.
        rdm2: Active part of 2-RDM.

    Returns:
        Orbital gradient.
    """
    inv_sqrt_2 = 1 / np.sqrt(2)
    gradient = np.zeros(len(q_idx))
    for idx, (t, u) in enumerate(q_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] += h_int[u, p] * RDM1(t, p, num_inactive_orbs, num_active_orbs, rdm1)
            gradient[idx] -= h_int[p, t] * RDM1(p, u, num_inactive_orbs, num_active_orbs, rdm1)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    gradient[idx] += g_int[u, p, q, r] * RDM2(
                        t, p, q, r, num_inactive_orbs, num_active_orbs, rdm1, rdm2
                    )
                    gradient[idx] -= g_int[p, q, r, t] * RDM2(
                        p, q, r, u, num_inactive_orbs, num_active_orbs, rdm1, rdm2
                    )
    return inv_sqrt_2 * gradient

@nb.jit(nopython=True)
def get_orbital_metric_qq_block(
    q_idx: list[tuple[int, int]] | np.ndarray,
    trial: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1: np.ndarray,
) -> np.ndarray:
    r"""Calculate the Sigma matrix orbital-orbital block.

    .. math::
        \Sum_{tu}\Sigma_{pq,tu}^{\hat{q},\hat{q}} = \left<0\left|\left[\hat{q}_{pq}^\dagger,\hat{q}_{tu}\right]\right|0\right>

    Args:
        q_idx: Orbital rotation parameter indices in spatial basis.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.
        rdm1: Active part of 1-RDM.

    Returns:
        Sigma matrix orbital-orbital block.
    """
    sigma = np.zeros(len(q_idx))
    for idx1, (u, t) in enumerate(q_idx):
        for idx2, (p, q) in enumerate(q_idx):
            if p == u:
                sigma[idx1] += RDM1(t, q, num_inactive_orbs, num_active_orbs, rdm1) * trial[idx2]
            if t == q:
                sigma[idx1] -= RDM1(p, u, num_inactive_orbs, num_active_orbs, rdm1) * trial[idx2]
    return - 0.5 * sigma

