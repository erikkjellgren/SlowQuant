import numpy as np
import scipy


def _pair(V: np.ndarray) -> np.ndarray:
    """[x; y] -> [y*; x*]. E2 pair(v) = pair(E2 v), S pair(v) = -pair(S v)."""
    n = V.shape[0] // 2
    return np.vstack((V[n:].conj(), V[:n].conj()))


class _Subspace:
    """Orthonormal trial space closed under pairing, with stored E2 @ V and S @ V."""

    def __init__(self, sigma, max_batch: int):
        self.sigma, self.max_batch = sigma, max_batch
        self.V = self.EV = self.SV = None

    def add(self, R: np.ndarray) -> int:
        for _ in range(2):
            R = R / np.maximum(np.linalg.norm(R, axis=0), 1e-300)
            if self.V is not None:
                R = R - self.V @ (self.V.conj().T @ R)
        B = _orth(R, 1e-8)
        if B.shape[1] == 0:
            return 0
        EB, SB = [], []
        for i in range(0, B.shape[1], self.max_batch):
            e, s = self.sigma(B[:, i : i + self.max_batch])
            EB.append(e)
            SB.append(s)
        EB, SB = np.hstack(EB), np.hstack(SB)
        P, EP, SP = _pair(B), _pair(EB), -_pair(SB)
        c = B.conj().T @ P
        P, EP, SP = P - B @ c, EP - EB @ c, SP - SB @ c
        _, s, vh = np.linalg.svd(P, full_matrices=False)
        M = vh[s > 1e-5].conj().T / s[s > 1e-5]
        new = [np.hstack((B, P @ M)), np.hstack((EB, EP @ M)), np.hstack((SB, SP @ M))]
        if self.V is None:
            self.V, self.EV, self.SV = new
        else:
            self.V, self.EV, self.SV = (np.hstack((a, b)) for a, b in zip((self.V, self.EV, self.SV), new))
        return new[0].shape[1]


def _orth(R: np.ndarray, thresh: float) -> np.ndarray:
    u, s, _ = np.linalg.svd(R, full_matrices=False)
    return u[:, s > thresh]


def _precondition(R, dA, dS, omega):
    den = np.concatenate((dA, dA))[:, None] - np.concatenate((dS, -dS))[:, None] * np.atleast_1d(omega)[None, :]
    den = np.where(np.abs(den) < 1e-8, 1e-8, den)
    return R / den


def davidson_eig(sigma, dA, dS, n_roots, tol=1e-6, max_iter=100, max_batch=16, max_space=400, shift=0.0):
    """Lowest positive roots of E2 v = omega S v (paired Davidson, matrix-free).

    E2 is indefinite (electron-positron rotations), so Rayleigh-Ritz gives spurious interior Ritz values.
    Harmonic Ritz extraction around `shift` is used instead: with (E2 - shift S) V = Q R,
    solve R u = nu Q^H S V u, and take omega as the Rayleigh quotient of X = V u.
    """
    ratio = np.where((dS > 1e-8) & (dA > shift), dA / np.where(dS > 1e-8, dS, 1.0), np.inf)
    start = np.argsort(ratio)[:n_roots]
    R = np.zeros((2 * len(dA), n_roots), dtype=complex)
    R[start, np.arange(n_roots)] = 1.0
    space = _Subspace(sigma, max_batch)
    for it in range(max_iter):
        if space.add(R) == 0:
            if it == 0:
                raise RuntimeError("Davidson (eigenvalues): no new trial vectors")
            print(f"WARNING: Davidson stopped (no new trial vectors), max residual {res.max():.3e}")
            return w, X
        Qw, Rw = np.linalg.qr(space.EV - shift * space.SV)
        _, u = scipy.linalg.eig(Rw, Qw.conj().T @ space.SV)
        X, EX, SX = space.V @ u, space.EV @ u, space.SV @ u
        norm = np.einsum("ij,ij->j", X.conj(), SX).real
        w = np.einsum("ij,ij->j", X.conj(), EX).real / np.where(np.abs(norm) > 1e-300, norm, 1.0)
        ok = np.all(np.isfinite(u), axis=0) & (w > 1e-8) & (norm > 1e-6 * np.linalg.norm(X, axis=0) ** 2)
        idx = np.where(ok)[0][np.argsort(w[ok])][:n_roots]
        w, sc = w[idx], 1 / np.sqrt(norm[idx])
        X, R = X[:, idx] * sc, (EX[:, idx] - SX[:, idx] * w) * sc
        res = np.linalg.norm(R, axis=0)
        print(f"Davidson {it + 1:3d} | subspace {space.V.shape[1]:4d} | max residual {res.max():.3e}")
        if res.max() < tol and len(w) == n_roots:
            return w, X
        R = _precondition(R[:, res >= tol], dA, dS, w[res >= tol])
        if space.V.shape[1] + 2 * R.shape[1] > max_space:
            space = _Subspace(sigma, max_batch)
            R = np.hstack((X, R))
    raise RuntimeError("Davidson (eigenvalues) did not converge")


def davidson_solve(sigma, dA, dS, V, omega=0.0, tol=1e-6, max_iter=100, max_batch=16, return_ex=False):
    """Solve (E2 - omega S) X = V (matrix-free subspace solver).

    The subspace solution minimizes the residual weighted with the inverse diagonal, so that the small-eigenvalue
    (electron-electron, CI) components are as well converged as the large electron-positron ones.
    Convergence: max_j ||W r_j|| / ||W v_j|| < tol with W = 1/|diag(E2 - omega S)|.
    return_ex: also return (E2 - omega S) X from the subspace.
    """
    V = V.reshape(V.shape[0], -1)
    w = 1 / np.maximum(np.abs(np.concatenate((dA, dA)) - omega * np.concatenate((dS, -dS))), 1e-2)
    vnorm = np.maximum(np.linalg.norm(w[:, None] * V, axis=0), 1e-300)
    R = _precondition(V, dA, dS, np.full(V.shape[1], omega))
    space = _Subspace(sigma, max_batch)
    for it in range(max_iter):
        space.add(R)
        Wm = space.EV - omega * space.SV
        u = np.linalg.lstsq(w[:, None] * Wm, w[:, None] * V, rcond=None)[0]
        X = space.V @ u
        Rfull = Wm @ u - V
        res = np.linalg.norm(w[:, None] * Rfull, axis=0) / vnorm
        print(f"Response {it + 1:3d} | subspace {space.V.shape[1]:4d} | max rel. residual {res.max():.3e}")
        if res.max() < tol:
            return (X, space.EV @ u - omega * (space.SV @ u)) if return_ex else X
        R = _precondition(Rfull[:, res >= tol], dA, dS, np.full(int((res >= tol).sum()), omega))
    raise RuntimeError("Davidson (linear response) did not converge")
