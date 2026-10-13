import numpy as np
import scipy.sparse as sp


class ActiveCI:
    r"""Sparse E_tu = a^\dagger_t a_u in the active determinant space of a CI_Info (generalized spin orbitals).

    Same determinant/bit and phase conventions as generalized_propagate_state.
    """

    def __init__(self, ci_info):
        dets = np.asarray(ci_info.idx2det, dtype=np.int64)
        n = self.n = 2 * ci_info.num_active_orbs
        self.ndet = nd = len(dets)
        order = np.argsort(dets)
        sdets = dets[order]
        full = (1 << n) - 1
        self.E = [[None] * n for _ in range(n)]
        coo = []
        for t in range(n):
            bt = np.int64(1 << (n - 1 - t))
            below_t = np.int64(full ^ ((1 << (n - t)) - 1))
            for u in range(n):
                bu = np.int64(1 << (n - 1 - u))
                below_u = np.int64(full ^ ((1 << (n - u)) - 1))
                d1 = dets ^ bu
                ok = ((dets & bu) != 0) & ((d1 & bt) == 0)
                d2 = d1[ok] | bt
                cnt = np.bitwise_count(d1[ok] & below_u) + np.bitwise_count(d2 & below_t)
                rows = order[np.searchsorted(sdets, d2)]
                cols = np.nonzero(ok)[0]
                vals = 1.0 - 2.0 * (cnt % 2)
                self.E[t][u] = sp.csr_matrix((vals.astype(complex), (rows, cols)), shape=(nd, nd))
                coo.append((rows, cols, vals, np.full(len(rows), t * n + u)))
        self._rows, self._cols, self._vals, self._pair = (np.concatenate(x) for x in zip(*coo))

    def hamiltonian(self, e: float, h: np.ndarray, g: np.ndarray) -> sp.csr_matrix:
        r"""e + \sum h_tu E_tu + 1/2 \sum g_tuvw (E_tu E_vw - \delta_uv E_tw) as sparse matrix."""
        n, nd = self.n, self.ndet
        hp = h - 0.5 * np.einsum("tuuw->tw", g)
        H = sp.csr_matrix((self._vals * hp.reshape(-1)[self._pair], (self._rows, self._cols)), shape=(nd, nd))
        H = H + e * sp.identity(nd, dtype=complex, format="csr")
        gf = 0.5 * g.reshape(n * n, n * n)
        for tu in range(n * n):
            M = sp.csr_matrix((self._vals * gf[tu][self._pair], (self._rows, self._cols)), shape=(nd, nd))
            H = H + self.E[tu // n][tu % n] @ M
        return H.tocsr()

    def phi(self, c: np.ndarray) -> np.ndarray:
        """E_tu |c>, shape (n*n, ndet) for one vector or (n*n, ndet, m) for c of shape (ndet, m)."""
        return np.array([self.E[t][u] @ c for t in range(self.n) for u in range(self.n)])

    def apply(self, ops, c: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
        r"""(\sum f_tu E_tu + 1/2 \sum g_tuvw (E_tu E_vw - \delta_uv E_tw)) |c> for operators [(f, g), ...], shape (m, ndet)."""
        n = self.n
        P = np.ascontiguousarray(self.phi(c) if phi is None else phi)
        F = np.array([(f - 0.5 * np.einsum("tuuw->tw", g)).reshape(-1) for f, g in ops])
        Gm = np.array([0.5 * g.reshape(n * n, n * n) for _, g in ops])
        out = F @ P
        Z = np.ascontiguousarray((Gm @ P).transpose(1, 2, 0))
        for tu in range(n * n):
            out += (self.E[tu // n][tu % n] @ Z[tu]).T
        return out

    def transition_rdms(self, L: np.ndarray, c: np.ndarray, phi: np.ndarray | None = None):
        r"""(<L|c>, <L|E_tu|c>, <L|e_tuvw|c>) for bra vectors L (m, ndet), e_tuvw = E_tu E_vw - \delta_uv E_tw."""
        n = self.n
        L = np.atleast_2d(L)
        m = len(L)
        P = self.phi(c) if phi is None else phi
        PT = np.ascontiguousarray(P.T)
        Lc = L.conj()
        s = Lc @ c
        d = (Lc @ PT).reshape(m, n, n)
        Lt = np.ascontiguousarray(L.T)
        Psi = np.empty((m, n, n, self.ndet), dtype=complex)
        for t in range(n):
            for u in range(n):
                Psi[:, t, u] = (self.E[u][t] @ Lt).T
        G = (Psi.reshape(m * n * n, -1).conj() @ PT).reshape(m, n, n, n, n)
        G -= np.einsum("uv,mtw->mtuvw", np.eye(n), d)
        return s, d, G


def ladder(src, dst, t: int, create: bool) -> sp.csr_matrix:
    r"""a^\dagger_t (create) or a_t between the active determinant spaces of CI_Infos src -> dst (same conventions as ActiveCI)."""
    n = 2 * src.num_active_orbs
    dets = np.asarray(src.idx2det, dtype=np.int64)
    ddets = np.asarray(dst.idx2det, dtype=np.int64)
    order = np.argsort(ddets)
    b = np.int64(1 << (n - 1 - t))
    below = np.int64(((1 << n) - 1) ^ ((1 << (n - t)) - 1))
    ok = (dets & b) == 0 if create else (dets & b) != 0
    cols = np.nonzero(ok)[0]
    new = dets[ok] ^ b
    rows = order[np.searchsorted(ddets[order], new)]
    vals = 1.0 - 2.0 * (np.bitwise_count(dets[ok] & below) % 2)
    return sp.csr_matrix((vals.astype(complex), (rows, cols)), shape=(len(ddets), len(dets)))
