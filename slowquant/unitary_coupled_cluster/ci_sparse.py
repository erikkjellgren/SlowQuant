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


def operator_matrix(ci_info, op) -> sp.csr_matrix:
    """Sparse matrix of a number-conserving FermionicOperator with active-space (unfolded) spin-orbital indices."""
    n = 2 * ci_info.num_active_orbs
    dets = np.asarray(ci_info.idx2det, dtype=np.int64)
    order = np.argsort(dets)
    sdets = dets[order]
    full = (1 << n) - 1
    rows, cols, vals = [], [], []
    for key, c in op.operators.items():
        d = dets.copy()
        ok = np.ones(len(d), dtype=bool)
        sign = np.ones(len(d))
        for orb, dag in reversed(key):
            b = np.int64(1 << (n - 1 - orb))
            occ = (d & b) != 0
            ok &= ~occ if dag else occ
            sign *= 1.0 - 2.0 * (np.bitwise_count(d & np.int64(full ^ ((1 << (n - orb)) - 1))) % 2)
            d = d ^ b
        j = np.nonzero(ok)[0]
        pos = np.searchsorted(sdets, d[j])
        keep = (pos < len(sdets)) & (sdets[np.minimum(pos, len(sdets) - 1)] == d[j])
        rows.append(order[pos[keep]])
        cols.append(j[keep])
        vals.append(c * sign[j[keep]])
    nd = len(dets)
    if not rows:
        return sp.csr_matrix((nd, nd), dtype=complex)
    return sp.csr_matrix(
        (np.concatenate(vals).astype(complex), (np.concatenate(rows), np.concatenate(cols))), shape=(nd, nd)
    )


class UPSEngine:
    r"""Fast unitary product state: U_k = exp(theta_k T_k - theta_k^* T_k^\dagger) with sparse T_k.

    U_k v = v + sin(r)/r A v + (1 - cos r)/r^2 A^2 v, r = |theta_k|, A^2 = -r^2 (T T^\dagger + T^\dagger T) (diagonal)
    when T_k^2 = 0. Same conventions as generalized_construct_ups_state_test_erik / generalized_get_grad_action.
    """

    def __init__(self, ci_info, ups_struct, op_builder):
        self.ci_info = ci_info
        self.ups = ups_struct
        self.build = op_builder
        self.T, self.Td, self.P = [], [], []
        self.update()

    def update(self):
        for k in range(len(self.T), len(self.ups.excitation_indices)):
            T = operator_matrix(self.ci_info, self.build(self.ups.excitation_operator_type[k], self.ups.excitation_indices[k]))
            Td = T.conj().T.tocsr()
            P = None
            if (T @ T).count_nonzero() == 0:
                P = np.real((T @ Td + Td @ T).diagonal())
            self.T.append(T)
            self.Td.append(Td)
            self.P.append(P)

    def _A(self, k, theta, v):
        return theta * (self.T[k] @ v) - np.conj(theta) * (self.Td[k] @ v)

    def _A2(self, k, theta, v, Av=None):
        if self.P[k] is not None:
            return -abs(theta) ** 2 * (self.P[k][:, None] if v.ndim == 2 else self.P[k]) * v
        return self._A(k, theta, self._A(k, theta, v) if Av is None else Av)

    def apply(self, k, theta, v, dagger=False):
        r = abs(theta)
        if r < 1e-28:
            return v.copy()
        th = -theta if dagger else theta
        Av = self._A(k, th, v)
        return v + np.sin(r) / r * Av + (1 - np.cos(r)) / r**2 * self._A2(k, th, v, Av)

    def state(self, v, thetas, dagger=False):
        self.update()
        out = np.array(v, dtype=complex)
        ks = range(len(thetas))
        for k in (reversed(ks) if dagger else ks):
            out = self.apply(k, thetas[k], out, dagger)
        return out

    def grad_action(self, k, theta, v):
        """(dU_k/dRe(theta) v, dU_k/dIm(theta) v)."""
        VR, VI = self._V(k, True, v), self._V(k, False, v)
        Av = theta.real * VR + theta.imag * VI
        A2v = self._A2(k, theta, v, Av)
        r = abs(theta)
        if r < 1e-8:
            f, g, c1, c2 = 1 - r**2 / 6, 0.5 - r**2 / 24, -1 / 3 + r**2 / 30, -1 / 12 + r**2 / 180
        else:
            f, g = np.sin(r) / r, (1 - np.cos(r)) / r**2
            c1, c2 = (r * np.cos(r) - np.sin(r)) / r**3, (r * np.sin(r) - 2 + 2 * np.cos(r)) / r**4
        out = []
        for real, V, x in ((True, VR, theta.real), (False, VI, theta.imag)):
            VAv = self._V(k, real, Av)
            out.append(f * V + g * (VAv + self._A(k, theta, V)) + x * c1 * Av + x * c2 * A2v)
        return out[0], out[1]

    def _V(self, k, real, w):
        Tw, Tdw = self.T[k] @ w, self.Td[k] @ w
        return Tw - Tdw if real else 1j * (Tw + Tdw)


def ups_theta_gradient(engine: UPSEngine, thetas: np.ndarray, ref: np.ndarray, hc: np.ndarray) -> np.ndarray:
    """dE/d(Re theta, Im theta) for |0> = U(theta)|ref> from hc = H|0> (one unitary peeled off per parameter,
    arXiv:2303.10825, Eq. 20)."""
    nt = len(thetas)
    g = np.zeros(2 * nt)
    bra = engine.state(hc, thetas, dagger=True)
    ket = np.array(ref, dtype=complex)
    for i in range(nt):
        bra = engine.apply(i, thetas[i], bra)
        dR, dI = engine.grad_action(i, thetas[i], ket)
        g[i] = 2 * np.vdot(bra, dR).real
        g[i + nt] = 2 * np.vdot(bra, dI).real
        ket = engine.apply(i, thetas[i], ket)
    return g
