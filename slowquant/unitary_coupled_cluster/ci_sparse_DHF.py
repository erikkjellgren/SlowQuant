import math
import numpy as np
import scipy.sparse as sp

try:
    from numba import get_num_threads, njit, prange
except ImportError:
    njit = None

PAR_MIN = 20000

if njit is not None:

    @njit(cache=True, parallel=True)
    def _rot_par(v, R, C, S, a, b, th, f, c):
        for p in prange(a, b):
            i, j, st = R[p], C[p], S[p] * th
            vi, vj = v[i], v[j]
            v[i] = c * vi + st * f * vj
            v[j] = c * vj - np.conj(st) * f * vi

    @njit(cache=True)
    def _rot(v, R, C, S, a, b, th):
        r = abs(th)
        if r < 1e-28:
            return
        f = np.sin(r) / r
        c = np.cos(r)
        if b - a > PAR_MIN:
            _rot_par(v, R, C, S, a, b, th, f, c)
            return
        for p in range(a, b):
            i, j, st = R[p], C[p], S[p] * th
            vi, vj = v[i], v[j]
            v[i] = c * vi + st * f * vj
            v[j] = c * vj - np.conj(st) * f * vi

    @njit(cache=True, parallel=True)
    def _e_scatter(rows, cols, vals, pair, X, out):
        for p in prange(len(rows)):
            a, r, c, v = pair[p], rows[p], cols[p], vals[p]
            for j in range(X.shape[1]):
                out[a, r, j] += v * X[c, j]

    @njit(cache=True, parallel=True)
    def _e_gather(rowptr, cols, vals, pair, Z, out):
        for r in prange(len(rowptr) - 1):
            for p in range(rowptr[r], rowptr[r + 1]):
                a, c, v = pair[p], cols[p], vals[p]
                for j in range(Z.shape[2]):
                    out[r, j] += v * Z[a, c, j]

    def quad_forms(indptr, indices, data, S, V, off, nd):
        """<v_q|H|v_q> for sparse vectors v_q (support S[off[q]:off[q+1]], values V) with H in CSR form."""
        return _quad_forms(indptr, indices, data, S, V, off, nd, 4 * get_num_threads())

    @njit(cache=True, parallel=True)
    def _quad_forms(indptr, indices, data, S, V, off, nd, nth):
        nq = len(off) - 1
        out = np.empty(nq, dtype=np.complex128)
        nch = max(1, min(nq, nth))
        for ch in prange(nch):
            w = np.zeros(nd, dtype=np.complex128)
            for q in range(ch * nq // nch, (ch + 1) * nq // nch):
                for p in range(off[q], off[q + 1]):
                    w[S[p]] = V[p]
                acc = 0j
                for p in range(off[q], off[q + 1]):
                    i = S[p]
                    hv = 0j
                    for z in range(indptr[i], indptr[i + 1]):
                        hv += data[z] * w[indices[z]]
                    acc += np.conj(V[p]) * hv
                out[q] = acc
                for p in range(off[q], off[q + 1]):
                    w[S[p]] = 0
        return out

    @njit(cache=True, parallel=True)
    def _h_fill(data, pos, nd, O, tu, col, sg, h, Wk, didx, dsgn, coef):
        n = O.shape[1]
        for i in prange(len(col)):
            acc = h[tu[i]]
            for q in range(n):
                acc += O[col[i], q] * Wk[tu[i], q]
            data[pos[nd + i]] = sg[i] * acc
        o = nd + len(col)
        for i in prange(len(didx)):
            data[pos[o + i]] = coef[didx[i]] * dsgn[i]

    @njit(cache=True)
    def _popc(x):
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        return (x * 0x0101010101010101) >> 56

    @njit(cache=True)
    def _string_pairs(dets, sdets, order, n, orbs, dags, lens, coef):
        nop, nd = len(lens), len(dets)
        full = (np.int64(1) << n) - 1
        cnt = np.zeros(nop + 1, dtype=np.int64)
        for pass_ in range(2):
            if pass_ == 1:
                for q in range(nop):
                    cnt[q + 1] += cnt[q]
                R = np.empty(cnt[nop], dtype=np.int64)
                C = np.empty(cnt[nop], dtype=np.int64)
                S = np.empty(cnt[nop], dtype=np.complex128)
            for q in range(nop):
                o = cnt[q]
                for i in range(nd):
                    d = dets[i]
                    sg = 1
                    ok = True
                    for z in range(lens[q] - 1, -1, -1):
                        b = np.int64(1) << (n - 1 - orbs[q, z])
                        occ = (d & b) != 0
                        if occ == dags[q, z]:
                            ok = False
                            break
                        if _popc(d & (full ^ ((np.int64(1) << (n - orbs[q, z])) - 1))) % 2:
                            sg = -sg
                        d = d ^ b
                    if not ok:
                        continue
                    p = np.searchsorted(sdets, d)
                    if p == nd or sdets[p] != d:
                        continue
                    if pass_ == 0:
                        cnt[q + 1] += 1
                    else:
                        R[o] = order[p]
                        C[o] = i
                        S[o] = coef[q] * sg
                        o += 1
        return R, C, S, cnt

    @njit(cache=True, parallel=True)
    def _pattern(dets, sdets, order, n, ne):
        nd = len(dets)
        nv = n - ne
        nsg = ne * nv
        ndb = (ne * (ne - 1) // 2) * (nv * (nv - 1) // 2)
        per = 1 + nsg + ndb
        full = (np.int64(1) << n) - 1
        cols = np.empty(nd * per, dtype=np.int32)
        kind = np.empty(nd * per, dtype=np.int8)
        sgn = np.empty(nd * per, dtype=np.int8)
        tu = np.empty(nd * per, dtype=np.int32)
        for r in prange(nd):
            occ = np.empty(ne, dtype=np.int64)
            emp = np.empty(nv, dtype=np.int64)
            rc = np.empty(per, dtype=np.int64)
            rk = np.empty(per, dtype=np.int8)
            rs = np.empty(per, dtype=np.int8)
            rt = np.empty(per, dtype=np.int32)
            dr = dets[r]
            a = 0
            e = 0
            for x in range(n):
                if dr & (np.int64(1) << (n - 1 - x)):
                    occ[a] = x
                    a += 1
                else:
                    emp[e] = x
                    e += 1
            m = 0
            rc[m], rk[m], rs[m], rt[m] = r, 0, 1, 0
            m += 1
            for ia in range(ne):
                t = occ[ia]
                for ib in range(nv):
                    u = emp[ib]
                    d = dr ^ (np.int64(1) << (n - 1 - t)) ^ (np.int64(1) << (n - 1 - u))
                    ex = _popc(d & (full ^ ((np.int64(1) << (n - u)) - 1)))
                    d1 = d ^ (np.int64(1) << (n - 1 - u))
                    ex += _popc(d1 & (full ^ ((np.int64(1) << (n - t)) - 1)))
                    rc[m], rk[m], rs[m], rt[m] = order[np.searchsorted(sdets, d)], 1, 1 - 2 * (ex % 2), t * n + u
                    m += 1
            for ia in range(ne):
                for ja in range(ia):
                    t, v = occ[ia], occ[ja]
                    for ib in range(nv):
                        for jb in range(ib):
                            u, w = emp[ib], emp[jb]
                            d = dr ^ (np.int64(1) << (n - 1 - t)) ^ (np.int64(1) << (n - 1 - v))
                            d = d ^ (np.int64(1) << (n - 1 - u)) ^ (np.int64(1) << (n - 1 - w))
                            ex = _popc(d & (full ^ ((np.int64(1) << (n - u)) - 1))) + _popc(d & (full ^ ((np.int64(1) << (n - w)) - 1)))
                            d2 = d ^ (np.int64(1) << (n - 1 - u)) ^ (np.int64(1) << (n - 1 - w))
                            ex += _popc(d2 & (full ^ ((np.int64(1) << (n - v)) - 1))) + _popc(d2 & (full ^ ((np.int64(1) << (n - t)) - 1))) + 1
                            rc[m], rk[m], rs[m], rt[m] = order[np.searchsorted(sdets, d)], 2, 1 - 2 * (ex % 2), ((t * n + v) * n + u) * n + w
                            m += 1
            srt = np.argsort(rc)
            for z in range(per):
                q = r * per + z
                cols[q], kind[q], sgn[q], tu[q] = rc[srt[z]], rk[srt[z]], rs[srt[z]], rt[srt[z]]
        return cols, kind, sgn, tu, per

    @njit(cache=True)
    def _ups_state(v, R, C, S, off, ths, dagger):
        n = len(ths)
        for q in range(n):
            k = n - 1 - q if dagger else q
            _rot(v, R, C, S, off[k], off[k + 1], -ths[k] if dagger else ths[k])

    @njit(cache=True, parallel=True)
    def _dots_par(bra, ket, R, C, S, a, b):
        bd, bij, bji = 0j, 0j, 0j
        for p in prange(a, b):
            i, j, s = R[p], C[p], S[p]
            bi, bj = np.conj(bra[i]), np.conj(bra[j])
            bd += bi * ket[i] + bj * ket[j]
            bij += bi * s * ket[j]
            bji += bj * np.conj(s) * ket[i]
        return bd, bij, bji

    @njit(cache=True)
    def _ups_grad(bra, ket, R, C, S, off, ths, g):
        nt = len(ths)
        for k in range(nt):
            th = ths[k]
            _rot(bra, R, C, S, off[k], off[k + 1], th)
            x, y, r = th.real, th.imag, abs(th)
            if r < 1e-8:
                f, c1 = 1 - r * r / 6, -1 / 3 + r * r / 30
            else:
                f, c1 = np.sin(r) / r, (r * np.cos(r) - np.sin(r)) / r**3
            if off[k + 1] - off[k] > PAR_MIN:
                bd, bij, bji = _dots_par(bra, ket, R, C, S, off[k], off[k + 1])
            else:
                bd, bij, bji = 0j, 0j, 0j
                for p in range(off[k], off[k + 1]):
                    i, j, s = R[p], C[p], S[p]
                    bi, bj = np.conj(bra[i]), np.conj(bra[j])
                    bd += bi * ket[i] + bj * ket[j]
                    bij += bi * s * ket[j]
                    bji += bj * np.conj(s) * ket[i]
            dx = -f * x * bd + (f + th * x * c1) * bij - (f + np.conj(th) * x * c1) * bji
            dy = -f * y * bd + (1j * f + th * y * c1) * bij - (-1j * f + np.conj(th) * y * c1) * bji
            g[k] = 2 * dx.real
            g[k + nt] = 2 * dy.real
            _rot(ket, R, C, S, off[k], off[k + 1], th)


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
        self.dets, self.sdets, self.order = dets, sdets, order

    def _hamiltonian_pattern(self):
        """Sparsity pattern of H (Slater-Condon): singles from E_tu, doubles with phases; CSR map built once."""
        n, nd, dets = self.n, self.ndet, self.dets
        bit = np.left_shift(np.int64(1), np.arange(n - 1, -1, -1, dtype=np.int64))
        occ = (dets[:, None] & bit[None, :]) != 0
        ne = int(occ[0].sum()) if nd else 0
        if njit is not None and nd and n < 63 and nd == math.comb(n, ne):
            cols, kind, sgn, tu, per = _pattern(dets, self.sdets, self.order, n, ne)
            q = [np.flatnonzero(kind == x) for x in range(3)]
            del kind
            pos = np.concatenate(q)
            if len(pos) < 2**31:
                pos = pos.astype(np.int32)
            singles = (tu[q[1]], cols[q[1]], sgn[q[1]].astype(float))
            didx, dsgn = tu[q[2]], sgn[q[2]]
            indptr = np.arange(nd + 1, dtype=np.int64) * per
            self._hpat = (occ.astype(float), singles, didx, dsgn, pos, cols, indptr)
            return
        below = np.cumsum(occ, axis=1) - occ
        rows, cols = [np.arange(nd, dtype=np.int32)], [np.arange(nd, dtype=np.int32)]
        singles = []
        for t in range(n):
            for u in range(n):
                if t != u:
                    E = self.E[t][u].tocoo()
                    singles.append((np.full(len(E.col), t * n + u, dtype=np.int32), E.col.astype(np.int32), E.data.real))
                    rows.append(E.row.astype(np.int32))
                    cols.append(E.col.astype(np.int32))
        tt, vv = np.tril_indices(n, -1)
        didx, dsgn = [], []
        for u in range(n):
            for w in range(u):
                S = np.nonzero(occ[:, u] & occ[:, w])[0]
                if len(S) == 0:
                    continue
                keep = (tt != u) & (tt != w) & (vv != u) & (vv != w)
                T_, V_ = tt[keep], vv[keep]
                ok = ~occ[S][:, T_] & ~occ[S][:, V_]
                b2 = below[S] - (np.arange(n) > u) - (np.arange(n) > w)
                ex = below[S, u][:, None] + below[S, w][:, None] + b2[:, V_] + b2[:, T_] + 1
                si, pi = np.nonzero(ok)
                d4 = (dets[S[si]] ^ bit[u] ^ bit[w]) | bit[T_[pi]] | bit[V_[pi]]
                rows.append(self.order[np.searchsorted(self.sdets, d4)].astype(np.int32))
                cols.append(S[si].astype(np.int32))
                didx.append((((T_[pi] * n + V_[pi]) * n + u) * n + w).astype(np.int32))
                dsgn.append((1 - 2 * (ex[si, pi] % 2)).astype(np.int8))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        perm = np.lexsort((cols, rows))
        indices = cols[perm]
        indptr = np.zeros(nd + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=nd), out=indptr[1:])
        del rows, cols
        pos = np.empty_like(perm)
        pos[perm] = np.arange(len(perm))
        del perm
        if len(pos) < 2**31:
            pos = pos.astype(np.int32)
        didx = np.concatenate(didx) if didx else np.zeros(0, np.int32)
        dsgn = np.concatenate(dsgn) if dsgn else np.zeros(0, np.int8)
        singles = tuple(np.concatenate(x) for x in zip(*singles)) if singles else (np.zeros(0, np.int32),) * 2 + (np.zeros(0),)
        self._hpat = (occ.astype(float), singles, didx, dsgn, pos, indices, indptr)

    def hamiltonian(self, e: float, h: np.ndarray, g: np.ndarray) -> sp.csr_matrix:
        r"""e + \sum h_tu E_tu + 1/2 \sum g_tuvw (E_tu E_vw - \delta_uv E_tw) as sparse matrix (Slater-Condon rules)."""
        if getattr(self, "_hpat", None) is None:
            self._hamiltonian_pattern()
        O, singles, didx, dsgn, pos, indices, indptr = self._hpat
        nd = self.ndet
        data = np.empty(len(pos), dtype=complex)
        JK = np.einsum("kkll->kl", g) - np.einsum("kllk->kl", g)
        data[pos[:nd]] = e + O @ np.diag(h) + 0.5 * np.einsum("dk,kl,dl->d", O, JK, O)
        Wk = (np.einsum("tukk->tuk", g) - np.einsum("tkku->tuk", g)).reshape(-1, O.shape[1])
        tu, col, sg = singles
        coef = (np.einsum("tuvw->tvuw", g) - np.einsum("twvu->tvuw", g)).reshape(-1)
        if njit is not None:
            _h_fill(data, pos, nd, O, tu, col, sg, np.ascontiguousarray(h, dtype=complex).reshape(-1), np.ascontiguousarray(Wk, dtype=complex), didx, dsgn, np.ascontiguousarray(coef, dtype=complex))
            return sp.csr_matrix((data, indices, indptr), shape=(nd, nd))
        o, step = nd, max(1, (1 << 27) // max(1, O.shape[1] ** 2))
        for r0 in range(0, nd, step):
            M = O[r0 : r0 + step] @ Wk.T
            m = (col >= r0) & (col < r0 + step)
            idx = np.flatnonzero(m)
            data[pos[o + idx]] = sg[idx] * (h.reshape(-1)[tu[idx]] + M[col[idx] - r0, tu[idx]])
        o += len(col)
        step = 1 << 22
        for a in range(0, len(didx), step):
            data[pos[o + a : o + a + step]] = coef[didx[a : a + step]] * dsgn[a : a + step]
        return sp.csr_matrix((data, indices, indptr), shape=(nd, nd))

    def _coo(self):
        if getattr(self, "_coo_cache", None) is None:
            n = self.n
            pt = (self._pair % n) * n + self._pair // n
            rows, cols, vals, pair = self._rows.astype(np.int64), self._cols.astype(np.int64), self._vals.real.copy(), self._pair.astype(np.int64)
            o = np.argsort(rows, kind="stable")
            rowptr = np.zeros(self.ndet + 1, dtype=np.int64)
            np.cumsum(np.bincount(rows, minlength=self.ndet), out=rowptr[1:])
            self._coo_cache = (rows, cols, vals, pair, pt.astype(np.int64))
            self._gather = (rowptr, cols[o], vals[o], pair[o])
        return self._coo_cache

    def phi(self, c: np.ndarray) -> np.ndarray:
        """E_tu |c>, shape (n*n, ndet) for one vector or (n*n, ndet, m) for c of shape (ndet, m)."""
        if njit is None:
            return np.array([self.E[t][u] @ c for t in range(self.n) for u in range(self.n)])
        r, cl, v, pr, _ = self._coo()
        X = np.ascontiguousarray(c.reshape(self.ndet, -1), dtype=complex)
        out = np.zeros((self.n**2, self.ndet, X.shape[1]), dtype=complex)
        _e_scatter(r, cl, v, pr, X, out)
        return out[:, :, 0] if c.ndim == 1 else out

    def apply(self, ops, c: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
        r"""(\sum f_tu E_tu + 1/2 \sum g_tuvw (E_tu E_vw - \delta_uv E_tw)) |c> for operators [(f, g), ...], shape (m, ndet)."""
        n = self.n
        P = np.ascontiguousarray(self.phi(c) if phi is None else phi)
        F = np.array([(f - 0.5 * np.einsum("tuuw->tw", g)).reshape(-1) for f, g in ops])
        Gm = np.array([0.5 * g.reshape(n * n, n * n) for _, g in ops])
        out = F @ P
        Z = np.ascontiguousarray((Gm @ P).transpose(1, 2, 0))
        if njit is None:
            for tu in range(n * n):
                out += (self.E[tu // n][tu % n] @ Z[tu]).T
            return out
        self._coo()
        res = np.zeros((self.ndet, Z.shape[2]), dtype=complex)
        _e_gather(*self._gather, Z, res)
        return out + res.T

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
        if njit is None:
            Psi = np.empty((m, n, n, self.ndet), dtype=complex)
            for t in range(n):
                for u in range(n):
                    Psi[:, t, u] = (self.E[u][t] @ Lt).T
            G = (Psi.reshape(m * n * n, -1).conj() @ PT).reshape(m, n, n, n, n)
        else:
            r, cl, v, _, pt = self._coo()
            Psi = np.zeros((n * n, self.ndet, m), dtype=complex)
            _e_scatter(r, cl, v, pt, np.ascontiguousarray(Lt.conj()), Psi)
            G = np.einsum("adm,db->mab", Psi, PT, optimize=True).reshape(m, n, n, n, n)
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

    When T_k^2 = 0 (all fUCCSD/generalized excitations), T_k is a signed partial permutation: each connected
    determinant pair (j -> i, T_ij = s) is rotated by a 2x2 unitary, all other determinants are unchanged:
    v_i <- cos r v_i + s theta sin(r)/r v_j,  v_j <- cos r v_j - s^* theta^* sin(r)/r v_i,  r = |theta|.
    Otherwise U_k v = v + sin(r)/r A v + (1 - cos r)/r^2 A^2 v with A = theta T - theta^* T^\dagger.
    Same conventions as generalized_construct_ups_state_test_erik / generalized_get_grad_action.
    """

    def __init__(self, ci_info, ups_struct, op_builder):
        self.ci_info = ci_info
        self.ups = ups_struct
        self.build = op_builder
        self.T, self.Td, self.P, self.pairs, self.ops = [], [], [], [], []
        self.update()

    @staticmethod
    def _simple(op):
        if len(op.operators) != 1:
            return False
        key = next(iter(op.operators))
        cre = [o for o, d in key if d]
        ann = [o for o, d in key if not d]
        return len(set(cre)) == len(cre) and len(set(ann)) == len(ann) and not set(cre) & set(ann)

    def update(self):
        K = len(self.ups.excitation_indices)
        k0 = len(self.pairs)
        if k0 == K:
            return
        ops = [self.build(self.ups.excitation_operator_type[k], self.ups.excitation_indices[k]) for k in range(k0, K)]
        self.ops += ops
        self.T += [None] * len(ops)
        self.Td += [None] * len(ops)
        self.P += [None] * len(ops)
        self.pairs += [None] * len(ops)
        fast = [i for i, o in enumerate(ops) if njit is not None and self._simple(o)]
        if fast:
            keys = [next(iter(ops[i].operators.items())) for i in fast]
            L = max(len(kc[0]) for kc in keys)
            orbs = np.zeros((len(fast), L), dtype=np.int64)
            dags = np.zeros((len(fast), L), dtype=np.bool_)
            lens = np.array([len(kc[0]) for kc in keys], dtype=np.int64)
            for q, (key, _) in enumerate(keys):
                orbs[q, : len(key)] = [o for o, _ in key]
                dags[q, : len(key)] = [d for _, d in key]
            coef = np.array([c for _, c in keys], dtype=complex)
            dets = np.asarray(self.ci_info.idx2det, dtype=np.int64)
            order = np.argsort(dets)
            R, C, S, cnt = _string_pairs(dets, dets[order], order, 2 * self.ci_info.num_active_orbs, orbs, dags, lens, coef)
            for q, i in enumerate(fast):
                a, b = cnt[q], cnt[q + 1]
                self.pairs[k0 + i] = (R[a:b], C[a:b], S[a:b])
        for i in set(range(len(ops))) - set(fast):
            self._mat(k0 + i)
            T, Td = self.T[k0 + i], self.Td[k0 + i]
            if (T @ T).count_nonzero() == 0:
                self.P[k0 + i] = np.real((T @ Td + Td @ T).diagonal())
                c = T.tocoo()
                self.pairs[k0 + i] = (c.row.astype(np.int64), c.col.astype(np.int64), c.data.astype(complex))

    def _mat(self, k):
        if self.T[k] is None:
            self.T[k] = operator_matrix(self.ci_info, self.ops[k])
            self.Td[k] = self.T[k].conj().T.tocsr()
        return self.T[k], self.Td[k]

    def _P(self, k):
        if self.P[k] is None and self.pairs[k] is not None:
            R, Cc, s = self.pairs[k]
            P = np.zeros(len(self.ci_info.idx2det))
            np.add.at(P, R, np.abs(s) ** 2)
            np.add.at(P, Cc, np.abs(s) ** 2)
            self.P[k] = P
        return self.P[k]

    def _A(self, k, theta, v):
        T, Td = self._mat(k)
        return theta * (T @ v) - np.conj(theta) * (Td @ v)

    def _A2(self, k, theta, v, Av=None):
        P = self._P(k)
        if P is not None:
            return -abs(theta) ** 2 * (P[:, None] if v.ndim == 2 else P) * v
        return self._A(k, theta, self._A(k, theta, v) if Av is None else Av)

    def apply_inplace(self, k, theta, v, dagger=False):
        """v <- U_k v (U_k^dagger if dagger), in place."""
        r = abs(theta)
        if r < 1e-28:
            return v
        th = -theta if dagger else theta
        if self.pairs[k] is None:
            Av = self._A(k, th, v)
            v += np.sin(r) / r * Av + (1 - np.cos(r)) / r**2 * self._A2(k, th, v, Av)
            return v
        R, Cc, s = self.pairs[k]
        if v.ndim == 2:
            s = s[:, None]
        f = np.sin(r) / r
        vi, vj = v[R], v[Cc]
        v[R] = np.cos(r) * vi + (s * th * f) * vj
        v[Cc] = np.cos(r) * vj - (np.conj(s) * np.conj(th) * f) * vi
        return v

    def apply(self, k, theta, v, dagger=False):
        return self.apply_inplace(k, theta, np.array(v, dtype=complex), dagger)

    def packed(self):
        """Concatenated pair arrays (R, C, s, offsets) for the compiled kernels, or None."""
        if njit is None or any(p is None for p in self.pairs):
            return None
        if getattr(self, "_packed", None) is None or len(self._packed[3]) != len(self.pairs) + 1:
            off = np.zeros(len(self.pairs) + 1, dtype=np.int64)
            off[1:] = np.cumsum([len(p[0]) for p in self.pairs])
            self._packed = tuple(np.concatenate([p[i] for p in self.pairs]) for i in range(3)) + (off,)
        return self._packed

    def state(self, v, thetas, dagger=False):
        self.update()
        out = np.array(v, dtype=complex)
        pk = self.packed()
        if pk is not None and out.ndim == 1:
            _ups_state(out, *pk, np.asarray(thetas, dtype=complex), dagger)
            return out
        ks = range(len(thetas))
        for k in (reversed(ks) if dagger else ks):
            self.apply_inplace(k, thetas[k], out, dagger)
        return out

    @staticmethod
    def _coeffs(r):
        if r < 1e-8:
            return 1 - r**2 / 6, 0.5 - r**2 / 24, -1 / 3 + r**2 / 30, -1 / 12 + r**2 / 180
        return np.sin(r) / r, (1 - np.cos(r)) / r**2, (r * np.cos(r) - np.sin(r)) / r**3, (r * np.sin(r) - 2 + 2 * np.cos(r)) / r**4

    def grad_dots(self, k, theta, bra, ket):
        """(Re<bra|dU_k/dRe(theta)|ket>, Re<bra|dU_k/dIm(theta)|ket>) for a pair-type generator."""
        R, Cc, s = self.pairs[k]
        x, y, r = theta.real, theta.imag, abs(theta)
        f, _, c1, _ = self._coeffs(r)
        bi, bj, ki, kj = bra[R].conj(), bra[Cc].conj(), ket[R], ket[Cc]
        bk_d = bi @ ki + bj @ kj
        bk_ij = bi @ (s * kj)
        bk_ji = bj @ (np.conj(s) * ki)
        dx = -f * x * bk_d + (f + theta * x * c1) * bk_ij - (f + np.conj(theta) * x * c1) * bk_ji
        dy = -f * y * bk_d + (1j * f + theta * y * c1) * bk_ij - (-1j * f + np.conj(theta) * y * c1) * bk_ji
        return dx.real, dy.real

    def grad_action(self, k, theta, v):
        """(dU_k/dRe(theta) v, dU_k/dIm(theta) v)."""
        VR, VI = self._V(k, True, v), self._V(k, False, v)
        Av = theta.real * VR + theta.imag * VI
        A2v = self._A2(k, theta, v, Av)
        f, g, c1, c2 = self._coeffs(abs(theta))
        out = []
        for real, V, x in ((True, VR, theta.real), (False, VI, theta.imag)):
            VAv = self._V(k, real, Av)
            out.append(f * V + g * (VAv + self._A(k, theta, V)) + x * c1 * Av + x * c2 * A2v)
        return out[0], out[1]

    def _V(self, k, real, w):
        T, Td = self._mat(k)
        Tw, Tdw = T @ w, Td @ w
        return Tw - Tdw if real else 1j * (Tw + Tdw)


def ups_theta_gradient(engine: UPSEngine, thetas: np.ndarray, ref: np.ndarray, hc: np.ndarray) -> np.ndarray:
    """dE/d(Re theta, Im theta) for |0> = U(theta)|ref> from hc = H|0> (one unitary peeled off per parameter,
    arXiv:2303.10825, Eq. 20)."""
    nt = len(thetas)
    g = np.zeros(2 * nt)
    bra = engine.state(hc, thetas, dagger=True)
    ket = np.array(ref, dtype=complex)
    pk = engine.packed()
    if pk is not None and bra.ndim == 1:
        _ups_grad(bra, ket, *pk, np.asarray(thetas, dtype=complex), g)
        return g
    for i in range(nt):
        engine.apply_inplace(i, thetas[i], bra)
        if engine.pairs[i] is not None:
            gx, gy = engine.grad_dots(i, thetas[i], bra, ket)
            g[i], g[i + nt] = 2 * gx, 2 * gy
        else:
            dR, dI = engine.grad_action(i, thetas[i], ket)
            g[i] = 2 * np.vdot(bra, dR).real
            g[i + nt] = 2 * np.vdot(bra, dI).real
        engine.apply_inplace(i, thetas[i], ket)
    return g
