import time

import numpy as np
import scipy.linalg
from pyscf.data import nist

from slowquant.unitary_coupled_cluster.ci_sparse_DHF import ActiveCI, ladder, operator_matrix
try:
    from slowquant.unitary_coupled_cluster.ci_sparse_DHF import quad_forms
except ImportError:
    quad_forms = None
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing_DHF
from slowquant.unitary_coupled_cluster.DHF_integral_direct import giao_integrals, giao_integrals_cd
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_propagate_state
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response.solvers_DHF import _precondition, davidson_eig, davidson_solve
from slowquant.unitary_coupled_cluster.operators import G1, G2, G3, G4, G5, G6
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2, iterate_t3, iterate_t4, iterate_t5, iterate_t6
from slowquant.unitary_coupled_cluster.printing import key_value, value_table, roots_table, timing_line


def _dag(rho):
    s, d, G = rho
    return np.conj(s), d.conj().T, G.conj().transpose(1, 0, 3, 2)


class LinearResponseDirect:
    r"""Integral-direct, matrix-free naive linear response for the 4c OO-UPS wave function.

    Operators: q_j = a^\dagger_{Q_j} a_{P_j}, (P_j, Q_j) in kappa_no_activeactive_spin_idx_resp, and G_j (active S, D, ...).
    E2 = [[A, B], [B*, A*]], S = [[Sigma, 0], [0, -Sigma*]] with
    A_ij = <0|[X_i^\dagger, [H, X_j]]|0>, B_ij = <0|[X_i^\dagger, [H, X_j^\dagger]]|0>, Sigma_ij = <0|[X_i^\dagger, X_j]|0>.
    qG blocks: plain double commutator <0|[G^\dagger, [H, q]]|0> (default; exact Hessian of E for exp(-kappa) exp(S)|0>,
    the orbital rotation applied last as in the OO wave function), or with symmetrize_qG=True the symmetrized
    1/2<0|[G^\dagger, [H, q]] + [[G^\dagger, H], q]|0> as in generalized_naive_DHF. GG and qq blocks: symmetrized.
    Property gradients: [<0|[X_i^\dagger, V]|0>; <0|[X_i, V]|0>].
    """

    def __init__(
        self,
        wave_function: GeneralizedWaveFunctionUPS,
        excitations: str = "SD",
        screen: float = 1e-8,
        symmetrize_qG: bool = False,
        two_level: int = 0,
    ) -> None:
        t0 = time.time()
        self.two_level = 200 if two_level is True else int(two_level)
        wf = self.wf = wave_function
        excitations = excitations.lower()
        occ, unocc = wf.active_occ_spin_idx, wf.active_unocc_spin_idx
        G_ops = []
        if "s" in excitations:
            G_ops += [G1(i, a) for a, i in iterate_t1(occ, unocc, is_spin_conserving=False)]
        if "d" in excitations:
            G_ops += [G2(i, j, a, b) for a, i, b, j in iterate_t2(occ, unocc, is_spin_conserving=False)]
        if "t" in excitations:
            G_ops += [G3(i, j, k, a, b, c) for a, i, b, j, c, k in iterate_t3(occ, unocc)]
        if "q" in excitations:
            G_ops += [G4(i, j, k, l, a, b, c, d) for a, i, b, j, c, k, d, l in iterate_t4(occ, unocc)]
        if "5" in excitations:
            G_ops += [G5(i, j, k, l, m, a, b, c, d, e) for a, i, b, j, c, k, d, l, e, m in iterate_t5(occ, unocc)]
        if "6" in excitations:
            G_ops += [
                G6(i, j, k, l, m, n, a, b, c, d, e, f)
                for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(occ, unocc)
            ]
        self.G_ops = G_ops
        ds = self.ds = wf.ds
        ds.set_rdms(wf.rdm1, wf.rdm2)
        ds.response_tensors()
        timing_line("Response integrals", time.time() - t0)
        self.N = ds.C.shape[1]
        self.pos = -np.ones(self.N, dtype=int)
        self.pos[ds.occ] = np.arange(len(ds.occ))
        self.Docc = np.zeros((len(ds.occ), len(ds.occ)), dtype=complex)
        nI = len(ds.I)
        self.Docc[:nI, :nI] = np.eye(nI)
        self.Docc[nI:, nI:] = wf.rdm1
        self.q_idx = np.array(wf.kappa_no_activeactive_spin_idx_resp, dtype=int).reshape(-1, 2)
        a_q, s_q = self._diag_q()
        self.q_idx = self.q_idx[(np.abs(a_q) > screen) | (np.abs(s_q) > screen)]
        self.nq, self.nG = len(self.q_idx), len(G_ops)
        self.n = self.nq + self.nG
        print(key_value("LR operators", f"{self.nq} orbital, {self.nG} active-space"))

        ci = self.ci = ActiveCI(wf.ci_info)
        c0 = self.c0 = wf.ci_coeffs
        self.phi0 = ci.phi(c0)
        H = ci.hamiltonian(*ds.active_integrals())
        ndet = len(c0)

        def prop(op, v):
            return generalized_propagate_state([op], v, wf.ci_info)

        h0 = H @ c0
        self.H = H
        shift = wf.active_spin_idx[0]
        Gm = [
            operator_matrix(wf.ci_info, FermionicOperator({tuple((o - shift, d) for o, d in k): v for k, v in G.operators.items()}))
            for G in G_ops
        ]
        self.gk = np.array([M @ c0 for M in Gm]).reshape(-1, ndet)
        self.gdk = np.array([M.conj().T @ c0 for M in Gm]).reshape(-1, ndet)
        self.Gh = np.array([M @ h0 for M in Gm]).reshape(-1, ndet)
        self.Gdh = np.array([M.conj().T @ h0 for M in Gm]).reshape(-1, ndet)
        self.gkc, self.Ghc = self.gk.conj(), self.Gh.conj()
        self.DA, self.DB = self._cross_terms(G_ops) if symmetrize_qG else (np.zeros((self.nG, self.nq)),) * 2
        timing_line("Linear response setup", time.time() - t0)

    def _cross_terms(self, G_ops) -> tuple[np.ndarray, np.ndarray]:
        r"""Symmetrization corrections DA_Gq = 1/2<0|H[q, G^\dagger]|0>, DB_Gq = 1/2<0|H[q, G]|0>^* (zero without an active index in q; virtual includes positronic).

        q = a^\dagger_v a_t: <0|H a^\dagger_v|X> = <zeta_v|X>, zeta_v = sum_u F^I_vu a_u|0> + sum g_vuxw E_xw a_u|0> (N-1).
        q = a^\dagger_t a_i: <0|H [a^\dagger_t, G] a_i|0> = <xi_i|[a^\dagger_t, G]|0>, xi_i = sum_u F^I_ui a^\dagger_u|0> + sum g_iuwx^* a^\dagger_u E_xw|0> (N+1).
        """
        wf, ds, c0 = self.wf, self.ds, self.c0
        nG, nq = len(G_ops), self.nq
        DA = np.zeros((nG, nq), dtype=complex)
        DB = np.zeros((nG, nq), dtype=complex)
        if nG == 0 or nq == 0:
            return DA, DB
        na, ne = len(ds.A), wf.num_active_elec
        loc = -np.ones(self.N, dtype=int)
        loc[ds.A] = np.arange(na)
        P, Q = self.q_idx.T
        inact = np.zeros(self.N, dtype=bool)
        inact[ds.I] = True
        tv = np.where((loc[P] >= 0) & (loc[Q] < 0) & ~inact[Q])[0]
        it = np.where(inact[P] & (loc[Q] >= 0))[0]
        legs = [sorted({loc[i] for key in G.operators for i, _ in key}) for G in G_ops]
        pairs = [(G, Gd, legs[k], k) for k, (G, Gd) in enumerate(zip(G_ops, [G.dagger for G in G_ops]))]

        def info(m):
            return get_indexing_DHF(wf.num_inactive_spin_orbs, na, wf.num_virtual_spin_orbs, m, 0, wf.num_spin_orbs_NES)

        def fill(rows, t_of, bra_of, lad, lc, sub):
            for G, Gd, lg, k in pairs:
                for t in lg:
                    sel = rows[t_of[rows] == t]
                    if len(sel) == 0:
                        continue
                    y = lad[t] @ self.gdk[k] - generalized_propagate_state([Gd], lc[t], sub)
                    yp = lad[t] @ self.gk[k] - generalized_propagate_state([G], lc[t], sub)
                    DA[k, sel] = bra_of(sel).conj() @ y
                    DB[k, sel] = (bra_of(sel).conj() @ yp).conj()

        g = ds.g_paaa
        if len(tv) and ne > 0:
            sub = info(ne - 1)
            lad = [ladder(wf.ci_info, sub, t, False) for t in range(na)]
            lc = np.array([a @ c0 for a in lad])
            Em = ActiveCI(sub).E
            Bv = np.empty((na, na, na, lc.shape[1]), dtype=complex)
            for x in range(na):
                for w in range(na):
                    Bv[:, x, w] = (Em[x][w] @ lc.T).T
            vs, vinv = np.unique(Q[tv], return_inverse=True)
            zeta = ds.FI[np.ix_(vs, ds.A)] @ lc + g[vs].reshape(len(vs), -1) @ Bv.reshape(na**3, -1)
            t_of = np.zeros(nq, dtype=int)
            t_of[tv] = loc[P[tv]]
            zrow = np.zeros(nq, dtype=int)
            zrow[tv] = vinv
            fill(tv, t_of, lambda sel: zeta[zrow[sel]], lad, lc, sub)
        if len(it) and ne < na:
            sub = info(ne + 1)
            lad = [ladder(wf.ci_info, sub, t, True) for t in range(na)]
            lc = np.array([a @ c0 for a in lad])
            e = self.phi0.reshape(na, na, -1)
            ins, iinv = np.unique(P[it], return_inverse=True)
            h = np.einsum("iuwx,xwd->iud", g[ins].conj(), e) + ds.FI[np.ix_(ds.A, ins)].T[:, :, None] * c0
            xi = sum((lad[u] @ h[:, u].T).T for u in range(na))
            t_of = np.zeros(nq, dtype=int)
            t_of[it] = loc[Q[it]]
            xrow = np.zeros(nq, dtype=int)
            xrow[it] = iinv
            fill(it, t_of, lambda sel: xi[xrow[sel]], lad, lc, sub)
        return 0.5 * DA, 0.5 * DB

    def _fget(self, F, rows, cols):
        p = self.pos[cols]
        return np.where(p >= 0, F[rows, np.maximum(p, 0)], 0.0)

    def _gg(self, x, y):
        gk, gdk, Gh, Gdh, gkc, Ghc, H = self.gk, self.gdk, self.Gh, self.Gdh, self.gkc, self.Ghc, self.H
        xc, yc = x.conj(), y.conj()
        Xc, Xdc, Xh, Xdh = gk.T @ x, gdk.T @ xc, Gh.T @ x, Gdh.T @ xc
        HXc, HXdc = H @ Xc, H @ Xdc
        s = gkc @ HXc + gdk @ HXdc.conj()
        s -= 0.5 * (gkc @ Xh + gdk @ Xdh.conj() + Gdh @ Xdc.conj() + Ghc @ Xc)
        Yc, Ydc = gdk.T @ y, gk.T @ yc
        HYc, Yh, Ydh, HYdc = H @ Yc, Gdh.T @ y, Gh.T @ yc, H @ Ydc
        s += gkc @ (HYc - Yh) + gdk @ (HYdc - Ydh).conj()
        return s

    def _sgg(self, x):
        return self.gkc @ (self.gk.T @ x) - self.gdk @ (self.gdk.T @ x.conj()).conj()

    def _metric_q(self, X):
        occ = self.ds.occ
        DX = np.zeros_like(X)
        XD = np.zeros_like(X)
        DX[occ] = self.Docc @ X[occ]
        XD[:, occ] = X[:, occ] @ self.Docc
        return DX - XD

    def sigma(self, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """E2 @ V and S @ V for trial vectors V of shape (2n, m). One J/K pass with 2 densities per vector."""
        ds, ci = self.ds, self.ci
        n, nq, nG, m, N = self.n, self.nq, self.nG, V.shape[1], self.N
        P, Q = self.q_idx.T
        x, y = V[:n], V[n:]
        E2V = np.zeros((2 * n, m), dtype=complex)
        SV = np.zeros((2 * n, m), dtype=complex)
        xG, yG = x[nq:], y[nq:]
        if nq:
            C, I = ds.C, ds.I
            Cd, CI = C.conj().T, C[:, I]
            if nG:
                L = np.hstack((self.gk.T @ xG + self.gdk.T @ yG, self.gdk.T @ xG.conj() + self.gk.T @ yG.conj())).T
                s, d, G = ci.transition_rdms(L, self.c0, self.phi0)
            ks, dms, rhos = [], [], []
            for j in range(m):
                k = np.zeros((N, N), dtype=complex)
                k[Q, P] = x[:nq, j]
                k[P, Q] += y[:nq, j]
                ks.append(k)
                dI, dIA = ds.transformed_densities(k, -k)
                dc = -dIA
                if nG:
                    rp0, rw0 = (s[j], d[j], G[j]), (s[m + j], d[m + j], G[m + j])
                    r0p, r0w = _dag(rp0), _dag(rw0)
                    rho1 = tuple(u - w for u, w in zip(r0p, rw0))
                    rho2 = tuple(w - u for w, u in zip(rp0, r0w))
                    rhos.append((rho1, rho2))
                    dc = dc + ds.transition_density(rho1[1])
                dms += [dI, dc]
            v = ds.direct.veff(np.array(dms), response=True)
            ops1, ops2 = [], []
            for j, k in enumerate(ks):
                vI, vc = v[2 * j], v[2 * j + 1]
                kd = k.conj().T
                F1, f1, g1 = ds.transformed_operator(k, -k, vI)
                F2, f2, g2 = ds.transformed_operator(kd, -kd, -vI.conj().T)
                M1, M2 = -F1, F2
                if nG:
                    M1 = M1 + ds.transition_fock(rhos[j][0])
                    M2 = M2 + ds.transition_fock(rhos[j][1])
                M1[:, : len(I)] += Cd @ vc @ CI
                M2[:, : len(I)] += Cd @ vc.conj().T @ CI
                E2V[:nq, j] = self._fget(M1, Q, P) - self._fget(M2, P, Q).conj()
                E2V[n : n + nq, j] = self._fget(M1, P, Q) - self._fget(M2, Q, P).conj()
                ops1.append((f1, g1))
                ops2.append((f2, g2))
                X = np.zeros((N, N), dtype=complex)
                X[P, Q] = x[:nq, j]
                SV[:nq, j] = self._metric_q(X)[P, Q]
                X[P, Q] = y[:nq, j].conj()
                SV[n : n + nq, j] = -self._metric_q(X)[P, Q].conj()
            if nG:
                o1 = ci.apply(ops1, self.c0, self.phi0)
                o2 = ci.apply(ops2, self.c0, self.phi0)
                gk, gdk = self.gk, self.gdk
                E2V[nq:n] = -(gdk @ o2.conj().T) - gk.conj() @ o1.T
                E2V[n + nq :] = -(gdk.conj() @ o1.T) - gk @ o2.conj().T
        if nG and nq:
            DA, DB, xq, yq = self.DA, self.DB, x[:nq], y[:nq]
            E2V[:nq] += DA.conj().T @ xG + DB.T @ yG
            E2V[nq:n] += DA @ xq + DB @ yq
            E2V[n : n + nq] += DB.conj().T @ xG + DA.T @ yG
            E2V[n + nq :] += DB.conj() @ xq + DA.conj() @ yq
        if nG:
            E2V[nq:n] += self._gg(xG, yG)
            E2V[n + nq :] += self._gg(yG.conj(), xG.conj()).conj()
            SV[nq:n] = self._sgg(xG)
            SV[n + nq :] = -self._sgg(yG.conj()).conj()
        return E2V, SV

    def _diag_q(self) -> tuple[np.ndarray, np.ndarray]:
        P, Q = self.q_idx.T
        ds = self.ds
        F = (ds.FI + ds.FA).diagonal().real
        Fg = ds.gen_fock().diagonal().real
        Dd = np.zeros(self.N)
        Dd[ds.occ] = self.Docc.diagonal().real
        dP, dQ = Dd[P], Dd[Q]
        return dP * F[Q] + dQ * F[P] - Fg[P] - Fg[Q], dP - dQ

    def diagonal(self) -> tuple[np.ndarray, np.ndarray]:
        """Approximate diagonal of A (orbital part from Fock energies) and diagonal of Sigma."""
        a_q, s_q = self._diag_q()
        gk, gdk, Gh, Gdh, H = self.gk, self.gdk, self.Gh, self.Gdh, self.H

        def dot(u, v):
            return np.einsum("md,md->m", u.conj(), v)

        def quad(V):
            if quad_forms is not None:
                r, cl = np.nonzero(V)
                off = np.zeros(len(V) + 1, dtype=np.int64)
                np.cumsum(np.bincount(r, minlength=len(V)), out=off[1:])
                Hc = H.tocsr()
                return quad_forms(Hc.indptr.astype(np.int64), Hc.indices.astype(np.int64), Hc.data, cl.astype(np.int64), V[r, cl], off, V.shape[1])
            out = np.empty(len(V), dtype=complex)
            for k, v in enumerate(V):
                S = np.flatnonzero(v)
                out[k] = np.vdot(v[S], H[S][:, S] @ v[S])
            return out

        a_G = quad(gk) + quad(gdk).conj() - 0.5 * (dot(gk, Gh) + dot(Gdh, gdk) + dot(gdk, Gdh) + dot(Gh, gk))
        s_G = dot(gk, gk) - dot(gdk, gdk)
        return np.concatenate((a_q, a_G.real)), np.concatenate((s_q, s_G.real))

    def explicit(self) -> tuple[np.ndarray, np.ndarray]:
        """Full E2 and S by applying sigma to unit vectors (testing/small systems only)."""
        return self.sigma(np.eye(2 * self.n, dtype=complex))

    def _projected_block(self, idx):
        """E2 and S restricted to the parameters idx (orbital rotations), from sigma with J/K projected onto the
        orbitals O they involve (+ active): v = S C_O [(OO|OO) contraction] C_O^+ S, exact on the O block."""
        ds = self.ds
        P, Q = self.q_idx[idx].T
        O = np.unique(np.concatenate((P, Q, ds.A)))
        CO = ds.C[:, O]
        g = ds.direct.eri(CO, CO, CO, CO)
        SC = ds.direct.mf.get_ovlp() @ CO

        class Projected:
            def __init__(self, direct):
                self.direct = direct

            def __getattr__(self, name):
                return getattr(self.direct, name)

            def veff(self, dms, response=False, hermi=0):
                X = np.einsum("mp,dmn,nq->dpq", SC.conj(), dms, SC, optimize=True)
                V = np.einsum("pqrs,dsr->dpq", g, X, optimize=True) - np.einsum("psrq,dsr->dpq", g, X, optimize=True)
                return np.einsum("mp,dpq,nq->dmn", SC, V, SC.conj(), optimize=True)

        n = self.n
        cols = np.concatenate((idx, n + idx))
        I = np.zeros((2 * n, len(cols)), dtype=complex)
        I[cols, np.arange(len(cols))] = 1.0
        direct, ds.direct = ds.direct, Projected(ds.direct)
        try:
            E, Sm = [], []
            for c0 in range(0, len(cols), 64):
                e, sm = self.sigma(I[:, c0 : c0 + 64])
                E.append(e[cols])
                Sm.append(sm[cols])
        finally:
            ds.direct = direct
        return cols, np.hstack(E), np.hstack(Sm)

    def preconditioner(self, n_orb: int = 200):
        """Two-level preconditioner: exact blocks for the active-space parameters (CI part only) and for the n_orb
        orbital rotations with the smallest diagonal (J/K projected onto their orbitals, no J/K pass); diagonal elsewhere."""
        nq, n, nG = self.nq, self.n, self.nG
        if getattr(self, "_blocks", None) is None:
            t0 = time.time()
            self._blocks = []
            dA, _ = self.diagonal()
            if nq and n_orb:
                idx = np.argsort(np.abs(dA[:nq]))[: min(n_orb, nq)]
                self._blocks.append(self._projected_block(idx))
            if nG:
                I, Z = np.eye(nG, dtype=complex), np.zeros((nG, nG), dtype=complex)
                A, B = self._gg(I, Z), self._gg(Z, I)
                Sx = self._sgg(I)
                self._blocks.append((np.r_[nq:n, n + nq : 2 * n], np.block([[A, B], [B.conj(), A.conj()]]), np.block([[Sx, Z], [Z, -Sx.conj()]])))
            self._lu = {}
            timing_line("LR preconditioner blocks", time.time() - t0)
        dA, dS = self.diagonal()

        def apply(R, omega):
            out = _precondition(R, dA, dS, omega)
            for b, (cols, E, Sm) in enumerate(self._blocks):
                for w in np.unique(omega):
                    if (b, w) not in self._lu:
                        self._lu[b, w] = scipy.linalg.lu_factor(E - w * Sm)
                    c = np.flatnonzero(omega == w)
                    out[np.ix_(cols, c)] = scipy.linalg.lu_solve(self._lu[b, w], R[np.ix_(cols, c)])
            return out

        return apply

    def max_batch(self) -> int:
        """Trial vectors per sigma call (one J/K integral pass each), from mol.max_memory; 16..128."""
        mol = self.wf.int_gen.int_obj
        nA = len(self.ds.occ) - len(self.ds.I)
        per = 16 * (6 * (2 * mol.nao_2c()) ** 2 + 3 * self.N**2 + 3 * nA**4)
        return int(np.clip(0.3 * mol.max_memory * 1e6 / per, 16, 128))

    def calc_excitation_energies(self, n_roots: int = 5, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
        """Lowest positive excitation energies via paired Davidson."""
        self.excitation_energies, self.response_vectors = davidson_eig(
            self.sigma, *self.diagonal(), n_roots, tol=tol, max_iter=max_iter, max_batch=self.max_batch()
        )
        roots_table("Excitation energies (linear response)", self.excitation_energies)
        return self.excitation_energies

    def solve_response(self, V: np.ndarray, omega: float = 0.0, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
        """Solve (E2 - omega S) X = V for all columns of V (one shared subspace)."""
        return davidson_solve(self.sigma, *self.diagonal(), V, omega, tol=tol, max_iter=max_iter, max_batch=self.max_batch(), precond=self.preconditioner(self.two_level) if getattr(self, "two_level", 0) else None)

    def operator_gradient(self, ops) -> np.ndarray:
        """Property gradients of Hermitian 1+2-electron operators given as (F[:, occ], F^I[A, A], g[A, A, A, A])."""
        n, nq = self.n, self.nq
        P, Q = self.q_idx.T
        out = np.zeros((2 * n, len(ops)), dtype=complex)
        for c, (F, _, _) in enumerate(ops):
            out[:nq, c] = self._fget(F, Q, P) - self._fget(F, P, Q).conj()
            out[n : n + nq, c] = self._fget(F, P, Q) - self._fget(F, Q, P).conj()
        if self.nG:
            o = self.ci.apply([(f, g) for _, f, g in ops], self.c0, self.phi0)
            gk, gdk = self.gk, self.gdk
            out[nq:n] = gk.conj() @ o.T - gdk @ o.conj().T
            out[n + nq :] = gdk.conj() @ o.T - gk @ o.conj().T
        return out

    def property_gradient(self, ao_ints: np.ndarray) -> np.ndarray:
        """Property gradients of one-electron AO operators (k, n4c, n4c) (Hermitian)."""
        ao_ints = np.asarray(ao_ints)
        shape = ao_ints.shape[:-2]
        ao_ints = ao_ints.reshape(-1, *ao_ints.shape[-2:])
        ds = self.ds
        C, occ, A = ds.C, ds.occ, ds.A
        Cd = C.conj().T
        na = len(A)
        zero = np.zeros((na, na, na, na), dtype=complex)
        ops = []
        for v_ao in ao_ints:
            v = Cd @ v_ao @ C
            ops.append((v[:, occ] @ self.Docc.T, v[np.ix_(A, A)], zero))
        return self.operator_gradient(ops).reshape(2 * self.n, *shape)

    def giao_operators(self) -> list[tuple]:
        r"""RMB-GIAO first-order magnetic operators (one per field component) in orthonormal MOs.

        h^B + g^B (GIAO derivative integrals) plus the connection term, i.e. the Hamiltonian one-index
        transformed with A = B = -S^B/2 (h -> h^B - (S^B h + h S^B)/2, same for all four indices of g).
        """
        t0 = time.time()
        wf, ds = self.wf, self.ds
        direct, ig = ds.direct, wf.int_gen
        C, I, A = ds.C, ds.I, ds.A
        Cd, CA, CI = C.conj().T, C[:, A], C[:, I]
        nI, N = len(I), self.N
        dA = CA @ wf.rdm1.T @ CA.conj().T
        use_cd = (
            (getattr(direct, "V", None) is not None or getattr(direct, "_ri_L", None) is not None)
            and getattr(direct, "piv", None) is not None
            and getattr(direct, "blocks", "") == "LS"
            and direct.ssss_giao_jk == direct.ssss_giao_g
            and getattr(direct, "cd_giao", True)
        )
        if use_cd:
            W, gB = giao_integrals_cd(direct, C, CA, np.array([ds.dI, dA]), consume=not direct.cd_jk)
        else:
            W, gB = giao_integrals(direct, C, CA, np.array([ds.dI, dA]))
        timing_line("GIAO two-electron integrals (CD)" if use_cd else "GIAO two-electron integrals", time.time() - t0)
        ts, dms = [], []
        for x in range(3):
            t = -0.5 * Cd @ ig.S_B[x] @ C
            ts.append(t)
            dms += ds.transformed_densities(t, t)
        vc = direct.veff(np.array(dms), response=True)
        ops = []
        for x in range(3):
            FIB = Cd @ (ig.h_B_RMB_GIAO[x] + W[0, x]) @ C
            F = np.empty((N, nI + len(A)), dtype=complex)
            F[:, :nI] = FIB[:, I] + Cd @ W[1, x] @ CI
            F[:, nI:] = FIB[:, A] @ wf.rdm1.T + np.einsum("nuvw,tuvw->nt", gB[x], wf.rdm2)
            Fc, fc, gc = ds.transformed_operator(ts[x], ts[x], vc[2 * x], vc[2 * x + 1])
            ops.append((F + Fc, FIB[np.ix_(A, A)] + fc, gB[x][A] + gc))
        timing_line("GIAO operators", time.time() - t0)
        return ops

    def giao_expectation(self) -> np.ndarray:
        r"""<0|h^{Bm}|0> with connection term, shape (natm, 3 (B), 3 (m))."""
        ds, ig = self.ds, self.wf.int_gen
        C, occ = ds.C, ds.occ
        Cd = C.conj().T
        h_Bm, h_m, S_B = ig.h_Bm_RMB_GIAO, ig.h_m, ig.S_B
        Docc = self.Docc
        Pao = C[:, occ] @ Docc.T @ C[:, occ].conj().T
        S = np.array([Cd @ s @ C for s in S_B])
        natm = h_m.shape[0]
        out = np.zeros((natm, 3, 3))
        for i in range(natm):
            for b in range(3):
                hm = Cd @ h_m[i][b] @ C
                for a in range(3):
                    conn = np.sum((S[a][occ] @ hm[:, occ] + hm[occ] @ S[a][:, occ]) * Docc)
                    out[i, a, b] = (np.einsum("pq,qp->", h_Bm[i, a, b], Pao) - 0.5 * conn).real
        return out

    def get_properties(
        self, shieldings: bool = True, sscc: bool = True, tol: float = 1e-5, max_iter: int = 100, nuclei=None, b_only=False
    ):
        """RMB-GIAO isotropic shieldings (ppm) and/or RKB isotropic reduced spin-spin couplings (Hz).

        One shared response solve for B (if shieldings) and the nuclear moments m_K of K in `nuclei` (default: all).
        Where both response vectors exist the second-order (error-cancelling) expression
        <<A;B>> = -(V_A^H X_B + X_A^H V_B - X_A^H E2 X_B) is used (error quadratic in the residuals), otherwise
        the linear one -V_A^H X_B. Shieldings of all nuclei need only X_B; a coupling K-L needs K or L in `nuclei`.
        Cost grows with the number of trial vectors (J/K contraction per density and CI part per vector).
        b_only: solve only the 3 B right-hand sides with tol/100 and use the linear expression (more iterations,
        far fewer trial vectors; cheaper for molecules with several nuclei, but needs the tight tolerance).
        """
        mb = self.max_batch()
        if b_only:
            nuclei, tol = [], tol * 1e-2
        h_m = self.wf.int_gen.h_m
        natm = h_m.shape[0]
        nuc = list(range(natm)) if nuclei is None else sorted(nuclei)
        sel = np.array([3 * K + a for K in nuc for a in range(3)], dtype=int)
        Vm = self.property_gradient(h_m.reshape(natm * 3, *h_m.shape[-2:]))
        V = Vm[:, sel]
        if shieldings:
            VB = self.operator_gradient(self.giao_operators())
            V = np.hstack((VB, V))
        X, EX = davidson_solve(self.sigma, *self.diagonal(), V, tol=tol, max_iter=max_iter, max_batch=mb, return_ex=True, precond=self.preconditioner(self.two_level) if getattr(self, "two_level", 0) else None)
        Xm, EXm = X[:, V.shape[1] - len(sel) :], EX[:, V.shape[1] - len(sel) :]
        res = {}
        if shieldings:
            XB, EXB = X[:, :3], EX[:, :3]
            resp = -(Vm.conj().T @ XB).real
            resp[sel] -= (Xm.conj().T @ VB - Xm.conj().T @ EXB).real
            resp = resp.reshape(natm, 3, 3).transpose(0, 2, 1)
            self.shielding_tensors = (resp + self.giao_expectation()) * nist.ALPHA**2 * 1e6
            self.shieldings = np.trace(self.shielding_tensors, axis1=1, axis2=2) / 3
            res["shieldings"] = self.shieldings
            mol = self.wf.int_gen.int_obj
            value_table(
                "Isotropic nuclear shieldings (RMB-GIAO)",
                [("Nucleus", 8, ""), ("sigma_iso [ppm]", 16, ".6f"), ("span [ppm]", 14, ".6f")],
                [
                    (f"{mol.atom_symbol(i)}{i + 1}", self.shieldings[i], np.ptp(np.linalg.eigvalsh(0.5 * (t + t.T))))
                    for i, t in enumerate(self.shielding_tensors)
                ],
            )
        if sscc:
            K = np.full((natm * 3, natm * 3), np.nan)
            K[:, sel] = -(Vm.conj().T @ Xm).real
            K[np.ix_(sel, sel)] -= (Xm.conj().T @ Vm[:, sel] - Xm.conj().T @ EXm).real
            K = np.where(np.isnan(K), K.T, K).reshape(natm, 3, natm, 3)
            nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
            au2Hz = nist.HARTREE2J / nist.PLANCK
            k = np.einsum("IaJa->IJ", K) / 3 * nist.ALPHA**4 * au2Hz * nuc_mag**2
            np.fill_diagonal(k, 0.0)
            self.sscc = k
            res["sscc"] = k
            mol = self.wf.int_gen.int_obj
            value_table(
                "Isotropic reduced spin-spin couplings (RKB)",
                [("Pair", 12, ""), ("K_iso [Hz]", 16, ".6f")],
                [(f"{mol.atom_symbol(i)}{i + 1}-{mol.atom_symbol(j)}{j + 1}", k[i, j]) for i in range(natm) for j in range(i + 1, natm) if np.isfinite(k[i, j])],
            )
        return res

    def get_shieldings_4comp_iso(self, tol: float = 1e-5) -> np.ndarray:
        return self.get_properties(True, False, tol)["shieldings"]

    def get_SSCC_4comp_iso(self, tol: float = 1e-5) -> np.ndarray:
        return self.get_properties(False, True, tol)["sscc"]
