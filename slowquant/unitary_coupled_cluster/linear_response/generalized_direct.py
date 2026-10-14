import time

import numpy as np

from slowquant.unitary_coupled_cluster.ci_sparse import ActiveCI, ladder
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing_generalized
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_propagate_state
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response.solvers import davidson_eig, davidson_solve
from slowquant.unitary_coupled_cluster.operators import G1, G2, G3, G4, G5, G6
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2, iterate_t3, iterate_t4, iterate_t5, iterate_t6
from slowquant.unitary_coupled_cluster.printing import key_value, value_table, roots_table, timing_line, HARTREE_EV


def _dag(rho):
    s, d, G = rho
    return np.conj(s), d.conj().T, G.conj().transpose(1, 0, 3, 2)


class LinearResponseDirect:
    r"""Integral-direct, matrix-free naive linear response for the generalized (GHF spin-orbital) OO-UPS wave function.

    Operators: q_j = a^\dagger_{Q_j} a_{P_j}, (P_j, Q_j) in kappa_no_activeactive_spin_idx, and G_j (active S, D, ...).
    E2 = [[A, B], [B*, A*]], S = [[Sigma, 0], [0, -Sigma*]] with
    A_ij = <0|[X_i^\dagger, [H, X_j]]|0>, B_ij = <0|[X_i^\dagger, [H, X_j^\dagger]]|0>, Sigma_ij = <0|[X_i^\dagger, X_j]|0>.
    All blocks use the symmetrized double commutator 1/2<0|[X_i^\dagger, [H, X_j]] + [[X_i^\dagger, H], X_j]|0> (as generalized_naive).
    The qG blocks are the plain double commutator (one-index transformed Hamiltonian) plus the symmetrization
    correction 1/2<0|H [q, G^\dagger]|0>, computed once with (N-1)/(N+1)-electron active-space intermediates.
    symmetrize_qG=False gives the plain qG blocks <0|[G^\dagger, [H, q]]|0> (exact Hessian for exp(-kappa) exp(S)|0>, as the 4c code).
    Property gradients: [<0|[X_i^\dagger, V]|0>; <0|[X_i, V]|0>].
    """

    def __init__(
        self,
        wave_function: GeneralizedWaveFunctionUPS,
        excitations: str = "SD",
        screen: float = 1e-8,
        symmetrize_qG: bool = True,
    ) -> None:
        t0 = time.time()
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
        self.q_idx = np.array(wf.kappa_no_activeactive_spin_idx, dtype=int).reshape(-1, 2)
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
        self.gk = np.array([prop(G, c0) for G in G_ops]).reshape(-1, ndet)
        self.gdk = np.array([prop(G.dagger, c0) for G in G_ops]).reshape(-1, ndet)
        self.Hg = (H @ self.gk.T).T
        self.Hgd = (H @ self.gdk.T).T
        self.Gh = np.array([prop(G, h0) for G in G_ops]).reshape(-1, ndet)
        self.Gdh = np.array([prop(G.dagger, h0) for G in G_ops]).reshape(-1, ndet)
        self.DA, self.DB = self._cross_terms(G_ops) if symmetrize_qG else (np.zeros((self.nG, self.nq)),) * 2
        timing_line("Linear response setup", time.time() - t0)

    def _cross_terms(self, G_ops) -> tuple[np.ndarray, np.ndarray]:
        r"""Symmetrization corrections DA_Gq = 1/2<0|H[q, G^\dagger]|0>, DB_Gq = 1/2<0|H[q, G]|0>^* (zero for inactive-virtual q).

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
            return get_indexing_generalized(wf.num_inactive_spin_orbs, na, wf.num_virtual_spin_orbs, m, 0)

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
        gk, gdk, Hg, Hgd, Gh, Gdh = self.gk, self.gdk, self.Hg, self.Hgd, self.Gh, self.Gdh
        xc, yc = x.conj(), y.conj()
        Xc, Xdc, HXc, HXdc, Xh, Xdh = gk.T @ x, gdk.T @ xc, Hg.T @ x, Hgd.T @ xc, Gh.T @ x, Gdh.T @ xc
        s = gk.conj() @ HXc + gdk @ HXdc.conj()
        s -= 0.5 * (gk.conj() @ Xh + gdk @ Xdh.conj() + Gdh @ Xdc.conj() + Gh.conj() @ Xc)
        HYc, Yh, Ydh, HYdc = Hgd.T @ y, Gdh.T @ y, Gh.T @ yc, Hg.T @ yc
        s += gk.conj() @ (HYc - Yh) + gdk @ (HYdc - Ydh).conj()
        return s

    def _sgg(self, x):
        return self.gk.conj() @ (self.gk.T @ x) - self.gdk @ (self.gdk.T @ x.conj()).conj()

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
        gk, gdk, Hg, Hgd, Gh, Gdh = self.gk, self.gdk, self.Hg, self.Hgd, self.Gh, self.Gdh

        def dot(u, v):
            return np.einsum("md,md->m", u.conj(), v)

        a_G = dot(gk, Hg) + dot(Hgd, gdk) - 0.5 * (dot(gk, Gh) + dot(Gdh, gdk) + dot(gdk, Gdh) + dot(Gh, gk))
        s_G = dot(gk, gk) - dot(gdk, gdk)
        return np.concatenate((a_q, a_G.real)), np.concatenate((s_q, s_G.real))

    def explicit(self) -> tuple[np.ndarray, np.ndarray]:
        """Full E2 and S by applying sigma to unit vectors (testing/small systems only)."""
        return self.sigma(np.eye(2 * self.n, dtype=complex))

    def calc_excitation_energies(
        self, n_roots: int = 5, tol: float = 1e-6, max_iter: int = 100, max_space: int = 400, n_extra: int = 4
    ) -> np.ndarray:
        """Lowest positive excitation energies via paired Davidson (n_extra unconverged buffer roots, subspace <= max_space)."""
        t0 = time.time()
        self.excitation_energies, self.response_vectors = davidson_eig(
            self.sigma, *self.diagonal(), n_roots, tol=tol, max_iter=max_iter, max_space=max_space, n_extra=n_extra
        )
        roots_table("Excitation energies (linear response)", self.excitation_energies)
        timing_line("Excitation energies", time.time() - t0)
        return self.excitation_energies

    def solve_response(self, V: np.ndarray, omega: float = 0.0, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
        """Solve (E2 - omega S) X = V for all columns of V (one shared subspace)."""
        return davidson_solve(self.sigma, *self.diagonal(), V, omega, tol=tol, max_iter=max_iter)

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
        """Property gradients of Hermitian one-electron AO operators (..., nao, nao) or (..., 2nao, 2nao)."""
        ao_ints = self._spin_ao(ao_ints)
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

    def _spin_ao(self, ao_ints) -> np.ndarray:
        ao_ints = np.asarray(ao_ints)
        nao = self.wf.int_gen.int_obj.nao
        if ao_ints.shape[-1] == nao:
            z = np.zeros_like(ao_ints)
            ao_ints = np.concatenate(
                (np.concatenate((ao_ints, z), axis=-1), np.concatenate((z, ao_ints), axis=-1)), axis=-2
            )
        return ao_ints

    def get_transition_dipole(self, dipole_integrals=None, x2c: bool = False) -> np.ndarray:
        """Transition dipoles <0|mu|n> (up to a phase) of the states from calc_excitation_energies, shape (n_roots, 3).

        dipole_integrals: (3, nao, nao) or (3, 2nao, 2nao) AO integrals, default int1e_r (origin (0,0,0)).
        x2c is not used (kept for API compatibility); pass picture-change transformed integrals if needed.
        """
        if dipole_integrals is None:
            dipole_integrals = self.wf.int_gen.int_obj.intor("int1e_r", comp=3)
        V = self.property_gradient(dipole_integrals)
        self.transition_dipoles = V.T @ self.response_vectors
        self.transition_dipoles = self.transition_dipoles.T
        return self.transition_dipoles

    def get_oscillator_strength(self, dipole_integrals=None, x2c: bool = False) -> np.ndarray:
        r"""f_n = 2/3 omega_n |<0|mu|n>|^2."""
        t0 = time.time()
        t = self.get_transition_dipole(dipole_integrals, x2c)
        self.oscillator_strengths = 2 / 3 * self.excitation_energies * np.sum(np.abs(t) ** 2, axis=1)
        timing_line("Oscillator strengths", time.time() - t0)
        return self.oscillator_strengths

    def get_formatted_oscillator_strength(self) -> str:
        output = f" {'Root':>4}  {'omega [Eh]':>14}  {'omega [eV]':>12}  {'f':>12}\n"
        for i, (e, f) in enumerate(zip(self.excitation_energies, self.oscillator_strengths)):
            output += f" {i + 1:>4d}  {e:>14.8f}  {e * HARTREE_EV:>12.6f}  {f:>12.6f}\n"
        return output

    def get_polarizability(self, omega: float = 0.0, dipole_integrals=None, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
        r"""Dipole polarizability alpha_ab(omega) = <<mu_a; mu_b>>_omega (3 x 3).

        Uses the error-cancelling expression V_a^H X_b + X_a^H V_b - X_a^H (E2 - omega S) X_b (error quadratic in the residual).
        """
        t0 = time.time()
        if dipole_integrals is None:
            dipole_integrals = self.wf.int_gen.int_obj.intor("int1e_r", comp=3)
        V = self.property_gradient(dipole_integrals)
        X, EX = davidson_solve(self.sigma, *self.diagonal(), V, omega, tol=tol, max_iter=max_iter, return_ex=True)
        self.polarizability = (V.conj().T @ X + X.conj().T @ V - X.conj().T @ EX).real
        value_table(
            f"Polarizability tensor, omega = {omega} Eh [a.u.]",
            [("", 3, ""), ("x", 14, ".6f"), ("y", 14, ".6f"), ("z", 14, ".6f")],
            [(c, *r) for c, r in zip("xyz", self.polarizability)],
            f"isotropic alpha = {np.trace(self.polarizability) / 3:.6f} a.u.",
        )
        timing_line("Polarizability", time.time() - t0)
        return self.polarizability
