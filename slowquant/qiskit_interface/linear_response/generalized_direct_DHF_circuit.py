import time

import numpy as np

from slowquant.qiskit_interface.fast_pauli import (
    _jw_ladder_product,
    commutator_terms,
    dagger_coefficients,
    double_commutator_terms,
    fermionic_to_xz,
    pauli_product,
)
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct_DHF import LinearResponseDirect
from slowquant.unitary_coupled_cluster.operators import G1, G2
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2


def _commutator_xz(ka, ca, kb, cb, nq):
    """Unique keys and coefficients of [A, B] for Pauli sums A, B (general, also for odd fermionic operators)."""
    k, s1 = pauli_product(ka[:, None], kb[None, :], nq)
    _, s2 = pauli_product(kb[None, :], ka[:, None], nq)
    c = ((s1 - s2) * ca[:, None] * cb[None, :]).ravel()
    uk, inv = np.unique(k.ravel(), return_inverse=True)
    uc = np.zeros(len(uk), complex)
    np.add.at(uc, inv, c)
    keep = np.abs(uc) > 1e-14
    return uk[keep], uc[keep]


class LinearResponseCircuitDirect(LinearResponseDirect):
    r"""Integral-direct naive linear response for GeneralizedWaveFunctionCircuitDHF, quantum-hardware compatible.

    Same E2/S and property expressions as LinearResponseDirect. Everything taken from the device is a Pauli
    expectation value of the reference circuit, measured in one batch:
    RDM1/RDM2, K1[j,t,u] = <[E_tu, G_j]>, K2[j,t,u,v,w] = <[e_tuvw, G_j]> (at most 3-body) and the GG matrices
    A_kl = <[G_k^+, [H, G_l]]> (symmetrized), B_kl = <[G_k^+, [H, G_l^+]]>, S_kl = <[G_k^+, G_l]> (4-body).
    G_j^+ quantities follow from <[O, G^+]> = -<[O^+, G]>^*.
    qG blocks plain by default; symmetrize_qG=True adds the symmetrization (see LinearResponseDirect._cross_terms) from the measured
    <a^+_u Y>, <a^+_u a^+_w a_x Y> (Y = [a_t, G^+], [a_t, G]) and <a_u Z>, <a^+_w a_x a_u Z> (Z = [a^+_t, G^+], [a^+_t, G]).
    """

    def __init__(
        self,
        wave_function,
        excitations: str = "SD",
        screen: float = 1e-8,
        chunk: int = 2_000_000,
        symmetrize_qG: bool = False,
    ) -> None:
        t0 = time.time()
        wf = self.wf = wave_function
        if not wf.pmap.fast:
            raise ValueError("LinearResponseCircuitDirect requires qiskit's JordanWignerMapper.")
        excitations = excitations.lower()
        occ, unocc = wf.active_occ_spin_idx, wf.active_unocc_spin_idx
        G_ops = []
        if "s" in excitations:
            G_ops += [G1(i, a) for a, i in iterate_t1(occ, unocc, is_spin_conserving=False)]
        if "d" in excitations:
            G_ops += [G2(i, j, a, b) for a, i, b, j in iterate_t2(occ, unocc, is_spin_conserving=False)]
        self.G_ops = G_ops
        ds = self.ds = wf.ds
        ds.set_rdms(wf.rdm1, wf.rdm2)
        ds.response_tensors()
        print(f"Response integrals: {time.time() - t0:.1f} s")
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
        print(f"Operators: {self.nq} orbital, {self.nG} active-space")
        self._chunk = chunk
        shift = wf.active_spin_idx[0]
        self.pairs = (
            [(k, t) for k, G in enumerate(G_ops) for t in sorted({i - shift for key in G.operators for i, _ in key})]
            if symmetrize_qG and self.nq
            else []
        )
        self._measure_active_space()
        self.DA, self.DB = self._cross_terms_measured()
        print(f"Linear response setup: {time.time() - t0:.1f} s")

    def _terms(self):
        """Yield (target, flat index, keys, coefficients): target.flat[index] += coefficients * <keys>."""
        wf, pm = self.wf, self.wf.pmap
        nq, na, nG = pm.nq, pm.na, self.nG
        shift = wf.active_spin_idx[0]
        G = [fermionic_to_xz(op, pm.num_orbs, nq, shift) for op in self.G_ops]
        Gd = [(k, dagger_coefficients(k, c, nq)) for k, c in G]
        r1, c1, v1, _ = pm.A1
        r2, c2, v2, _ = pm.A2
        k1, k2 = pm.keys[c1], pm.keys[c2]
        n2 = len(pm.idx2)
        _, h, g = self.ds.active_integrals()
        hc = pm.hamiltonian_coefficients(0.0, h, g)
        nz = np.abs(hc) > 1e-12
        kh, ch = pm.keys[nz], hc[nz]
        step2 = max(1, self._chunk // 16)
        for j, (kg, cg) in enumerate(G):
            ia, _, k, c = commutator_terms(k1, v1, kg, cg, nq)
            yield "K1", j * na * na + r1[ia], k, c
            for s0 in range(0, len(k2), step2):
                sl = slice(s0, s0 + step2)
                ia, _, k, c = commutator_terms(k2[sl], v2[sl], kg, cg, nq)
                yield "K2", j * n2 + r2[sl][ia], k, c
        num_orbs = pm.num_orbs
        I = np.arange(na)
        u3 = np.array(np.meshgrid(I, I, I, indexing="ij")).reshape(3, -1).T
        bras = {}
        for name, idx, dag in (
            ("L1", I[:, None], (True,)),
            ("L2", u3, (True, True, False)),
            ("M1", I[:, None], (False,)),
            ("M2", u3[:, [1, 2, 0]], (True, False, False)),
        ):
            x, z, c = _jw_ladder_product(idx, dag, num_orbs)
            bras[name] = ((x * (1 << nq) + z), c)
        stepb = max(1, self._chunk // 64)
        for j, (l, t) in enumerate(self.pairs):
            T = t + shift
            Gdl = self.G_ops[l].dagger
            for dag, names in ((False, ("L1", "L2")), (True, ("M1", "M2"))):
                xt, zt, ct = _jw_ladder_product(np.array([[t]]), (dag,), num_orbs)
                kt, ct = (xt * (1 << nq) + zt).ravel(), ct.ravel()
                for suffix, op in (("A", Gdl), ("B", self.G_ops[l])):
                    ky, cy = _commutator_xz(kt, ct, *fermionic_to_xz(op, num_orbs, nq, shift), nq)
                    if len(ky) == 0:
                        continue
                    for name in names:
                        kb, cb = bras[name]
                        nb = len(kb)
                        for b0 in range(0, nb, stepb):
                            kbs, cbs = kb[b0 : b0 + stepb], cb[b0 : b0 + stepb]
                            k, sg = pauli_product(kbs[:, :, None], ky[None, None, :], nq)
                            c = sg * cbs[:, :, None] * cy[None, None, :]
                            rows = np.broadcast_to((j * nb + b0 + np.arange(len(kbs)))[:, None, None], k.shape)
                            yield name + suffix, rows.ravel(), k.ravel(), c.ravel()
        ka = np.concatenate([k for k, _ in Gd])
        ca = np.concatenate([c for _, c in Gd])
        rows = np.repeat(np.arange(nG), [len(k) for k, _ in Gd])
        stepa = max(1, self._chunk // max(1, 8 * len(kh)))
        for l in range(nG):
            ia, _, k, c = commutator_terms(ka, ca, *G[l], nq)
            yield "S", rows[ia] * nG + l, k, c
            for name, (kb, cb) in (("A", G[l]), ("B", Gd[l])):
                for a0 in range(0, len(ka), stepa):
                    a = slice(a0, a0 + stepa)
                    ia, _, k, c = double_commutator_terms(
                        ka[a], ca[a], kh, ch, kb, cb, nq, symmetrized=name == "A"
                    )
                    yield name, rows[a][ia] * nG + l, k, c

    def _measure_active_space(self):
        t0 = time.time()
        na, nG = self.wf.pmap.na, self.nG
        out = {
            "K1": np.zeros(nG * na * na, complex),
            "K2": np.zeros(nG * len(self.wf.pmap.idx2), complex),
            "S": np.zeros(nG * nG, complex),
            "A": np.zeros(nG * nG, complex),
            "B": np.zeros(nG * nG, complex),
        }
        npair = len(self.pairs)
        for s_ in "AB":
            out["L1" + s_] = np.zeros(npair * na, complex)
            out["M1" + s_] = np.zeros(npair * na, complex)
            out["L2" + s_] = np.zeros(npair * na**3, complex)
            out["M2" + s_] = np.zeros(npair * na**3, complex)

        def add(name, idx, v):
            o = out[name]
            o += np.bincount(idx, weights=v.real, minlength=len(o)) + 1j * np.bincount(
                idx, weights=v.imag, minlength=len(o)
            )

        est = self.wf.estimator
        if est.exact:
            psi = est.statevector()
            for name, idx, k, c in self._terms():
                add(name, idx, c * est._exact(k, psi))
        else:
            keys = [np.zeros(0, np.int64)]
            acc = 0
            for _, _, k, _ in self._terms():
                keys.append(np.unique(k))
                acc += len(k)
                if acc > 20 * self._chunk:
                    keys = [np.unique(np.concatenate(keys))]
                    acc = 0
            keys = np.unique(np.concatenate(keys))
            vals = est(keys)
            print(f"Active-space LR quantities: {len(keys)} Paulis measured ({time.time() - t0:.1f} s)")
            for name, idx, k, c in self._terms():
                add(name, idx, c * vals[np.searchsorted(keys, k)])
        self.K1 = out["K1"].reshape(nG, na, na)
        t, u, v, w = self.wf.pmap.idx2.T
        K2 = np.zeros((nG, na, na, na, na), complex)
        k2 = out["K2"].reshape(nG, -1)
        K2[:, t, u, v, w] = k2
        K2[:, v, w, t, u] = k2
        K2[:, v, u, t, w] = -k2
        K2[:, t, w, v, u] = -k2
        self.K2 = K2
        self.S_GG, self.A_GG, self.B_GG = (out[x].reshape(nG, nG) for x in ("S", "A", "B"))
        npair = len(self.pairs)
        self._ext = {k: out[k].reshape(npair, -1) for k in out if k[0] in "LM"} if npair else {}
        print(f"Active-space LR setup: {time.time() - t0:.1f} s")

    def _cross_terms_measured(self) -> tuple[np.ndarray, np.ndarray]:
        """DA_Gq = 1/2<0|H[q, G^+]|0>, DB_Gq = 1/2<0|H[q, G]|0>^* from the measured extended transition tensors."""
        ds, nG, nq = self.ds, self.nG, self.nq
        DA = np.zeros((nG, nq), dtype=complex)
        DB = np.zeros((nG, nq), dtype=complex)
        if nG == 0 or nq == 0:
            return DA, DB
        na = len(ds.A)
        loc = -np.ones(self.N, dtype=int)
        loc[ds.A] = np.arange(na)
        P, Q = self.q_idx.T
        inact = np.zeros(self.N, dtype=bool)
        inact[ds.I] = True
        tv = np.where((loc[P] >= 0) & (loc[Q] < 0) & ~inact[Q])[0]
        it = np.where(inact[P] & (loc[Q] >= 0))[0]
        g, FI, A, E = ds.g_paaa, ds.FI, ds.A, self._ext
        for j, (k, t) in enumerate(self.pairs):
            sel = tv[loc[P[tv]] == t]
            if len(sel):
                v = Q[sel]
                W1 = FI[np.ix_(A, v)].T
                W2 = g[v].conj().transpose(0, 1, 3, 2).reshape(len(v), -1)
                DA[k, sel] = W1 @ E["L1A"][j] + W2 @ E["L2A"][j]
                DB[k, sel] = (W1 @ E["L1B"][j] + W2 @ E["L2B"][j]).conj()
            sel = it[loc[Q[it]] == t]
            if len(sel):
                i = P[sel]
                W1 = FI[np.ix_(i, A)]
                W2 = g[i].reshape(len(i), -1)
                DA[k, sel] = W1 @ E["M1A"][j] + W2 @ E["M2A"][j]
                DB[k, sel] = (W1 @ E["M1B"][j] + W2 @ E["M2B"][j]).conj()
        return 0.5 * DA, 0.5 * DB

    def _km(self, K1, K2):
        """<[O, G^+]> tensors from <[O, G]>: K1m[j,t,u] = -conj(K1[j,u,t]), K2m[j,t,u,v,w] = -conj(K2[j,u,t,w,v])."""
        return -K1.conj().transpose(0, 2, 1), -K2.conj().transpose(0, 2, 1, 4, 3)

    def _comm_G(self, ops):
        """(<[G_k^+, O]>, <[G_k, O]>) for operators O = sum f_tu E_tu + 1/2 sum g_tuvw e_tuvw, shape (nG, m)."""
        K1m, K2m = self._km(self.K1, self.K2)
        nG = self.nG
        F = np.array([f for f, _ in ops]).reshape(len(ops), -1)
        Gm = 0.5 * np.array([g for _, g in ops]).reshape(len(ops), -1)
        gd = -(K1m.reshape(nG, -1) @ F.T + K2m.reshape(nG, -1) @ Gm.T)
        g = -(self.K1.reshape(nG, -1) @ F.T + self.K2.reshape(nG, -1) @ Gm.T)
        return gd, g

    def sigma(self, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """E2 @ V and S @ V for trial vectors V of shape (2n, m). One J/K pass with 2 densities per vector."""
        ds = self.ds
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
                K1m, K2m = self._km(self.K1, self.K2)
                d1 = np.einsum("jm,jtu->mtu", xG, self.K1) + np.einsum("jm,jtu->mtu", yG, K1m)
                G1_ = np.einsum("jm,jtuvw->mtuvw", xG, self.K2) + np.einsum("jm,jtuvw->mtuvw", yG, K2m)
                d2 = -np.einsum("jm,jtu->mtu", xG.conj(), K1m) - np.einsum("jm,jtu->mtu", yG.conj(), self.K1)
                G2_ = -np.einsum("jm,jtuvw->mtuvw", xG.conj(), K2m) - np.einsum(
                    "jm,jtuvw->mtuvw", yG.conj(), self.K2
                )
            ks, dms = [], []
            for j in range(m):
                k = np.zeros((N, N), dtype=complex)
                k[Q, P] = x[:nq, j]
                k[P, Q] += y[:nq, j]
                ks.append(k)
                dI, dIA = ds.transformed_densities(k, -k)
                dc = -dIA
                if nG:
                    dc = dc + ds.transition_density(d1[j])
                dms += [dI, dc]
            v = ds.direct.veff(np.array(dms), response=True)
            ops1 = []
            for j, k in enumerate(ks):
                vI, vc = v[2 * j], v[2 * j + 1]
                kd = k.conj().T
                F1, f1, g1 = ds.transformed_operator(k, -k, vI)
                F2, _, _ = ds.transformed_operator(kd, -kd, -vI.conj().T)
                M1, M2 = -F1, F2
                if nG:
                    M1 = M1 + ds.transition_fock((0.0, d1[j], G1_[j]))
                    M2 = M2 + ds.transition_fock((0.0, d2[j], G2_[j]))
                M1[:, : len(I)] += Cd @ vc @ CI
                M2[:, : len(I)] += Cd @ vc.conj().T @ CI
                E2V[:nq, j] = self._fget(M1, Q, P) - self._fget(M2, P, Q).conj()
                E2V[n : n + nq, j] = self._fget(M1, P, Q) - self._fget(M2, Q, P).conj()
                ops1.append((f1, g1))
                X = np.zeros((N, N), dtype=complex)
                X[P, Q] = x[:nq, j]
                SV[:nq, j] = self._metric_q(X)[P, Q]
                X[P, Q] = y[:nq, j].conj()
                SV[n : n + nq, j] = -self._metric_q(X)[P, Q].conj()
            if nG:
                cd, c = self._comm_G(ops1)
                E2V[nq:n] = -cd
                E2V[n + nq :] = -c
        if nG and nq:
            DA, DB, xq, yq = self.DA, self.DB, x[:nq], y[:nq]
            E2V[:nq] += DA.conj().T @ xG + DB.T @ yG
            E2V[nq:n] += DA @ xq + DB @ yq
            E2V[n : n + nq] += DB.conj().T @ xG + DA.T @ yG
            E2V[n + nq :] += DB.conj() @ xq + DA.conj() @ yq
        if nG:
            E2V[nq:n] += self.A_GG @ xG + self.B_GG @ yG
            E2V[n + nq :] += self.B_GG.conj() @ xG + self.A_GG.conj() @ yG
            SV[nq:n] = self.S_GG @ xG
            SV[n + nq :] = -self.S_GG.conj() @ yG
        return E2V, SV

    def diagonal(self) -> tuple[np.ndarray, np.ndarray]:
        a_q, s_q = self._diag_q()
        return np.concatenate((a_q, self.A_GG.diagonal().real)), np.concatenate(
            (s_q, self.S_GG.diagonal().real)
        )

    def operator_gradient(self, ops) -> np.ndarray:
        """Property gradients of Hermitian 1+2-electron operators given as (F[:, occ], F^I[A, A], g[A, A, A, A])."""
        n, nq = self.n, self.nq
        P, Q = self.q_idx.T
        out = np.zeros((2 * n, len(ops)), dtype=complex)
        for c, (F, _, _) in enumerate(ops):
            out[:nq, c] = self._fget(F, Q, P) - self._fget(F, P, Q).conj()
            out[n : n + nq, c] = self._fget(F, P, Q) - self._fget(F, Q, P).conj()
        if self.nG:
            cd, cg = self._comm_G([(f, g) for _, f, g in ops])
            out[nq:n] = cd
            out[n + nq :] = cg
        return out
