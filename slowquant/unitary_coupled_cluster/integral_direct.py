import os
import tempfile
import time

import h5py
import numpy as np
import scipy
from pyscf import ao2mo, lib, scf
from pyscf.scf import ghf, hf

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import a_op_spin


class GHFDirect:
    """Integral-direct GHF (spin-orbital) building blocks with the scalar Coulomb operator (PySCF backend).

    AO basis: 2 nao functions ordered (alpha AOs, beta AOs). h_ao: 2nao x 2nao one-electron Hamiltonian
    (non-relativistic, X2C, ECP or finite field); default T + V.
    Conventions: (pq|rs) = sum C*_mp C_nq C*_ls C_sr (mn|ls), J[D]_pq = sum (pq|rs) D_sr, K[D]_pq = sum (ps|rq) D_sr.
    screen_tol: Schwarz/density screening threshold of the J/K builds (PySCF direct_scf_tol).
    incore_memory: J/K and MO transformations use in-memory 8-fold AO integrals if they fit in this many MB
    (0: always integral-direct J/K and out-of-core MO transformations).
    """

    def __init__(
        self,
        mol,
        h_ao: np.ndarray | None = None,
        max_memory: float | None = None,
        tmpdir: str | None = None,
        screen_tol: float = 1e-13,
        incore_memory: float = 256.0,
    ):
        self.mol = mol
        self.nao = mol.nao
        if h_ao is None:
            h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            h_ao = scipy.linalg.block_diag(h, h)
        self.h_ao = np.asarray(h_ao, dtype=complex)
        mf = scf.RHF(mol)
        mf.direct_scf_tol = screen_tol
        self.screen_tol = screen_tol
        self.opt = mf.init_direct_scf(mol)
        self.max_memory = mol.max_memory if max_memory is None else max_memory
        self.incore_memory = incore_memory
        self._eri_ao = None
        self.tmpdir = lib.param.TMPDIR if tmpdir is None else tmpdir

    def describe(self) -> str:
        mode = "in-core AO integrals" if self._incore() else f"direct, screen_tol = {self.screen_tol:.0e}"
        return f"GHFDirect (exact), {mode}"

    def _jk_exact(self, dms):
        def build(mol, dm, hermi, with_j, with_k, omega=None):
            if self._incore():
                return hf.dot_eri_dm(self._eri_ao, dm, hermi, with_j, with_k)
            return hf.get_jk(mol, dm, hermi, self.opt, with_j, with_k, omega)

        return ghf.get_jk(self.mol, dms, hermi=0, jkbuild=build)

    def jk(self, dms: np.ndarray, response: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """J and K for a batch of general (non-Hermitian, complex) AO densities, one integral pass."""
        return self._jk_exact(np.asarray(dms, dtype=complex))

    def veff(self, dms: np.ndarray, response: bool = False) -> np.ndarray:
        vj, vk = self.jk(dms, response)
        return vj - vk

    def eri(self, c1, c2, c3, c4) -> np.ndarray:
        """(12|34) in the MO sets c1..c4, shape (n1, n2, n3, n4). Out-of-core real ao2mo per spin pair, accumulated in place."""
        return self._eri_exact(c1, c2, c3, c4)

    def _eri_exact(self, c1, c2, c3, c4):
        n = self.nao
        n1, n2, n3, n4 = shape = (c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1])
        out = np.zeros((n1 * n2, n3 * n4), dtype=complex)
        ncol = 4 * n3 * n4
        blk = max(1, int(self.max_memory * 1e6 / 8 / 16 / max(ncol * 2 * n2, 1)))
        for s in (0, n):
            for t in (0, n):
                cs = [np.hstack((c[o : o + n].real, c[o : o + n].imag)) for c, o in zip((c1, c2, c3, c4), (s, s, t, t))]
                if self._incore():
                    self._accumulate(ao2mo.incore.general(self._eri_ao, cs, compact=False), out, shape, blk)
                    continue
                fd, fname = tempfile.mkstemp(suffix=".h5", dir=self.tmpdir)
                os.close(fd)
                try:
                    ao2mo.outcore.general(
                        self.mol, cs, fname, "eri", intor="int2e", compact=False, max_memory=self.max_memory, verbose=0
                    )
                    with h5py.File(fname, "r") as f:
                        self._accumulate(f["eri"], out, shape, blk)
                finally:
                    os.remove(fname)
        return out.reshape(shape)

    def _incore(self) -> bool:
        """Keep the 8-fold symmetric AO integrals in memory when they need less than incore_memory MB."""
        if self._eri_ao is None:
            npair = self.nao * (self.nao + 1) // 2
            if npair * (npair + 1) // 2 * 8e-6 > self.incore_memory:
                return False
            self._eri_ao = self.mol.intor("int2e", aosym="s8")
        return True

    @staticmethod
    def _accumulate(d, out, shape, blk):
        n1, n2, n3, n4 = shape
        for p0 in range(0, n1, blk):
            p1 = min(n1, p0 + blk)
            ra = d[p0 * 2 * n2 : p1 * 2 * n2].reshape(p1 - p0, 2 * n2, -1)
            rb = d[(n1 + p0) * 2 * n2 : (n1 + p1) * 2 * n2].reshape(p1 - p0, 2 * n2, -1)
            x = ra[:, :n2] + rb[:, n2:] + 1j * (ra[:, n2:] - rb[:, :n2])
            x = x.reshape(p1 - p0, n2, 2 * n3, 2 * n4)
            y = x[:, :, :n3, :n4] + x[:, :, n3:, n4:] + 1j * (x[:, :, :n3, n4:] - x[:, :, n3:, :n4])
            out[p0 * n2 : p1 * n2] += y.reshape(-1, n3 * n4)


class GHFCholesky(GHFDirect):
    r"""Pivoted Cholesky decomposition of the scalar Coulomb operator, drop-in for GHFDirect.

    (mn|ls) ~ sum_K V_K[m, n] V_K[l, s] with real symmetric V_K. cd_jk=False: exact (direct) J/K, Cholesky vectors
    only for the MO integrals (eri). tol: largest neglected residual diagonal. span: pivots inside one shell-pair
    column block are taken while diag > span * (largest diag at block start).
    """

    def __init__(self, mol, tol: float = 1e-6, span: float = 1e-2, cd_jk: bool = True, **kwargs):
        super().__init__(mol, **kwargs)
        self.cd_jk = cd_jk
        self.tol = tol
        t0 = time.time()
        self.V = self._decompose(tol, span)
        print(f"Cholesky decomposition: {len(self.V)} vectors (tol {tol:.0e}), {time.time() - t0:.1f} s")

    def describe(self) -> str:
        jk = "CD" if self.cd_jk else ("exact, in-core" if self._incore() else f"exact, screen_tol = {self.screen_tol:.0e}")
        return f"GHFCholesky, tol = {self.tol:.0e}, {len(self.V)} vectors, J/K: {jk}"

    def _decompose(self, tol, span):
        mol, n = self.mol, self.nao
        loc = mol.ao_loc_nr()
        nb = mol.nbas
        shell = np.repeat(np.arange(nb), np.diff(loc))
        d = np.zeros((n, n))
        for i in range(nb):
            for j in range(nb):
                v = mol.intor("int2e", shls_slice=(i, i + 1, j, j + 1, i, i + 1, j, j + 1))
                d[loc[i] : loc[i + 1], loc[j] : loc[j + 1]] = np.einsum("abab->ab", v)
        d = d.ravel()
        npair, cs = n * n, 256
        L = []
        k = 0
        while True:
            p = int(np.argmax(d))
            dmax = d[p]
            if dmax < tol:
                break
            a, b = divmod(p, n)
            ci, cj = shell[a], shell[b]
            ra, rb = np.arange(loc[ci], loc[ci + 1]), np.arange(loc[cj], loc[cj + 1])
            cols = (ra[:, None] * n + rb[None, :]).ravel()
            keep = d[cols] >= max(tol, span * dmax)
            cols = cols[keep]
            G = mol.intor("int2e", shls_slice=(0, nb, 0, nb, ci, ci + 1, cj, cj + 1)).reshape(npair, -1)[:, keep]
            for i, Lc in enumerate(L):
                m = min(cs, k - i * cs)
                G -= Lc[:m].T @ Lc[:m, cols]
            while True:
                dc = G[cols, np.arange(len(cols))]
                j = int(np.argmax(dc))
                if dc[j] < max(tol, span * dmax):
                    break
                v = G[:, j] / np.sqrt(dc[j])
                if k % cs == 0:
                    L.append(np.empty((cs, npair)))
                L[-1][k % cs] = v
                k += 1
                G -= np.outer(v, v[cols])
                d -= v**2
                d[cols[j]] = 0.0
                np.maximum(d, 0.0, out=d)
        V = np.empty((k, npair))
        for i in range(len(L)):
            m = min(cs, k - i * cs)
            V[i * cs : i * cs + m] = L[i][:m]
            L[i] = None
        return V.reshape(k, n, n)

    def _jk_cd(self, dms):
        n, V = self.nao, self.V
        nK, nd = len(V), len(dms)
        vj = np.zeros_like(dms)
        vk = np.zeros_like(dms)
        if nK == 0:
            return vj, vk
        Vf = V.reshape(nK, -1)
        dt = (dms[:, :n, :n] + dms[:, n:, n:]).transpose(0, 2, 1).reshape(nd, -1)
        j = ((Vf @ dt.T).T @ Vf).reshape(nd, n, n).transpose(0, 2, 1)
        vj[:, :n, :n] = j
        vj[:, n:, n:] = j
        chunk = max(1, int(self.max_memory * 1e6 / 16 / (2 * n * n)))
        for k0 in range(0, nK, chunk):
            Vc = V[k0 : k0 + chunk]
            m = len(Vc)
            Vh = Vc.transpose(1, 0, 2).reshape(n, m * n)
            Vv = Vc.reshape(m * n, n)
            for i in range(nd):
                for s in (0, n):
                    for t in (0, n):
                        A = (dms[i, s : s + n, t : t + n] @ Vh).reshape(n, m, n).transpose(1, 0, 2).reshape(m * n, n)
                        vk[i, s : s + n, t : t + n] += Vv.T @ A
        return vj, vk

    def jk(self, dms: np.ndarray, response: bool = False) -> tuple[np.ndarray, np.ndarray]:
        dms = np.asarray(dms, dtype=complex)
        return self._jk_cd(dms) if self.cd_jk else self._jk_exact(dms)

    def eri(self, c1, c2, c3, c4) -> np.ndarray:
        n, V = self.nao, self.V
        nK = len(V)
        shape = (c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1])
        out = np.zeros((shape[0] * shape[1], shape[2] * shape[3]), dtype=complex)
        if nK == 0:
            return out.reshape(shape)
        per = 16 * (n * n + n * max(shape) + shape[0] * shape[1] + shape[2] * shape[3])
        chunk = max(1, int(self.max_memory * 1e6 / 4 / per))
        for k0 in range(0, nK, chunk):
            Vc = V[k0 : k0 + chunk]
            m = len(Vc)
            B12 = sum(c1[s : s + n].conj().T @ (Vc @ c2[s : s + n]) for s in (0, n))
            B34 = sum(c3[s : s + n].conj().T @ (Vc @ c4[s : s + n]) for s in (0, n))
            out += B12.reshape(m, -1).T @ B34.reshape(m, -1)
        return out.reshape(shape)


class DirectState:
    """Integral-direct quantities for one set of MO coefficients C.

    inactive/active are absolute MO index lists. With rdm1 given, F^I and F^A come from one J/K pass.
    """

    def __init__(self, direct: GHFDirect, C: np.ndarray, inactive, active, rdm1: np.ndarray | None = None):
        self.direct = direct
        self.C = C
        self.I = np.array(inactive, dtype=int)
        self.A = np.array(active, dtype=int)
        self.occ = np.concatenate((self.I, self.A))
        CI, CA = C[:, self.I], C[:, self.A]
        Cd = C.conj().T
        self.dI = CI @ CI.conj().T
        dms = [self.dI]
        if rdm1 is not None:
            dms.append(CA @ rdm1.T @ CA.conj().T)
        v = direct.veff(np.array(dms))
        self.e_core = np.einsum("ij,ji->", direct.h_ao + 0.5 * v[0], self.dI).real
        self.h = Cd @ direct.h_ao @ C
        self.FI = self.h + Cd @ v[0] @ C
        self._rdm1 = None
        if rdm1 is not None:
            self.FA = Cd @ v[1] @ C
            self._rdm1 = rdm1.copy()
        self.g_paaa = direct.eri(CA, CA, C, CA).transpose(2, 3, 0, 1)
        self.g_aaaa = self.g_paaa[self.A]
        self._H = None
        self._Hci = None
        self._T = None

    def set_rdms(self, rdm1: np.ndarray, rdm2: np.ndarray) -> None:
        if self._rdm1 is None or not np.array_equal(rdm1, self._rdm1):
            CA = self.C[:, self.A]
            self.FA = self.C.conj().T @ self.direct.veff((CA @ rdm1.T @ CA.conj().T)[None])[0] @ self.C
            self._rdm1 = rdm1.copy()
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.Q = np.einsum("nuvw,tuvw->nt", self.g_paaa, rdm2)

    def energy(self, rdm1: np.ndarray, rdm2: np.ndarray) -> float:
        ia = np.ix_(self.A, self.A)
        return (
            self.e_core
            + np.einsum("tu,tu->", self.FI[ia], rdm1)
            + 0.5 * np.einsum("tuvw,tuvw->", self.g_aaaa, rdm2)
        ).real

    def gen_fock(self) -> np.ndarray:
        r"""F_{NM} = \sum_P h_{NP}D_{MP} + \sum_{PQR} g_{NPQR}\Gamma_{MPQR}, full N x N (zero for unoccupied M)."""
        F = np.zeros_like(self.FI)
        F[:, self.I] = self.FI[:, self.I] + self.FA[:, self.I]
        F[:, self.A] = self.FI[:, self.A] @ self.rdm1.T + self.Q
        return F

    def active_integrals(self) -> tuple[float, np.ndarray, np.ndarray]:
        return self.e_core, self.FI[np.ix_(self.A, self.A)], self.g_aaaa

    def ci_hamiltonian(self, ci):
        """Sparse active-space CI Hamiltonian (incl. E_core) for an ActiveCI engine."""
        if self._Hci is None:
            self._Hci = ci.hamiltonian(*self.active_integrals())
        return self._Hci

    def hamiltonian(self) -> FermionicOperator:
        """Active-space Hamiltonian as FermionicOperator (kept for compatibility; use ActiveCI for speed)."""
        if self._H is None:
            self._H = active_hamiltonian(self.e_core, self.FI[np.ix_(self.A, self.A)], self.g_aaaa, self.A)
        return self._H

    def response_tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """T0 = (N x|v w), T1 = (N u|x w), T2 = (N u|v x) for all MOs N, x and active u, v, w (three ao2mo passes)."""
        if self._T is None:
            C, CA = self.C, self.C[:, self.A]
            self._T = (
                np.ascontiguousarray(self.direct.eri(CA, CA, C, C).transpose(2, 3, 0, 1)),
                self.direct.eri(C, CA, C, CA),
                self.direct.eri(C, CA, CA, C),
            )
        return self._T

    def transformed_densities(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""AO densities (D_I, D_I + D_A) whose J - K give the 2e parts of the one-index transformed F^I and F^I + F^A.

        One-index transformation: h -> A h + h B, g_pqrs -> A on p and r, B on q and s.
        """
        C, CI, CA = self.C, self.C[:, self.I], self.C[:, self.A]
        Cd, r1t = C.conj().T, self._rdm1.T
        dI = CI @ (A[self.I] @ Cd) + (C @ B[:, self.I]) @ CI.conj().T
        dA = CA @ r1t @ (A[self.A] @ Cd) + (C @ B[:, self.A]) @ r1t @ CA.conj().T
        return dI, dI + dA

    def transformed_operator(self, A, B, vI, vIA=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generalized Fock F[:, occ], F^I[A, A] and g[A, A, A, A] of the one-index transformed Hamiltonian.

        vI, vIA: J - K of transformed_densities(A, B). With vIA=None the J - K part of the inactive
        columns of F is left out (to be added by the caller). Requires set_rdms.
        """
        C, I, Ai = self.C, self.I, self.A
        nI, na, N = len(I), len(Ai), C.shape[1]
        T0, T1, T2 = self.response_tensors()
        Cd, r1t, G2 = C.conj().T, self.rdm1.T, self.rdm2
        AA, BA = A[Ai], B[:, Ai]
        FIt = np.empty((N, nI + na), dtype=complex)
        FIt[:, nI:] = A @ self.FI[:, Ai] + self.FI @ B[:, Ai] + Cd @ vI @ C[:, Ai]
        FIt[:, :nI] = A @ (self.FI[:, I] + self.FA[:, I]) + (self.FI + self.FA) @ B[:, I]
        if vIA is not None:
            FIt[:, :nI] += Cd @ vIA @ C[:, I]
        Q = A @ self.Q
        Q += T0.reshape(N, -1) @ np.einsum("xu,tuvw->xvwt", BA, G2).reshape(-1, na)
        Q += T1.reshape(N, -1) @ np.einsum("vx,tuvw->uxwt", AA, G2).reshape(-1, na)
        Q += T2.reshape(N, -1) @ np.einsum("xw,tuvw->uvxt", BA, G2).reshape(-1, na)
        F = np.empty((N, nI + na), dtype=complex)
        F[:, :nI] = FIt[:, :nI]
        F[:, nI:] = FIt[:, nI:] @ r1t + Q
        gt = (
            np.einsum("tx,xuvw->tuvw", AA, T0[:, Ai])
            + np.einsum("txvw,xu->tuvw", T0[Ai], BA)
            + np.einsum("vx,tuxw->tuvw", AA, T1[Ai])
            + np.einsum("tuvx,xw->tuvw", T2[Ai], BA)
        )
        return F, FIt[Ai, nI:], gt

    def transition_fock(self, rho, v=None) -> np.ndarray:
        """Generalized Fock F[:, occ] for a transition density rho = (<L|R>, d_tu, Gamma_tuvw).

        v = J - K of transition_density(d); with v=None the J - K part (inactive columns) is left out.
        """
        s, d, G = rho
        C, I, A = self.C, self.I, self.A
        F = np.empty((C.shape[1], len(I) + len(A)), dtype=complex)
        F[:, : len(I)] = s * self.FI[:, I]
        if v is not None:
            F[:, : len(I)] += C.conj().T @ v @ C[:, I]
        F[:, len(I) :] = self.FI[:, A] @ d.T + np.einsum("nuvw,tuvw->nt", self.g_paaa, G)
        return F

    def transition_density(self, d: np.ndarray) -> np.ndarray:
        CA = self.C[:, self.A]
        return CA @ d.T @ CA.conj().T


def active_hamiltonian(e_core, h_act, g_act, active) -> FermionicOperator:
    H = FermionicOperator({(): e_core})
    for t, T in enumerate(active):
        for u, U in enumerate(active):
            if abs(h_act[t, u]) > 1e-14:
                H += h_act[t, u] * (a_op_spin(T, True) * a_op_spin(U, False))
    for t, T in enumerate(active):
        for u, U in enumerate(active):
            for v, V in enumerate(active):
                for w, W in enumerate(active):
                    if abs(g_act[t, u, v, w]) > 1e-14:
                        H += (0.5 * g_act[t, u, v, w]) * (
                            a_op_spin(T, True) * a_op_spin(V, True) * a_op_spin(W, False) * a_op_spin(U, False)
                        )
    return H


def orbital_gradient(F: np.ndarray, kappa_idx: list[tuple[int, int]]) -> np.ndarray:
    """Real/imag orbital gradient from the generalized Fock matrix, same layout as get_orbital_gradient_generalized_real_imag."""
    M, N = np.array(kappa_idx, dtype=int).reshape(-1, 2).T
    gr = 2 * (F[N, M] - F[M, N]).real
    gi = -2 * (F[N, M] + F[M, N]).imag
    return np.concatenate((gr, gi))


def expm_antihermitian(K: np.ndarray) -> np.ndarray:
    """exp(K) for anti-Hermitian K via eigh of the Hermitian iK (faster than scipy expm)."""
    w, V = np.linalg.eigh(1j * K)
    return (V * np.exp(-1j * w)) @ V.conj().T
