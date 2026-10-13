import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from pyscf import lib, scf
from pyscf.ao2mo import r_outcore

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import a_op_spin

# Dirac-Coulomb blocks: (intor, block of (12|34), scale power of (1/2c)^2)
_DC_TERMS = (
    ("int2e_spinor", "LLLL", 0),
    ("int2e_spsp1spsp2_spinor", "SSSS", 2),
    ("int2e_spsp2_spinor", "LLSS", 1),
    ("int2e_spsp1_spinor", "SSLL", 1),
)

# GIAO/RMB derivative of the Dirac-Coulomb operator, same terms as IntegralManager.g_B.
# (intor, block of X, scale power, form): X_abcd = Y_abcd ("id"), Y_cdab, conj(Y_badc), conj(Y_dcba)
_GIAO_TERMS = (
    ("int2e_g1_spinor", "LLLL", 0, "id"),
    ("int2e_g1_spinor", "LLLL", 0, "cdab"),
    ("int2e_spgsp1spsp2_spinor", "SSSS", 2, "id"),
    ("int2e_spgsp1spsp2_spinor", "SSSS", 2, "cdab"),
    ("int2e_giao_sa10sp1spsp2_spinor", "SSSS", 2, "id"),
    ("int2e_giao_sa10sp1spsp2_spinor", "SSSS", 2, "badc"),
    ("int2e_giao_sa10sp1spsp2_spinor", "SSSS", 2, "cdab"),
    ("int2e_giao_sa10sp1spsp2_spinor", "SSSS", 2, "dcba"),
    ("int2e_spgsp1_spinor", "SSLL", 1, "id"),
    ("int2e_g1spsp2_spinor", "SSLL", 1, "cdab"),
    ("int2e_giao_sa10sp1_spinor", "SSLL", 1, "id"),
    ("int2e_giao_sa10sp1_spinor", "SSLL", 1, "badc"),
    ("int2e_spgsp1_spinor", "LLSS", 1, "cdab"),
    ("int2e_g1spsp2_spinor", "LLSS", 1, "id"),
    ("int2e_giao_sa10sp1_spinor", "LLSS", 1, "cdab"),
    ("int2e_giao_sa10sp1_spinor", "LLSS", 1, "dcba"),
)


# Where (SS|SS) is kept: ground-state J/K (F^I, F^A), response J/K (trial densities, GIAO connection),
# MO integrals (active space, response tensors), GIAO J^B - K^B, GIAO (N u|v w)^B
_SSSS_MODES = {
    "full": (True, True, True, True, True),
    "mf": (True, True, False, True, False),
    "ground": (True, False, False, False, False),
    "nogiao": (True, True, False, False, False),
    "noresp": (True, False, False, True, False),
    "none": (False, False, False, False, False),
}


class DHFDirect:
    """Integral-direct 4c Dirac-Coulomb building blocks (PySCF backend).

    Conventions: (pq|rs) = sum C*_mp C_nq C*_ls C_sr (mn|ls), J[D]_pq = sum (pq|rs) D_sr, K[D]_pq = sum (ps|rq) D_sr.
    screen_tol: Schwarz/density screening threshold of the J/K builds (PySCF direct_scf_tol).
    ssss: "full" | "mf" | "ground" | "none", see _SSSS_MODES.
    """

    def __init__(
        self,
        mol,
        max_memory: float | None = None,
        tmpdir: str | None = None,
        screen_tol: float = 1e-13,
        ssss: str = "full",
    ):
        self.mol = mol
        self.mf = scf.dhf.DHF(mol)
        self.mf.verbose = 0
        self.mf.direct_scf_tol = screen_tol
        self.n2c = mol.nao_2c()
        self.h_ao = self.mf.get_hcore()
        self.opt = self.mf.init_direct_scf(mol)
        self.max_memory = mol.max_memory if max_memory is None else max_memory
        self.tmpdir = lib.param.TMPDIR if tmpdir is None else tmpdir
        self.c2 = (0.5 / lib.param.LIGHT_SPEED) ** 2
        self.ssss = ssss
        self.ssss_ground, self.ssss_resp, self.ssss_eri, self.ssss_giao_jk, self.ssss_giao_g = _SSSS_MODES[ssss]

    def _jk_exact(self, dms, ssss):
        opt_ssll = self.opt[1] if len(dms) == 1 else None
        return scf.dhf.get_jk_coulomb(self.mol, dms, 0, "SSSS" if ssss else "SSLL", self.opt[0], opt_ssll, self.opt[2])

    def jk(self, dms: np.ndarray, response: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """J and K for a batch of general (non-Hermitian) AO densities, one integral pass."""
        return self._jk_exact(np.asarray(dms), self.ssss_resp if response else self.ssss_ground)

    def veff(self, dms: np.ndarray, response: bool = False) -> np.ndarray:
        vj, vk = self.jk(dms, response)
        return vj - vk

    def eri(self, c1, c2, c3, c4) -> np.ndarray:
        """(12|34) in the MO sets c1..c4, shape (n1, n2, n3, n4). Out-of-core per block, accumulated in place."""
        return self._eri_exact(c1, c2, c3, c4, self.ssss_eri)

    def _eri_exact(self, c1, c2, c3, c4, ssss=True):
        n = self.n2c
        shape = (c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1])
        out = np.zeros((shape[0] * shape[1], shape[2] * shape[3]), dtype=complex)
        blk = max(1, int(self.max_memory * 1e6 / 4 / 16 / max(out.shape[1], 1)))
        for intor, sel, p in _DC_TERMS:
            if sel == "SSSS" and not ssss:
                continue
            cs = [c[:n] if b == "L" else c[n:] for c, b in zip((c1, c2, c3, c4), sel)]
            fd, fname = tempfile.mkstemp(suffix=".h5", dir=self.tmpdir)
            os.close(fd)
            try:
                r_outcore.general(self.mol, cs, fname, "eri", intor=intor, aosym="s4", max_memory=self.max_memory, verbose=0)
                with h5py.File(fname, "r") as f:
                    d = f["eri"]
                    for i in range(0, out.shape[0], blk):
                        out[i : i + blk] += self.c2**p * d[i : i + blk]
            finally:
                os.remove(fname)
        return out.reshape(shape)


_CD_INTOR = {
    ("L", "L"): ("int2e_spinor", 0),
    ("S", "L"): ("int2e_spsp1_spinor", 1),
    ("L", "S"): ("int2e_spsp2_spinor", 1),
    ("S", "S"): ("int2e_spsp1spsp2_spinor", 2),
}


def _intor4(mol, intor, shls):
    """(ij|kl) for a shell slice as a C-ordered (ni, nj, nk, nl) array (PySCF returns sliced spinor ints in i-fastest order)."""
    loc = mol.ao_loc_2c()
    dims = [loc[shls[2 * i + 1]] - loc[shls[2 * i]] for i in range(4)]
    return mol.intor(intor, shls_slice=shls).ravel(order="K").reshape(dims[::-1]).transpose(3, 2, 1, 0)


class DHFCholesky(DHFDirect):
    r"""Pivoted Cholesky decomposition of the Dirac-Coulomb operator, drop-in for DHFDirect.

    Pair metric G_{(ab),(cd)} = (ba|cd) (Hermitian PSD) over the LL and SS pair spaces ("LS") or the SS pair
    space only ("S", i.e. only (SS|SS) is decomposed, the rest exact). (mn|ls) ~ sum_K V_K[n, m] V_K[l, s]^*.
    cd_jk=False: exact (direct) J/K, Cholesky vectors only for the MO integrals (eri). eri_tol: separate LL+SS
    decomposition used for the MO integrals (e.g. blocks="S" for the (SS|SS) J/K and eri_tol for eri). tol: largest neglected
    residual diagonal. span: pivots inside one shell-pair column block are taken while
    diag > span * (largest diag at block start).
    """

    def __init__(
        self, mol, tol: float = 1e-6, blocks: str = "LS", span: float = 1e-2, cd_jk: bool = True, eri_tol=None, **kwargs
    ):
        super().__init__(mol, **kwargs)
        if blocks == "LS":
            assert self.ssss == "full"
        self.blocks = blocks
        self.cd_jk = cd_jk
        self.V = self._decompose(tol, span, blocks)
        self.V_eri = None if eri_tol is None else self._decompose(eri_tol, span, "LS")

    def _decompose(self, tol, span, blocks):
        mol, n, c2 = self.mol, self.n2c, self.c2
        loc = mol.ao_loc_2c()
        nb, nx = mol.nbas, len(blocks)
        shell = np.repeat(np.arange(nb), np.diff(loc))
        d = np.zeros((nx, n, n))
        for x, X in enumerate(blocks):
            intor, p = _CD_INTOR[X, X]
            for i in range(nb):
                for j in range(nb):
                    v = _intor4(mol, intor, (j, j + 1, i, i + 1, i, i + 1, j, j + 1))
                    d[x, loc[i] : loc[i + 1], loc[j] : loc[j + 1]] = c2**p * np.einsum("baab->ab", v).real
        d = d.ravel()
        npair, cs = nx * n * n, 256
        L = []
        k = 0
        while True:
            p = int(np.argmax(d))
            dmax = d[p]
            if dmax < tol:
                break
            y, a, b = np.unravel_index(p, (nx, n, n))
            ci, cj = shell[a], shell[b]
            ra, rb = np.arange(loc[ci], loc[ci + 1]), np.arange(loc[cj], loc[cj + 1])
            cols = (y * n * n + ra[:, None] * n + rb[None, :]).ravel()
            G = np.empty((nx, n, n, len(ra), len(rb)), dtype=complex)
            for x, X in enumerate(blocks):
                intor, pw = _CD_INTOR[X, blocks[y]]
                G[x] = c2**pw * _intor4(mol, intor, (0, nb, 0, nb, ci, ci + 1, cj, cj + 1)).transpose(1, 0, 2, 3)
            keep = d[cols] >= max(tol, span * dmax)
            cols = cols[keep]
            G = G.reshape(-1, len(keep))[:, keep]
            for i, Lc in enumerate(L):
                m = min(cs, k - i * cs)
                G -= Lc[:m].T @ Lc[:m, cols].conj()
            while True:
                dc = G[cols, np.arange(len(cols))].real
                j = int(np.argmax(dc))
                if dc[j] < max(tol, span * dmax):
                    break
                v = G[:, j] / np.sqrt(dc[j])
                if k % cs == 0:
                    L.append(np.empty((cs, npair), dtype=complex))
                L[-1][k % cs] = v
                k += 1
                G -= np.outer(v, v[cols].conj())
                d -= np.abs(v) ** 2
                d[cols[j]] = 0.0
                np.maximum(d, 0.0, out=d)
        V = np.empty((k, npair), dtype=complex)
        for i in range(len(L)):
            m = min(cs, k - i * cs)
            V[i * cs : i * cs + m] = L[i][:m]
            L[i] = None
        return V.reshape(k, nx, n, n)

    def _sub(self, dms, x, y):
        n = self.n2c
        o = {"L": 0, "S": n}
        return dms[:, o[x] : o[x] + n, o[y] : o[y] + n]

    def _jk_cd(self, dms, chunk=None):
        n, V = self.n2c, self.V
        nK, nd = len(V), len(dms)
        vj = np.zeros_like(dms)
        vk = np.zeros_like(dms)
        if nK == 0:
            return vj, vk
        bl = self.blocks
        c = sum(V[:, x].reshape(nK, -1).conj() @ self._sub(dms, X, X).transpose(0, 2, 1).reshape(nd, -1).T for x, X in enumerate(bl))
        for x, X in enumerate(bl):
            self._sub(vj, X, X)[:] = (c.T @ V[:, x].reshape(nK, -1)).reshape(nd, n, n).transpose(0, 2, 1)
        if chunk is None:
            chunk = max(1, int(self.max_memory * 1e6 / 16 / (2 * n * n)))
        for k0 in range(0, nK, chunk):
            Vc = V[k0 : k0 + chunk]
            m = len(Vc)
            for y, Y in enumerate(bl):
                Vy = Vc[:, y].conj().transpose(1, 0, 2).reshape(n, m * n)
                for i in range(nd):
                    for x, X in enumerate(bl):
                        A = (self._sub(dms, X, Y)[i] @ Vy).reshape(n, m, n).transpose(1, 0, 2).reshape(m * n, n)
                        self._sub(vk, X, Y)[i] += Vc[:, x].reshape(m * n, n).T @ A
        return vj, vk

    def jk(self, dms: np.ndarray, response: bool = False) -> tuple[np.ndarray, np.ndarray]:
        dms = np.asarray(dms)
        if not self.cd_jk:
            return super().jk(dms, response)
        if self.blocks == "LS":
            return self._jk_cd(dms)
        ssss = self.ssss_resp if response else self.ssss_ground
        vj, vk = self._jk_exact(dms, False)
        if ssss:
            j, k = self._jk_cd(dms)
            vj += j
            vk += k
        return vj, vk

    def _eri_cd(self, c1, c2, c3, c4, V, blocks):
        n = self.n2c
        nK = len(V)
        shape = (c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1])
        out = np.zeros((shape[0] * shape[1], shape[2] * shape[3]), dtype=complex)
        o = {"L": slice(0, n), "S": slice(n, 2 * n)}
        per = 16 * (n * n + n * max(shape) + shape[0] * shape[1] + shape[2] * shape[3])
        chunk = max(1, int(self.max_memory * 1e6 / 4 / per))
        for k0 in range(0, nK, chunk):
            Vc = V[k0 : k0 + chunk]
            m = len(Vc)
            B12 = sum(c1[o[X]].conj().T @ (Vc[:, x].transpose(0, 2, 1) @ c2[o[X]]) for x, X in enumerate(blocks))
            B34 = sum(c3[o[X]].conj().T @ (Vc[:, x].conj() @ c4[o[X]]) for x, X in enumerate(blocks))
            out += B12.reshape(m, -1).T @ B34.reshape(m, -1)
        return out.reshape(shape)

    def eri(self, c1, c2, c3, c4) -> np.ndarray:
        if self.V_eri is not None:
            return self._eri_cd(c1, c2, c3, c4, self.V_eri, "LS")
        if self.blocks == "LS":
            return self._eri_cd(c1, c2, c3, c4, self.V, "LS")
        out = self._eri_exact(c1, c2, c3, c4, False)
        if self.ssss_eri:
            out += self._eri_cd(c1, c2, c3, c4, self.V, "S")
        return out


_GIAO_SAME_CENTRE_ZERO = {"int2e_g1_spinor", "int2e_spgsp1_spinor", "int2e_g1spsp2_spinor", "int2e_spgsp1spsp2_spinor"}


def _giao_batches(mol, skip_same_atom: bool, max_elem: float):
    """Shell batches (i0, i1, j0, j1) over the first pair; same-atom pairs skipped if requested."""
    atom = mol._bas[:, 0]
    loc = mol.ao_loc_2c()
    n2c = loc[-1]
    nb = mol.nbas
    for i in range(nb):
        ni = loc[i + 1] - loc[i]
        cap = max(1, int(max_elem / (3 * ni * n2c * n2c)))
        j = 0
        while j < nb:
            if skip_same_atom and atom[j] == atom[i]:
                j += 1
                continue
            j1 = j
            while j1 < nb and loc[j1 + 1] - loc[j] <= cap and not (skip_same_atom and atom[j1] == atom[i]):
                j1 += 1
            j1 = max(j1, j + 1)
            yield i, i + 1, j, j1
            j = j1


def _half(W, P, Q):
    """T[x, b, a, j, i] = sum_{l,k} W[x, l, k, j, i] P[k, a] Q[l, b] (smaller coefficient set contracted first)."""
    x, nl, nk, nj, ni = W.shape
    na, nb = P.shape[1], Q.shape[1]
    if nb <= na:
        A = np.matmul(Q.T, W.reshape(x, nl, -1)).reshape(x, nb, nk, nj * ni)
        T = np.matmul(P.T, A)
    else:
        A = np.matmul(P.T, W.reshape(x, nl, nk, nj * ni))
        T = np.matmul(Q.T, A.reshape(x, nl, -1)).reshape(x, nb, na, nj * ni)
    return T.reshape(x, nb, na, nj, ni)


def _ij(T, R1, R2):
    """sum_{j,i} T[x, b, a, j, i] R1[i, p] R2[j, q] -> (x, b, a, q, p)."""
    x, nb, na, nj, ni = T.shape
    t = (T.reshape(-1, ni) @ R1).reshape(x * nb * na, nj, -1)
    t = np.ascontiguousarray(t.transpose(1, 0, 2)).reshape(nj, -1)
    return (R2.T @ t).reshape(R2.shape[1], x, nb, na, -1).transpose(1, 2, 3, 0, 4)


def giao_integrals(direct, C: np.ndarray, CA: np.ndarray, dms: np.ndarray, max_memory: float | None = None):
    r"""One pass over the GIAO/RMB two-electron derivative integrals (same terms as IntegralManager.g_B).

    Returns W (n_dm, 3, n4c, n4c) = J^B - K^B of the AO densities dms (general, no symmetry assumed)
    and g (3, N, na, na, na) = (N u|v w)^B. First-pair same-centre shell pairs are skipped where the
    GIAO integrals vanish exactly (g1, spgsp1, g1spsp2, spgsp1spsp2).
    """
    mol = direct.mol
    n2c = direct.n2c
    n4c = 2 * n2c
    loc = mol.ao_loc_2c()
    nb = mol.nbas
    nd, N, na = len(dms), C.shape[1], CA.shape[1]
    max_elem = (direct.max_memory if max_memory is None else max_memory) * 1e6 / 16 / 4
    sl = {"L": slice(0, n2c), "S": slice(n2c, n4c)}
    Cb = {"L": C[:n2c], "S": C[n2c:]}
    Ab = {"L": CA[:n2c], "S": CA[n2c:]}
    Db = {(a, b): np.ascontiguousarray(dms[:, sl[a], sl[b]]) for a in "LS" for b in "LS"}
    J = np.zeros((nd, 3, n4c, n4c), dtype=complex)
    Jc = np.zeros((nd, 3, n4c, n4c), dtype=complex)
    g = np.zeros((3, N, na, na, na), dtype=complex)
    gc = np.zeros((3, N, na, na, na), dtype=complex)
    nworkers = max(1, lib.num_threads())
    window = 2 * nworkers
    max_elem /= window + 1
    keep_s = direct.ssss_giao_jk or direct.ssss_giao_g
    all_terms = [t for t in _GIAO_TERMS if t[1] != "SSSS" or keep_s]
    jobs = [
        (intor, b)
        for intor in dict.fromkeys(t[0] for t in all_terms)
        for b in _giao_batches(mol, intor in _GIAO_SAME_CENTRE_ZERO, max_elem)
    ]

    def compute(job):
        intor, (i0, i1, j0, j1) = job
        return mol.intor(intor, comp=3, shls_slice=(i0, i1, j0, j1, 0, nb, 0, nb)).ravel(order="K")

    pool = ThreadPoolExecutor(max_workers=nworkers)
    futures = [pool.submit(compute, job) for job in jobs[:window]]
    for n_job, (intor, (i0, i1, j0, j1)) in enumerate(jobs):
        raw = futures[n_job].result()
        futures[n_job] = None
        if n_job + window < len(jobs):
            futures.append(pool.submit(compute, jobs[n_job + window]))
        terms = [(blk, p, form) for name, blk, p, form in all_terms if name == intor]
        a0, a1, b0, b1 = loc[i0], loc[i1], loc[j0], loc[j1]
        ni, nj = a1 - a0, b1 - b0
        W = raw.reshape(3, n2c, n2c, nj, ni)
        Wm = W.reshape(3, n2c * n2c, nj * ni)
        Wt = None
        ri, rj = slice(a0, a1), slice(b0, b1)
        for blk, p, form in terms:
            fac = direct.c2**p
            fj = fac if blk != "SSSS" or direct.ssss_giao_jk else 0.0
            fg = fac if blk != "SSSS" or direct.ssss_giao_g else 0.0
            B1, B2, B3, B4 = blk
            o1, o2, o4 = sl[B1].start, sl[B2].start, sl[B4].start
            if form == "id":
                E = Db[B4, B3].reshape(nd, -1)
                R = np.matmul(E, Wm).reshape(3, nd, nj, ni).transpose(1, 0, 3, 2)
                J[:, :, o1 + a0 : o1 + a1, o2 + b0 : o2 + b1] += fj * R
                E = Db[B2, B3][:, rj].transpose(0, 2, 1).reshape(nd, -1)
                R = np.matmul(W.reshape(3, n2c, n2c * nj, ni).transpose(0, 1, 3, 2), E.T)
                J[:, :, o1 + a0 : o1 + a1, sl[B4]] -= fj * R.transpose(3, 0, 2, 1)
                T = _half(W, Ab[B3].conj(), Ab[B4])
                g += fg * _ij(T, Cb[B1][ri].conj(), Ab[B2][rj]).transpose(0, 4, 3, 2, 1)
            elif form == "cdab":
                E = Db[B4, B3][:, rj, ri].reshape(nd, -1)
                R = np.matmul(Wm, E.T).reshape(3, n2c, n2c, nd).transpose(3, 0, 2, 1)
                J[:, :, sl[B1], sl[B2]] += fj * R
                E = Db[B2, B3][:, :, ri].reshape(nd, -1)
                if Wt is None:
                    Wt = W.reshape(3, n2c, n2c * nj, ni).transpose(0, 2, 1, 3).reshape(3, n2c * nj, n2c * ni)
                R = np.matmul(Wt, E.T).reshape(3, n2c, nj, nd).transpose(3, 0, 1, 2)
                J[:, :, sl[B1], o4 + b0 : o4 + b1] -= fj * R
                T = _half(W, Cb[B1].conj(), Ab[B2])
                g += fg * _ij(T, Ab[B3][ri].conj(), Ab[B4][rj]).transpose(0, 2, 1, 4, 3)
            elif form == "badc":
                E = Db[B4, B3].conj().transpose(0, 2, 1).reshape(nd, -1)
                R = np.matmul(E, Wm).reshape(3, nd, nj, ni).transpose(1, 0, 2, 3)
                Jc[:, :, o1 + b0 : o1 + b1, o2 + a0 : o2 + a1] += fj * R
                E = Db[B2, B3][:, ri].conj().transpose(0, 2, 1).reshape(nd, -1)
                if Wt is None:
                    Wt = W.reshape(3, n2c, n2c * nj, ni).transpose(0, 2, 1, 3).reshape(3, n2c * nj, n2c * ni)
                R = np.matmul(Wt, E.T).reshape(3, n2c, nj, nd).transpose(3, 0, 2, 1)
                Jc[:, :, o1 + b0 : o1 + b1, sl[B4]] -= fj * R
                T = _half(W, Ab[B4].conj(), Ab[B3])
                gc += fg * _ij(T, Ab[B2][ri].conj(), Cb[B1][rj]).transpose(0, 3, 4, 1, 2)
            else:
                E = Db[B4, B3][:, ri, rj].conj().transpose(0, 2, 1).reshape(nd, -1)
                R = np.matmul(Wm, E.T).reshape(3, n2c, n2c, nd).transpose(3, 0, 1, 2)
                Jc[:, :, sl[B1], sl[B2]] += fj * R
                E = Db[B2, B3][:, :, rj].conj().reshape(nd, -1)
                R = np.matmul(W.reshape(3, n2c, n2c * nj, ni).transpose(0, 1, 3, 2), E.T)
                Jc[:, :, sl[B1], o4 + a0 : o4 + a1] -= fj * R.transpose(3, 0, 1, 2)
                T = _half(W, Ab[B2].conj(), Cb[B1])
                gc += fg * _ij(T, Ab[B4][ri].conj(), Ab[B3][rj])
    pool.shutdown()
    return J + Jc.conj(), g + gc.conj()


class DirectState:
    """Integral-direct quantities for one set of MO coefficients C.

    inactive/active are absolute MO index lists. With rdm1 given, F^I and F^A come from one J/K pass.
    """

    def __init__(self, direct: DHFDirect, C: np.ndarray, inactive, active, rdm1: np.ndarray | None = None):
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
