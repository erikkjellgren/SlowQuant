import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import scipy.linalg
from pyscf import lib, scf
from pyscf.gto import moleintor
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

    def describe(self) -> str:
        return f"DHFDirect (exact), screen_tol = {self.mf.direct_scf_tol:.0e}, SSSS: {self.ssss}"

    def _jk_exact(self, dms, ssss, hermi=0):
        opt_ssll = self.opt[1] if len(dms) == 1 else None
        return scf.dhf.get_jk_coulomb(self.mol, dms, hermi, "SSSS" if ssss else "SSLL", self.opt[0], opt_ssll, self.opt[2])

    def jk(self, dms: np.ndarray, response: bool = False, hermi: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """J and K for a batch of AO densities (hermi=1: all Hermitian), one integral pass."""
        return self._jk_exact(np.asarray(dms), self.ssss_resp if response else self.ssss_ground, hermi)

    def veff(self, dms: np.ndarray, response: bool = False, hermi: int = 0) -> np.ndarray:
        vj, vk = self.jk(dms, response, hermi)
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
        self.tol = tol
        self.V, self.piv = self._decompose(tol, span, blocks)
        self.V_eri = None if eri_tol is None else self._decompose(eri_tol, span, "LS")[0]

    def describe(self) -> str:
        jk = "CD" if self.cd_jk else f"exact, screen_tol = {self.mf.direct_scf_tol:.0e}"
        nv = "released" if self.V is None else f"{len(self.V)} vectors"
        return f"DHFCholesky, tol = {self.tol:.0e}, {nv} ({self.blocks}), J/K: {jk}, SSSS: {self.ssss}"

    def release(self) -> None:
        """Free the Cholesky vectors (e.g. after the LR setup). Later MO integrals and J/K use the exact path."""
        self.V = self.V_eri = None

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
        floor = np.maximum(tol, 128 * np.finfo(float).eps * d)
        npair, cs, rows, max_cols = nx * n * n, 256, 65536, 64
        L = []
        piv = []
        k = 0
        t0 = tlast = time.time()
        while True:
            p = int(np.argmax(np.where(d >= floor, d, 0.0)))
            dmax = d[p]
            if dmax < floor[p]:
                break
            y, a, b = np.unravel_index(p, (nx, n, n))
            ci, cj = shell[a], shell[b]
            ra, rb = np.arange(loc[ci], loc[ci + 1]), np.arange(loc[cj], loc[cj + 1])
            cols = (y * n * n + ra[:, None] * n + rb[None, :]).ravel()
            keep = np.flatnonzero(d[cols] >= np.maximum(floor[cols], span * dmax))
            keep = keep[np.argsort(-d[cols[keep]])[:max_cols]]
            cols = cols[keep]
            G = np.empty((npair, len(cols)), dtype=complex)
            for x, X in enumerate(blocks):
                intor, pw = _CD_INTOR[X, blocks[y]]
                raw = _intor4(mol, intor, (0, nb, 0, nb, ci, ci + 1, cj, cj + 1))
                G[x * n * n : (x + 1) * n * n] = c2**pw * raw.transpose(1, 0, 2, 3).reshape(n * n, -1)[:, keep]
                del raw
            S = G[cols].copy()
            for i, Lc in enumerate(L):
                m = min(cs, k - i * cs)
                S -= Lc[:m, cols].T @ Lc[:m, cols].conj()
            lim = np.maximum(floor[cols], span * dmax)
            q, Lq = [], []
            while True:
                dc = S.diagonal().real
                dc = np.where(dc >= lim, dc, 0.0)
                j = int(np.argmax(dc))
                if dc[j] == 0.0:
                    break
                l = S[:, j] / np.sqrt(dc[j])
                S -= np.outer(l, l.conj())
                q.append(j)
                Lq.append(l)
            if q:
                Gq = G[:, q]
                for i, Lc in enumerate(L):
                    m = min(cs, k - i * cs)
                    B = Lc[:m, cols[q]].conj()
                    for r0 in range(0, npair, rows):
                        Gq[r0 : r0 + rows] -= Lc[:m, r0 : r0 + rows].T @ B
                Lqq = np.array(Lq)[:, q].T
                Vn = scipy.linalg.solve_triangular(Lqq.conj(), Gq.T, lower=True)
                for v, j in zip(Vn, q):
                    if k % cs == 0:
                        L.append(np.empty((cs, npair), dtype=complex))
                    L[-1][k % cs] = v
                    piv.append(int(cols[j]))
                    k += 1
                d -= (np.abs(Vn) ** 2).sum(axis=0)
                np.maximum(d, 0.0, out=d)
            d[cols] = np.maximum(S.diagonal().real, 0.0)
            if time.time() - tlast > 60:
                tlast = time.time()
                print(f"Cholesky: {k} vectors ({k * npair * 16 / 1e9:.1f} GB), max residual {dmax:.1e}, {tlast - t0:.0f} s", flush=True)
        V = np.empty((k, npair), dtype=complex)
        for i in range(len(L)):
            m = min(cs, k - i * cs)
            V[i * cs : i * cs + m] = L[i][:m]
            L[i] = None
        return V.reshape(k, nx, n, n), np.array(piv, dtype=np.int64)

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

    def jk(self, dms: np.ndarray, response: bool = False, hermi: int = 0) -> tuple[np.ndarray, np.ndarray]:
        dms = np.asarray(dms)
        if not self.cd_jk or self.V is None:
            return super().jk(dms, response, hermi)
        if self.blocks == "LS":
            return self._jk_cd(dms)
        ssss = self.ssss_resp if response else self.ssss_ground
        vj, vk = self._jk_exact(dms, False, hermi)
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
        if self.V is None:
            return self._eri_exact(c1, c2, c3, c4, self.ssss_eri)
        if self.V_eri is not None:
            return self._eri_cd(c1, c2, c3, c4, self.V_eri, "LS")
        if self.blocks == "LS":
            return self._eri_cd(c1, c2, c3, c4, self.V, "LS")
        out = self._eri_exact(c1, c2, c3, c4, False)
        if self.ssss_eri:
            out += self._eri_cd(c1, c2, c3, c4, self.V, "S")
        return out


_GIAO_SAME_CENTRE_ZERO = {"int2e_g1_spinor", "int2e_spgsp1_spinor", "int2e_g1spsp2_spinor", "int2e_spgsp1spsp2_spinor"}


def _giao_batches(mol, skip_same_atom: bool, max_elem: float, lower: bool = False):
    """Shell batches (i0, i1, j0, j1) over the first pair; same-atom pairs skipped if requested, j < i if lower."""
    atom = mol._bas[:, 0]
    loc = mol.ao_loc_2c()
    n2c = loc[-1]
    nb = mol.nbas
    for i in range(nb):
        ni = loc[i + 1] - loc[i]
        cap = max(1, int(max_elem / (3 * ni * n2c * n2c)))
        jend = i if lower else nb
        j = 0
        while j < jend:
            if skip_same_atom and atom[j] == atom[i]:
                j += 1
                continue
            j1 = j
            while j1 < jend and loc[j1 + 1] - loc[j] <= cap and not (skip_same_atom and atom[j1] == atom[i]):
                j1 += 1
            j1 = max(j1, j + 1)
            yield i, i + 1, j, j1
            j = j1


def _shell_blocks(loc, nblk: int) -> list[int]:
    """Shell boundaries splitting the spinor functions into about nblk blocks."""
    nb = len(loc) - 1
    target = loc[-1] / nblk
    out = [0]
    for s in range(1, nb):
        if loc[s] - loc[out[-1]] >= target:
            out.append(s)
    out.append(nb)
    return out


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
    Time-reversal symmetry: all classes obey (ij|kl) = s_k s_l (ij|p(l) p(k)) (only k-block >= l-block is
    computed), the four derivative classes also (ij|kl) = -s_i s_j (p(j) p(i)|kl) (only j < i is computed).
    """
    mol = direct.mol
    n2c = direct.n2c
    n4c = 2 * n2c
    loc = mol.ao_loc_2c()
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
    tr = mol.time_reversal_map()
    perm, sgn = np.abs(tr) - 1, np.sign(tr).astype(float)
    blk = _shell_blocks(loc, max(1, min(8, mol.nbas)))
    nblk = len(blk) - 1
    nworkers = max(1, lib.num_threads())
    budget = 2 * max_elem
    max_elem = budget / (nworkers + 1)
    keep_s = direct.ssss_giao_jk or direct.ssss_giao_g
    all_terms = [t for t in _GIAO_TERMS if t[1] != "SSSS" or keep_s]
    jobs = [
        (intor, b)
        for intor in dict.fromkeys(t[0] for t in all_terms)
        for b in _giao_batches(mol, intor in _GIAO_SAME_CENTRE_ZERO, max_elem, intor in _GIAO_SAME_CENTRE_ZERO)
    ]

    def compute(job):
        intor, (i0, i1, j0, j1) = job
        ni, nj = loc[i1] - loc[i0], loc[j1] - loc[j0]
        W = np.empty((3, n2c, n2c, nj, ni), dtype=complex)
        for kb in range(nblk):
            k0, k1 = blk[kb], blk[kb + 1]
            for lb in range(kb + 1):
                l0, l1 = blk[lb], blk[lb + 1]
                x = mol.intor(intor, comp=3, shls_slice=(i0, i1, j0, j1, k0, k1, l0, l1))
                W[:, loc[l0] : loc[l1], loc[k0] : loc[k1]] = x.ravel(order="K").reshape(3, loc[l1] - loc[l0], loc[k1] - loc[k0], nj, ni)
        for kb in range(nblk):
            K = slice(loc[blk[kb]], loc[blk[kb + 1]])
            for lb in range(kb + 1, nblk):
                L = slice(loc[blk[lb]], loc[blk[lb + 1]])
                src = W[:, perm[K][:, None], perm[L][None, :]].transpose(0, 2, 1, 3, 4)
                W[:, L, K] = src * (sgn[L][:, None] * sgn[K][None, :])[None, :, :, None, None]
        return W

    def contract(W, terms, a0, a1, b0, b1):
        ni, nj = a1 - a0, b1 - b0
        ri, rj = slice(a0, a1), slice(b0, b1)
        Wm = W.reshape(3, n2c * n2c, nj * ni)
        Wk = W.reshape(3, n2c, n2c * nj, ni)
        jl, jr, k1, k2, gq = [], [], [], [], []
        for blk_, p, form in terms:
            fac = direct.c2**p
            fj = fac if blk_ != "SSSS" or direct.ssss_giao_jk else 0.0
            fg = fac if blk_ != "SSSS" or direct.ssss_giao_g else 0.0
            B1, B2, B3, B4 = blk_
            o1, o2, o4 = sl[B1].start, sl[B2].start, sl[B4].start
            if form == "id":
                jl.append((fj, Db[B4, B3].reshape(nd, -1), J, (o1 + a0, o1 + a1), (o2 + b0, o2 + b1), (1, 0, 3, 2)))
                k1.append((fj, Db[B2, B3][:, rj].transpose(0, 2, 1).reshape(nd, -1), J, (o1 + a0, o1 + a1), sl[B4], (3, 0, 2, 1)))
                gq.append((fg, Ab[B4], Ab[B3].conj(), Cb[B1][ri].conj(), Ab[B2][rj], (0, 4, 3, 2, 1), g))
            elif form == "cdab":
                jr.append((fj, Db[B4, B3][:, rj, ri].reshape(nd, -1), J, sl[B1], sl[B2], (3, 0, 2, 1)))
                k2.append((fj, Db[B2, B3][:, :, ri], J, sl[B1], (o4 + b0, o4 + b1), (3, 0, 1, 2)))
                gq.append((fg, Ab[B2], Cb[B1].conj(), Ab[B3][ri].conj(), Ab[B4][rj], (0, 2, 1, 4, 3), g))
            elif form == "badc":
                jl.append((fj, Db[B4, B3].conj().transpose(0, 2, 1).reshape(nd, -1), Jc, (o1 + b0, o1 + b1), (o2 + a0, o2 + a1), (1, 0, 2, 3)))
                k2.append((fj, Db[B2, B3][:, ri].conj().transpose(0, 2, 1), Jc, (o1 + b0, o1 + b1), sl[B4], (3, 0, 2, 1)))
                gq.append((fg, Ab[B3], Ab[B4].conj(), Ab[B2][ri].conj(), Cb[B1][rj], (0, 3, 4, 1, 2), gc))
            else:
                jr.append((fj, Db[B4, B3][:, ri, rj].conj().transpose(0, 2, 1).reshape(nd, -1), Jc, sl[B1], sl[B2], (3, 0, 1, 2)))
                k1.append((fj, Db[B2, B3][:, :, rj].conj().reshape(nd, -1), Jc, sl[B1], (o4 + a0, o4 + a1), (3, 0, 1, 2)))
                T = _half(W, Ab[B2].conj(), Cb[B1])
                gc[...] += fg * _ij(T, Ab[B4][ri].conj(), Ab[B3][rj])

        def put(out, r0, r1, R):
            out[:, :, slice(*r0) if isinstance(r0, tuple) else r0, slice(*r1) if isinstance(r1, tuple) else r1] += R

        if jl:
            R = np.matmul(np.concatenate([e for _, e, *_ in jl]), Wm).reshape(3, len(jl), nd, nj, ni)
            for n, (f, _, out, r0, r1, tp) in enumerate(jl):
                put(out, r0, r1, f * (R[:, n].transpose(1, 0, 3, 2) if tp == (1, 0, 3, 2) else R[:, n].transpose(1, 0, 2, 3)))
        if jr:
            R = np.matmul(Wm, np.concatenate([e for _, e, *_ in jr]).T).reshape(3, n2c, n2c, len(jr), nd)
            for n, (f, _, out, r0, r1, tp) in enumerate(jr):
                put(out, r0, r1, f * R[:, :, :, n].transpose(*tp))
        if k1:
            R = np.matmul(Wk.transpose(0, 1, 3, 2), np.concatenate([e for _, e, *_ in k1]).T).reshape(3, n2c, ni, len(k1), nd)
            for n, (f, _, out, r0, r1, tp) in enumerate(k1):
                put(out, r0, r1, -f * R[:, :, :, n].transpose(*tp))
        if k2:
            E = np.concatenate([e for _, e, *_ in k2]).transpose(1, 2, 0)
            R = np.matmul(Wk, E).sum(axis=1).reshape(3, n2c, nj, len(k2), nd)
            for n, (f, _, out, r0, r1, tp) in enumerate(k2):
                put(out, r0, r1, -f * R[:, :, :, n].transpose(*tp))
        if gq:
            A = np.matmul(np.concatenate([q for _, q, *_ in gq], axis=1).T, W.reshape(3, n2c, -1)).reshape(3, len(gq), na, n2c, nj * ni)
            for n, (f, _, P, R1, R2, tp, out) in enumerate(gq):
                T = np.matmul(P.T, A[:, n]).reshape(3, na, P.shape[1], nj, ni)
                out[...] += f * _ij(T, R1, R2).transpose(*tp)

    sizes = [3 * n2c * n2c * (loc[i1] - loc[i0]) * (loc[j1] - loc[j0]) for _, (i0, i1, j0, j1) in jobs]
    pool = ThreadPoolExecutor(max_workers=nworkers)
    futures, state = {}, {"next": 0, "inflight": 0}

    def fill():
        while state["next"] < len(jobs) and len(futures) < nworkers + 1 and (
            not futures or state["inflight"] + sizes[state["next"]] <= budget
        ):
            futures[state["next"]] = pool.submit(compute, jobs[state["next"]])
            state["inflight"] += sizes[state["next"]]
            state["next"] += 1

    fill()
    for n_job, (intor, (i0, i1, j0, j1)) in enumerate(jobs):
        W = futures.pop(n_job).result()
        terms = [(blk_, p, form) for name, blk_, p, form in all_terms if name == intor]
        a0, a1 = loc[i0], loc[i1]
        contract(W, terms, a0, a1, loc[j0], loc[j1])
        if intor in _GIAO_SAME_CENTRE_ZERO:
            b0, b1 = loc[j0], loc[j1]
            pI, pJ = perm[a0:a1] - a0, perm[b0:b1] - b0
            Wp = np.empty((3, n2c, n2c, a1 - a0, b1 - b0), dtype=complex)
            Wp[...] = W[:, :, :, pJ[None, :], pI[:, None]]
            Wp *= -(sgn[a0:a1][:, None] * sgn[b0:b1][None, :])
            del W
            contract(Wp, terms, b0, b1, a0, a1)
            del Wp
        W = None
        state["inflight"] -= sizes[n_job]
        fill()
    pool.shutdown()
    return J + Jc.conj(), g + gc.conj()


def _ints3(mol, intor, shls, opt):
    """GIAO derivative integrals of a shell slice as (3, ni, nj, nk, nl); opt: libcint optimizer for intor."""
    loc = mol.ao_loc_2c()
    d = [loc[shls[2 * i + 1]] - loc[shls[2 * i]] for i in range(4)]
    raw = moleintor.getints(intor, mol._atm, mol._bas, mol._env, shls, 3, 0, "s1", loc, opt)
    return raw.ravel(order="K").reshape(3, d[3], d[2], d[1], d[0]).transpose(0, 4, 3, 2, 1)


def giao_integrals_cd(
    direct, C: np.ndarray, CA: np.ndarray, dms: np.ndarray, max_memory: float | None = None, consume: bool = False
):
    r"""giao_integrals from the derivative of the RI-equivalent pivoted Cholesky decomposition (Nottoli et al. 2022).

    Pair metric G_{(ab),(cd)} = (ba|cd) over LL and SS pairs, G ~ M M^+, M = V^T, L = M[P, :] (pivots P).
    G^B ~ A W + W^+ A^+ - W^+ A_PP W with A = G^B[:, P] (derivative integrals at the pivot columns only)
    and W = G_PP^-1 G[P, :] = L^-+ V^*. Same output as giao_integrals: W_out = J^B - K^B of the Hermitian
    densities dms (low-rank factorized for K) and g = (N u|v w)^B.
    consume=True: W overwrites the Cholesky vectors in place (no extra memory); the vectors are released afterwards.
    """
    mol, n, c2 = direct.mol, direct.n2c, direct.c2
    loc = mol.ao_loc_2c()
    nb = mol.nbas
    shell = np.repeat(np.arange(nb), np.diff(loc))
    piv = direct.piv
    k = len(piv)
    npair = 2 * n * n
    nd, N, na = len(dms), C.shape[1], CA.shape[1]
    budget = (direct.max_memory if max_memory is None else max_memory) * 1e6 / 16 / 4
    Vf = direct.V.reshape(k, npair)
    L = np.ascontiguousarray(Vf[:, piv].T)
    W = Vf if consume else np.empty((k, npair), dtype=complex)
    step = max(1, int(budget / max(k, 1)))
    for c0 in range(0, npair, step):
        W[:, c0 : c0 + step] = scipy.linalg.solve_triangular(L.conj().T, Vf[:, c0 : c0 + step].conj(), lower=False)
    if consume:
        direct.V = None
    Wb = W.reshape(k, 2, n, n)
    sl = [slice(0, n), slice(n, 2 * n)]
    Cx, Ax = [C[:n], C[n:]], [CA[:n], CA[n:]]
    terms = [t for t in _GIAO_TERMS if t[1] != "SSSS" or direct.ssss_giao_jk]
    opts = {t[0]: moleintor.make_cintopt(mol._atm, mol._bas, mol._env, t[0]) for t in terms}
    dvec = np.stack([dms[:, s, s].transpose(0, 2, 1) for s in sl], 1).reshape(nd, -1)
    wd = dvec @ W.T
    fac_d = []
    for d in dms:
        e, U = np.linalg.eigh(d)
        m = np.abs(e) > 1e-12 * max(1.0, np.abs(e).max())
        X = U[:, m]
        Uk = [np.ascontiguousarray((Wb[:, x].transpose(0, 2, 1) @ X[sl[x]].conj()).conj()) for x in range(2)]
        fac_d.append((e[m], X, Uk))
    wP = sum(np.einsum("lv,pls,sw->pvw", Ax[x].conj(), Wb[:, x], Ax[x], optimize=True) for x in range(2))
    omP = sum(np.einsum("mN,pnm,nu->pNu", Cx[x].conj(), Wb[:, x].conj(), Ax[x], optimize=True) for x in range(2))
    t1 = np.zeros((nd, 3, npair), dtype=complex)
    u = np.zeros((nd, 3, k), dtype=complex)
    App = np.zeros((3, k, k), dtype=complex)
    Kout = np.zeros((nd, 3, 2 * n, 2 * n), dtype=complex)
    g = np.zeros((3, N, na, na, na), dtype=complex)
    y_, a_, b_ = np.unravel_index(piv, (2, n, n))
    tr = mol.time_reversal_map()
    perm, sgn = np.abs(tr) - 1, np.sign(tr).astype(float)
    blks = _shell_blocks(loc, max(1, min(8, nb)))
    nblk = len(blks) - 1
    groups = {}
    for K_, key in enumerate(zip(y_, shell[a_], shell[b_])):
        groups.setdefault(key, []).append(K_)
    g1set = _GIAO_SAME_CENTRE_ZERO
    pin = [np.flatnonzero(piv // (n * n) == x) for x in range(2)]
    pin_loc = [piv[pin[x]] - x * n * n for x in range(2)]

    def place(Ag, x, v, Ra, Rb, sym, fac):
        Ag[:, x, Ra, Rb] += fac * v
        if sym and Ra != Rb:
            ia, ib = perm[Ra] - Ra.start, perm[Rb] - Rb.start
            w = v[:, ia][:, :, ib].transpose(0, 2, 1, 3)
            Ag[:, x, Rb, Ra] += (sym * fac) * (sgn[perm[Rb]][:, None] * sgn[perm[Ra]][None, :])[None, :, :, None] * w

    done = set()
    for key in groups:
        if key in done:
            continue
        y, cs, ds = key
        keys = [key] + ([(y, ds, cs)] if cs != ds and (y, ds, cs) in groups else [])
        done.update(keys)
        info = []
        for y_g, c_s, d_s in keys:
            Ks = np.array(groups[(y_g, c_s, d_s)])
            info.append((Ks, a_[Ks], b_[Ks]))
        (K1, c1, d1) = info[0]
        cl1, dl1 = c1 - loc[cs], d1 - loc[ds]
        pc1, pd1 = perm[c1] - loc[cs], perm[d1] - loc[ds]
        s1 = sgn[c1] * sgn[d1]
        if len(info) > 1:
            (K2, c2_, d2_) = info[1]
            cl2, dl2 = c2_ - loc[ds], d2_ - loc[cs]
            pc2, pd2 = perm[c2_] - loc[ds], perm[d2_] - loc[cs]
            s2 = sgn[c2_] * sgn[d2_]
        for xs in range(2):
            As = [np.zeros((3, 1, n, n, len(Ks)), dtype=complex) for Ks, _, _ in info]
            A1 = As[0]
            A2 = As[1] if len(As) > 1 else None
            for intor in dict.fromkeys(t[0] for t in terms):
                tl = [(blk, p, form) for name, blk, p, form in terms if name == intor and blk[2] == "LS"[y] and blk[0] == "LS"[xs]]
                idt = [t for t in tl if t[2] in ("id", "badc")]
                brt = [t for t in tl if t[2] in ("cdab", "dcba")]
                g1c = intor in g1set
                if idt:
                    rs = -1.0 if g1c else 0.0
                    for i1 in range(nblk):
                        for i2 in range(nblk):
                            if rs and i2 > i1:
                                continue
                            S1, S2 = (blks[i1], blks[i1 + 1]), (blks[i2], blks[i2 + 1])
                            R1, R2 = slice(loc[S1[0]], loc[S1[1]]), slice(loc[S2[0]], loc[S2[1]])
                            Y = _ints3(mol, intor, (*S1, *S2, cs, cs + 1, ds, ds + 1), opts[intor])
                            for blk, p, form in idt:
                                x, fac = "LS".index(blk[0]), c2**p
                                if x != xs:
                                    continue
                                if form == "id":
                                    place(A1, 0, Y[:, :, :, cl1, dl1].transpose(0, 2, 1, 3), R2, R1, rs, fac)
                                    if len(info) > 1:
                                        place(A2, 0, (s2 * Y[:, :, :, pd2, pc2]).transpose(0, 2, 1, 3), R2, R1, rs, fac)
                                else:
                                    place(A1, 0, (s1 * Y[:, :, :, pc1, pd1]).conj(), R1, R2, 0.0, fac)
                                    if len(info) > 1:
                                        place(A2, 0, Y[:, :, :, dl2, cl2].conj(), R1, R2, 0.0, fac)
                if brt:
                    need2 = (not g1c) and (len(info) > 1 or any(t[2] == "dcba" for t in brt))
                    for i1 in range(nblk):
                        for i2 in range(i1 + 1):
                            S1, S2 = (blks[i1], blks[i1 + 1]), (blks[i2], blks[i2 + 1])
                            R1, R2 = slice(loc[S1[0]], loc[S1[1]]), slice(loc[S2[0]], loc[S2[1]])
                            Yb1 = _ints3(mol, intor, (cs, cs + 1, ds, ds + 1, *S1, *S2), opts[intor])
                            Yb2 = _ints3(mol, intor, (ds, ds + 1, cs, cs + 1, *S1, *S2), opts[intor]) if need2 else None
                            for blk, p, form in brt:
                                x, fac = "LS".index(blk[0]), c2**p
                                if x != xs:
                                    continue
                                if form == "cdab":
                                    place(A1, 0, Yb1[:, cl1, dl1].transpose(0, 3, 2, 1), R2, R1, 1.0, fac)
                                    if len(info) > 1:
                                        v = -s2[:, None, None] * Yb1[:, pd2, pc2] if g1c else Yb2[:, cl2, dl2]
                                        place(A2, 0, v.transpose(0, 3, 2, 1), R2, R1, 1.0, fac)
                                else:
                                    place(A1, 0, Yb2[:, dl1, cl1].transpose(0, 2, 3, 1).conj(), R1, R2, 1.0, fac)
                                    if len(info) > 1:
                                        place(A2, 0, Yb1[:, dl2, cl2].transpose(0, 2, 3, 1).conj(), R1, R2, 1.0, fac)
            for (Ks, _, _), A in zip(info, As):
                kb = len(Ks)
                Af = A[:, 0].reshape(3, n * n, kb)
                rx = slice(xs * n * n, (xs + 1) * n * n)
                t1[:, :, rx] += (Af @ wd[:, Ks].T).transpose(2, 0, 1)
                u[:, :, Ks] += (dvec[:, rx].conj() @ Af).conj().transpose(1, 0, 2)
                App[:, pin[xs], Ks[:, None]] = Af[:, pin_loc[xs], :].transpose(0, 2, 1)
                AT = A[:, 0].reshape(3, n, n * kb).transpose(0, 2, 1)
                for i, (lam, X, Uk) in enumerate(fac_d):
                    r = len(lam)
                    Pm = (AT @ X[sl[xs]]).reshape(3, n, kb * r)
                    for yy in range(2):
                        F = Pm @ (Uk[yy][Ks].conj().transpose(0, 2, 1) * lam[:, None]).reshape(kb * r, n)
                        Kout[i, :, sl[xs], sl[yy]] += F
                        Kout[i, :, sl[yy], sl[xs]] += F.conj().transpose(0, 2, 1)
                Ta = (AT @ Ax[xs]).reshape(3, n, kb * na)
                a = (Cx[xs].conj().T @ Ta).reshape(3, N, kb, na).transpose(0, 1, 3, 2).reshape(3 * N * na, kb)
                g += (a @ wP[Ks].reshape(kb, na * na)).reshape(3, N, na, na, na)
                al = (Ta.transpose(0, 2, 1) @ Ax[xs].conj()).conj().reshape(3, kb, na * na)
                g += (omP[Ks].reshape(kb, N * na).T @ al).reshape(3, N, na, na, na)
    s = np.einsum("xpq,dq->dxp", App, wd, optimize=True)
    J = t1 + np.einsum("kp,dxk->dxp", W.conj(), u - s, optimize=True)
    qb = max(1, int(budget / max(1, 3 * n * max(len(f[0]) for f in fac_d))))
    for i, (lam, X, Uk) in enumerate(fac_d):
        for q0 in range(0, k, qb):
            Q = slice(q0, q0 + qb)
            r = len(lam)
            AQ = App[:, :, Q].transpose(0, 2, 1)
            q = AQ.shape[1]
            Ut = [(AQ @ Uk[x].reshape(k, n * r)).reshape(3, q, n, r).transpose(0, 2, 1, 3).reshape(3, n, q * r) for x in range(2)]
            for yy in range(2):
                B = (Uk[yy][Q].conj().transpose(0, 2, 1) * lam[:, None]).reshape(q * r, n)
                for x in range(2):
                    Kout[i, :, sl[x], sl[yy]] -= Ut[x] @ B
    g -= np.einsum("pNu,xpq,qvw->xNuvw", omP, App, wP, optimize=True)
    out = -Kout
    Jb = J.reshape(nd, 3, 2, n, n)
    for x in range(2):
        out[:, :, sl[x], sl[x]] += Jb[:, :, x].transpose(0, 1, 3, 2)
    return out, g


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
        v = direct.veff(np.array(dms), hermi=1)
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
            self.FA = self.C.conj().T @ self.direct.veff((CA @ rdm1.T @ CA.conj().T)[None], hermi=1)[0] @ self.C
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
        Q += T0.reshape(N, -1) @ np.einsum("xu,tuvw->xvwt", BA, G2, optimize=True).reshape(-1, na)
        Q += T1.reshape(N, -1) @ np.einsum("vx,tuvw->uxwt", AA, G2, optimize=True).reshape(-1, na)
        Q += T2.reshape(N, -1) @ np.einsum("xw,tuvw->uvxt", BA, G2, optimize=True).reshape(-1, na)
        F = np.empty((N, nI + na), dtype=complex)
        F[:, :nI] = FIt[:, :nI]
        F[:, nI:] = FIt[:, nI:] @ r1t + Q
        gt = (
            np.einsum("tx,xuvw->tuvw", AA, T0[:, Ai], optimize=True)
            + np.einsum("txvw,xu->tuvw", T0[Ai], BA, optimize=True)
            + np.einsum("vx,tuxw->tuvw", AA, T1[Ai], optimize=True)
            + np.einsum("tuvx,xw->tuvw", T2[Ai], BA, optimize=True)
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
        F[:, len(I) :] = self.FI[:, A] @ d.T + np.einsum("nuvw,tuvw->nt", self.g_paaa, G, optimize=True)
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
