"""Fast Pauli-level measurement layer (hardware compatible).

Paulis are stored as integer key x * 2**n + z for the operator X^x Z^z (X^x Z^z = (-i)^{n_Y} P_label).
All fermionic quantities are linear maps (sparse matrices) from the vector of Pauli expectation values.
"""

import itertools

import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import PauliList, SparsePauliOp, Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing_generalized
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import G1_generalized, G2_generalized, a_op_spin
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2

try:
    from slowquant.unitary_coupled_cluster.ci_sparse import UPSEngine, ups_theta_gradient
except ImportError:
    from slowquant.unitary_coupled_cluster.ci_sparse_DHF import UPSEngine, ups_theta_gradient


def _popcount_parity(a):
    return (np.bitwise_count(a) & 1).astype(np.float64)


def _jw_ladder_product(idx, dag, num_orbs):
    """X^x Z^z expansion of prod_l a^(dag_l)_{idx[:, l]} for each row of idx (Jordan-Wigner, qiskit qubit order)."""
    idx = np.asarray(idx, dtype=np.int64)
    q = idx // 2 + (idx % 2) * num_orbs
    e = np.left_shift(np.int64(1), q)
    m = e - 1
    M = len(idx)
    x = np.zeros((M, 1), np.int64)
    z = np.zeros((M, 1), np.int64)
    c = np.ones((M, 1), complex)
    for l, d in enumerate(dag):
        el, ml = e[:, l : l + 1], m[:, l : l + 1]
        sgn = 1 - 2 * ((z & el) != 0)
        x = np.concatenate((x ^ el, x ^ el), axis=1)
        z = np.concatenate((z ^ ml, z ^ (ml | el)), axis=1)
        c = 0.5 * np.concatenate((c * sgn, c * sgn * (1.0 if d else -1.0)), axis=1)
    return x, z, c


def _label_to_xz(label):
    x = z = 0
    for j, ch in enumerate(reversed(label)):
        if ch in "XY":
            x |= 1 << j
        if ch in "ZY":
            z |= 1 << j
    return x, z, label.count("Y")


def xz_to_label(x, z, n):
    letters = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    return "".join(letters[(x >> j) & 1, (z >> j) & 1] for j in range(n - 1, -1, -1))


def group_qwc(xs, zs):
    """Greedy qubit-wise-commuting grouping. Returns head x masks, head z masks and member index arrays."""
    nmax = len(xs)
    cx = np.zeros(nmax, np.int64)
    cz = np.zeros(nmax, np.int64)
    head = np.empty(nmax, np.int64)
    nh = 0
    for i in range(nmax):
        x, z = xs[i], zs[i]
        ok = np.flatnonzero(((cx[:nh] ^ x) | (cz[:nh] ^ z)) & (x | z) & (cx[:nh] | cz[:nh]) == 0)
        if len(ok):
            j = ok[0]
            cx[j] |= x
            cz[j] |= z
        else:
            j = nh
            cx[j], cz[j] = x, z
            nh += 1
        head[i] = j
    order = np.argsort(head, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(head, minlength=nh))[:-1])
    return cx[:nh].tolist(), cz[:nh].tolist(), members


def fermionic_to_xz(op: FermionicOperator, num_orbs: int, nq: int, shift: int = 0):
    """Jordan-Wigner (qiskit convention) expansion of a FermionicOperator (indices minus shift): unique keys x*2**nq+z and X^x Z^z coefficients."""
    groups = {}
    for key, c in op.operators.items():
        groups.setdefault(tuple(d for _, d in key), []).append(([i - shift for i, _ in key], c))
    keys, cs = [np.zeros(1, np.int64)], [np.zeros(1, complex)]
    for dag, items in groups.items():
        coef = np.array([c for _, c in items], dtype=complex)
        if len(dag) == 0:
            cs[0] += coef.sum()
            continue
        x, z, c = _jw_ladder_product(np.array([i for i, _ in items]), dag, num_orbs)
        keys.append((x * (1 << nq) + z).ravel())
        cs.append((c * coef[:, None]).ravel())
    keys, cs = np.concatenate(keys), np.concatenate(cs)
    uk, inv = np.unique(keys, return_inverse=True)
    uc = np.zeros(len(uk), complex)
    np.add.at(uc, inv, cs)
    keep = np.abs(uc) > 1e-14
    return uk[keep], uc[keep]


def fermionic_to_sparse_pauli(op: FermionicOperator, num_orbs: int, nq: int) -> SparsePauliOp:
    """Same result as JordanWignerMapper().map(FermionicOp(op.get_qiskit_form(num_orbs), 2 * num_orbs))."""
    keys, c = fermionic_to_xz(op, num_orbs, nq)
    if len(keys) == 0:
        return SparsePauliOp("I" * nq, [0.0])
    x, z = keys >> nq, keys & ((1 << nq) - 1)
    bits = np.arange(nq, dtype=np.int64)
    xb = ((x[:, None] >> bits) & 1).astype(bool)
    zb = ((z[:, None] >> bits) & 1).astype(bool)
    return SparsePauliOp(PauliList.from_symplectic(zb, xb), c * (-1j) ** np.bitwise_count(x & z))


class FermionPauliMap:
    """Sparse maps from Pauli expectation values to active-space fermionic expectation values.

    Rows are products of ladder operators on active spin orbitals (indices 0..n_act-1, slowquant
    alpha/beta interleaved order). Works for qiskit's JordanWignerMapper directly, other mappers through QI.op_to_qbit.
    """

    def __init__(self, quantum_interface, num_active_spin_orbs):
        self.QI = quantum_interface
        self.na = num_active_spin_orbs
        self.num_orbs = num_active_spin_orbs // 2
        self.nq = quantum_interface.num_qubits
        self.fast = type(quantum_interface.mapper) is JordanWignerMapper
        self.keys = np.zeros(0, np.int64)
        self._pos = {}
        na = self.na
        self.idx1 = np.array(list(itertools.product(range(na), repeat=2)), dtype=np.int64)
        pairs = np.array(list(itertools.combinations(range(na), 2)), dtype=np.int64).reshape(-1, 2)
        self.pairs = pairs
        P = len(pairs)
        cr, an = np.repeat(pairs, P, axis=0), np.tile(pairs, (P, 1))
        self.idx2 = np.stack((cr[:, 0], an[:, 0], cr[:, 1], an[:, 1]), axis=1)  # (t, u, v, w), t<v, u<w
        self.A1 = self._build(self.idx1[:, [0, 1]], (True, False))
        self.A2 = self._build(self.idx2[:, [0, 2, 3, 1]], (True, True, False, False))
        self.identity = self._index(np.zeros(1, np.int64))[0]

    def _index(self, keys):
        new = [k for k in np.unique(keys) if k not in self._pos]
        for k in new:
            self._pos[k] = len(self._pos)
        if new:
            self.keys = np.concatenate((self.keys, np.array(new, dtype=np.int64)))
        return np.array([self._pos[k] for k in keys], dtype=np.int64)

    def _expand(self, idx, dag):
        if self.fast:
            x, z, c = _jw_ladder_product(idx, dag, self.num_orbs)
            rows = np.repeat(np.arange(len(idx)), x.shape[1])
            return rows, (x * (1 << self.nq) + z).ravel(), c.ravel()
        rows, keys, coef = [], [], []
        for r, ind in enumerate(idx):
            op = a_op_spin(int(ind[0]), dag[0])
            for k, d in zip(ind[1:], dag[1:]):
                op = op * a_op_spin(int(k), d)
            mapped = self.QI.op_to_qbit(op)
            for label, cf in zip(mapped.paulis.to_labels(), mapped.coeffs):
                x, z, ny = _label_to_xz(label)
                rows.append(r)
                keys.append(x * (1 << self.nq) + z)
                coef.append(cf * 1j**ny)
        return np.array(rows), np.array(keys, dtype=np.int64), np.array(coef, dtype=complex)

    def _build(self, idx, dag):
        rows, keys, coef = self._expand(idx, dag)
        keep = np.abs(coef) > 1e-14
        rows, keys, coef = rows[keep], keys[keep], coef[keep]
        cols = self._index(keys)
        return (rows, cols, coef, len(idx))

    def matrix(self, A):
        rows, cols, coef, nrow = A
        return sp.csr_matrix((coef, (rows, cols)), shape=(nrow, len(self.keys)))

    def rdms(self, values):
        """RDM1[t, u] = <a+_t a_u>, RDM2[t, u, v, w] = <a+_t a+_v a_w a_u> from Pauli values (aligned with self.keys)."""
        na = self.na
        d1 = (self.matrix(self.A1) @ values).reshape(na, na)
        g = self.matrix(self.A2) @ values
        t, u, v, w = self.idx2.T
        d2 = np.zeros((na, na, na, na), complex)
        d2[t, u, v, w] = g
        d2[v, w, t, u] = g
        d2[v, u, t, w] = -g
        d2[t, w, v, u] = -g
        return d1, d2

    def hamiltonian_coefficients(self, e_core, h, g):
        """Pauli coefficients c (aligned with self.keys) with <H> = c . <P> for H = e_core + h_tu E_tu + 1/2 g_tuvw e_tuvw."""
        t, u, v, w = self.idx2.T
        G = 0.5 * (g[t, u, v, w] + g[v, w, t, u] - g[v, u, t, w] - g[t, w, v, u])
        c = self.matrix(self.A1).T @ h.ravel() + self.matrix(self.A2).T @ G
        c[self.identity] += e_core
        return c


class _Layout:
    def __init__(self):
        self.excitation_indices = []
        self.excitation_operator_type = []


class FermionSimulator:
    """Exact simulation of fUCC circuits (QI ansatz fUCCSD / fUCCD, JW) in the N-electron determinant space.

    Each circuit block (norm r_k, phase phi_k) is exp(theta_k T_k - theta_k^* T_k^dagger), theta_k = r_k e^{i phi_k};
    states and gradients come from the sparse UPS engine (cost ~ n_params x n_det instead of n_gates x 2^n_qubits).
    The equivalence with the circuit is checked against qiskit at construction (ValueError if it does not hold).
    """

    def __init__(self, QI):
        if type(QI.mapper) is not JordanWignerMapper or QI.ansatz not in ("fUCCSD", "fUCCD"):
            raise ValueError("FermionSimulator: only fUCCSD / fUCCD with JW")
        n, el = QI.num_spin_orbs, QI.num_elec
        self.n = n
        opt = QI.ansatz_options
        isc = opt.get("is_spin_conserving", True)
        lay = _Layout()
        for _ in range(opt.get("n_layers", 1)):
            if opt.get("S", False):
                for a, i in iterate_t1(QI.occ_spin_idx, QI.unocc_spin_idx, is_spin_conserving=isc):
                    lay.excitation_indices.append((i, a))
                    lay.excitation_operator_type.append("single")
            if opt.get("D", False):
                for a, i, b, j in iterate_t2(QI.occ_spin_idx, QI.unocc_spin_idx, is_spin_conserving=isc):
                    lay.excitation_indices.append((i, j, a, b))
                    lay.excitation_operator_type.append("double")
        nk = len(lay.excitation_indices)
        names = [str(p) for p in QI.circuit.parameters]
        if len(names) != 2 * nk:
            raise ValueError("FermionSimulator: parameter count mismatch")
        self.ir = np.array([names.index(f"norm{k:09d}") for k in range(nk)])
        self.ip = np.array([names.index(f"phi{k:09d}") for k in range(nk)])
        ci = get_indexing_generalized(0, n, 0, el[0], el[1])
        dets = np.asarray(ci.idx2det, dtype=np.int64)
        occ = np.array([[t for t in range(n) if (d >> (n - 1 - t)) & 1] for d in dets], dtype=np.int64)
        num_orbs = n // 2
        q = occ // 2 + (occ % 2) * num_orbs
        self.mask = np.sum(np.left_shift(np.int64(1), q), axis=1)
        x, _, c = _jw_ladder_product(occ, (True,) * occ.shape[1], num_orbs)
        self.sign = np.sum(np.where(x == self.mask[:, None], c, 0.0), axis=1)
        ops = {"single": G1_generalized, "double": G2_generalized}
        self.engine = UPSEngine(ci, lay, lambda t, idx: ops[t](*[int(v) for v in idx]))
        self.index = -np.ones(1 << n, dtype=np.int64)
        self.index[self.mask] = np.arange(len(dets))
        sv0 = Statevector(QI.circuit.assign_parameters(np.zeros(len(names)))).data
        self.ref = sv0[self.mask] / self.sign
        if abs(np.linalg.norm(self.ref) - 1) > 1e-10:
            raise ValueError("FermionSimulator: reference state outside the N-electron space")
        self._hcache = None
        rng = np.random.default_rng(11)
        for _ in range(2):
            p = rng.uniform(-0.8, 0.8, len(names))
            a = Statevector(QI.circuit.assign_parameters(p)).data
            b = self.qubit(self.state(p))
            ov = np.vdot(b, a)
            if abs(abs(ov) - 1) > 1e-10 or np.abs(b * ov / abs(ov) - a).max() > 1e-10:
                raise ValueError("FermionSimulator: circuit is not the fermionic fUCC product")

    def thetas(self, p):
        p = np.asarray(p, dtype=float)
        return p[self.ir] * np.exp(1j * p[self.ip])

    def state(self, p):
        return self.engine.state(self.ref, self.thetas(p))

    def qubit(self, c):
        out = np.zeros(1 << self.n, complex)
        out[self.mask] = self.sign * c
        return out

    def hamiltonian(self, keys, coeffs, xs, W):
        """Sparse N-electron matrix of sum_k coeffs_k X^x Z^z (cached for the last keys/coeffs)."""
        h = self._hcache
        if h is not None and np.array_equal(h[0], keys) and np.array_equal(h[1], coeffs):
            return h[2]
        rows, cols, vals = [], [], []
        for x, w in zip(xs, W):
            t = self.index[self.mask ^ x]
            ok = t >= 0
            rows.append(t[ok])
            cols.append(np.nonzero(ok)[0])
            vals.append(self.sign[t[ok]] * self.sign[ok] * w[self.mask[ok]])
        nd = len(self.mask)
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nd, nd))
        self._hcache = (keys.copy(), np.array(coeffs).copy(), H)
        return H

    def energy_gradient(self, H, p):
        p = np.asarray(p, dtype=float)
        th = self.thetas(p)
        c = self.engine.state(self.ref, th)
        hc = H @ c
        g = ups_theta_gradient(self.engine, th, self.ref, hc)
        nk = len(th)
        gR, gI = g[:nk], g[nk:]
        grad = np.zeros(len(p))
        ph = p[self.ip]
        grad[self.ir] = gR * np.cos(ph) + gI * np.sin(ph)
        grad[self.ip] = -gR * th.imag + gI * th.real
        return np.vdot(c, hc).real, grad


class PauliEstimator:
    """Expectation values <X^x Z^z> of the QI circuit.

    exact=True: statevector of the bound circuit (ideal simulation, fast). exact=False: QI sampler with
    qubit-wise-commuting cliques, one device call per batch, QI error mitigation applied.
    Values are cached per parameter set; new Paulis are taken from already measured cliques when possible.
    """

    def __init__(self, quantum_interface, exact=None):
        self.QI = quantum_interface
        self.n = quantum_interface.num_qubits
        self.exact = quantum_interface.shots is None and not quantum_interface.ISA if exact is None else exact
        self._params = None
        self._cache = {}
        self._heads = []
        self._psi = None
        self._sim = None
        self._sim_circuit = None
        self._fsim = None
        self._fsim_circuit = None
        self._wcache = None

    def _reset(self, params):
        self._params = np.array(params, dtype=float)
        self._cache = {}
        self._heads = []
        self._psi = None

    def __call__(self, keys, params=None):
        params = self.QI.parameters if params is None else params
        if self._params is None or not np.array_equal(self._params, params):
            self._reset(params)
        keys = np.asarray(keys, dtype=np.int64)
        missing = np.array([k for k in np.unique(keys) if k not in self._cache], dtype=np.int64)
        if len(missing):
            vals = self._exact(missing) if self.exact else self._sampled(missing)
            self._cache.update(zip(missing.tolist(), vals))
        return np.array([self._cache[k] for k in keys.tolist()], dtype=complex)

    def expectation_multi(self, keys, coeffs, params_list, max_circuits=2000):
        """sum_k coeffs_k <P_k> for each parameter set (independent of the cache); sampler: batched device calls."""
        keys = np.asarray(keys, dtype=np.int64)
        if self.exact:
            fs = self._fermion()
            if fs is not None:
                H = fs.hamiltonian(keys, coeffs, *self._weights(keys, coeffs))
                cs = [fs.state(p) for p in params_list]
                return np.array([np.vdot(c, H @ c) for c in cs])
            xs, W = self._weights(keys, coeffs)
            b = np.arange(1 << self.n, dtype=np.int64)
            out = np.empty(len(params_list), complex)
            step = max(1, (1 << 22) // (1 << self.n))
            for p0 in range(0, len(params_list), step):
                S = self._svs(params_list[p0 : p0 + step])
                out[p0 : p0 + len(S)] = sum(np.einsum("pb,pb,b->p", S[:, b ^ x].conj(), S, w) for x, w in zip(xs, W))
            return out
        n = self.n
        xs, zs = keys >> n, keys & ((1 << n) - 1)
        order = np.argsort(-np.bitwise_count(xs | zs), kind="stable")
        cx, cz, members = group_qwc(xs[order], zs[order])
        members = [order[m] for m in members]
        labels = [xz_to_label(x, z, n) for x, z in zip(cx, cz)]
        ph = coeffs * (-1j) ** np.bitwise_count(xs & zs)
        out = np.zeros(len(params_list), complex)
        shots = (self.QI.shots or 0) * self.QI._circuit_multipl
        budget = max_circuits if shots == 0 else min(max_circuits, max(1, int(2e7 // shots)))
        step = max(1, budget // len(labels))
        QI = self.QI
        for p0 in range(0, len(params_list), step):
            plist = [list(p) for p in params_list[p0 : p0 + step]]
            distr = QI.sampler_distributions(labels, plist)
            if QI.mitigation_flags.do_M_mitigation:
                QI._apply_M_mitigation(distr, QI.circuit)
            if QI.mitigation_flags.do_postselection:
                QI._apply_postselection(distr, [l for l in labels for _ in plist], QI.ansatz_options)
            for h, mem in enumerate(members):
                s = (xs[mem] | zs[mem])[:, None]
                for c in range(len(plist)):
                    d = distr[h * len(plist) + c]
                    k = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
                    pr = np.fromiter(d.values(), dtype=float, count=len(d))
                    out[p0 + c] += ph[mem] @ ((1 - 2 * _popcount_parity(k[None, :] & s)) @ pr)
        return out

    def _weights(self, keys, coeffs):
        """sum_k c_k <X^x Z^z> = sum_x sum_b conj(psi[b^x]) psi[b] W_x[b]; W cached for the last (keys, coeffs)."""
        c = self._wcache
        if c is not None and np.array_equal(c[0], keys) and np.array_equal(c[1], coeffs):
            return c[2], c[3]
        n = self.n
        b = np.arange(1 << n, dtype=np.int64)
        xk, zk = keys >> n, keys & ((1 << n) - 1)
        xs = np.unique(xk)
        W = []
        for x in xs:
            sel = xk == x
            W.append((1 - 2 * _popcount_parity(b[None, :] & zk[sel, None])).T @ coeffs[sel])
        self._wcache = (keys.copy(), np.array(coeffs).copy(), xs, W)
        return xs, W

    def _simulator(self):
        qc = self.QI.circuit
        if self._sim_circuit is not qc:
            self._sim_circuit = qc
            try:
                sim = CircuitSimulator(qc)
                p = np.random.default_rng(7).uniform(-1, 1, qc.num_parameters)
                ref = Statevector(qc.assign_parameters(p)).data
                self._sim = sim if np.abs(sim.statevectors(p)[0] - ref).max() < 1e-10 else None
            except ValueError:
                self._sim = None
        return self._sim

    def _fermion(self):
        qc = self.QI.circuit
        if self._fsim_circuit is not qc:
            self._fsim_circuit = qc
            try:
                self._fsim = FermionSimulator(self.QI)
            except (ValueError, AttributeError, KeyError):
                self._fsim = None
        return self._fsim

    def energy_gradient(self, keys, coeffs, params):
        """Exact (E, dE/dparams) of sum_k coeffs_k X^x Z^z (exact path only): fermionic fUCC simulation if
        possible, else adjoint differentiation of the gate-level circuit. None if neither applies."""
        fs = self._fermion()
        if fs is not None:
            return fs.energy_gradient(fs.hamiltonian(keys, coeffs, *self._weights(keys, coeffs)), params)
        sim = self._simulator()
        if sim is None:
            return None
        xs, W = self._weights(keys, coeffs)
        b = np.arange(1 << self.n, dtype=np.int64)

        def apply_h(psi):
            out = np.zeros_like(psi)
            for x, w in zip(xs, W):
                out += (w * psi)[b ^ x]
            return out

        return sim.energy_gradient(params, apply_h)

    def _svs(self, params_list):
        fs = self._fermion()
        if fs is not None:
            return np.array([fs.qubit(fs.state(p)) for p in params_list])
        sim = self._simulator()
        if sim is None:
            return np.array([self._sv(p) for p in params_list])
        return sim.statevectors(params_list)

    def _sv(self, params):
        fs = self._fermion()
        if fs is not None:
            return fs.qubit(fs.state(params))
        sim = self._simulator()
        if sim is not None:
            return sim.statevectors(params)[0]
        qc = self.QI.circuit.assign_parameters(np.asarray(params, dtype=float))
        return np.ascontiguousarray(Statevector(qc).data)

    def statevector(self):
        if self._psi is None:
            self._psi = self._sv(self._params)
        return self._psi

    def _exact(self, keys, psi=None):
        psi = self.statevector() if psi is None else psi
        n = self.n
        b = np.arange(1 << n, dtype=np.int64)
        xs, zs = keys >> n, keys & ((1 << n) - 1)
        out = np.empty(len(keys), complex)
        for x in np.unique(xs):
            sel = np.where(xs == x)[0]
            v = psi[b ^ x].conj() * psi
            if len(sel) > n:
                f = v.copy()
                h = 1
                while h < len(f):
                    f = f.reshape(-1, 2, h)
                    f = np.stack((f[:, 0] + f[:, 1], f[:, 0] - f[:, 1]), axis=1).reshape(-1)
                    h *= 2
                out[sel] = f[zs[sel]]
            else:
                sgn = 1 - 2 * _popcount_parity(b[None, :] & zs[sel, None])
                out[sel] = sgn @ v
        return out

    def _sampled(self, keys):
        n = self.n
        xs, zs = keys >> n, keys & ((1 << n) - 1)
        out = np.empty(len(keys), complex)
        hx = np.array([h[0] for h in self._heads], dtype=np.int64)
        hz = np.array([h[1] for h in self._heads], dtype=np.int64)
        assign = -np.ones(len(keys), dtype=np.int64)
        if len(hx):
            for i, (x, z) in enumerate(zip(xs, zs)):
                ok = np.flatnonzero(((hx ^ x) | (hz ^ z)) & (x | z) == 0)
                if len(ok):
                    assign[i] = ok[0]
        new = np.where(assign < 0)[0]
        order = new[np.argsort(-np.bitwise_count(xs[new] | zs[new]), kind="stable")]
        cx, cz, members = group_qwc(xs[order], zs[order])
        members = [order[m] for m in members]
        if cx:
            labels = [xz_to_label(x, z, n) for x, z in zip(cx, cz)]
            QI = self.QI
            distr = QI.sampler_distributions(labels, [list(self._params)])
            if QI.mitigation_flags.do_M_mitigation:
                QI._apply_M_mitigation(distr, QI.circuit)
            if QI.mitigation_flags.do_postselection:
                QI._apply_postselection(distr, labels, QI.ansatz_options)
            for x, z, d in zip(cx, cz, distr):
                k = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
                p = np.fromiter(d.values(), dtype=float, count=len(d))
                self._heads.append((x, z, k, p))
            for j, mem in enumerate(members):
                assign[mem] = len(self._heads) - len(cx) + j
        for h in np.unique(assign):
            sel = np.where(assign == h)[0]
            _, _, k, p = self._heads[h]
            s = (xs[sel] | zs[sel])[:, None]
            out[sel] = (1 - 2 * _popcount_parity(k[None, :] & s)) @ p
        return out * (-1j) ** np.bitwise_count(xs & zs)


class CircuitSimulator:
    """Batched statevector simulation of a parameterized circuit (many parameter sets per numpy pass).

    Gate list compiled once; consecutive one-qubit gates are fused per qubit (constant parts pre-multiplied).
    Supports one- and two-qubit gates with angles linear in one parameter. Raises ValueError for anything else
    (caller falls back to qiskit). energy_gradient: exact gradient by adjoint differentiation (two sweeps).
    """

    _fixed2 = {"cx", "cz"}

    def __init__(self, qc):
        self.n = qc.num_qubits
        self.params = list(qc.parameters)
        pidx = {p: i for i, p in enumerate(self.params)}
        self.gphase = self._lin(qc.global_phase, pidx)
        self.ops = []
        self.gates = []
        pend = {}

        def flush(q):
            items = pend.pop(q, [])
            merged = []
            for it in items:
                if it[0] == "m" and merged and merged[-1][0] == "m":
                    merged[-1] = ("m", it[1] @ merged[-1][1])
                else:
                    merged.append(it)
            if len(merged) == 1 and merged[0][0] == "m":
                self.ops.append(("1c", q, merged[0][1]))
            elif merged:
                self.ops.append(("1q", q, merged))

        for ins in qc.data:
            op, qs = ins.operation, [qc.find_bit(q).index for q in ins.qubits]
            name = op.name
            if name in ("barrier", "id", "delay"):
                continue
            if len(qs) == 1:
                if op.params and name in ("rx", "ry", "rz", "p", "u1"):
                    item = ("p" if name == "u1" else name, self._lin(op.params[0], pidx))
                elif not any(hasattr(x, "parameters") for x in op.params):
                    item = ("m", np.asarray(op.to_matrix(), dtype=complex))
                else:
                    raise ValueError(f"unsupported parameterized gate {name}")
                pend.setdefault(qs[0], []).append(item)
                self.gates.append(("1", qs[0], item))
            elif len(qs) == 2 and not any(hasattr(x, "parameters") for x in op.params):
                flush(qs[0])
                flush(qs[1])
                if name in self._fixed2:
                    g = (name, qs[0], qs[1])
                else:
                    g = ("2q", qs, np.asarray(op.to_matrix(), dtype=complex).reshape(2, 2, 2, 2))
                self.ops.append(g)
                self.gates.append(g)
            else:
                raise ValueError(f"unsupported gate {name}")
        for q in list(pend):
            flush(q)

    @staticmethod
    def _lin(expr, pidx):
        if not hasattr(expr, "parameters") or len(expr.parameters) == 0:
            return (-1, 0.0, float(expr))
        if len(expr.parameters) != 1:
            raise ValueError("gate angle depends on several parameters")
        (p,) = expr.parameters
        b = complex(expr.bind({p: 0.0}))
        a = complex(expr.bind({p: 1.0})) - b
        if abs(complex(expr.bind({p: 2.0})) - b - 2 * a) > 1e-12 or abs(a.imag) + abs(b.imag) > 1e-14:
            raise ValueError("non-linear gate angle")
        return (pidx[p], a.real, b.real)

    @staticmethod
    def _angle(lin, P):
        i, a, b = lin
        return np.full(len(P), b) if i < 0 else a * P[:, i] + b

    @staticmethod
    def _rot(kind, t):
        """Rotation matrices (m, 2, 2) and their angle derivatives."""
        m = len(t)
        G = np.zeros((m, 2, 2), complex)
        D = np.zeros((m, 2, 2), complex)
        c, s = np.cos(t / 2), np.sin(t / 2)
        if kind == "rx":
            G[:, 0, 0] = G[:, 1, 1] = c
            G[:, 0, 1] = G[:, 1, 0] = -1j * s
            D[:, 0, 0] = D[:, 1, 1] = -0.5 * s
            D[:, 0, 1] = D[:, 1, 0] = -0.5j * c
        elif kind == "ry":
            G[:, 0, 0] = G[:, 1, 1] = c
            G[:, 0, 1], G[:, 1, 0] = -s, s
            D[:, 0, 0] = D[:, 1, 1] = -0.5 * s
            D[:, 0, 1], D[:, 1, 0] = -0.5 * c, 0.5 * c
        elif kind == "rz":
            G[:, 0, 0], G[:, 1, 1] = np.exp(-0.5j * t), np.exp(0.5j * t)
            D[:, 0, 0], D[:, 1, 1] = -0.5j * np.exp(-0.5j * t), 0.5j * np.exp(0.5j * t)
        else:
            G[:, 0, 0], G[:, 1, 1] = 1.0, np.exp(1j * t)
            D[:, 1, 1] = 1j * np.exp(1j * t)
        return G, D

    def _mat(self, items, P):
        M = None
        for kind, val in items:
            G = val if kind == "m" else self._rot(kind, self._angle(val, P))[0]
            M = G if M is None else G @ M
        return np.broadcast_to(M, (len(P), 2, 2)) if M.ndim == 2 else M

    def _apply1(self, psi, q, M):
        m = psi.shape[0]
        v = psi.reshape(m, -1, 2, 1 << q)
        a0, a1 = v[:, :, 0], v[:, :, 1]
        if M.ndim == 2:
            return np.stack((M[0, 0] * a0 + M[0, 1] * a1, M[1, 0] * a0 + M[1, 1] * a1), axis=2).reshape(m, -1)
        M = M[:, :, :, None, None]
        return np.stack((M[:, 0, 0] * a0 + M[:, 0, 1] * a1, M[:, 1, 0] * a0 + M[:, 1, 1] * a1), axis=2).reshape(m, -1)

    def _apply2(self, psi, g, adjoint=False):
        m, n = psi.shape[0], self.n
        if g[0] in ("cx", "cz"):
            c, t = g[1], g[2]
            hi, lo = max(c, t), min(c, t)
            v = psi.reshape(m, 1 << (n - 1 - hi), 2, 1 << (hi - lo - 1), 2, 1 << lo)
            if g[0] == "cz":
                v[:, :, 1, :, 1] *= -1
            elif c == hi:
                x = v[:, :, 1]
                x[...] = x[:, :, :, ::-1].copy()
            else:
                x = v[:, :, :, :, 1]
                x[...] = x[:, :, ::-1].copy()
            return psi
        (q0, q1), U = g[1], g[2]
        if adjoint:
            U = U.reshape(4, 4).conj().T.reshape(2, 2, 2, 2)
        x = np.moveaxis(psi.reshape((m,) + (2,) * n), (n - q1, n - q0), (-2, -1))
        return np.ascontiguousarray(
            np.moveaxis(np.einsum("p...cd,abcd->p...ab", x, U), (-2, -1), (n - q1, n - q0))
        ).reshape(m, -1)

    def statevectors(self, params) -> np.ndarray:
        P = np.atleast_2d(np.asarray(params, dtype=float))
        m, n = len(P), self.n
        psi = np.zeros((m, 1 << n), complex)
        psi[:, 0] = 1.0
        for op in self.ops:
            if op[0] == "1c":
                psi = self._apply1(psi, op[1], op[2])
            elif op[0] == "1q":
                psi = self._apply1(psi, op[1], self._mat(op[2], P))
            else:
                psi = self._apply2(psi, op)
        return psi * np.exp(1j * self._angle(self.gphase, P))[:, None]

    def energy_gradient(self, params, apply_h):
        """E = <psi|H|psi> and dE/dparams by adjoint differentiation; apply_h(psi) returns H psi (1-D)."""
        p = np.asarray(params, dtype=float)
        P = p[None]
        psi = self.statevectors(P)
        lam = apply_h(psi[0])[None]
        e = np.vdot(psi[0], lam[0]).real
        grad = np.zeros(len(p))
        for g in reversed(self.gates):
            if g[0] != "1":
                psi = self._apply2(psi, g, adjoint=True)
                lam = self._apply2(lam, g, adjoint=True)
                continue
            q, (kind, val) = g[1], g[2]
            if kind == "m":
                Md = val.conj().T
                psi = self._apply1(psi, q, Md)
                lam = self._apply1(lam, q, Md)
                continue
            G, D = self._rot(kind, self._angle(val, P))
            Gd = G[0].conj().T
            psi = self._apply1(psi, q, Gd)
            if val[0] >= 0:
                grad[val[0]] += 2 * val[1] * np.vdot(lam[0], self._apply1(psi, q, D[0])[0]).real
            lam = self._apply1(lam, q, Gd)
        return e, grad

def pauli_product(k1, k2, nq):
    """Keys and signs of X^x1 Z^z1 X^x2 Z^z2 = sign X^(x1^x2) Z^(z1^z2) (broadcasting)."""
    m = (1 << nq) - 1
    return k1 ^ k2, 1 - 2 * (np.bitwise_count((k1 & m) & (k2 >> nq)) & 1).astype(np.int8)


def anticommute(k1, k2, nq):
    """True where the Paulis anticommute."""
    m = (1 << nq) - 1
    return (np.bitwise_count(((k1 & m) & (k2 >> nq)) ^ ((k1 >> nq) & (k2 & m))) & 1).astype(bool)


def dagger_coefficients(keys, coeffs, nq):
    """Coefficients of O^dagger for O = sum_k coeffs_k X^x Z^z."""
    m = (1 << nq) - 1
    return coeffs.conj() * (1.0 - 2.0 * (np.bitwise_count((keys >> nq) & keys & m) & 1))


def commutator_terms(ka, ca, kb, cb, nq):
    """Nonzero Pauli terms of [A, B]: (index into A terms, index into B terms, key, coefficient)."""
    ia, ib = np.nonzero(anticommute(ka[:, None], kb[None, :], nq))
    k, s = pauli_product(ka[ia], kb[ib], nq)
    return ia, ib, k, 2.0 * s * ca[ia] * cb[ib]


def double_commutator_terms(ka, ca, kh, ch, kb, cb, nq, symmetrized=True):
    """Nonzero Pauli terms of [A,[H,B]] or (1/2)([A,[H,B]] + [[A,H],B]): (index into A terms, index into B terms, key, coefficient)."""
    ih, ib = np.nonzero(anticommute(kh[:, None], kb[None, :], nq))
    khb, s_hb = pauli_product(kh[ih], kb[ib], nq)
    ia, j = np.nonzero(anticommute(ka[:, None], khb[None, :], nq))
    k, s = pauli_product(ka[ia], khb[j], nq)
    w = (2.0 if symmetrized else 4.0) * s * s_hb[j]
    out_a, out_b, out_k, out_c = [ia], [ib[j]], [k], [w * ca[ia] * ch[ih[j]] * cb[ib[j]]]
    if symmetrized:
        ia, ih = np.nonzero(anticommute(ka[:, None], kh[None, :], nq))
        kah, s_ah = pauli_product(ka[ia], kh[ih], nq)
        i2, ib = np.nonzero(anticommute(kah[:, None], kb[None, :], nq))
        k, s = pauli_product(kah[i2], kb[ib], nq)
        out_a.append(ia[i2])
        out_b.append(ib)
        out_k.append(k)
        out_c.append(2.0 * s * s_ah[i2] * ca[ia[i2]] * ch[ih[i2]] * cb[ib])
    return np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_k), np.concatenate(out_c)
