import time

import numpy as np
import scipy

from slowquant.qiskit_interface.fast_pauli import FermionPauliMap, PauliEstimator
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.integral_direct import orbital_gradient


def shift_rule(R: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Shifts and weights of the general parameter-shift rule for E(x) with frequencies 1..R (Wierichs et al., Quantum 6, 677 (2022)).

    order 1: E'(0) = sum_mu w_mu E(x_mu), x_mu = (2mu-1)pi/(2R), mu = 1..2R.
    order 2: E''(0) = sum_mu w_mu E(x_mu), x_0 = 0, x_mu = mu pi/R, mu = 1..2R-1.
    """
    if order == 1:
        mu = np.arange(1, 2 * R + 1)
        x = (2 * mu - 1) * np.pi / (2 * R)
        return x, (-1.0) ** (mu - 1) / (4 * R * np.sin(x / 2) ** 2)
    mu = np.arange(1, 2 * R)
    x = mu * np.pi / R
    return np.concatenate(([0.0], x)), np.concatenate(
        ([-(2 * R**2 + 1) / 6], (-1.0) ** (mu - 1) / (2 * np.sin(x / 2) ** 2))
    )


class GeneralizedWaveFunctionCircuitDirect(GeneralizedWaveFunctionUPS):
    """Generalized (GHF spin-orbital) orbital-optimized circuit wave function, integral-direct and quantum-hardware compatible.

    Supports the same one-electron Hamiltonians as GeneralizedWaveFunctionUPS (non-relativistic, X2C, ECP, finite B).

    The device only delivers Pauli expectation values of the ansatz circuit. RDMs, energies and orbital
    gradients are classical contractions (integral-direct Fock matrices, no MO integrals beyond (N a|a a)).
    Theta parameters are the QI circuit parameters (norm r and phase phi per excitation).
    """

    def __init__(
        self,
        cas: tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,
        mol,
        quantum_interface,
        do_x2c: bool = False,
        include_active_kappa: bool = False,
        ecp: bool = False,
        B: list | None = None,
        orig: tuple | None = None,
        direct=None,
        exact: bool | None = None,
    ) -> None:
        """exact: True = ideal statevector Pauli expectation values, False = QI sampler (shots, mitigation),
        None = statevector if QI.shots is None and no ISA transpilation. direct: integral backend
        (default GHFDirect; e.g. GHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10))."""
        super().__init__(
            cas, np.asarray(mo_coeffs, dtype=complex), mol, "adapt", do_x2c, {}, include_active_kappa, ecp, B, orig
        )
        if direct is not None:
            self.set_integral_backend(direct)
        self.QI = quantum_interface
        self.QI.construct_circuit(
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            self.num_active_spin_orbs,
            (self.num_active_elec_alpha, self.num_active_elec_beta),
        )
        Rr, Rp = self.QI.grad_param_R_r, self.QI.grad_param_R_phi
        self.R = np.array([Rr[n] if n in Rr else Rp[n] for n in self.QI.param_names], dtype=int)
        self.pmap = FermionPauliMap(self.QI, self.num_active_spin_orbs)
        self.estimator = PauliEstimator(self.QI, exact)
        n = self.QI.circuit.num_parameters // 2
        self._thetas_real = [0.0] * n
        self._thetas_imag = [0.0] * n
        self.QI.parameters = [0.0] * (2 * n)
        self._hk = None
        self._fixed_rdms = False
        self.num_energy_evals = 0

    @property
    def thetas(self) -> list[float]:
        return list(self.QI.parameters)

    def set_thetas(self, theta_r: list[float], theta_phi: list[float]) -> None:
        """Set circuit parameters (norms r and phases phi)."""
        self._thetas_real = [float(x) for x in theta_r]
        self._thetas_imag = [float(x) for x in theta_phi]
        self.QI.parameters = self._thetas_real + self._thetas_imag
        self._rdm1 = None
        self._rdm2 = None
        self._energy_elec = None

    def set_thetas_initial(self, theta_real: list[float], theta_imag: list[float]) -> None:
        """Set parameters from Cartesian theta = theta_real + i theta_imag."""
        self.set_thetas(np.hypot(theta_real, theta_imag), np.arctan2(theta_imag, theta_real))

    def set_integral_backend(self, direct) -> None:
        super().set_integral_backend(direct)
        self._hk = None

    def _set_orbitals(self, C: np.ndarray) -> None:
        super()._set_orbitals(C)
        self._hk = None

    def _make_rdms(self) -> None:
        self._rdm1, self._rdm2 = self.pmap.rdms(self.estimator(self.pmap.keys))

    @property
    def energy_elec(self) -> float:
        if self._energy_elec is None:
            self._energy_elec = self.ds.energy(self.rdm1, self.rdm2)
        return self._energy_elec

    def _hamiltonian_paulis(self) -> tuple[np.ndarray, np.ndarray]:
        if self._hk is None:
            c = self.pmap.hamiltonian_coefficients(*self.ds.active_integrals())
            nz = np.abs(c) > 1e-12
            self._hk = (self.pmap.keys[nz], c[nz])
        return self._hk

    def theta_energy(self, p) -> float:
        """<H> at circuit parameters p for the current orbitals."""
        keys, c = self._hamiltonian_paulis()
        self.num_energy_evals += 1
        return float((self.estimator(keys, p) @ c).real)

    def _energies(self, points) -> np.ndarray:
        keys, c = self._hamiltonian_paulis()
        self.num_energy_evals += len(points)
        return self.estimator.expectation_multi(keys, c, points).real

    def theta_gradient(self, p=None) -> np.ndarray:
        """Parameter-shift gradient (all shifted circuits in one batch)."""
        p = np.array(self.thetas if p is None else p, dtype=float)
        pts, w, idx = [], [], []
        for i, R in enumerate(self.R):
            for x, wt in zip(*shift_rule(R, 1)):
                q = p.copy()
                q[i] += x
                pts.append(q)
                w.append(wt)
                idx.append(i)
        g = np.zeros(len(p))
        np.add.at(g, idx, np.array(w) * self._energies(pts))
        return g

    def theta_hessian(self, p=None) -> np.ndarray:
        """Parameter-shift Hessian: second-derivative rule on the diagonal, product of first-derivative rules off it."""
        p = np.array(self.thetas if p is None else p, dtype=float)
        n = len(p)
        pts, w, ij = [], [], []
        for i in range(n):
            for x, wt in zip(*shift_rule(self.R[i], 2)):
                q = p.copy()
                q[i] += x
                pts.append(q)
                w.append(wt)
                ij.append((i, i))
            xi, wi = shift_rule(self.R[i], 1)
            for j in range(i + 1, n):
                xj, wj = shift_rule(self.R[j], 1)
                for a, wa in zip(xi, wi):
                    for b, wb in zip(xj, wj):
                        q = p.copy()
                        q[i] += a
                        q[j] += b
                        pts.append(q)
                        w.append(wa * wb)
                        ij.append((i, j))
        H = np.zeros((n, n))
        ij = np.array(ij)
        np.add.at(H, (ij[:, 0], ij[:, 1]), np.array(w) * self._energies(pts))
        return H + np.triu(H, 1).T

    def _optimize_thetas(self, gtol: float, maxiter: int, method: str = "l-bfgs-b") -> float:
        """Theta optimization at fixed orbitals. method: any scipy.optimize.minimize method; Newton-type methods
        ('trust-exact', 'trust-krylov', 'newton-cg') use the parameter-shift Hessian."""
        if self._fixed_rdms:
            return self.energy_elec
        p0 = np.array(self.thetas, dtype=float)
        opts = {"maxiter": maxiter}
        kw = {}
        m = method.lower()
        if m in (
            "l-bfgs-b",
            "bfgs",
            "cg",
            "slsqp",
            "tnc",
            "trust-exact",
            "trust-krylov",
            "trust-ncg",
            "newton-cg",
        ):
            kw["jac"] = self.theta_gradient
            opts["gtol"] = gtol
        if m == "l-bfgs-b":
            opts.update({"ftol": 1e-16, "maxcor": 30})
        if m in ("trust-exact", "trust-krylov", "trust-ncg", "newton-cg"):
            kw["hess"] = self.theta_hessian
            if m == "newton-cg":
                opts.pop("gtol")
        res = scipy.optimize.minimize(self.theta_energy, p0, method=method, options=opts, **kw)
        n = len(p0) // 2
        self.set_thetas(res.x[:n], res.x[n:])
        return float(res.fun)

    def orbital_gradient_max(self) -> float:
        ds = self.ds
        ds.set_rdms(self.rdm1, self.rdm2)
        g = orbital_gradient(ds.gen_fock(), [tuple(i) for i in self.kappa_spin_idx])
        return float(np.abs(g).max()) if len(g) else 0.0

    def run_wf_optimization_2step_direct(
        self,
        tol: float = 1e-5,
        maxiter: int = 50,
        theta_method: str = "l-bfgs-b",
        theta_gtol: float = 1e-8,
        theta_maxiter: int = 500,
        orbital_maxiter: int = 50,
        **orbital_options,
    ) -> None:
        """Hardware-friendly two-step OO: theta optimization (quantum) then orbital optimization at fixed measured
        RDMs (classical, integral-direct L-BFGS, see run_wf_optimization_direct).
        Quantum cost per macro iteration: theta optimization + one RDM measurement."""
        print(
            f"### Circuit OO: {len(self.kappa_spin_idx)} orbital rotations, {len(self.thetas)} circuit parameters"
        )
        print(
            "Macro | Time [s] | E after theta opt. [Hartree] | max |orbital grad| | E after orbital opt. | energy evals"
        )
        for it in range(maxiter):
            t0 = time.time()
            e_theta = self._optimize_thetas(theta_gtol, theta_maxiter, theta_method)
            gmax = self.orbital_gradient_max()
            if gmax < tol:
                print(
                    f"{it + 1:^5} | {time.time() - t0:^8.2f} | {e_theta:^29.12f} | {gmax:^18.3e} | converged"
                )
                break
            self._fixed_rdms = True
            try:
                self.run_wf_optimization_direct(tol=tol, maxiter=orbital_maxiter, **orbital_options)
            finally:
                self._fixed_rdms = False
            self._energy_elec = None
            e_orb = self.ds.energy(self.rdm1, self.rdm2)
            print(
                f"{it + 1:^5} | {time.time() - t0:^8.2f} | {e_theta:^29.12f} | {gmax:^18.3e} | {e_orb:^20.12f} | {self.num_energy_evals}"
            )
        else:
            print("WARNING: circuit OO did not converge")
        self._energy_elec = None
