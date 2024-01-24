import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import scipy
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, QNSPSA, SLSQP, SPSA

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.qiskit_interface.base import FermionicOperator
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.operators import Epq, hamiltonian_pauli_0i_0a
from slowquant.qiskit_interface.optimizers import RotoSolve
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_electronic_energy,
    get_orbital_gradient,
)


class WaveFunction:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        quantum_interface: QuantumInterface,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UCC wave function.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            c_orthonormal: Initial orbital coefficients.
            h_ao: One-electron integrals in AO for Hamiltonian.
            g_ao: Two-electron integrals in AO.
            include_active_kappa: Include active-active orbital rotations.
        """
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        self._c_orthonormal = c_orthonormal
        self.h_ao = h_ao
        self.g_ao = g_ao
        self.inactive_spin_idx = []
        self.virtual_spin_idx = []
        self.active_spin_idx = []
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        self.active_spin_idx_shifted = []
        self.active_occ_spin_idx_shifted = []
        self.active_unocc_spin_idx_shifted = []
        self.num_elec = num_elec
        self.num_spin_orbs = num_spin_orbs
        self.num_orbs = num_spin_orbs // 2
        self._include_active_kappa = include_active_kappa
        self.num_active_elec = 0
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
        self.num_virtual_spin_orbs = 0
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._h_mo = None
        self._g_mo = None
        active_space = []
        orbital_counter = 0
        for i in range(num_elec - cas[0], num_elec):
            active_space.append(i)
            orbital_counter += 1
        for i in range(num_elec, num_elec + 2 * cas[1] - orbital_counter):
            active_space.append(i)
        for i in range(num_elec):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_occ_spin_idx.append(i)
                self.num_active_spin_orbs += 1
                self.num_active_elec += 1
            else:
                self.inactive_spin_idx.append(i)
                self.num_inactive_spin_orbs += 1
        for i in range(num_elec, num_spin_orbs):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_unocc_spin_idx.append(i)
                self.num_active_spin_orbs += 1
            else:
                self.virtual_spin_idx.append(i)
                self.num_virtual_spin_orbs += 1
        if len(self.active_spin_idx) != 0:
            active_shift = np.min(self.active_spin_idx)
            for active_idx in self.active_spin_idx:
                self.active_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_spin_idx:
                self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_spin_idx:
                self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)
        self.num_inactive_orbs = self.num_inactive_spin_orbs // 2
        self.num_active_orbs = self.num_active_spin_orbs // 2
        self.num_virtual_orbs = self.num_virtual_spin_orbs // 2
        # Contruct spatial idx
        self.inactive_idx: list[int] = []
        self.virtual_idx: list[int] = []
        self.active_idx: list[int] = []
        self.active_occ_idx: list[int] = []
        self.active_unocc_idx: list[int] = []
        for idx in self.inactive_spin_idx:
            if idx // 2 not in self.inactive_idx:
                self.inactive_idx.append(idx // 2)
        for idx in self.active_spin_idx:
            if idx // 2 not in self.active_idx:
                self.active_idx.append(idx // 2)
        for idx in self.virtual_spin_idx:
            if idx // 2 not in self.virtual_idx:
                self.virtual_idx.append(idx // 2)
        for idx in self.active_occ_spin_idx:
            if idx // 2 not in self.active_occ_idx:
                self.active_occ_idx.append(idx // 2)
        for idx in self.active_unocc_spin_idx:
            if idx // 2 not in self.active_unocc_idx:
                self.active_unocc_idx.append(idx // 2)
        # Find non-redundant kappas
        self.kappa = []
        self.kappa_idx = []
        self.kappa_idx_dagger = []
        self.kappa_redundant = []
        self.kappa_redundant_idx = []
        self._kappa_old = []
        self._kappa_redundant_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self.kappa_redundant.append(0.0)
                    self._kappa_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self.kappa_redundant.append(0.0)
                    self._kappa_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self.kappa_redundant.append(0.0)
                        self._kappa_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                if include_active_kappa:
                    if p in self.active_occ_idx and q in self.active_occ_idx:
                        self.kappa_redundant.append(0.0)
                        self._kappa_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                    if p in self.active_unocc_idx and q in self.active_unocc_idx:
                        self.kappa_redundant.append(0.0)
                        self._kappa_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                self.kappa.append(0.0)
                self._kappa_old.append(0.0)
                self.kappa_idx.append([p, q])
                self.kappa_idx_dagger.append([q, p])
        # HF like orbital rotation indecies
        self.kappa_hf_like_idx = []
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append([p, q])
                elif p in self.inactive_idx and q in self.active_unocc_idx:
                    self.kappa_hf_like_idx.append([p, q])
                elif p in self.active_occ_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append([p, q])
        self._energy_elec = None
        # Setup Qiskit stuff
        self.QI = quantum_interface
        self.QI.construct_circuit(self.num_active_orbs, self.num_active_elec)

    @property
    def c_orthonormal(self) -> np.ndarray:
        """Get orthonormalization coefficients (MO coefficients).

        Returns:
            Orthonormalization coefficients.
        """
        return self._c_orthonormal

    @c_orthonormal.setter
    def c_orthonormal(self, c: np.ndarray) -> None:
        """Set orthonormalization coefficients.

        Args:
            c: Orthonormalization coefficients.
        """
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._c_orthonormal = c

    @property
    def c_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        kappa_mat = np.zeros_like(self._c_orthonormal)
        if len(self.kappa) != 0:
            if np.max(np.abs(self.kappa)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        if len(self.kappa_redundant) != 0:
            if np.max(np.abs(self.kappa_redundant)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa_redundant, self.kappa_redundant_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        return np.matmul(self._c_orthonormal, scipy.linalg.expm(-kappa_mat))

    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = one_electron_integral_transform(self.c_trans, self.h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_trans, self.g_ao)
        return self._g_mo

    @property
    def ansatz_parameters(self) -> list[float]:
        return self.QI.parameters

    @ansatz_parameters.setter
    def ansatz_parameters(self, parameters: list[float]) -> None:
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._energy_elec = None
        self.QI.parameters = parameters

    @property
    def rdm1(self) -> np.ndarray:
        """Calcuate one-electron reduced density matrix.

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    rdm1_op = Epq(p, q).get_folded_operator(
                        self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                    )
                    val = self.QI.quantum_expectation_value(rdm1_op)
                    self._rdm1[p_idx, q_idx] = val
                    self._rdm1[q_idx, p_idx] = val
            # trace = 0
            # for i in range(self.num_active_orbs):
            #    trace += self._rdm1[i, i]
            # for i in range(self.num_active_orbs):
            #    self._rdm1[i, i] = self._rdm1[i, i] * self.num_active_elec / trace
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        """Calcuate two-electron reduced density matrix.

        Returns:
            Two-electron reduced density matrix.
        """
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                )
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_idx = r - self.num_inactive_orbs
                        if p == q:
                            s_lim = r + 1
                        elif p == r:
                            s_lim = q + 1
                        elif q < r:
                            s_lim = p
                        else:
                            s_lim = p + 1
                        for s in range(self.num_inactive_orbs, s_lim):
                            s_idx = s - self.num_inactive_orbs
                            pdm2_op = (Epq(p, q) * Epq(r, s)).get_folded_operator(
                                self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                            )
                            val = self.QI.quantum_expectation_value(pdm2_op)
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val
            # trace = 0
            # for i in range(self.num_active_orbs):
            #    for j in range(self.num_active_orbs):
            #        trace += self._rdm2[i, i, j, j]
            # for i in range(self.num_active_orbs):
            #    for j in range(self.num_active_orbs):
            #        self._rdm2[i, i, j, j] = (
            #            self._rdm2[i, i, j, j] * self.num_active_elec * (self.num_active_elec - 1) / trace
            #        )
        return self._rdm2

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            I = C_\text{MO}S\C_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = one_electron_integral_transform(self.c_trans, overlap_integral)
        one = np.identity(len(S_ortho))
        diff = np.abs(S_ortho - one)
        print("Max ortho-normal diff:", np.max(diff))

    @property
    def energy_elec(self) -> float:
        if self._energy_elec is None:
            H = hamiltonian_pauli_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            self._energy_elec = calc_energy_theta(self.ansatz_parameters, H, self.QI)
        return self._energy_elec

    def run_vqe_2step(
        self,
        ansatz_optimizer: str,
        orbital_optimization: bool = False,
        tol: float = 1e-8,
        maxiter: int = 1000,
        is_silent_subiterations: bool = False,
    ) -> None:
        """Run VQE of wave function."""
        global iteration
        global start

        def print_progress(x, energy_func, silent: bool) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration
            global start
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            if not silent:
                e_str = f"{energy_func(x):3.16f}"
                print(f"--------{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            start = time.time()  # type: ignore

        def print_progress_SPSA(
            ___,
            theta,
            f_val,
            _,
            __,
            silent: bool,
        ) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration
            global start
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{f_val:3.12f}"
            if not silent:
                print(f"--------{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            start = time.time()  # type: ignore

        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree]")
        for full_iter in range(0, int(maxiter)):
            full_start = time.time()
            iteration = 0  # type: ignore
            start = time.time()  # type: ignore

            # Do ansatz optimization
            if not is_silent_subiterations:
                print("--------Ansatz optimization")
                print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree]")
            H = hamiltonian_pauli_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            energy_theta = partial(
                calc_energy_theta,
                operator=H,
                quantum_interface=self.QI,
            )
            gradient_theta = partial(ansatz_parameters_gradient, operator=H, quantum_interface=self.QI)
            if ansatz_optimizer.lower() == "slsqp":
                print_progress_ = partial(
                    print_progress, energy_func=energy_theta, silent=is_silent_subiterations
                )
                optimizer = SLSQP(maxiter=maxiter, ftol=tol, callback=print_progress_)
            elif ansatz_optimizer.lower() == "l_bfgs_b":
                print_progress_ = partial(
                    print_progress, energy_func=energy_theta, silent=is_silent_subiterations
                )
                optimizer = L_BFGS_B(maxiter=maxiter, tol=tol, callback=print_progress_)
            elif ansatz_optimizer.lower() == "cobyla":
                print_progress_ = partial(
                    print_progress, energy_func=energy_theta, silent=is_silent_subiterations
                )
                optimizer = COBYLA(maxiter=maxiter, tol=tol, callback=print_progress_)
            elif ansatz_optimizer.lower() == "rotosolve":
                print_progress_ = partial(
                    print_progress, energy_func=energy_theta, silent=is_silent_subiterations
                )
                optimizer = RotoSolve(maxiter=maxiter, tol=tol, callback=print_progress_)
            elif ansatz_optimizer.lower() == "spsa":
                print("WARNING: Convergence tolerence cannot be set for SPSA; using qiskit default")
                print_progress_SPSA_ = partial(print_progress_SPSA, silent=is_silent_subiterations)
                optimizer = SPSA(maxiter=maxiter, callback=print_progress_SPSA_)
            elif ansatz_optimizer.lower() == "qnspsa":
                optimizer = QNSPSA(
                    QNSPSA.get_fidelity(self.QI.ansatz, sampler=self.qiskit_sampler),
                    maxiter=maxiter,
                    tol=tol,
                    callback=print_progress_SPSA,
                )
            else:
                raise ValueError(f"Unknown optimizer: {ansatz_optimizer}")
            res = optimizer.minimize(energy_theta, self.ansatz_parameters, jac=gradient_theta)
            self.ansatz_parameters = res.x.tolist()

            if orbital_optimization and len(self.kappa) != 0:
                iteration = 0  # type: ignore
                start = time.time()  # type: ignore
                if not is_silent_subiterations:
                    print("--------Orbital optimization")
                    print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree]")
                energy_oo = partial(
                    calc_energy_oo,
                    wf=self,
                )
                gradiet_oo = partial(
                    orbital_rotation_gradient,
                    wf=self,
                )

                print_progress_ = partial(
                    print_progress, energy_func=energy_oo, silent=is_silent_subiterations
                )
                optimizer = L_BFGS_B(maxiter=maxiter, tol=tol, callback=print_progress_)
                res = optimizer.minimize(energy_oo, [0.0] * len(self.kappa_idx), jac=gradiet_oo)
                for i in range(len(self.kappa)):
                    self.kappa[i] = 0.0
                    self._kappa_old[i] = 0.0
                for i in range(len(self.kappa_redundant)):
                    self.kappa_redundant[i] = 0.0
                    self._kappa_redundant_old[i] = 0.0
            else:
                # If theres is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa) == 0:
                    print(
                        "WARNING: No orbital optimization performed, because there is no non-redundant orbital parameters"
                    )
                break

            e_new = res.fun
            time_str = f"{time.time() - full_start:7.2f}"  # type: ignore
            e_str = f"{e_new:3.12f}"
            print(f"{str(full_iter+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            if abs(e_new - e_old) < tol:
                break
            e_old = e_new
        self._energy_elec = e_new

    def run_vqe_1step(
        self,
        optimizer_name: str,
        orbital_optimization: bool = False,
        tol: float = 1e-8,
        maxiter: int = 1000,
    ) -> None:
        """Run VQE of wave function."""
        if not orbital_optimization:
            raise ValueError("Does only work with orbital optimization right now")
        global iteration
        global start
        iteration = 0  # type: ignore
        start = time.time()  # type: ignore

        def print_progress(x, energy_func) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration
            global start
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{energy_func(x):3.12f}"
            print(f"{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            start = time.time()  # type: ignore

        def print_progress_SPSA(
            ___,
            theta,
            f_val,
            _,
            __,
        ) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration
            global start
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{f_val:3.12f}"
            print(f"{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            start = time.time()  # type: ignore

        print("Iteration # | Iteration time [s] | Electronic energy [Hartree]")
        energy_both = partial(
            calc_energy_both,
            wf=self,
        )
        gradient_both = partial(
            calc_gradient_both,
            wf=self,
        )
        if optimizer_name.lower() == "slsqp":
            print_progress_ = partial(print_progress, energy_func=energy_both)
            optimizer = SLSQP(maxiter=maxiter, ftol=tol, callback=print_progress_)
        elif optimizer_name.lower() == "l_bfgs_b":
            print_progress_ = partial(print_progress, energy_func=energy_both)
            optimizer = L_BFGS_B(maxiter=maxiter, tol=tol, callback=print_progress_)
        elif optimizer_name.lower() == "cobyla":
            print_progress_ = partial(print_progress, energy_func=energy_both)
            optimizer = COBYLA(maxiter=maxiter, tol=tol, callback=print_progress_)
        elif optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa) != 0:
                raise ValueError(
                    "Cannot use rotosolve together with orbital optimization in the one-step solver."
                )
            print_progress_ = partial(print_progress, energy_func=energy_both)
            optimizer = RotoSolve(maxiter=maxiter, tol=tol, callback=print_progress_)
        elif optimizer_name.lower() == "spsa":
            print("WARNING: Convergence tolerence cannot be set for SPSA; using qiskit default")
            optimizer = SPSA(maxiter=maxiter, callback=print_progress_SPSA)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        parameters = self.kappa + self.ansatz_parameters
        res = optimizer.minimize(energy_both, parameters, jac=gradient_both)
        self.ansatz_parameters = res.x[len(self.kappa) :].tolist()
        for i in range(len(self.kappa)):
            self.kappa[i] = 0.0
            self._kappa_old[i] = 0.0
        for i in range(len(self.kappa_redundant)):
            self.kappa_redundant[i] = 0.0
            self._kappa_redundant_old[i] = 0.0
        self._energy_elec = res.fun


def calc_energy_theta(parameters, operator: FermionicOperator, quantum_interface: QuantumInterface) -> float:
    quantum_interface.parameters = parameters
    return quantum_interface.quantum_expectation_value(operator)


def calc_energy_oo(kappa, wf) -> float:
    kappa_mat = np.zeros_like(wf.c_orthonormal)
    for kappa_val, (p, q) in zip(np.array(kappa) - np.array(wf._kappa_old), wf.kappa_idx):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    if len(wf.kappa_redundant) != 0:
        if np.max(np.abs(wf.kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(
                np.array(wf.kappa_redundant) - np.array(wf._kappa_redundant_old), wf.kappa_redundant_idx
            ):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(wf.c_orthonormal, scipy.linalg.expm(-kappa_mat))
    wf._kappa_old = kappa.copy()
    wf._kappa_redundant_old = wf.kappa_redundant.copy()
    # Moving expansion point of kappa
    wf.c_orthonormal = c_trans
    rdms = ReducedDenstiyMatrix(
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_active_orbs,
        rdm1=wf.rdm1,
        rdm2=wf.rdm2,
    )
    energy = get_electronic_energy(rdms, wf.h_mo, wf.g_mo, wf.num_inactive_orbs, wf.num_active_orbs)
    return energy


def calc_energy_both(parameters, wf) -> float:
    kappa = parameters[: len(wf.kappa)]
    theta = parameters[len(wf.kappa) :]
    assert len(theta) == len(wf.ansatz_parameters)
    # Do orbital partial
    kappa_mat = np.zeros_like(wf.c_orthonormal)
    for kappa_val, (p, q) in zip(np.array(kappa) - np.array(wf._kappa_old), wf.kappa_idx):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    if len(wf.kappa_redundant) != 0:
        if np.max(np.abs(wf.kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(
                np.array(wf.kappa_redundant) - np.array(wf._kappa_redundant_old), wf.kappa_redundant_idx
            ):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(wf.c_orthonormal, scipy.linalg.expm(-kappa_mat))
    wf._kappa_old = kappa.copy()
    wf._kappa_redundant_old = wf.kappa_redundant.copy()
    # Moving expansion point of kappa
    wf.c_orthonormal = c_trans
    # Build operator
    wf.ansatz_parameters = theta.copy()  # Reset rdms
    H = hamiltonian_pauli_0i_0a(wf.h_mo, wf.g_mo, wf.num_inactive_orbs, wf.num_active_orbs)
    H = H.get_folded_operator(wf.num_inactive_orbs, wf.num_active_orbs, wf.num_virtual_orbs)
    return wf.QI.quantum_expectation_value(H)


def orbital_rotation_gradient(
    placeholder,
    wf,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to orbital rotations.

    Args:
        wf: Wave function object.

    Return:
        Electronic gradient with respect to orbital rotations.
    """
    rdms = ReducedDenstiyMatrix(
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_active_orbs,
        rdm1=wf.rdm1,
        rdm2=wf.rdm2,
    )
    gradient = get_orbital_gradient(
        rdms, wf.h_mo, wf.g_mo, wf.kappa_idx, wf.num_inactive_orbs, wf.num_active_orbs
    )
    return gradient


def ansatz_parameters_gradient(
    parameters: list[float], operator, quantum_interface: QuantumInterface
) -> np.ndarray:
    """Calculate gradient with respect to ansatz parameters.

    The gradient is calculated using parameter-shift assuming three eigenvalues of the generators.
    This works for fermionic generators of the type:

    .. math::
        \\hat{G}_{pq} = \\hat{a}^\\dagger_p \\hat{a}_q - \\hat{a}_q^\\dagger \\hat{a}_p

    and,

    .. math::
        \\hat{G}_{pqrs} = \\hat{a}^\\dagger_p \\hat{a}^\\dagger_q \\hat{a}_r \\hat{a}_s - \\hat{a}^\\dagger_s \\hat{a}^\\dagger_r \\hat{a}_p \\hat{a}_q

    The parameter-shift rule is implemented as:

    .. math::
        \\begin{align}
        \\frac{\\partial E(\\boldsymbol{\\theta})}{\\partial \\theta_i} &=
        \\frac{\\sqrt{2}+1}{2\\sqrt{2}}\\left(E\\left(\\theta_i += \\frac{\\pi}{2} - \\frac{\\pi}{4}\\right) - E\\left(\\theta_i += -\\frac{\\pi}{2} + \\frac{\\pi}{4}\\right)\\right)\\\\
        &- \\frac{\\sqrt{2}-1}{2\\sqrt{2}}\\left(E\\left(\\theta_i += \\frac{\\pi}{2} + \\frac{\\pi}{4}\\right) - E\\left(\\theta_i += -\\frac{\\pi}{2} - \\frac{\\pi}{4}\\right)\\right)
        \\end{align}


    #. 10.1088/1367-2630/ac2cb3: Eq. (F14)
    #. 10.22331/q-2022-03-30-677: Eq. (8)

    Args:
        parameters: Ansatz parameters.
        operator: Operator which the derivative is with respect to.
        quantum_interface: Interface to call quantum device.

    Returns:
        Gradient with repsect to ansatz parameters.
    """
    gradient = np.zeros(len(parameters))
    h = np.pi / 2 - np.pi / 4
    h2 = np.pi / 2 + np.pi / 4
    for i in range(len(parameters)):
        parameters[i] += h
        Ep = quantum_interface.quantum_expectation_value(operator, custom_parameters=parameters)
        parameters[i] -= h
        parameters[i] += h2
        Ep2 = quantum_interface.quantum_expectation_value(operator, custom_parameters=parameters)
        parameters[i] -= h2
        parameters[i] -= h
        Em = quantum_interface.quantum_expectation_value(operator, custom_parameters=parameters)
        parameters[i] += h
        parameters[i] -= h2
        Em2 = quantum_interface.quantum_expectation_value(operator, custom_parameters=parameters)
        parameters[i] += h2
        gradient[i] = (2 ** (1 / 2) + 1) / (2 * 2 ** (1 / 2)) * (Ep - Em) - (2 ** (1 / 2) - 1) / (
            2 * 2 ** (1 / 2)
        ) * (Ep2 - Em2)
    return gradient


def calc_gradient_both(parameters, wf) -> np.ndarray:
    gradient = np.zeros(len(parameters))
    theta = parameters[len(wf.kappa) :]
    assert len(theta) == len(wf.ansatz_parameters)
    kappa_grad = orbital_rotation_gradient(0, wf)
    gradient[: len(wf.kappa)] = kappa_grad
    H = hamiltonian_pauli_0i_0a(wf.h_mo, wf.g_mo, wf.num_inactive_orbs, wf.num_active_orbs)
    H = H.get_folded_operator(wf.num_inactive_orbs, wf.num_active_orbs, wf.num_virtual_orbs)
    theta_grad = ansatz_parameters_gradient(theta, H, wf.QI)
    gradient[len(wf.kappa) :] = theta_grad
    return gradient
