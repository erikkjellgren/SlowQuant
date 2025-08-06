# pylint: disable=too-many-lines
import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import scipy
from qiskit import QuantumCircuit
from qiskit.primitives import (
    BaseEstimator,
    BaseEstimatorV2,
    BaseSamplerV1,
    BaseSamplerV2,
)
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.optimizers import RotoSolve
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_electronic_energy,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    Epq,
    hamiltonian_0i_0a,
    one_elec_op_0i_0a,
)


class WaveFunctionSA:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        states: tuple[list[list[float]], list[list[str]]],
        quantum_interface: QuantumInterface,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for circuit based state-averaged wave function.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            c_orthonormal: Initial orbital coefficients.
            h_ao: One-electron integrals in AO for Hamiltonian.
            g_ao: Two-electron integrals in AO.
            states: States to include in the state-averaged expansion.
                    Tuple of lists containing weights and determinants.
                    Each state in SA can be constructed of several dets.
                    Ordering: left-to-right, alpha-beta alternating.
            quantum_interface: QuantumInterface.
            include_active_kappa: Include active-active orbital rotations.
        """
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        if isinstance(quantum_interface.ansatz, QuantumCircuit):
            print("WARNING: A QI with a custom Ansatz was passed. VQE will only work with COBYLA optimizer.")
        if cas[0] % 2 == 1:
            raise ValueError(
                f"Wave function only implemented for an even number of active electrons. Got; {cas[0]}"
            )
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
        self.num_active_elec = cas[0]
        self.num_active_elec_alpha = self.num_active_elec // 2
        self.num_active_elec_beta = self.num_active_elec // 2
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
        self.num_virtual_spin_orbs = 0
        self._rdm1 = None
        self._rdm2 = None
        self._h_mo = None
        self._g_mo = None
        self.do_trace_corrected = True
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
        self.kappa_no_activeactive_idx = []
        self.kappa_no_activeactive_idx_dagger = []
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
                if not (p in self.active_idx and q in self.active_idx):
                    self.kappa_no_activeactive_idx.append([p, q])
                    self.kappa_no_activeactive_idx_dagger.append([q, p])
                self.kappa.append(0.0)
                self._kappa_old.append(0.0)
                self.kappa_idx.append([p, q])
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
        self._energy_elec: float | None = None
        self.num_states = len(states[0])
        self.states = states
        # Setup Qiskit stuff
        self.QI = quantum_interface
        self.QI.construct_circuit(
            self.num_active_orbs, (self.num_active_elec_alpha, self.num_active_elec_beta)
        )

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
        self._state_energies = None
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
        """Getter for ansatz parameters.

        Returns:
            Ansatz parameters.
        """
        return self.QI.parameters

    @ansatz_parameters.setter
    def ansatz_parameters(self, parameters: list[float]) -> None:
        """Setter for ansatz paramters.

        Args:
            parameters: New ansatz paramters.
        """
        self._rdm1 = None
        self._rdm2 = None
        self._state_energies = None
        self._state_ci_coeffs = None
        self._ci_coeffs = None
        self.QI.parameters = parameters

    def change_primitive(self, primitive: BaseSamplerV1 | BaseSamplerV2, verbose: bool = True) -> None:
        """Change the primitive expectation value calculator.

        Args:
            primitive: Primitive object.
            verbose: Print more info.
        """
        if verbose:
            print(
                "Using this function is only recommended for switching from ideal simulator to shot-noise or quantum hardware.\n \
                Multiple switching back and forth can lead to un-expected outcomes and is an experimental feature.\n"
            )

        if isinstance(primitive, (BaseEstimator, BaseEstimatorV2)):
            raise ValueError("Estimator is not supported.")
        self.QI._primitive = primitive  # pylint: disable=protected-access
        if verbose:
            if self.QI.mitigation_flags.do_M_ansatz0:
                print("Reset RDMs, energies, QI metrics, and correlation matrix.")
            else:
                print("Reset RDMs, energies, and QI metrics.")
        self._rdm1 = None
        self._rdm2 = None
        self._energy_elec = None
        self.QI.total_device_calls = 0
        self.QI.total_shots_used = 0
        self.QI.total_paulis_evaluated = 0

        # Reset circuit and initiate re-transpiling
        ISA_old = self.QI.ISA
        self._reconstruct_circuit()  # Reconstruct circuit but keeping parameters
        self.QI._transpiled = False  # pylint: disable=protected-access
        self.QI.ISA = ISA_old  # Redo ISA including transpilation if requested
        self.QI.shots = self.QI.shots  # Redo shots parameter check

        if verbose:
            self.QI.get_info()

    def _reconstruct_circuit(self) -> None:
        """Construct circuit again."""
        self.QI.construct_circuit(
            self.num_active_orbs, (self.num_active_elec // 2, self.num_active_elec // 2)
        )

    @property
    def rdm1(self) -> np.ndarray:
        r"""Calcuate one-electron reduced density matrix.

        The trace condition is enforced:

        .. math::
            \sum_i\Gamma^{[1]}_{ii} = N_e

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
                    val = 0.0
                    for coeffs, csf in zip(self.states[0], self.states[1]):
                        val += self.QI.quantum_expectation_value_csfs((coeffs, csf), rdm1_op, (coeffs, csf))
                    val = val / self.num_states
                    self._rdm1[p_idx, q_idx] = val  # type: ignore [index]
                    self._rdm1[q_idx, p_idx] = val  # type: ignore [index]
            if self.do_trace_corrected:
                trace = 0.0
                for i in range(self.num_active_orbs):
                    trace += self._rdm1[i, i]  # type: ignore [index]
                for i in range(self.num_active_orbs):
                    self._rdm1[i, i] = self._rdm1[i, i] * self.num_active_elec / trace  # type: ignore [index]
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        r"""Calcuate two-electron reduced density matrix.

        The trace condition is enforced:

        .. math::
            \sum_{ij}\Gamma^{[2]}_{iijj} = N_e(N_e-1)

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
                            val = 0.0
                            for coeffs, csf in zip(self.states[0], self.states[1]):
                                val += self.QI.quantum_expectation_value_csfs(
                                    (coeffs, csf), pdm2_op, (coeffs, csf)
                                )
                            val = val / self.num_states
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore [index]
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val  # type: ignore [index]
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val  # type: ignore [index]
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val  # type: ignore [index]
            if self.do_trace_corrected:
                trace = 0.0
                for i in range(self.num_active_orbs):
                    for j in range(self.num_active_orbs):
                        trace += self._rdm2[i, i, j, j]  # type: ignore [index]
                for i in range(self.num_active_orbs):
                    for j in range(self.num_active_orbs):
                        self._rdm2[i, i, j, j] = (  # type: ignore [index]
                            self._rdm2[i, i, j, j] * self.num_active_elec * (self.num_active_elec - 1) / trace  # type: ignore [index]
                        )
        return self._rdm2

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = one_electron_integral_transform(self.c_trans, overlap_integral)
        one = np.identity(len(S_ortho))
        diff = np.abs(S_ortho - one)
        print("Max ortho-normal diff:", np.max(diff))

    def run_vqe_2step(
        self,
        ansatz_optimizer: str,
        orbital_optimization: bool = False,
        tol: float = 1e-8,
        maxiter: int = 1000,
        is_silent_subiterations: bool = False,
    ) -> None:
        """Run VQE of wave function."""
        global iteration  # pylint: disable=global-variable-undefined
        global start  # pylint: disable=global-variable-undefined

        if isinstance(self.QI.ansatz, QuantumCircuit) and not ansatz_optimizer.lower() == "cobyla":
            raise ValueError("Custom Ansatz in QI only works with COBYLA as optimizer")

        def print_progress(x, energy_func, silent: bool) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
                energy_func: Function to calculate energy.
                silent: Supress print.
            """
            global iteration  # pylint: disable=global-variable-undefined
            global start  # pylint: disable=global-variable-undefined
            time_str = f"{time.time() - start:7.2f}"  # type: ignore [name-defined] # pylint: disable=used-before-assignment
            if not silent:
                e_str = f"{energy_func(x):3.16f}"
                print(
                    f"--------{str(iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}"  # type: ignore [name-defined] # pylint: disable=used-before-assignment
                )
            iteration += 1  # type: ignore
            start = time.time()  # type: ignore

        def print_progress_SPSA(
            ___,
            theta,  # pylint: disable=unused-argument
            f_val,
            _,
            __,
            silent: bool,
        ) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                theta: Wave function parameters.
                f_val: Function value at theta.
                silent: Supress print.
            """
            global iteration  # pylint: disable=global-variable-undefined
            global start  # pylint: disable=global-variable-undefined
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{f_val:3.12f}"
            if not silent:
                print(f"--------{str(iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
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
            H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
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
            elif ansatz_optimizer.lower() == "slsqp_nograd":
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
                optimizer = RotoSolve(
                    self.QI.grad_param_R,
                    self.QI.param_names,
                    maxiter=maxiter,
                    tol=tol,
                    callback=print_progress_,
                )
            elif ansatz_optimizer.lower() == "spsa":
                print("WARNING: Convergence tolerence cannot be set for SPSA; using qiskit default")
                print_progress_SPSA_ = partial(print_progress_SPSA, silent=is_silent_subiterations)
                optimizer = SPSA(maxiter=maxiter, callback=print_progress_SPSA_)
            else:
                raise ValueError(f"Unknown optimizer: {ansatz_optimizer}")
            if ansatz_optimizer.lower() == "slsqp_nograd":
                res = optimizer.minimize(energy_theta, self.ansatz_parameters)
            else:
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
                for i in range(len(self.kappa)):  # pylint: disable=consider-using-enumerate
                    self.kappa[i] = 0.0
                    self._kappa_old[i] = 0.0
                for i in range(len(self.kappa_redundant)):  # pylint: disable=consider-using-enumerate
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
            print(f"{str(full_iter + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
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
        global iteration  # pylint: disable=global-variable-undefined
        global start  # pylint: disable=global-variable-undefined
        iteration = 0  # type: ignore
        start = time.time()  # type: ignore

        if isinstance(self.QI.ansatz, QuantumCircuit) and not optimizer_name.lower() == "cobyla":
            raise ValueError("Custom Ansatz in QI only works with COBYLA as optimizer")

        def print_progress(x, energy_func) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
                energy_func: Function to calculate energy.
            """
            global iteration  # pylint: disable=global-variable-undefined
            global start  # pylint: disable=global-variable-undefined
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{energy_func(x):3.12f}"
            print(f"{str(iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            start = time.time()  # type: ignore

        def print_progress_SPSA(
            ___,
            theta,  # pylint: disable=unused-argument
            f_val,
            _,
            __,
        ) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                theta: Wave function parameters.
                f_val: Function value at theta.
            """
            global iteration  # pylint: disable=global-variable-undefined
            global start  # pylint: disable=global-variable-undefined
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{f_val:3.12f}"
            print(f"{str(iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
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
            optimizer = RotoSolve(
                self.QI.grad_param_R, self.QI.param_names, maxiter=maxiter, tol=tol, callback=print_progress_
            )
        elif optimizer_name.lower() == "spsa":
            print("WARNING: Convergence tolerence cannot be set for SPSA; using qiskit default")
            optimizer = SPSA(maxiter=maxiter, callback=print_progress_SPSA)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        parameters = self.kappa + self.ansatz_parameters
        res = optimizer.minimize(energy_both, parameters, jac=gradient_both)
        self.ansatz_parameters = res.x[len(self.kappa) :].tolist()
        for i in range(len(self.kappa)):  # pylint: disable=consider-using-enumerate
            self.kappa[i] = 0.0
            self._kappa_old[i] = 0.0
        for i in range(len(self.kappa_redundant)):  # pylint: disable=consider-using-enumerate
            self.kappa_redundant[i] = 0.0
            self._kappa_redundant_old[i] = 0.0
        self._energy_elec = res.fun

    def _do_state_ci(self) -> None:
        r"""Do subspace diagonalisation.

        #. 10.1103/PhysRevLett.122.230401, Eq. 2
        """
        state_H = np.zeros((self.num_states, self.num_states))
        H = hamiltonian_0i_0a(
            self.h_mo,
            self.g_mo,
            self.num_inactive_orbs,
            self.num_active_orbs,
        ).get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
        for i, (coeffs_i, csf_i) in enumerate(zip(self.states[0], self.states[1])):
            for j, (coeffs_j, csf_j) in enumerate(zip(self.states[0], self.states[1])):
                if j > i:
                    continue
                state_H[i, j] = state_H[j, i] = self.QI.quantum_expectation_value_csfs(
                    (coeffs_i, csf_i), H, (coeffs_j, csf_j)
                )
        eigval, eigvec = scipy.linalg.eig(state_H)
        sorting = np.argsort(eigval)
        self._state_energies = np.real(eigval[sorting])
        self._state_ci_coeffs = np.real(eigvec[:, sorting])

    @property
    def energy_states(self) -> np.ndarray:
        """Get state specific energies.

        Returns:
            State specific energies.
        """
        if self._state_energies is None:
            self._do_state_ci()
        if self._state_energies is None:
            raise ValueError("_state_energies is None")
        return self._state_energies

    @property
    def excitation_energies(self) -> np.ndarray:
        r"""Get excitation energies.

        .. math::
            \varepsilon_n = E_n - E_0

        Returns:
            Excitation energies.
        """
        energies = np.zeros(self.num_states - 1)
        for i, energy in enumerate(self.energy_states[1:]):
            energies[i] = energy - self.energy_states[0]
        return energies

    def get_transition_property(self, ao_integral: np.ndarray) -> np.ndarray:
        r"""Get transition property.

        .. math::
            t_n = \left<0\left|\hat{O}\right|n\right>

        Args:
            ao_integral: Operator integrals in AO basis.

        Returns:
            Transition property.
        """
        if self._state_ci_coeffs is None:
            self._do_state_ci()
        if self._state_ci_coeffs is None:
            raise ValueError("_state_ci_coeffs is None")
        mo_integral = one_electron_integral_transform(self.c_trans, ao_integral)
        transition_property = np.zeros(self.num_states - 1)
        state_op = np.zeros((self.num_states, self.num_states))
        op = one_elec_op_0i_0a(mo_integral, self.num_inactive_orbs, self.num_active_orbs).get_folded_operator(
            self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
        )
        for i, (coeffs_i, csf_i) in enumerate(zip(self.states[0], self.states[1])):
            for j, (coeffs_j, csf_j) in enumerate(zip(self.states[0], self.states[1])):
                state_op[i, j] = self.QI.quantum_expectation_value_csfs(
                    (coeffs_i, csf_i), op, (coeffs_j, csf_j)
                )
        for i in range(self.num_states - 1):
            transition_property[i] = self._state_ci_coeffs[:, i + 1] @ state_op @ self._state_ci_coeffs[:, 0]
        return transition_property

    def get_oscillator_strenghts(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        r"""Get oscillator strengths between ground state and excited states.

        .. math::
            f_n = \frac{2}{3}\varepsilon_n\left|\left<0\left|\hat{\boldsymbol{\mu}}\right|n\right>\right|^2

        Args:
            dipole_integrals: Dipole integrals in AO basis.

        Returns:
            Oscillator strengths.
        """
        transition_dipole_x = self.get_transition_property(dipole_integrals[0])
        transition_dipole_y = self.get_transition_property(dipole_integrals[1])
        transition_dipole_z = self.get_transition_property(dipole_integrals[2])
        osc_strs = np.zeros(self.num_states - 1)
        for idx, (excitation_energy, td_x, td_y, td_z) in enumerate(
            zip(self.excitation_energies, transition_dipole_x, transition_dipole_y, transition_dipole_z)
        ):
            osc_strs[idx] = 2 / 3 * excitation_energy * (td_x**2 + td_y**2 + td_z**2)
        return osc_strs

    def _calc_energy_elec(self, ISA_csfs_option: int = 0, rep: int = 1) -> list[float] | float:
        """Run electronic energy simulation.

        Args:
            ISA_csfs_option: Option for how to deal with superposition state circuits.
            rep: Repeat energy calculation for statistics.

        Returns:
            Electronic energy.
        """
        H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
        if rep == 1:
            energy = 0.0
            for coeffs, csf in zip(self.states[0], self.states[1]):
                energy += self.QI.quantum_expectation_value_csfs(
                    (coeffs, csf),
                    H,
                    (coeffs, csf),
                    ISA_csfs_option=ISA_csfs_option,
                )

            return energy / self.num_states
        energies = []
        for _ in range(rep):
            energy = 0.0
            for coeffs, csf in zip(self.states[0], self.states[1]):
                energy += self.QI.quantum_expectation_value_csfs(
                    (coeffs, csf),
                    H,
                    (coeffs, csf),
                    ISA_csfs_option=ISA_csfs_option,
                )
            energy = energy / self.num_states
            energies.append(energy)
        print("Mean: ", np.mean(energies))
        print("Std: ", np.std(energies))

        return energies


def calc_energy_theta(
    parameters: list[float], operator: FermionicOperator, quantum_interface: QuantumInterface
) -> float:
    """Calculate electronic energy using expectation values.

    Args:
        paramters: Ansatz paramters.
        operator: Hamiltonian operator.
        quantum_interface: QuantumInterface.

    Returns:
        Electronic energy.
    """
    quantum_interface.parameters = parameters
    return quantum_interface.quantum_expectation_value(operator)


def calc_energy_oo(kappa: list[float], wf: WaveFunctionSA) -> float:
    """Calculate electronic energy using RDMs.

    Args:
        kappa: Orbital rotation parameters.
        wf: Wave function object.

    Returns:
        Electronic energy.
    """
    kappa_mat = np.zeros_like(wf.c_orthonormal)
    for kappa_val, (p, q) in zip(
        np.array(kappa) - np.array(wf._kappa_old), wf.kappa_idx  # pylint: disable=protected-access
    ):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    if len(wf.kappa_redundant) != 0:
        if np.max(np.abs(wf.kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(
                np.array(wf.kappa_redundant)
                - np.array(wf._kappa_redundant_old),  # pylint: disable=protected-access
                wf.kappa_redundant_idx,
            ):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(wf.c_orthonormal, scipy.linalg.expm(-kappa_mat))
    wf._kappa_old = kappa.copy()  # pylint: disable=protected-access
    wf._kappa_redundant_old = wf.kappa_redundant.copy()  # pylint: disable=protected-access
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


def calc_energy_both(parameters: list[float], wf: WaveFunctionSA) -> float:
    """Calculate electronic energy.

    Args:
        parameters: Ansatz and orbital rotation parameters.
        wf: Wave function object.

    Returns:
        Electronic energy.
    """
    kappa = parameters[: len(wf.kappa)]
    theta = parameters[len(wf.kappa) :]
    assert len(theta) == len(wf.ansatz_parameters)
    # Do orbital partial
    kappa_mat = np.zeros_like(wf.c_orthonormal)
    for kappa_val, (p, q) in zip(
        np.array(kappa) - np.array(wf._kappa_old), wf.kappa_idx  # pylint: disable=protected-access
    ):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    if len(wf.kappa_redundant) != 0:
        if np.max(np.abs(wf.kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(
                np.array(wf.kappa_redundant)
                - np.array(wf._kappa_redundant_old),  # pylint: disable=protected-access
                wf.kappa_redundant_idx,
            ):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(wf.c_orthonormal, scipy.linalg.expm(-kappa_mat))
    wf._kappa_old = kappa.copy()  # pylint: disable=protected-access
    wf._kappa_redundant_old = wf.kappa_redundant.copy()  # pylint: disable=protected-access
    # Moving expansion point of kappa
    wf.c_orthonormal = c_trans
    # Build operator
    wf.ansatz_parameters = theta.copy()  # Reset rdms
    H = hamiltonian_0i_0a(wf.h_mo, wf.g_mo, wf.num_inactive_orbs, wf.num_active_orbs)
    H = H.get_folded_operator(wf.num_inactive_orbs, wf.num_active_orbs, wf.num_virtual_orbs)
    return wf.QI.quantum_expectation_value(H)


def orbital_rotation_gradient(
    placeholder,  # pylint: disable=unused-argument
    wf: WaveFunctionSA,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to orbital rotations.

    Args:
        placeholder: Placeholder for kappa parameters, these are fetched OOP style instead.
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
    parameters: list[float], operator: FermionicOperator, quantum_interface: QuantumInterface
) -> np.ndarray:
    r"""Calculate gradient with respect to ansatz parameters.

    Args:
        parameters: Ansatz parameters.
        operator: Operator which the derivative is with respect to.
        quantum_interface: Interface to call quantum device.

    Returns:
        Gradient with repsect to ansatz parameters.
    """
    gradient = np.zeros(len(parameters))
    for i in range(len(parameters)):  # pylint: disable=consider-using-enumerate
        R = quantum_interface.grad_param_R[quantum_interface.param_names[i]]
        e_vals_grad = get_energy_evals_for_grad(operator, quantum_interface, parameters, i, R)
        grad = 0.0
        for j, mu in enumerate(list(range(1, 2 * R + 1))):
            x_mu = (2 * mu - 1) / (2 * R) * np.pi
            grad += e_vals_grad[j] * (-1) ** (mu - 1) / (4 * R * (np.sin(1 / 2 * x_mu)) ** 2)
        gradient[i] = grad
    return gradient


def get_energy_evals_for_grad(
    operator: FermionicOperator,
    quantum_interface: QuantumInterface,
    parameters: list[float],
    idx: int,
    R: int,
) -> list[float]:
    r"""Get energy evaluations needed for the gradient calculation.

    The gradient formula is defined for x=0,
    so x_shift is used to shift ensure we can get the energy in the point we actually want.

    Args:
        operator: Operator which the derivative is with respect to.
        parameters: Paramters.
        idx: Parameter idx.
        R: Parameter to control we get the needed points.

    Returns:
        Energies in a few fixed points.
    """
    e_vals = []
    x = parameters.copy()
    x_shift = x[idx]
    for mu in range(1, 2 * R + 1):
        x_mu = (2 * mu - 1) / (2 * R) * np.pi
        x[idx] = x_mu + x_shift
        e_vals.append(quantum_interface.quantum_expectation_value(operator, custom_parameters=x))
    return e_vals


def calc_gradient_both(parameters: list[float], wf: WaveFunctionSA) -> np.ndarray:
    """Calculate electronic gradient.

    Args:
        parameters: Ansatz and orbital rotation parameters.
        wf: Wave function object.

    Returns:
        Electronic gradient.
    """
    gradient = np.zeros(len(parameters))
    theta = parameters[len(wf.kappa) :]
    assert len(theta) == len(wf.ansatz_parameters)
    kappa_grad = orbital_rotation_gradient(0, wf)
    gradient[: len(wf.kappa)] = kappa_grad
    H = hamiltonian_0i_0a(wf.h_mo, wf.g_mo, wf.num_inactive_orbs, wf.num_active_orbs)
    H = H.get_folded_operator(wf.num_inactive_orbs, wf.num_active_orbs, wf.num_virtual_orbs)
    theta_grad = ansatz_parameters_gradient(theta, H, wf.QI)
    gradient[len(wf.kappa) :] = theta_grad
    return gradient
