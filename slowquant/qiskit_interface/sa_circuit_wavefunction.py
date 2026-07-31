import time
from functools import partial

import numpy as np
import pyscf
import scipy
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV1, BaseEstimatorV2, BaseSamplerV1, BaseSamplerV2

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.SlowQuant import SlowQuant
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.integral_manager import IntegralManager
from slowquant.unitary_coupled_cluster.operators import (
    Epq,
    hamiltonian_0i_0a,
    one_elec_op_0i_0a,
)
from slowquant.unitary_coupled_cluster.optimizers import Optimizers


class WaveFunctionSACircuit:
    def __init__(
        self,
        active_space: tuple[int, int] | tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,
        integral_generator: SlowQuant | pyscf.gto.mole.Mole,
        states: tuple[list[list[float]], list[list[str]]],
        quantum_interface: QuantumInterface,
        include_active_kappa: bool = True,
        resolve_unpaired_idx: str = "both",
    ) -> None:
        """Initialize circuit based state-averaged UPS wave function.

        Args:
            active_space: (num_active_elec, num_active_orbs) or ((num_active_elec_alpha, num_active_elec_beta), num_active_orbs),
                 orbitals are counted in spatial basis.
            mo_coeffs: Initial orbital coefficients.
            integral_generator: Integral generator object.
            states: States to include in the state-averaged expansion.
                    Tuple of lists containing weights and determinants.
                    Each state in SA can be constructed of several dets.
                    Ordering: left-to-right, alpha-beta alternating.
            quantum_interface: QuantumInterface.
            include_active_kappa: Include active-active orbital rotations.
            resolve_unpaired_idx: Specify how to resolve spatial index with respect to occupation of reference determinant.
                                  'both', include spatial index in both occ and unocc idx list.
                                  'occ', include spatial index only in occ idx list.
                                  'unocc', include spatial index only in unocc idx list.
        """
        if isinstance(quantum_interface.ansatz, QuantumCircuit):
            print("WARNING: A QI with a custom Ansatz was passed. VQE will only work with COBYLA optimizer.")
        if len(active_space) != 2:
            raise ValueError(f"cas must have two elements, got {len(active_space)} elements.")
        if isinstance(active_space[0], int):
            if active_space[0] % 2 == 0:
                cas = ((active_space[0] // 2, active_space[0] // 2), active_space[1])
            else:
                # Odd number of electrons mean one electron must be unpaired.
                cas = ((active_space[0] // 2 + 1, active_space[0] // 2), active_space[1])
        else:
            cas = ((active_space[0][0], active_space[0][1]), active_space[1])
        # Init stuff
        self.int_gen = IntegralManager(integral_generator)
        if np.sum(cas[0]) > self.int_gen.num_elec:
            raise ValueError("More active electrons than total number electrons.")
        if (np.sum(cas[0]) % 2 == 0 and self.int_gen.num_elec % 2 == 1) or (
            np.sum(cas[0]) % 2 == 1 and self.int_gen.num_elec % 2 == 0
        ):
            raise ValueError("Specified CAS gives odd number of electrons in inactive space.")
        self.num_inactive_spin_orbs = self.int_gen.num_elec - int(np.sum(cas[0]))
        self.num_inactive_orbs = self.num_inactive_spin_orbs // 2
        self.num_active_orbs = cas[1]
        self.num_active_spin_orbs = 2 * self.num_active_orbs
        self.num_active_orbs = self.num_active_spin_orbs // 2
        self.num_virtual_orbs = (
            len(self.int_gen.kinetic_energy) - self.num_inactive_orbs - self.num_active_orbs
        )
        if self.num_virtual_orbs < 0:
            raise ValueError(
                "Number of inactive + number of active orbitals is larger than total number of orbitals."
            )
        self.num_virtual_spin_orbs = 2 * self.num_virtual_orbs
        self.num_orbs = self.num_inactive_orbs + self.num_active_orbs + self.num_virtual_orbs
        self.num_spin_orbs = 2 * self.num_orbs
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_elec = self.num_active_elec_alpha + self.num_active_elec_beta
        self._rdm1 = None
        self._rdm2 = None
        self._h_mo = None
        self._g_mo = None
        self._sa_energy: float | None = None
        self._state_energies = None
        self.num_energy_evals = 0
        self._c_mo = mo_coeffs
        # Construct spin orbital idx
        self.inactive_spin_idx = [x for x in range(self.num_inactive_spin_orbs)]
        self.active_spin_idx = [x + self.num_inactive_spin_orbs for x in range(self.num_active_spin_orbs)]
        self.virtual_spin_idx = [
            x + self.num_inactive_spin_orbs + self.num_active_spin_orbs
            for x in range(self.num_virtual_spin_orbs)
        ]
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        for i, orb_idx in enumerate(self.active_spin_idx):
            n_ones = 0
            n_zeros = 0
            for state in states[1]:
                for det in state:
                    if det[i] == "1":
                        n_ones += 1
                    else:
                        n_zeros += 1
            if n_zeros == 0:
                # Always occupied in reference
                self.active_occ_spin_idx.append(orb_idx)
            elif n_ones == 0:
                # Always unoccupied in refence
                self.active_unocc_spin_idx.append(orb_idx)
            elif resolve_unpaired_idx == "both":
                self.active_occ_spin_idx.append(orb_idx)
                self.active_unocc_spin_idx.append(orb_idx)
            elif resolve_unpaired_idx == "occ":
                self.active_occ_spin_idx.append(orb_idx)
            elif resolve_unpaired_idx == "unocc":
                self.active_unocc_spin_idx.append(orb_idx)
            else:
                raise ValueError(
                    f"Got unknown option for resolve_unpaired_idx, {resolve_unpaired_idx}, excepted 'both', 'occ' or 'unocc'."
                )
        self.active_spin_idx_shifted = [x - self.num_inactive_spin_orbs for x in self.active_spin_idx]
        self.active_occ_spin_idx_shifted = [x - self.num_inactive_spin_orbs for x in self.active_occ_spin_idx]
        self.active_unocc_spin_idx_shifted = [
            x - self.num_inactive_spin_orbs for x in self.active_unocc_spin_idx
        ]
        # Construct spatial idx
        self.inactive_idx = [x for x in range(self.num_inactive_orbs)]
        self.active_idx = [x + self.num_inactive_orbs for x in range(self.num_active_orbs)]
        self.virtual_idx = [
            x + self.num_inactive_orbs + self.num_active_orbs for x in range(self.num_virtual_orbs)
        ]
        self.active_occ_idx = []
        self.active_unocc_idx = []
        for i, orb_idx in enumerate(self.active_idx):
            n_ones = 0
            n_zeros = 0
            for state in states[1]:
                for det in state:
                    if det[2 * i] == "1" and det[2 * i + 1] == "1":
                        n_ones += 1
                    elif det[2 * i] == "0" and det[2 * i + 1] == "0":
                        n_zeros += 1
                    else:
                        n_zeros += 1
                        n_ones += 1
            if n_zeros == 0:
                # Always occupied in reference
                self.active_occ_idx.append(orb_idx)
            elif n_ones == 0:
                # Always unoccupied in refence
                self.active_unocc_idx.append(orb_idx)
            elif resolve_unpaired_idx == "both":
                self.active_occ_idx.append(orb_idx)
                self.active_unocc_idx.append(orb_idx)
            elif resolve_unpaired_idx == "occ":
                self.active_occ_idx.append(orb_idx)
            elif resolve_unpaired_idx == "unocc":
                self.active_unocc_idx.append(orb_idx)
            else:
                raise ValueError(
                    f"Got unknown option for resolve_unpaired_idx, {resolve_unpaired_idx}, excepted 'both', 'occ' or 'unocc'."
                )
        self.active_idx_shifted = [x - self.num_inactive_orbs for x in self.active_idx]
        self.active_occ_idx_shifted = [x - self.num_inactive_orbs for x in self.active_occ_idx]
        self.active_unocc_idx_shifted = [x - self.num_inactive_orbs for x in self.active_unocc_idx]
        # Find non-redundant kappas
        self._kappa = []
        kappa_idx = []
        kappa_idx_dagger = []
        self._kappa_old = []
        # kappa can be optimized in spatial basis
        # Loop over all q>p orb combinations.
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                # find redundant kappas
                if p in self.inactive_idx and q in self.inactive_idx:
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        continue
                # the rest is non-redundant
                self._kappa.append(0.0)
                self._kappa_old.append(0.0)
                kappa_idx.append((p, q))
                kappa_idx_dagger.append((q, p))
        self.kappa_idx = np.array(kappa_idx, dtype=int)
        # States
        self.num_states = len(states[0])
        self.states = states
        # Setup Qiskit stuff
        self.QI = quantum_interface
        self.QI.construct_circuit(
            self.active_occ_idx_shifted,
            self.active_unocc_idx_shifted,
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            self.num_active_orbs,
            (self.num_active_elec_alpha, self.num_active_elec_beta),
        )

    @property
    def kappa(self) -> list[float]:
        """Get orbital rotation parameters."""
        return self._kappa.copy()

    @kappa.setter
    def kappa(self, k: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._h_mo = None
        self._g_mo = None
        self._sa_energy = None
        self._state_energies = None
        self._kappa = k.copy()
        # Move current expansion point.
        self._c_mo = self.c_mo
        self._kappa_old = self.kappa

    @property
    def c_mo(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        kappa_mat = np.zeros_like(self._c_mo)
        if len(self.kappa) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa) - np.array(self._kappa_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(self.kappa, self._kappa_old, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = one_electron_integral_transform(self.c_mo, self.int_gen.h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_mo, self.int_gen.electron_electron_repulsion)
        return self._g_mo

    @property
    def thetas(self) -> list[float]:
        """Getter for ansatz parameters.

        Returns:
            Ansatz parameters.
        """
        return self.QI.parameters

    @thetas.setter
    def thetas(self, parameters: list[float]) -> None:
        """Setter for ansatz paramters.

        Args:
            parameters: New ansatz paramters.
        """
        self._rdm1 = None
        self._rdm2 = None
        self._sa_energy = None
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

        if isinstance(primitive, (BaseEstimatorV1, BaseEstimatorV2)):
            raise ValueError("Estimator is not supported.")
        elif not isinstance(primitive, (BaseSamplerV1, BaseSamplerV2)):
            raise TypeError(f"Unsupported Qiskit primitive, {type(primitive)}")
        self.QI._primitive = primitive
        if verbose:
            if self.QI.mitigation_flags.do_M_ansatz0:
                print("Reset RDMs, energies, QI metrics, and correlation matrix.")
            else:
                print("Reset RDMs, energies, and QI metrics.")
        self._rdm1 = None
        self._rdm2 = None
        self._sa_energy = None
        self._state_energies = None
        self.QI.total_device_calls = 0
        self.QI.total_shots_used = 0
        self.QI.total_paulis_evaluated = 0

        # Reset circuit and initiate re-transpiling
        ISA_old = self.QI.ISA
        self._reconstruct_circuit()  # Reconstruct circuit but keeping parameters
        self.QI._transpiled = False
        self.QI.ISA = ISA_old  # Redo ISA including transpilation if requested
        self.QI.shots = self.QI.shots  # Redo shots parameter check

        if verbose:
            self.QI.get_info()

    def _reconstruct_circuit(self) -> None:
        """Construct circuit again."""
        self.QI.construct_circuit(
            self.active_occ_idx_shifted,
            self.active_unocc_idx_shifted,
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            self.num_active_orbs,
            (self.num_active_elec_alpha, self.num_active_elec_beta),
        )

    @property
    def rdm1(self) -> np.ndarray:
        r"""Calculate one-electron reduced density matrix.

        The trace condition is enforced:

        .. math::
            \sum_i\Gamma^{[1]}_{ii} = N_e

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs), dtype=float)
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
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        r"""Calculate two-electron reduced density matrix.

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
                ),
                dtype=float,
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
        return self._rdm2

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = one_electron_integral_transform(self.c_mo, overlap_integral)
        one = np.identity(len(S_ortho))
        diff = np.abs(S_ortho - one)
        print("Max ortho-normal diff:", np.max(diff))

    @property
    def sa_energy(self) -> float:
        """Get the state-averaged electronic energy.

        Returns:
            State-averaged electronic energy.
        """
        if self._sa_energy is None:
            H = hamiltonian_0i_0a(
                self.h_mo,
                self.g_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
            )
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            sa_energy = 0.0
            for coeffs, csf in zip(self.states[0], self.states[1]):
                sa_energy += self.QI.quantum_expectation_value_csfs((coeffs, csf), H, (coeffs, csf))
            self._sa_energy = sa_energy / self.num_states
        return self._sa_energy

    def _calc_energy_elec(self) -> float:
        """Run electronic energy simulation, regardless of self._sa_energy_elec variable.

        Returns:
            State-averaged electronic energy.
        """
        H = hamiltonian_0i_0a(
            self.h_mo,
            self.g_mo,
            self.num_inactive_orbs,
            self.num_active_orbs,
        )
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
        sa_energy = 0.0
        for coeffs, csf in zip(self.states[0], self.states[1]):
            sa_energy += self.QI.quantum_expectation_value_csfs((coeffs, csf), H, (coeffs, csf))
        return sa_energy / self.num_states

    def run_wf_optimization(
        self,
        tol: float = 1e-10,
        maxiter: int = 1000,
        orbital_optimization: bool = False,
        theta_optimization: bool = True,
        one_step_optimizer: str = "BFGS",
        theta_optimizer: str = "BFGS",
        orbital_optimizer: str = "BFGS",
        opt_type: str = "1step",
        is_silent_subiterations: bool = False,
    ) -> None:
        """Run variational optimization of wavefunction.

        Args:
            tol: Tolerance for finishing the optimization.
            maxiter: Maximum number of iterations.
            orbital_optimization: Perform orbital optimization.
            theta_optimization: Perform theta optimization.
            one_step_optimizer: Optimizer used for 1step optimizer.
            theta_optimizer: Optimizer used for theta optimization in 2step optimizer.
            orbital_optimizer: Optimizer used for orbital optimization in 2step optimizer.
            opt_type: Optimization type, can be '1step' or '2step'.
            is_silent_subiterations: Silence sub iterations in 2step.
        """
        if len(self.kappa) == 0 and orbital_optimization:
            print("No kappa parameters turning off orbital optimization.")
            orbital_optimization = False
        if len(self.thetas) == 0 and theta_optimization:
            print("No thetas parameters turning off theta optimization.")
            theta_optimization = False
        if opt_type.lower() == "2step" and (not orbital_optimization or not theta_optimization):
            if not orbital_optimization:
                print("Orbital optimization not requested changing optimizer type to 1step.")
                opt_type = "1step"
            elif not theta_optimization:
                print("theta optimization not requested changing optimizer type to 1step.")
                opt_type = "1step"
        if theta_optimization and isinstance(self.QI.ansatz, QuantumCircuit):
            if opt_type.lower() == "1step" and one_step_optimizer.lower() not in ("cobyla", "cobyqa"):
                raise ValueError("Custom Ansatz in QI only works with COBYLA and COBYQA as optimizer.")
            elif opt_type.lower() == "2step" and theta_optimizer.lower() not in ("cobyla", "cobyqa"):
                raise ValueError("Custom Ansatz in QI only works with COBYLA and COBYQA as optimizer.")
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa)}")
        if theta_optimization:
            print("### Number theta: {len(self.thetas)}")
        if opt_type.lower() == "1step":
            self._run_wf_optimization_1step(
                tol, maxiter, orbital_optimization, theta_optimization, one_step_optimizer
            )
        elif opt_type.lower() == "2step":
            self._run_wf_optimization_2step(
                tol, maxiter, theta_optimizer, orbital_optimizer, is_silent_subiterations
            )
        else:
            raise ValueError(f"Got unknown 'opt_type', {opt_type} excpected '1step' or '2step'.")

    def _run_wf_optimization_2step(
        self,
        tol: float,
        maxiter: int,
        theta_optimizer: str,
        orbital_optimizer: str,
        is_silent_subiterations: bool,
    ) -> None:
        """Run two step optimization of wave function.

        This function should not be called from the outside.
        See 'run_wf_optimization' for argument description.
        """
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        for full_iter in range(0, maxiter):
            full_start = time.time()
            # Do ansatz optimization
            if not is_silent_subiterations:
                print("--------Ansatz optimization")
                print(
                    "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                )
            if theta_optimizer.lower() == "rotosolve":
                # For RotoSolve type solvers the energy per state is needed in the optimization,
                # instead of only the state-averaged energy.
                energy_theta = partial(
                    self._calc_energy_optimization,
                    theta_optimization=True,
                    kappa_optimization=False,
                    return_all_states=True,
                )
            else:
                energy_theta = partial(
                    self._calc_energy_optimization,
                    theta_optimization=True,
                    kappa_optimization=False,
                )
            gradient_theta = partial(
                self._calc_gradient_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            optimizer = Optimizers(
                energy_theta,
                theta_optimizer,
                grad=gradient_theta,
                maxiter=maxiter,
                tol=tol,
                is_silent=is_silent_subiterations,
                energy_eval_callback=lambda: self.num_energy_evals,
            )
            self._old_opt_parameters = np.zeros_like(self.thetas) + 10**20
            self._E_opt_old = 0.0
            res = optimizer.minimize(
                self.thetas,
                extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names},
            )
            self.thetas = res.x.tolist()

            if not is_silent_subiterations:
                print("--------Orbital optimization")
                print(
                    "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                )
            energy_oo = partial(
                self._calc_energy_optimization,
                theta_optimization=False,
                kappa_optimization=True,
            )
            gradient_oo = partial(
                self._calc_gradient_optimization,
                theta_optimization=False,
                kappa_optimization=True,
            )

            optimizer = Optimizers(
                energy_oo,
                orbital_optimizer,
                grad=gradient_oo,
                maxiter=maxiter,
                tol=tol,
                is_silent=is_silent_subiterations,
                energy_eval_callback=lambda: self.num_energy_evals,
            )
            self._old_opt_parameters = np.zeros(len(self.kappa_idx)) + 10**20
            self._E_opt_old = 0.0
            res = optimizer.minimize([0.0] * len(self.kappa_idx))
            for i in range(len(self.kappa)):
                self._kappa[i] = 0.0
                self._kappa_old[i] = 0.0
            e_new = res.fun
            time_str = f"{time.time() - full_start:7.2f}"
            e_str = f"{e_new:3.12f}"
            print(
                f"{str(full_iter + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)} | {str(self.num_energy_evals).center(11)}"
            )
            if abs(e_new - e_old) < tol:
                break
            e_old = e_new
        # Subspace diagonalization
        self._do_state_ci()
        self._sa_energy = res.fun

    def _run_wf_optimization_1step(
        self,
        tol: float,
        maxiter: int,
        orbital_optimization: bool,
        theta_optimization: bool,
        one_step_optimizer: str,
    ) -> None:
        """Run one step optimization of wave function.

        This function should not be called from the outside.
        See 'run_wf_optimization' for argument description.
        """
        if one_step_optimizer.lower() == "rotosolve" and orbital_optimization:
            raise ValueError(
                "Cannot use RotoSolve together with orbital optimization in the one-step solver."
            )
        print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        energy = partial(
            self._calc_energy_optimization,
            theta_optimization=theta_optimization,
            kappa_optimization=orbital_optimization,
        )
        gradient = partial(
            self._calc_gradient_optimization,
            theta_optimization=theta_optimization,
            kappa_optimization=orbital_optimization,
        )
        if orbital_optimization:
            if theta_optimization:
                parameters = self.kappa + self.thetas
            else:
                parameters = self.kappa
        else:
            parameters = self.thetas
        if one_step_optimizer.lower() == "rotosolve":
            # For RotoSolve type solvers the energy per state is needed in the optimization,
            # instead of only the state-averaged energy.
            energy = partial(
                self._calc_energy_optimization,
                theta_optimization=True,
                kappa_optimization=False,
                return_all_states=True,
            )
        optimizer = Optimizers(
            energy,
            one_step_optimizer,
            grad=gradient,
            maxiter=maxiter,
            tol=tol,
            energy_eval_callback=lambda: self.num_energy_evals,
        )
        self._old_opt_parameters = np.zeros_like(parameters) + 10**20
        self._E_opt_old = 0.0
        res = optimizer.minimize(
            parameters,
            extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names},
        )
        if orbital_optimization:
            if theta_optimization:
                self.thetas = res.x[len(self.kappa) :].tolist()
            for i in range(len(self.kappa)):
                self._kappa[i] = 0.0
                self._kappa_old[i] = 0.0
        else:
            self.thetas = res.x.tolist()
        # Subspace diagonalization
        self._do_state_ci()
        self._sa_energy = res.fun

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
        r"""Get transition property with one-electron operator.

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
        mo_integral = one_electron_integral_transform(self.c_mo, ao_integral)
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

    def get_oscillator_strenghts(self) -> np.ndarray:
        r"""Get oscillator strengths between ground state and excited states.

        .. math::
            f_n = \frac{2}{3}\varepsilon_n\left|\left<0\left|\hat{\boldsymbol{\mu}}\right|n\right>\right|^2

        Returns:
            Oscillator strengths.
        """
        dipole_integrals = self.int_gen.electric_dipole
        transition_dipole_x = self.get_transition_property(dipole_integrals[0])
        transition_dipole_y = self.get_transition_property(dipole_integrals[1])
        transition_dipole_z = self.get_transition_property(dipole_integrals[2])
        osc_strs = np.zeros(self.num_states - 1)
        for idx, (excitation_energy, td_x, td_y, td_z) in enumerate(
            zip(self.excitation_energies, transition_dipole_x, transition_dipole_y, transition_dipole_z)
        ):
            osc_strs[idx] = 2 / 3 * excitation_energy * (td_x**2 + td_y**2 + td_z**2)
        return osc_strs

    def _calc_energy_optimization(
        self,
        parameters: list[float],
        theta_optimization: bool,
        kappa_optimization: bool,
        return_all_states: bool = False,
    ) -> float | np.ndarray:
        r"""Calculate electronic energy of SA-UPS wave function.

        .. math::
            E = \left<0\left|\hat{H}\right|0\right>

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: If used in theta optimization.
            kappa_optimization: If used in kappa optimization.
            return_all_states: Return the energy for all states instead of only the averaged energy.

        Returns:
            State-averaged electronic energy.
        """
        # Avoid recalculating energy in callback
        if np.max(np.abs(np.array(self._old_opt_parameters) - np.array(parameters))) < 10**-14:
            return self._E_opt_old
        num_kappa = 0
        if kappa_optimization:
            num_kappa = len(self.kappa_idx)
            self.kappa = parameters[:num_kappa]
        if theta_optimization:
            self.thetas = parameters[num_kappa:]
        H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
        energy_states = []
        for coeffs, csf in zip(self.states[0], self.states[1]):
            energy_states.append(self.QI.quantum_expectation_value_csfs((coeffs, csf), H, (coeffs, csf)))
        self.num_energy_evals += self.num_states  # count one measurement per state
        if return_all_states:
            return np.array(energy_states)
        E = np.mean(energy_states)
        self._E_opt_old = E
        self._old_opt_parameters = np.copy(parameters)
        return E

    def _calc_gradient_optimization(
        self, parameters: list[float], theta_optimization: bool, kappa_optimization: bool
    ) -> np.ndarray:
        r"""Calculate electronic gradient.

        For theta part,

        #. 10.48550/arXiv.2303.10825, Eq. 17-21 (appendix - v1)

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: If used in theta optimization.
            kappa_optimization: If used in kappa optimization.

        Returns:
            State-averaged electronic gradient.
        """
        gradient = np.zeros(len(parameters))
        num_kappa = 0
        if kappa_optimization:
            num_kappa = len(self.kappa_idx)
            self.kappa = parameters[:num_kappa]
        if theta_optimization:
            self.thetas = parameters[num_kappa:]
        if kappa_optimization:
            gradient[:num_kappa] = get_orbital_gradient(
                self.h_mo,
                self.g_mo,
                self.kappa_idx,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.rdm1,
                self.rdm2,
            )
        if theta_optimization:
            H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            for i in range(len(self.thetas)):
                R = self.QI.grad_param_R[self.QI.param_names[i]]
                e_vals_grad = get_energy_evals_for_grad(H, self.QI, self.thetas, i, R, self.states)
                grad = 0.0
                for j, mu in enumerate(list(range(1, 2 * R + 1))):
                    x_mu = (2 * mu - 1) / (2 * R) * np.pi
                    grad += e_vals_grad[j] * (-1) ** (mu - 1) / (4 * R * (np.sin(1 / 2 * x_mu)) ** 2)
                gradient[i + num_kappa] += grad
            self.num_energy_evals += (
                2 * np.sum(list(self.QI.grad_param_R.values())) * self.num_states
            )  # Count energy measurements for all gradients
        return gradient


def get_energy_evals_for_grad(
    operator: FermionicOperator,
    quantum_interface: QuantumInterface,
    parameters: list[float],
    idx: int,
    R: int,
    states: tuple[list[list[float]], list[list[str]]],
) -> list[float]:
    r"""Get energy evaluations needed for the gradient calculation.

    The gradient formula is defined for x=0,
    so x_shift is used to shift ensure we can get the energy in the point we actually want.

    Args:
        operator: Operator which the derivative is with respect to.
        quantum_interface: Quantum interface object
        parameters: Paramters.
        idx: Parameter idx.
        R: Parameter to control we get the needed points.
        states: Reference states in the SA wave function.

    Returns:
        Energies in a few fixed points.
    """
    e_vals = []
    x = parameters.copy()
    x_shift = x[idx]
    for mu in range(1, 2 * R + 1):
        x_mu = (2 * mu - 1) / (2 * R) * np.pi
        x[idx] = x_mu + x_shift
        energy_states = []
        for coeffs, csf in zip(states[0], states[1]):
            energy_states.append(
                quantum_interface.quantum_expectation_value_csfs(
                    (coeffs, csf), operator, (coeffs, csf), custom_parameters=x
                )
            )
        e_vals.append(np.mean(energy_states))
    return e_vals
