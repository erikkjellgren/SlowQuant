import time
from functools import partial

import numpy as np
import scipy
from qiskit import QuantumCircuit
from qiskit.primitives import (
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSamplerV1,
    BaseSamplerV2,
)
from qiskit.quantum_info import SparsePauliOp

from slowquant.molecularintegrals.integralfunctions import (
    generalized_one_electron_transform,
    generalized_two_electron_transform,
)
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.generalized_density_matrix import (
    get_electronic_energy_generalized,
    get_orbital_gradient_generalized_real_imag,
)
from slowquant.unitary_coupled_cluster.generalized_operators import (
    a_op_spin,
    generalized_hamiltonian_full_space,
)
from slowquant.unitary_coupled_cluster.optimizers import Optimizers


class WaveFunctionCircuit:
    def __init__(
        self,
        num_elec: int,
        cas: tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        quantum_interface: QuantumInterface,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UCC wave function.

        Args:
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spin basis.
            mo_coeffs: Initial orbital coefficients.
            h_ao: One-electron integrals in AO for Hamiltonian.
            g_ao: Two-electron integrals in AO.
            quantum_interface: QuantumInterface.
            include_active_kappa: Include active-active orbital rotations.
        """
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        if len(cas[0]) != 2:
            raise ValueError(
                "Number of electrons in the active space must be specified as a tuple of (alpha, beta)."
            )
        if cas[1] > mo_coeffs.shape[1]:
            raise ValueError(
                f"More spatial active orbitls than total orbitals. Got {cas[1]} active orbitals, and {mo_coeffs.shape[1]} total orbitals."
            )
        if np.sum(cas[0]) > num_elec:
            raise ValueError(
                f"More active electrons than total electrons. Got {np.sum(cas[0])} active electrons, and {num_elec} total electrons."
            )
        if isinstance(quantum_interface.ansatz, QuantumCircuit):
            print(
                "WARNING: A QI with a custom Ansatz was passed. VQE will only work with COBYLA and COBYQA optimizer."
            )
        self._c_mo = np.copy(mo_coeffs).astype(np.complex128)
        self._h_ao = h_ao
        self._g_ao = g_ao
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._h_mo = None
        self._g_mo = None
        self._energy_elec: float | None = None
        # Construct spin orbital spaces and indices
        self.inactive_spin_idx = []
        self.virtual_spin_idx = []
        self.active_spin_idx = []
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        self.active_spin_idx_shifted = []
        self.active_occ_spin_idx_shifted = []
        self.active_unocc_spin_idx_shifted = []
        self.num_energy_evals = 0  # number of energy measurements on quanutm
        self.num_elec = num_elec
        self.num_spin_orbs = 2 * len(h_ao)
        self.num_orbs = len(h_ao)
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_elec = self.num_active_elec_alpha + self.num_active_elec_beta
        self.num_active_spin_orbs = cas[1]
        self.num_inactive_spin_orbs = num_elec - self.num_active_elec
        self.num_virtual_spin_orbs = (
            self.num_spin_orbs - self.num_active_spin_orbs - self.num_inactive_spin_orbs
        )
        self.inactive_spin_idx = [x for x in range(self.num_inactive_spin_orbs)]
        self.active_spin_idx = [x + self.num_inactive_spin_orbs for x in range(self.num_active_spin_orbs)]
        self.virtual_spin_idx = [
            x + self.num_inactive_spin_orbs + self.num_active_spin_orbs
            for x in range(self.num_virtual_spin_orbs)
        ]
        self.active_occ_spin_idx = []
        for i in range(self.num_active_elec_alpha):
            self.active_occ_spin_idx.append(2 * i + self.num_inactive_spin_orbs)
        for i in range(self.num_active_elec_beta):
            self.active_occ_spin_idx.append(2 * i + 1 + self.num_inactive_spin_orbs)
        self.active_occ_spin_idx.sort()
        self.active_unocc_spin_idx = []
        for i in range(self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs):
            if i not in self.active_occ_spin_idx:
                self.active_unocc_spin_idx.append(i)
        # Make shifted indices
        if len(self.active_spin_idx) != 0:
            active_shift = np.min(self.active_spin_idx)
            for active_idx in self.active_spin_idx:
                self.active_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_spin_idx:
                self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_spin_idx:
                self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)
        # Find non-redundant kappas
        self._kappa_real = []
        self._kappa_imag = []
        self.kappa_spin_idx = []
        self.kappa_no_activeactive_spin_idx = []
        self.kappa_no_activeactive_spin_idx_dagger = []
        self._kappa_real_redundant = []
        self._kappa_imag_redundant = []
        self.kappa_redundant_spin_idx = []
        self._kappa_real_old = []
        self._kappa_imag_old = []
        self._kappa_real_redundant_old = []
        self._kappa_imag_redundant_old = []
        for P in range(0, self.num_spin_orbs):
            for Q in range(P, self.num_spin_orbs):
                if P in self.inactive_spin_idx and Q in self.inactive_spin_idx:
                    self._kappa_real_redundant.append(0.0)
                    self._kappa_imag_redundant.append(0.0)
                    self._kappa_real_redundant_old.append(0.0)
                    self._kappa_imag_redundant_old.append(0.0)
                    self.kappa_redundant_spin_idx.append((P, Q))
                    continue
                if P in self.virtual_spin_idx and Q in self.virtual_spin_idx:
                    self._kappa_real_redundant.append(0.0)
                    self._kappa_imag_redundant.append(0.0)
                    self._kappa_real_redundant_old.append(0.0)
                    self._kappa_imag_redundant_old.append(0.0)
                    self.kappa_redundant_spin_idx.append((P, Q))
                    continue
                if not include_active_kappa:
                    if P in self.active_spin_idx and Q in self.active_spin_idx:
                        self._kappa_real_redundant.append(0.0)
                        self._kappa_imag_redundant.append(0.0)
                        self._kappa_real_redundant_old.append(0.0)
                        self._kappa_imag_redundant_old.append(0.0)
                        self.kappa_redundant_spin_idx.append((P, Q))
                        continue
                if include_active_kappa:
                    if P in self.active_occ_spin_idx and Q in self.active_occ_spin_idx:
                        if P != Q:
                            self._kappa_real_redundant.append(0.0)
                            self._kappa_imag_redundant.append(0.0)
                            self._kappa_real_redundant_old.append(0.0)
                            self._kappa_imag_redundant_old.append(0.0)
                            self.kappa_redundant_spin_idx.append((P, Q))
                            continue
                    if P in self.active_unocc_spin_idx and Q in self.active_unocc_spin_idx:
                        self._kappa_real_redundant.append(0.0)
                        self._kappa_imag_redundant.append(0.0)
                        self._kappa_real_redundant_old.append(0.0)
                        self._kappa_imag_redundant_old.append(0.0)
                        self.kappa_redundant_spin_idx.append((P, Q))
                        continue
                if not (P in self.active_spin_idx and Q in self.active_spin_idx):
                    self.kappa_no_activeactive_spin_idx.append((P, Q))
                    self.kappa_no_activeactive_spin_idx_dagger.append((Q, P))
                self._kappa_real.append(0.0)
                self._kappa_imag.append(0.0)
                self._kappa_real_old.append(0.0)
                self._kappa_imag_old.append(0.0)
                self.kappa_spin_idx.append((P, Q))
        # Setup Qiskit stuff
        self.QI = quantum_interface
        self.QI.construct_circuit(
            2 * self.num_active_spin_orbs, (self.num_active_elec_alpha, self.num_active_elec_beta)
        )

    @property
    def kappa_real(self) -> list[float]:
        """Get real orbital rotation parameters."""
        return self._kappa_real.copy()

    @property
    def kappa_imag(self) -> list[float]:
        """Get imaginary orbital rotation parameters."""
        return self._kappa_imag.copy()

    def set_kappa_cep(self, k_real: list[float], k_imag: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._kappa_real = k_real.copy()
        self._kappa_imag = k_imag.copy()
        if isinstance(self._kappa_real, np.ndarray):
            self._kappa_real = self._kappa_real.tolist()
        if isinstance(self._kappa_imag, np.ndarray):
            self._kappa_img = self._kappa_imag.tolist()
        # Move current expansion point.
        self._c_mo = self.c_mo
        self._kappa_real_old = self.kappa_real
        self._kappa_imag_old = self.kappa_imag

    @property
    def c_mo(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_mo)
        if len(self.kappa_real) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_real) - np.array(self._kappa_real_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_real, self._kappa_real_old, self.kappa_spin_idx
                ):
                    if p == q:
                        continue
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
            if np.max(np.abs(np.array(self.kappa_imag) - np.array(self._kappa_imag_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_imag, self._kappa_imag_old, self.kappa_spin_idx
                ):
                    kappa_mat[p, q] += (kappa_val - kappa_old) * 1.0j
                    kappa_mat[q, p] += (kappa_val - kappa_old) * 1.0j
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = generalized_one_electron_transform(self.c_mo, self._h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = generalized_two_electron_transform(self.c_mo, self._g_ao)
        return self._g_mo

    @property
    def thetas_real(self) -> list[float]:
        """Get real theta values.

        Returns:
            theta values.
        """
        return self._thetas_real.copy()

    @property
    def thetas_imag(self) -> list[float]:
        """Get imaginary theta values.

        Returns:
            theta values.
        """
        return self._thetas_imag.copy()

    @property
    def thetas(self) -> list[complex]:
        """Get complex theta values.

        Returns:
            theta values.
        """
        return self.QI.parameters

    def set_thetas(self, theta_real: list[float], theta_imag: list[float]) -> None:
        """Set theta values.

        Args:
            theta_vals: theta values.
        """
        if len(theta_real) != len(self._thetas_real):
            raise ValueError(f"Expected {len(self._thetas_real)} real theta values got {len(theta_real)}")
        # Remove this warning for running with real valued thetas:
        if len(theta_imag) != len(self._thetas_imag):
            raise ValueError(
                f"Expected {len(self._thetas_imag)} imaginary theta values got {len(theta_imag)}"
            )
        self._rdm1 = None
        self._rdm2 = None
        self._energy_elec = None
        self._thetas_real = theta_real.copy()
        self._thetas_imag = theta_imag.copy()
        if isinstance(self._thetas_real, np.ndarray):
            self._thetas_real = self._thetas_real.tolist()
        if isinstance(self._thetas_imag, np.ndarray):
            self._thetas_img = self._thetas_imag.tolist()
        self.QI.parameters = (np.array(self.thetas_real) + 1.0j * np.array(self.thetas_imag)).tolist()

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
            raise TypeError(f"Unsupported primitive, {type(primitive)}")
        self.QI._primitive = primitive
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
        self.QI._transpiled = False
        self.QI.ISA = ISA_old  # Redo ISA including transpilation if requested
        self.QI.shots = self.QI.shots  # Redo shots parameter check

        if verbose:
            self.QI.get_info()

    def _reconstruct_circuit(self) -> None:
        """Construct circuit again."""
        self.QI.construct_circuit(
            2 * self.num_active_spin_orbs, (self.num_active_elec_alpha, self.num_active_elec_beta)
        )

    @property
    def rdm1(self) -> np.ndarray:
        r"""Calculate one-electron reduced density matrix.

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_spin_orbs, self.num_active_spin_orbs), dtype=np.complex128)
            for P in range(
                self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
            ):
                P_idx = P - self.num_inactive_spin_orbs
                for Q in range(self.num_inactive_spin_orbs, P + 1):
                    Q_idx = Q - self.num_inactive_spin_orbs
                    rdm1_op = (a_op_spin(P, True) * a_op_spin(Q, False)).get_folded_operator(
                        self.num_inactive_spin_orbs // 2,
                        self.num_active_spin_orbs // 2,
                        self.num_virtual_spin_orbs // 2,
                    )
                    val = self.QI.quantum_expectation_value_complex(rdm1_op)
                    self._rdm1[P_idx, Q_idx] = val  # type: ignore
                    self._rdm1[Q_idx, P_idx] = val.conjugate()  # type: ignore (1.7.7 EST)
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        r"""Calculate two-electron reduced density matrix.

        Returns:
            Two-electron reduced density matrix.
        """
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                ),
                dtype=np.complex128,
            )
            for P in range(
                self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
            ):
                P_idx = P - self.num_inactive_spin_orbs
                for Q in range(self.num_inactive_spin_orbs, P + 1):
                    Q_idx = Q - self.num_inactive_spin_orbs
                    for R in range(self.num_inactive_spin_orbs, P + 1):
                        R_idx = R - self.num_inactive_spin_orbs
                        if P == Q:
                            S_lim = R + 1
                        elif P == R:
                            S_lim = Q + 1
                        elif Q < R:
                            S_lim = P  # Not sure I understand this limit? Why not R?
                        else:
                            S_lim = P + 1
                        for S in range(self.num_inactive_spin_orbs, S_lim):
                            S_idx = S - self.num_inactive_spin_orbs
                            pdm2_op = (
                                a_op_spin(P, dagger=True)
                                * a_op_spin(R, dagger=True)
                                * a_op_spin(S, dagger=False)
                                * a_op_spin(Q, dagger=False)
                            ).get_folded_operator(
                                self.num_inactive_spin_orbs // 2,
                                self.num_active_spin_orbs // 2,
                                self.num_virtual_spin_orbs // 2,
                            )
                            val = self.QI.quantum_expectation_value_complex(pdm2_op)
                            self._rdm2[P_idx, Q_idx, R_idx, S_idx] = val  # type: ignore
                            self._rdm2[Q_idx, P_idx, S_idx, R_idx] = val.conjugate()  # type: ignore
                            self._rdm2[R_idx, S_idx, P_idx, Q_idx] = val  # type: ignore
                            self._rdm2[S_idx, R_idx, Q_idx, P_idx] = val.conjugate()  # type: ignore
        return self._rdm2

    def precalc_rdm_paulis(self, rdm_order: int) -> None:
        """Pre-calculate all Paulis used to construct RDMs up to a certain order.

        This utilizes the saving feature in QuantumInterface when using the Sampler primitive.
        If saving is turned up in QuantumInterface this function will do nothing but waste device time.

        Args:
            rdm_order: Max order RDM.
        """
        if not isinstance(
            self.QI._primitive,
            (BaseSamplerV1, BaseSamplerV2),
        ):
            raise TypeError(
                f"This feature is only supported for Sampler got {type(self.QI._primitive)} from QuantumInterface"
            )
        if rdm_order > 2:
            raise ValueError(f"Precalculation only supported up to order 4 got {rdm_order}")
        if rdm_order < 1:
            raise ValueError(f"Precalculation need at least an order of 1 got {rdm_order}")
        cumulated_paulis = None
        if rdm_order >= 1:
            self._rdm1 = None
            for P in range(
                self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
            ):
                for Q in range(self.num_inactive_spin_orbs, P + 1):
                    rdm1_op = (a_op_spin(P, True) * a_op_spin(Q, False)).get_folded_operator(
                        self.num_inactive_spin_orbs // 2,
                        self.num_active_spin_orbs // 2,
                        self.num_virtual_spin_orbs // 2,
                    )
                    mapped_op = self.QI.op_to_qbit(rdm1_op)
                    if cumulated_paulis is None:
                        cumulated_paulis = set(mapped_op.paulis)
                    else:
                        cumulated_paulis = cumulated_paulis.union(mapped_op.paulis)
        if rdm_order >= 2:
            self._rdm2 = None
            for P in range(
                self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
            ):
                for Q in range(self.num_inactive_spin_orbs, P + 1):
                    for R in range(self.num_inactive_spin_orbs, P + 1):
                        if P == Q:
                            S_lim = R + 1
                        elif P == R:
                            S_lim = Q + 1
                        elif Q < R:
                            S_lim = P  # Not sure I understand this limit? Why not R?
                        else:
                            S_lim = P + 1
                        for S in range(self.num_inactive_spin_orbs, S_lim):
                            pdm2_op = (
                                a_op_spin(P, dagger=True)
                                * a_op_spin(R, dagger=True)
                                * a_op_spin(S, dagger=False)
                                * a_op_spin(Q, dagger=False)
                            ).get_folded_operator(
                                self.num_inactive_spin_orbs // 2,
                                self.num_active_spin_orbs // 2,
                                self.num_virtual_spin_orbs // 2,
                            )
                            mapped_op = self.QI.op_to_qbit(pdm2_op)
                            cumulated_paulis = cumulated_paulis.union(mapped_op.paulis)  # type: ignore[union-attr]
        # Calling expectation value to put all Paulis in cliques
        # and compute distributions for the cliques.
        # The coefficients are set to one, so the Paulis cannot cancel out.
        _ = self.QI._sampler_quantum_expectation_value(
            SparsePauliOp(cumulated_paulis, np.ones(len(cumulated_paulis)))  # type: ignore[arg-type]
        )

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = generalized_one_electron_transform(self.c_mo, overlap_integral)
        one = np.identity(len(S_ortho))
        diff = np.abs(S_ortho - one)
        print("Max ortho-normal diff:", np.max(diff))

    @property
    def energy_elec(self) -> float:
        """Get electronic energy.

        Returns:
            Electronic energy.
        """
        if self._energy_elec is None:
            self._energy_elec = self._calc_energy_elec()
        return self._energy_elec

    def _calc_energy_elec(self) -> float:
        """Run electronic energy simulation, regardless of self._energy_elec variable.

        Returns:
            Electronic energy.
        """
        # H = generalized_hamiltonian_0i_0a(
        #     self.h_mo,
        #     self.g_mo,
        #     self.num_inactive_spin_orbs,
        #     self.num_active_spin_orbs,
        # )
        H = generalized_hamiltonian_full_space(
            self.h_mo,
            self.g_mo,
            self.num_spin_orbs,
        )
        H = H.get_folded_operator(
            self.num_inactive_spin_orbs // 2, self.num_active_spin_orbs // 2, self.num_virtual_spin_orbs // 2
        )
        energy_elec = self.QI.quantum_expectation_value_complex(H)
        # The Hamiltonian is Hermitian so the return should be real.
        return energy_elec.real

    def run_wf_optimization_2step(
        self,
        optimizer_name: str,
        orbital_optimization: bool = False,
        tol: float = 1e-10,
        maxiter: int = 1000,
        is_silent_subiterations: bool = False,
    ) -> None:
        """Run two step optimization of wave function.

        Args:
            optimizer_name: Name of optimizer.
            orbital_optimization: Perform orbital optimization.
            tol: Convergence tolerance.
            maxiter: Maximum number of iterations.
            is_silent_subiterations: Silence subiterations.
        """
        if isinstance(self.QI.ansatz, QuantumCircuit) and optimizer_name.lower() not in ("cobyla", "cobyqa"):
            raise ValueError("Custom Ansatz in QI only works with COBYLA and COBYQA as optimizer.")
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa_real)}")
        print(f"### Number theta: {len(self.thetas)}")
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        for full_iter in range(0, int(maxiter)):
            full_start = time.time()

            # Do ansatz optimization
            if not is_silent_subiterations:
                print("--------Ansatz optimization")
                print(
                    "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                )
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
                optimizer_name,
                grad=gradient_theta,
                maxiter=maxiter,
                tol=tol,
                is_silent=is_silent_subiterations,
                energy_eval_callback=lambda: self.num_energy_evals,
            )
            thetas = self.thetas_real + self.thetas_imag
            res = optimizer.minimize(
                thetas,
                extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names},
            )
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(res.x[i])
                thetas_i.append(res.x[i + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)

            if orbital_optimization and len(self.kappa_real) != 0:
                if not is_silent_subiterations:
                    print("--------Orbital optimization")
                    print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree]")
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
                    "l-bfgs-b",
                    grad=gradient_oo,
                    maxiter=maxiter,
                    tol=tol,
                    is_silent=is_silent_subiterations,
                    energy_eval_callback=lambda: self.num_energy_evals,
                )
                res = optimizer.minimize([0.0] * 2 * len(self.kappa_spin_idx))
                for i in range(len(self.kappa_real)):
                    self._kappa_real[i] = 0.0
                    self._kappa_imag[i] = 0.0
                    self._kappa_real_old[i] = 0.0
                    self._kappa_imag_old[i] = 0.0
            else:
                # If there is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa_real) == 0:
                    print(
                        "WARNING: No orbital optimization performed, because there is no non-redundant orbital parameters"
                    )
                break

            e_new = res.fun
            time_str = f"{time.time() - full_start:7.2f}"  # type: ignore
            e_str = f"{e_new:3.12f}"
            print(
                f"{str(full_iter + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)} | {str(self.num_energy_evals).center(11)}"
            )  # type: ignore
            if abs(e_new - e_old) < tol:
                break
            e_old = e_new
        self._energy_elec = e_new

    def run_wf_optimization_1step(
        self,
        optimizer_name: str,
        orbital_optimization: bool = False,
        tol: float = 1e-10,
        maxiter: int = 1000,
    ) -> None:
        """Run one step optimization of wave function.

        Args:
            optimizer_name: Name of optimizer.
            orbital_optimization: Perform orbital optimization.
            tol: Convergence tolerance.
            maxiter: Maximum number of iterations.
        """
        if isinstance(self.QI.ansatz, QuantumCircuit) and optimizer_name.lower() not in ("cobyla", "cobyqa"):
            raise ValueError("Custom Ansatz in QI only works with COBYLA and COBYQA as optimizer.")
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa_real)}")
        print(f"### Number theta: {len(self.thetas)}")
        if optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa_real) != 0:
                raise ValueError(
                    "Cannot use RotoSolve together with orbital optimization in the one-step solver."
                )

        print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        if orbital_optimization:
            if len(self.thetas) > 0:
                energy = partial(
                    self._calc_energy_optimization,
                    theta_optimization=True,
                    kappa_optimization=True,
                )
                gradient = partial(
                    self._calc_gradient_optimization,
                    theta_optimization=True,
                    kappa_optimization=True,
                )
            else:
                energy = partial(
                    self._calc_energy_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )
                gradient = partial(
                    self._calc_gradient_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )
        else:
            energy = partial(
                self._calc_energy_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            gradient = partial(
                self._calc_gradient_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
        if orbital_optimization:
            if len(self.thetas) > 0:
                thetas = self.thetas_real + self.thetas_imag
                parameters = np.zeros(2 * len(self.kappa_real), dtype=float).tolist() + thetas
            else:
                parameters = np.zeros(2 * len(self.kappa_real), dtype=float).tolist()
        else:
            parameters = self.thetas_real + self.thetas_imag
        optimizer = Optimizers(
            energy,
            optimizer_name,
            grad=gradient,
            maxiter=maxiter,
            tol=tol,
            energy_eval_callback=lambda: self.num_energy_evals,
        )
        res = optimizer.minimize(
            parameters, extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names}
        )
        # AWE two R lists
        if orbital_optimization:
            if len(self.thetas) > 0:
                thetas_r = []
                thetas_i = []
                for i in range(len(self.thetas)):
                    thetas_r.append(res.x[i + 2 * len(self.kappa_real)])
                    thetas_i.append(res.x[i + 2 * len(self.kappa_real) + len(self.thetas)])
                self.set_thetas(thetas_r, thetas_i)
            for i in range(len(self.kappa_real)):
                self._kappa_real[i] = 0.0
                self._kappa_imag[i] = 0.0
                self._kappa_real_old[i] = 0.0
                self._kappa_imag_old[i] = 0.0
        else:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(res.x[i])
                thetas_i.append(res.x[i + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)
        #self._energy_elec = res.fun AWE og AE har udkommenteret
        self._energy_elec = res.fun

    def _calc_energy_optimization(
        self, parameters: list[complex], theta_optimization: bool, kappa_optimization: bool
    ) -> float:
        """Calculate electronic energy.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: Doing theta optimization.
            kappa_optimization: Doing kappa optimization.

        Returns:
            Electronic energy.
        """
        num_kappa = 0
        self.num_energy_evals += 1  # count one measurement
        if kappa_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real)])
            self.set_kappa_cep(kappa_r, kappa_i)
        if theta_optimization:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(parameters[i + num_kappa])
                thetas_i.append(parameters[i + num_kappa + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)
        if kappa_optimization:
            # RDM is more expensive than evaluation of the Hamiltonian.
            # Thus only construct these if orbital-optimization is turned on,
            # since the RDMs will be reused in the oo gradient calculation.
            E = get_electronic_energy_generalized(
                self.h_mo,
                self.g_mo,
                self.num_inactive_spin_orbs,
                self.num_active_spin_orbs,
                self.rdm1,
                self.rdm2,
            )
        else:
            # Build operator
            # H = generalized_hamiltonian_0i_0a(
            #     self.h_mo,
            #     self.g_mo,
            #     self.num_inactive_spin_orbs,
            #     self.num_active_spin_orbs,
            # )
            H = generalized_hamiltonian_full_space(
                self.h_mo,
                self.g_mo,
                self.num_spin_orbs,
            )
            H = H.get_folded_operator(
                self.num_inactive_spin_orbs // 2,
                self.num_active_spin_orbs // 2,
                self.num_virtual_spin_orbs // 2,
            )  # Skal det deles med 2? Ser fishy ud AWE
            # Hermitian so expecation value should be real
            E = self.QI.quantum_expectation_value_complex(H)
        # Hermitian so expecation value should be real
        return E.real

    def _calc_gradient_optimization(
        self, parameters: list[complex], theta_optimization: bool, kappa_optimization: bool
    ) -> np.ndarray:
        """Calculate electronic gradient.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: Doing theta optimization.
            kappa_optimization: Doing kappa optimization.

        Returns:
            Electronic gradient.
        """
        num_kappa = 0
        gradient = np.zeros(len(parameters))
        num_kappa = 0
        if kappa_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real)])
            self.set_kappa_cep(kappa_r, kappa_i)
        if theta_optimization:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(parameters[i + num_kappa])
                # Silence the imaginary part if you wish to run with real-valued thetas:
                thetas_i.append(parameters[i + num_kappa + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)
        if kappa_optimization:
            gradient[:num_kappa] = get_orbital_gradient_generalized_real_imag(
                self.h_mo,
                self.g_mo,
                self.kappa_spin_idx,
                self.num_inactive_spin_orbs,
                self.num_active_spin_orbs,
                self.rdm1,
                self.rdm2,
            )
        if theta_optimization:
            # H = generalized_hamiltonian_0i_0a(
            #     self.h_mo,
            #     self.g_mo,
            #     self.num_inactive_spin_orbs,
            #     self.num_active_spin_orbs,
            # )
            H = generalized_hamiltonian_full_space(
                self.h_mo,
                self.g_mo,
                self.num_spin_orbs,
            )
            H = H.get_folded_operator(
                self.num_inactive_spin_orbs // 2,
                self.num_active_spin_orbs // 2,
                self.num_virtual_spin_orbs // 2,
            )
            #
            # Here we need to implement parameter-shift for complex.
            #
            
            for i in range(len(parameters[num_kappa:])):
               R = self.QI.grad_param_R[self.QI.param_names[i]]
               e_vals_grad = _get_energy_evals_for_grad(H, self.QI, parameters, i, R)
               grad = 0.0
               for j, mu in enumerate(list(range(1, 2 * R + 1))):
                   x_mu = (2 * mu - 1) / (2 * R) * np.pi
                   grad += e_vals_grad[j] * (-1) ** (mu - 1) / (4 * R * (np.sin(1 / 2 * x_mu)) ** 2)
               gradient[num_kappa + i] = grad
            self.num_energy_evals += 2 * np.sum(
               list(self.QI.grad_param_R.values())
            )  # Count energy measurements for all gradients

    
        return gradient


def _get_energy_evals_for_grad(
    operator: FermionicOperator,
    quantum_interface: QuantumInterface,
    parameters: list[complex],
    idx: int,
    R: int,
) -> list[float]:
    """Get energy evaluations needed for the gradient calculation.

    The gradient formula is defined for x=0.
    The x_shift variable is used to shift the energy function, such that current parameter value is in zero.

    Args:
        operator: Operator which the derivative is with respect to.
        quantum_interface: Quantum interface class object.
        parameters: Parameters.
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
        # Real? because Hermitian?
        e_vals.append(quantum_interface.quantum_expectation_value_complex(operator, custom_parameters=x).real)
    return e_vals

