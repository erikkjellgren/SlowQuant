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
    one_electron_integral_transform,
    two_electron_integral_transform,
    two_electron_integral_transform_split,
)
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import a_op
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.unrestricted_density_matrix import (
    get_electronic_energy_unrestricted,
    get_orbital_gradient_unrestricted,
)
from slowquant.unitary_coupled_cluster.unrestricted_operators import (
    unrestricted_hamiltonian_0i_0a,
)


class UnrestrictedWaveFunctionCircuit:
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
                 orbitals are counted in spatial basis.
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
        if isinstance(quantum_interface.ansatz, QuantumCircuit):
            print(
                "WARNING: A QI with a custom Ansatz was passed. VQE will only work with COBYLA and COBYQA optimizer."
            )
        self._c_a_mo = mo_coeffs[0]
        self._c_b_mo = mo_coeffs[1]
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
        self.active_idx_shifted = []
        self.active_occ_idx_shifted = []
        self.active_unocc_idx_shifted = []
        self.num_elec = num_elec
        self.num_elec_alpha = (num_elec - np.sum(cas[0])) // 2 + cas[0][0]
        self.num_elec_beta = (num_elec - np.sum(cas[0])) // 2 + cas[0][1]
        self.num_spin_orbs = 2 * len(h_ao)
        self.num_orbs = len(h_ao)
        self._include_active_kappa = include_active_kappa
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_elec = self.num_active_elec_alpha + self.num_active_elec_beta
        self.num_active_spin_orbs = 2 * cas[1]
        self.num_inactive_spin_orbs = self.num_elec - self.num_active_elec
        self.num_virtual_spin_orbs = 2 * len(h_ao) - self.num_inactive_spin_orbs - self.num_active_spin_orbs
        self._rdm1aa = None
        self._rdm1bb = None
        self._rdm2aaaa = None
        self._rdm2bbbb = None
        self._rdm2aabb = None
        self._rdm2bbaa = None
        self._haa_mo = None
        self._hbb_mo = None
        self._gaaaa_mo = None
        self._gbbbb_mo = None
        self._gaabb_mo = None
        self._gbbaa_mo = None
        self.inactive_spin_idx = [x for x in range(self.num_inactive_spin_orbs)]
        self.active_spin_idx = [x + self.num_inactive_spin_orbs for x in range(self.num_active_spin_orbs)]
        self.virtual_spin_idx = [
            x + self.num_inactive_spin_orbs + self.num_virtual_spin_orbs
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
        # Make shifted indices
        if len(self.active_spin_idx) != 0:
            active_shift = np.min(self.active_spin_idx)
            for active_idx in self.active_spin_idx:
                self.active_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_spin_idx:
                self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_spin_idx:
                self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)
        if len(self.active_idx) != 0:
            active_shift = np.min(self.active_idx)
            for active_idx in self.active_idx:
                self.active_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_idx:
                self.active_occ_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_idx:
                self.active_unocc_idx_shifted.append(active_idx - active_shift)
        # Find non-redundant kappas
        self._kappa_a = []
        self._kappa_b = []
        self.kappa_idx = []
        self.kappa_no_activeactive_idx = []
        self.kappa_no_activeactive_idx_dagger = []
        self._kappa_a_redundant = []
        self._kappa_b_redundant = []
        self.kappa_redundant_idx = []
        self._kappa_a_old = []
        self._kappa_b_old = []
        self._kappa_a_redundant_old = []
        self._kappa_b_redundant_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self._kappa_a_redundant.append(0.0)
                    self._kappa_b_redundant.append(0.0)
                    self._kappa_a_redundant_old.append(0.0)
                    self._kappa_b_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append((p, q))
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self._kappa_a_redundant.append(0.0)
                    self._kappa_b_redundant.append(0.0)
                    self._kappa_a_redundant_old.append(0.0)
                    self._kappa_b_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append((p, q))
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self._kappa_a_redundant.append(0.0)
                        self._kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append((p, q))
                        continue
                if include_active_kappa:
                    if p in self.active_occ_idx and q in self.active_occ_idx:
                        self._kappa_a_redundant.append(0.0)
                        self._kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append((p, q))
                        continue
                    if p in self.active_unocc_idx and q in self.active_unocc_idx:
                        self._kappa_a_redundant.append(0.0)
                        self._kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append((p, q))
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    self.kappa_no_activeactive_idx.append((p, q))
                    self.kappa_no_activeactive_idx_dagger.append((q, p))
                self._kappa_a.append(0.0)
                self._kappa_b.append(0.0)
                self._kappa_a_old.append(0.0)
                self._kappa_b_old.append(0.0)
                self.kappa_idx.append((p, q))
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
    def kappa_a(self) -> list[float]:
        """Get orbital rotation parameters."""
        return self._kappa_a.copy()

    @property
    def kappa_b(self) -> list[float]:
        """Get orbital rotation parameters."""
        return self._kappa_b.copy()

    @kappa_a.setter
    def kappa_a(self, k: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._haa_mo = None
        self._gaaaa_mo = None
        self._gaabb_mo = None
        self._gbbaa_mo = None
        self._energy_elec = None
        self._kappa_a = k.copy()
        if isinstance(self._kappa_a, np.ndarray):
            self._kappa_a = self._kappa_a.tolist()
        # Move current expansion point.
        self._c_a_mo = self.c_a_mo
        self._kappa_a_old = self.kappa_a

    @kappa_b.setter
    def kappa_b(self, k: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._hbb_mo = None
        self._gbbbb_mo = None
        self._gaabb_mo = None
        self._gbbaa_mo = None
        self._energy_elec = None
        self._kappa_b = k.copy()
        if isinstance(self._kappa_b, np.ndarray):
            self._kappa_b = self._kappa_b.tolist()
        # Move current expansion point.
        self._c_b_mo = self.c_b_mo
        self._kappa_b_old = self.kappa_b

    @property
    def thetas(self) -> list[float]:
        """Get theta values.

        Returns:
            theta values.
        """
        return self.QI.parameters

    @thetas.setter
    def thetas(self, theta_vals: list[float]) -> None:
        """Set theta values.

        Args:
            theta_vals: theta values.
        """
        self._rdm1aa = None
        self._rdm1bb = None
        self._rdm2aaaa = None
        self._rdm2bbbb = None
        self._rdm2aabb = None
        self._rdm2bbaa = None
        self._energy_elec = None
        self.QI.parameters = theta_vals.copy()
        if isinstance(self.QI.parameters, np.ndarray):
            self.QI.parameters = self.QI.parameters.tolist()

    @property
    def c_a_mo(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_a_mo)
        if len(self.kappa_a) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_a) - np.array(self._kappa_a_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(self.kappa_a, self._kappa_a_old, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_a_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def c_b_mo(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_b_mo)
        if len(self.kappa_b) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_b) - np.array(self._kappa_b_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(self.kappa_b, self._kappa_b_old, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_b_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def haa_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._haa_mo is None:
            self._haa_mo = one_electron_integral_transform(self.c_a_mo, self.h_ao)
        return self._haa_mo

    @property
    def hbb_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._hbb_mo is None:
            self._hbb_mo = one_electron_integral_transform(self.c_b_mo, self.h_ao)
        return self._hbb_mo

    @property
    def h_mo(self) -> tuple[np.ndarray, np.ndarray]:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        return (self.haa_mo, self.hbb_mo)

    @property
    def gaaaa_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gaaaa_mo is None:
            self._gaaaa_mo = two_electron_integral_transform(self.c_a_mo, self.g_ao)
        return self._gaaaa_mo

    @property
    def gbbbb_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gbbbb_mo is None:
            self._gbbbb_mo = two_electron_integral_transform(self.c_b_mo, self.g_ao)
        return self._gbbbb_mo

    @property
    def gaabb_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gaabb_mo is None:
            self._gaabb_mo = two_electron_integral_transform_split(self.c_a_mo, self.c_b_mo, self.g_ao)
        return self._gaabb_mo

    @property
    def gbbaa_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gbbaa_mo is None:
            self._gbbaa_mo = two_electron_integral_transform_split(self.c_b_mo, self.c_a_mo, self.g_ao)
        return self._gbbaa_mo

    @property
    def g_mo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        return (self.gaaaa_mo, self.gbbbb_mo, self.gaabb_mo)

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
        self._rdm1aa = None
        self._rdm1bb = None
        self._rdm2aaaa = None
        self._rdm2bbbb = None
        self._rdm2aabb = None
        self._rdm2bbaa = None
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
            self.num_active_orbs, (self.num_active_elec_alpha, self.num_active_elec_beta)
        )

    def _calculate_rdm1(self, spin) -> np.ndarray:
        """Calcuate one-electron reduced density matrix.

        Returns:
            One-electron reduced density matrix.
        """
        # if self._calculate_rdm1 is None:
        calculate_rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
        for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
            p_idx = p - self.num_inactive_orbs
            for q in range(self.num_inactive_orbs, p + 1):
                q_idx = q - self.num_inactive_orbs
                rdm1_op = (a_op(p, spin, True) * a_op(q, spin, False)).get_folded_operator(
                    self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                )
                val = self.QI.quantum_expectation_value(rdm1_op)
                calculate_rdm1[p_idx, q_idx] = val
                calculate_rdm1[q_idx, p_idx] = val
        return calculate_rdm1

    @property
    def rdm1_C(self) -> np.ndarray:
        if self._rdm1aa is None:
            self._rdm1aa = self._calculate_rdm1("alpha")
        if self._rdm1bb is None:
            self._rdm1bb = self._calculate_rdm1("beta")
        return self._rdm1aa + self._rdm1bb

    @property
    def rdm1_S(self) -> np.ndarray:
        if self._rdm1aa is None:
            self._rdm1aa = self._calculate_rdm1("alpha")
        if self._rdm1bb is None:
            self._rdm1bb = self._calculate_rdm1("beta")
        return self._rdm1aa - self._rdm1bb

    @property
    def rdm1aa(self) -> np.ndarray:
        if self._rdm1aa is None:
            self._rdm1aa = self._calculate_rdm1("alpha")
        return self._rdm1aa

    @property
    def rdm1bb(self) -> np.ndarray:
        if self._rdm1bb is None:
            self._rdm1bb = self._calculate_rdm1("beta")
        return self._rdm1bb

    def _calculate_rdm2(self, spin1, spin2) -> np.ndarray:
        """Calcuate two-electron unrestricted reduced density matrix.

        Returns:
            Two-electron unrestricted reduced density matrix.
        """
        calculate_rdm2 = np.zeros(
            (self.num_active_orbs, self.num_active_orbs, self.num_active_orbs, self.num_active_orbs)
        )
        for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
            p_idx = p - self.num_inactive_orbs
            for q in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                q_idx = q - self.num_inactive_orbs
                for r in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                    r_idx = r - self.num_inactive_orbs
                    for s in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                        s_idx = s - self.num_inactive_orbs
                        rdm2_op = (
                            a_op(p, spin1, True)
                            * a_op(r, spin2, True)
                            * a_op(s, spin2, False)
                            * a_op(q, spin1, False)
                        ).get_folded_operator(
                            self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                        )
                        val = self.QI.quantum_expectation_value(rdm2_op)
                        calculate_rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                        # calculate_rdm2[r_idx, s_idx, p_idx, q_idx] = val # type: ignore
                        # calculate_rdm2[q_idx, p_idx, s_idx, r_idx] = val # type: ignore
                        # calculate_rdm2[s_idx, r_idx, q_idx, p_idx] = val # type: ignore
        return calculate_rdm2

    @property
    def rdm2_C(self) -> np.ndarray:
        if self._rdm2aaaa is None:
            self._rdm2aaaa = self._calculate_rdm2("alpha", "alpha")
        if self._rdm2bbbb is None:
            self._rdm2bbbb = self._calculate_rdm2("beta", "beta")
        if self._rdm2aabb is None:
            self._rdm2aabb = self._calculate_rdm2("alpha", "beta")
        return self._rdm2aaaa + self._rdm2bbbb + 2 * self._rdm2aabb

    @property
    def rdm2aaaa(self) -> np.ndarray:
        if self._rdm2aaaa is None:
            self._rdm2aaaa = self._calculate_rdm2("alpha", "alpha")
        return self._rdm2aaaa

    @property
    def rdm2bbbb(self) -> np.ndarray:
        if self._rdm2bbbb is None:
            self._rdm2bbbb = self._calculate_rdm2("beta", "beta")
        return self._rdm2bbbb

    @property
    def rdm2aabb(self) -> np.ndarray:
        if self._rdm2aabb is None:
            self._rdm2aabb = self._calculate_rdm2("alpha", "beta")
        return self._rdm2aabb

    @property
    def rdm2bbaa(self) -> np.ndarray:
        if self._rdm2bbaa is None:
            self._rdm2bbaa = self._calculate_rdm2("beta", "alpha")
        return self._rdm2bbaa

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
            raise ValueError(f"Precalculation only supported up to order 2 got {rdm_order}")
        if rdm_order < 1:
            raise ValueError(f"Precalculation need at least an order of 1 got {rdm_order}")
        cumulated_paulis = None
        if rdm_order >= 1:
            self._rdm1aa = None
            self._rdm1bb = None
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, p + 1):
                    for spin in ("alpha", "beta"):
                        rdm1_op = (a_op(p, spin, True) * a_op(q, spin, False)).get_folded_operator(
                            self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                        )
                        mapped_op = self.QI.op_to_qbit(rdm1_op)
                        if cumulated_paulis is None:
                            cumulated_paulis = set(mapped_op.paulis)
                        else:
                            cumulated_paulis = cumulated_paulis.union(mapped_op.paulis)
        if rdm_order >= 2:
            self._rdm2aaaa = None
            self._rdm2bbbb = None
            self._rdm2aabb = None
            self._rdm2bbaa = None
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                    for r in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                        for s in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                            for spin1 in ("alpha", "beta"):
                                for spin2 in ("alpha", "beta"):
                                    rdm2_op = (
                                        a_op(p, spin1, True)
                                        * a_op(r, spin2, True)
                                        * a_op(s, spin2, False)
                                        * a_op(q, spin1, False)
                                    ).get_folded_operator(
                                        self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                                    )
                                    mapped_op = self.QI.op_to_qbit(rdm2_op)
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
        S_ortho = one_electron_integral_transform(self.c_mo, overlap_integral)
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

    def _get_hamiltonian(self) -> FermionicOperator:
        """Return electronic Hamiltonian as FermionicOperator.

        Returns:
            FermionicOperator.
        """
        H = unrestricted_hamiltonian_0i_0a(
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.num_inactive_orbs,
            self.num_active_orbs,
        )
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)

        return H

    def _calc_energy_elec(self) -> float:
        """Run electronic energy simulation, regardless of self._energy_elec variable.

        Returns:
            Electronic energy.
        """
        H = unrestricted_hamiltonian_0i_0a(
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.num_inactive_orbs,
            self.num_active_orbs,
        )
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
        energy_elec = self.QI.quantum_expectation_value(H)
        return energy_elec

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
            print(f"### Number kappa: {len(self.kappa_a) + len(self.kappa_b)}")
        print(f"### Number theta: {len(self.thetas)}")
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree]")
        for full_iter in range(0, int(maxiter)):
            full_start = time.time()

            # Do ansatz optimization
            if not is_silent_subiterations:
                print("--------Ansatz optimization")
                print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree]")
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
            )
            res = optimizer.minimize(
                self.thetas,
                extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names},
            )
            self.thetas = res.x.tolist()

            if orbital_optimization and len(self.kappa_a) + len(self.kappa_b) != 0:
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
                )
                res = optimizer.minimize([0.0] * (len(self.kappa_a) + len(self.kappa_b)))
                for i in range(len(self.kappa_a)):
                    self._kappa_a[i] = 0.0
                    self._kappa_a_old[i] = 0.0
                for i in range(len(self.kappa_b)):
                    self._kappa_b[i] = 0.0
                    self._kappa_b_old[i] = 0.0
            else:
                # If there is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa_a) + len(self.kappa_b) == 0:
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
            print(f"### Number kappa: {len(self.kappa_a) + len(self.kappa_b)}")
        print(f"### Number theta: {len(self.thetas)}")
        if optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa_a) + len(self.kappa_b) != 0:
                raise ValueError(
                    "Cannot use RotoSolve together with orbital optimization in the one-step solver."
                )

        print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree]")
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
                parameters = self.kappa_a + self.kappa_b + self.thetas
            else:
                parameters = self.kappa_a + self.kappa_b
        else:
            parameters = self.thetas
        optimizer = Optimizers(energy, optimizer_name, grad=gradient, maxiter=maxiter, tol=tol)
        res = optimizer.minimize(
            parameters, extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names}
        )
        if orbital_optimization:
            self.thetas = res.x[len(self.kappa_a) + len(self.kappa_b) :].tolist()
            for i in range(len(self.kappa_a)):
                self._kappa_a[i] = 0.0
                self._kappa_a_old[i] = 0.0
            for i in range(len(self.kappa_b)):
                self._kappa_b[i] = 0.0
                self._kappa_b_old[i] = 0.0
        else:
            self.thetas = res.x.tolist()
        self._energy_elec = res.fun

    def _calc_energy_optimization(
        self, parameters: list[float], theta_optimization: bool, kappa_optimization: bool
    ) -> float:
        """Calculate electronic energy.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: Doing theta optimization.
            kappa_optimization: Doing kappa optimization.

        Returns:
            Electronic energy.
        """
        num_kappa_a = 0
        num_kappa_b = 0
        if kappa_optimization:
            num_kappa_a = len(self.kappa_a)
            num_kappa_b = len(self.kappa_b)
            self.kappa_a = parameters[:num_kappa_a]
            self.kappa_b = parameters[num_kappa_a : num_kappa_a + num_kappa_b]
        if theta_optimization:
            self.thetas = parameters[num_kappa_a + num_kappa_b :]
            # Build operator
            H = unrestricted_hamiltonian_0i_0a(
                self.haa_mo,
                self.hbb_mo,
                self.gaaaa_mo,
                self.gbbbb_mo,
                self.gaabb_mo,
                self.gbbaa_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
            )
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            return self.QI.quantum_expectation_value(H)
        # RDM is more expensive than evaluation of the Hamiltonian.
        # Thus only construct these if orbital-optimization is turned on,
        # since the RDMs will be reused in the oo gradient calculation.
        return get_electronic_energy_unrestricted(
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.rdm1aa,
            self.rdm1bb,
            self.rdm2aaaa,
            self.rdm2bbbb,
            self.rdm2aabb,
            self.rdm2bbaa,
        )

    def _calc_gradient_optimization(
        self, parameters: list[float], theta_optimization: bool, kappa_optimization: bool
    ) -> np.ndarray:
        """Calculate electronic gradient.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: Doing theta optimization.
            kappa_optimization: Doing kappa optimization.

        Returns:
            Electronic gradient.
        """
        gradient = np.zeros(len(parameters))
        num_kappa_a = 0
        num_kappa_b = 0
        if kappa_optimization:
            num_kappa_a = len(self.kappa_a)
            num_kappa_b = len(self.kappa_b)
            self.kappa_a = parameters[:num_kappa_a]
            self.kappa_b = parameters[num_kappa_a : num_kappa_a + num_kappa_b]
        if theta_optimization:
            self.thetas = parameters[num_kappa_a + num_kappa_b :]
        if kappa_optimization:
            gradient[: num_kappa_a + num_kappa_b] = get_orbital_gradient_unrestricted(
                self.haa_mo,
                self.hbb_mo,
                self.gaaaa_mo,
                self.gbbbb_mo,
                self.gaabb_mo,
                self.gbbaa_mo,
                self.kappa_idx,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.rdm1aa,
                self.rdm1bb,
                self.rdm2aaaa,
                self.rdm2bbbb,
                self.rdm2aabb,
                self.rdm2bbaa,
            )
        if theta_optimization:
            H = unrestricted_hamiltonian_0i_0a(
                self.haa_mo,
                self.hbb_mo,
                self.gaaaa_mo,
                self.gbbbb_mo,
                self.gaabb_mo,
                self.gbbaa_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
            )
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            for i in range(len(parameters[num_kappa_a + num_kappa_b :])):
                R = self.QI.grad_param_R[self.QI.param_names[i]]
                e_vals_grad = _get_energy_evals_for_grad(H, self.QI, self.thetas, i, R)
                grad = 0.0
                for j, mu in enumerate(list(range(1, 2 * R + 1))):
                    x_mu = (2 * mu - 1) / (2 * R) * np.pi
                    grad += e_vals_grad[j] * (-1) ** (mu - 1) / (4 * R * (np.sin(1 / 2 * x_mu)) ** 2)
                gradient[num_kappa_a + num_kappa_b + i] = grad
        return gradient


def _get_energy_evals_for_grad(
    operator: FermionicOperator,
    quantum_interface: QuantumInterface,
    parameters: list[float],
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
        e_vals.append(quantum_interface.quantum_expectation_value(operator, custom_parameters=x))
    return e_vals
