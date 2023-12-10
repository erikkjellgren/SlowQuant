import os
import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import scipy
import scipy.optimize

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.base import StateVector
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    epq_hybrid,
    expectation_value_hybrid,
    expectation_value_hybrid_flow,
    hamiltonian_hybrid_0i_0a,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    epq_pauli,
    expectation_value_pauli,
    hamiltonian_pauli_0i_0a,
)
from slowquant.unitary_coupled_cluster.util import ThetaPicker, construct_ucc_u


def eE(wf, E, p, q):
    return expectation_value_hybrid_flow(wf.state_vector, [E[(p, q)]], wf.state_vector)


def eEE(wf, E, p, q, r, s):
    return expectation_value_hybrid_flow(wf.state_vector, [E[(p, q)], E[(r, s)]], wf.state_vector)


def eEEE(wf, E, p, q, r, s, t, u):
    return expectation_value_hybrid_flow(wf.state_vector, [E[(p, q)], E[(r, s)], E[(t, u)]], wf.state_vector)


class WaveFunctionUCC:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
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
        o = np.array([0, 1])
        z = np.array([1, 0])
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
        inactive_on_vector = []
        active_on_vector = []
        virtual_on_vector = []
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
        self._u_number_spin_conserving = None
        self._u = None
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
                active_on_vector.append(o)
                self.num_active_spin_orbs += 1
                self.num_active_elec += 1
            else:
                self.inactive_spin_idx.append(i)
                inactive_on_vector.append(o)
                self.num_inactive_spin_orbs += 1
        for i in range(num_elec, num_spin_orbs):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_unocc_spin_idx.append(i)
                active_on_vector.append(z)
                self.num_active_spin_orbs += 1
            else:
                self.virtual_spin_idx.append(i)
                virtual_on_vector.append(z)
                self.num_virtual_spin_orbs += 1
        if len(self.active_spin_idx) != 0:
            active_shift = np.min(self.active_spin_idx)
            for active_idx in self.active_spin_idx:
                self.active_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_spin_idx:
                self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_spin_idx:
                self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)
        self.state_vector = StateVector(inactive_on_vector, active_on_vector, virtual_on_vector)
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
        self.singlet_excitation_operator_generator = ThetaPicker(
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            is_spin_conserving=True,
        )
        # Construct theta1
        self._theta1 = []
        for _ in self.singlet_excitation_operator_generator.get_t1_generator_sa(0):
            self._theta1.append(0.0)
        # Construct theta2
        self._theta2 = []
        for _ in self.singlet_excitation_operator_generator.get_t2_generator_sa(0):
            self._theta2.append(0.0)
        # Construct theta3
        self._theta3 = []
        for _ in self.singlet_excitation_operator_generator.get_t3_generator(0):
            self._theta3.append(0.0)
        # Construct theta4
        self._theta4 = []
        for _ in self.singlet_excitation_operator_generator.get_t4_generator(0):
            self._theta4.append(0.0)
        # Construct theta5
        self._theta5 = []
        for _ in self.singlet_excitation_operator_generator.get_t5_generator(0):
            self._theta5.append(0.0)
        # Construct theta6
        self._theta6 = []
        for _ in self.singlet_excitation_operator_generator.get_t6_generator(0):
            self._theta6.append(0.0)

    def save_wavefunction(self, filename: str, force_overwrite: bool = False) -> None:
        """Save the wave function to a compressed NumPy object.

        Args:
            filename: Filename of compressed NumPy object without file extension.
            force_overwrite: Overwrite file if it already exists.
        """
        if os.path.exists(f"{filename}.npz") and not force_overwrite:
            raise ValueError(f"{filename}.npz already exists and force_overwrite is False.")
        np.savez_compressed(
            f"{filename}.npz",
            theta1=self.theta1,
            theta2=self.theta2,
            theta3=self.theta3,
            theta4=self.theta4,
            theta5=self.theta5,
            theta6=self.theta6,
            c_trans=self.c_trans,
            h_ao=self.h_ao,
            g_ao=self.g_ao,
            num_spin_orbs=self.num_spin_orbs,
            num_elec=self.num_elec,
            num_active_elec=self.num_active_elec,
            num_active_orbs=self.num_active_orbs,
            include_active_kappa=self._include_active_kappa,
            energy_elec=self.energy_elec,
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
        self._c_orthonormal = c

    @property
    def u_number_spin_conserving(self) -> np.ndarray:
        """Get unitary that only work on number and spin conserving states.

        Return:
            Unitary that only works on number and spin conserving states.
        """
        if self._u_number_spin_conserving is None:
            thetas = []
            if "s" in self._excitations:
                thetas += self.theta1
            if "d" in self._excitations:
                thetas += self.theta2
            if "t" in self._excitations:
                thetas += self.theta3
            if "q" in self._excitations:
                thetas += self.theta4
            if "5" in self._excitations:
                thetas += self.theta5
            if "6" in self._excitations:
                thetas += self.theta6
            self._u_number_spin_conserving = construct_ucc_u(
                self.num_active_spin_orbs,
                thetas,
                self.singlet_excitation_operator_generator,
                self._excitations,
                allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
            )
        return self._u_number_spin_conserving

    @property
    def u(self) -> np.ndarray:
        """Get unitary ansatz.

        Return:
            Unitary ansatz.
        """
        if self._u is None:
            thetas = []
            if "s" in self._excitations:
                thetas += self.theta1
            if "d" in self._excitations:
                thetas += self.theta2
            if "t" in self._excitations:
                thetas += self.theta3
            if "q" in self._excitations:
                thetas += self.theta4
            if "5" in self._excitations:
                thetas += self.theta5
            if "6" in self._excitations:
                thetas += self.theta6
            self._u = construct_ucc_u(
                self.num_active_spin_orbs,
                thetas,
                self.singlet_excitation_operator_generator,
                self._excitations,
            )
        return self._u

    @property
    def theta1(self) -> list[float]:
        """Get theta1 values.

        Returns:
            theta1 values.
        """
        return self._theta1

    @theta1.setter
    def theta1(self, theta: list[float]) -> None:
        """Set theta1 values.

        Args:
            theta: theta1 values.
        """
        if len(theta) != len(self._theta1):
            raise ValueError(f"Expected {len(self._theta1)} theta1 values got {len(theta)}")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._u_number_spin_conserving = None
        self._u = None
        self._theta1 = theta.copy()
        self.state_vector.new_u(
            self.u_number_spin_conserving,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )

    @property
    def theta2(self) -> list[float]:
        """Get theta2 values.

        Returns:
            theta2 values.
        """
        return self._theta2

    @theta2.setter
    def theta2(self, theta: list[float]) -> None:
        """Set theta2 values.

        Args:
            theta: theta2 values.
        """
        if len(theta) != len(self._theta2):
            raise ValueError(f"Expected {len(self._theta2)} theta2 values got {len(theta)}")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._u_number_spin_conserving = None
        self._u = None
        self._theta2 = theta.copy()
        self.state_vector.new_u(
            self.u_number_spin_conserving,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )

    @property
    def theta3(self) -> list[float]:
        """Get theta3 values.

        Returns:
            theta3 values.
        """
        return self._theta3

    @theta3.setter
    def theta3(self, theta: list[float]) -> None:
        """Set theta3 values.

        Args:
            theta: theta3 values.
        """
        if len(theta) != len(self._theta3):
            raise ValueError(f"Expected {len(self._theta3)} theta3 values got {len(theta)}")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._u_number_spin_conserving = None
        self._u = None
        self._theta3 = theta.copy()
        self.state_vector.new_u(
            self.u_number_spin_conserving,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )

    @property
    def theta4(self) -> list[float]:
        """Get theta4 values.

        Returns:
            theta4 values.
        """
        return self._theta4

    @theta4.setter
    def theta4(self, theta: list[float]) -> None:
        """Set theta4 values.

        Args:
            theta: theta4 values.
        """
        if len(theta) != len(self._theta4):
            raise ValueError(f"Expected {len(self._theta4)} theta4 values got {len(theta)}")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._u_number_spin_conserving = None
        self._u = None
        self._theta4 = theta.copy()
        self.state_vector.new_u(
            self.u_number_spin_conserving,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )

    @property
    def theta5(self) -> list[float]:
        """Get theta5 values.

        Returns:
            theta5 values.
        """
        return self._theta5

    @theta5.setter
    def theta5(self, theta: list[float]) -> None:
        """Set theta5 values.

        Args:
            theta: theta5 values.
        """
        if len(theta) != len(self._theta5):
            raise ValueError(f"Expected {len(self._theta5)} theta5 values got {len(theta)}")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._u_number_spin_conserving = None
        self._u = None
        self._theta5 = theta.copy()
        self.state_vector.new_u(
            self.u_number_spin_conserving,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )

    @property
    def theta6(self) -> list[float]:
        """Get theta6 values.

        Returns:
            theta6 values.
        """
        return self._theta6

    @theta6.setter
    def theta6(self, theta: list[float]) -> None:
        """Set theta6 values.

        Args:
            theta: theta6 values.
        """
        if len(theta) != len(self._theta6):
            raise ValueError(f"Expected {len(self._theta6)} theta6 values got {len(theta)}")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._u_number_spin_conserving = None
        self._u = None
        self._theta6 = theta.copy()
        self.state_vector.new_u(
            self.u_number_spin_conserving,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )

    def add_multiple_theta(self, theta: dict[str, list[float]], excitations: str) -> None:
        """Add multiple ranks of thetas.

        Args:
            theta: Dictionary of thetas.
            excitations: Excitations to be included.
        """
        if "s" in excitations:
            self._theta1 = theta["theta1"].copy()
        if "d" in excitations:
            self._theta2 = theta["theta2"].copy()
        if "t" in excitations:
            self._theta3 = theta["theta3"].copy()
        if "q" in excitations:
            self._theta4 = theta["theta4"].copy()
        if "5" in excitations:
            self._theta5 = theta["theta5"].copy()
        if "6" in excitations:
            self._theta6 = theta["theta6"].copy()
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._u_number_spin_conserving = None
        self._u = None
        self.state_vector.new_u(
            self.u_number_spin_conserving,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )

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
                    val = expectation_value_pauli(
                        self.state_vector,
                        epq_pauli(p, q, self.num_spin_orbs),
                        self.state_vector,
                    )
                    self._rdm1[p_idx, q_idx] = val
                    self._rdm1[q_idx, p_idx] = val
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
            E = {}
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                    E[(p, q)] = epq_hybrid(
                        p,
                        q,
                        self.num_inactive_spin_orbs,
                        self.num_active_spin_orbs,
                        self.num_virtual_spin_orbs,
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
                            val = expectation_value_hybrid_flow(
                                self.state_vector,
                                [E[(p, q)], E[(r, s)]],
                                self.state_vector,
                            )
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val
        return self._rdm2

    @property
    def rdm3(self) -> np.ndarray:
        """Calcuate three-electron reduced density matrix.

        Currently not utilizing the full symmetry.

        Returns:
            Three-electron reduced density matrix.
        """
        if self._rdm3 is None:
            self._rdm3 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                )
            )

            E = {}
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                    E[(p, q)] = epq_hybrid(
                        p,
                        q,
                        self.num_inactive_spin_orbs,
                        self.num_active_spin_orbs,
                        self.num_virtual_spin_orbs,
                    )

            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_idx = r - self.num_inactive_orbs
                        for s in range(self.num_inactive_orbs, p + 1):
                            s_idx = s - self.num_inactive_orbs
                            for t in range(self.num_inactive_orbs, r + 1):
                                t_idx = t - self.num_inactive_orbs
                                for u in range(self.num_inactive_orbs, p + 1):
                                    u_idx = u - self.num_inactive_orbs
                                    val = expectation_value_hybrid_flow(
                                        self.state_vector,
                                        [E[(p, q)], E[(r, s)], E[(t, u)]],
                                        self.state_vector,
                                    )
                                    if t == s:
                                        val -= expectation_value_hybrid_flow(
                                            self.state_vector, [E[(p, q)], E[(r, u)]], self.state_vector
                                        )
                                    if r == q:
                                        val -= expectation_value_hybrid_flow(
                                            self.state_vector, [E[(p, s)], E[(t, u)]], self.state_vector
                                        )
                                    if r == q and t == s:
                                        val += expectation_value_hybrid_flow(
                                            self.state_vector, [E[(p, u)]], self.state_vector
                                        )
                                    if t == q:
                                        val -= expectation_value_hybrid_flow(
                                            self.state_vector, [E[(p, u)], E[(r, s)]], self.state_vector
                                        )
                                    if t == q and r == u:
                                        val += expectation_value_hybrid_flow(
                                            self.state_vector, [E[(p, s)]], self.state_vector
                                        )
                                    self._rdm3[p_idx, q_idx, r_idx, s_idx, t_idx, u_idx] = val
                                    self._rdm3[p_idx, q_idx, t_idx, u_idx, r_idx, s_idx] = val
                                    self._rdm3[r_idx, s_idx, p_idx, q_idx, t_idx, u_idx] = val
                                    self._rdm3[r_idx, s_idx, t_idx, u_idx, p_idx, q_idx] = val
                                    self._rdm3[t_idx, u_idx, p_idx, q_idx, r_idx, s_idx] = val
                                    self._rdm3[t_idx, u_idx, r_idx, s_idx, p_idx, q_idx] = val
                                    self._rdm3[q_idx, p_idx, s_idx, r_idx, u_idx, t_idx] = val
                                    self._rdm3[q_idx, p_idx, u_idx, t_idx, s_idx, r_idx] = val
                                    self._rdm3[s_idx, r_idx, q_idx, p_idx, u_idx, t_idx] = val
                                    self._rdm3[s_idx, r_idx, u_idx, t_idx, q_idx, p_idx] = val
                                    self._rdm3[u_idx, t_idx, q_idx, p_idx, s_idx, r_idx] = val
                                    self._rdm3[u_idx, t_idx, s_idx, r_idx, q_idx, p_idx] = val
        return self._rdm3

    @property
    def rdm4(self) -> np.ndarray:
        """Calcuate four-electron reduced density matrix.

        Currently not utilizing the full symmetry.

        Returns:
            Four-electron reduced density matrix.
        """
        if self._rdm4 is None:
            self._rdm4 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                )
            )

            E = {}
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                    E[(p, q)] = epq_hybrid(
                        p,
                        q,
                        self.num_inactive_spin_orbs,
                        self.num_active_spin_orbs,
                        self.num_virtual_spin_orbs,
                    )

            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_idx = r - self.num_inactive_orbs
                        for s in range(self.num_inactive_orbs, p + 1):
                            s_idx = s - self.num_inactive_orbs
                            for t in range(self.num_inactive_orbs, r + 1):
                                t_idx = t - self.num_inactive_orbs
                                for u in range(self.num_inactive_orbs, p + 1):
                                    u_idx = u - self.num_inactive_orbs
                                    for m in range(self.num_inactive_orbs, t + 1):
                                        m_idx = m - self.num_inactive_orbs
                                        for n in range(self.num_inactive_orbs, p + 1):
                                            n_idx = n - self.num_inactive_orbs
                                            val = expectation_value_hybrid_flow(
                                                self.state_vector,
                                                [E[(p, q)], E[(r, s)], E[(t, u)], E[(m, n)]],
                                                self.state_vector,
                                            )
                                            if m == u:
                                                val += -eEEE(self, E, p, q, r, s, t, n)
                                            if t == s:
                                                val += -eEEE(self, E, p, q, r, u, m, n)
                                            if t == s and m == u:
                                                val += eEE(self, E, p, q, r, n)
                                            if m == s:
                                                val += -eEEE(self, E, p, q, r, n, t, u)
                                            if m == s and t == n:
                                                val += eEE(self, E, p, q, r, u)
                                            if r == q:
                                                val += -eEEE(self, E, p, s, t, u, m, n)
                                            if r == q and m == u:
                                                val += eEE(self, E, p, s, t, n)
                                            if r == q and t == s:
                                                val += eEE(self, E, p, u, m, n)
                                            if r == q and t == s and m == u:
                                                val -= eE(self, E, p, n)
                                            if r == q and m == s:
                                                val += eEE(self, E, p, n, t, u)
                                            if r == q and m == s and t == n:
                                                val -= eE(self, E, p, u)
                                            if t == q:
                                                val += -eEEE(self, E, p, u, r, s, m, n)
                                            if t == q and m == s:
                                                val += eEE(self, E, p, u, r, n)
                                            if t == q and r == u:
                                                val += eEE(self, E, p, s, m, n)
                                            if t == q and r == u and m == s:
                                                val -= eE(self, E, p, n)
                                            if t == q and m == u:
                                                val += eEE(self, E, p, n, r, s)
                                            if t == q and m == u and r == n:
                                                val -= eE(self, E, p, s)
                                            if m == q:
                                                val += -eEEE(self, E, p, n, r, s, t, u)
                                            if m == q and t == s:
                                                val += eEE(self, E, p, n, r, u)
                                            if m == q and r == n:
                                                val += eEE(self, E, p, s, t, u)
                                            if m == q and r == n and t == s:
                                                val -= eE(self, E, p, u)
                                            if m == q and t == n:
                                                val += eEE(self, E, p, u, r, s)
                                            if m == q and t == n and r == u:
                                                val -= eE(self, E, p, s)
                                            self._rdm4[
                                                p_idx, q_idx, r_idx, s_idx, t_idx, u_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[
                                                p_idx, q_idx, r_idx, s_idx, m_idx, n_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[
                                                p_idx, q_idx, t_idx, u_idx, r_idx, s_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[
                                                p_idx, q_idx, t_idx, u_idx, m_idx, n_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[
                                                p_idx, q_idx, m_idx, n_idx, r_idx, s_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[
                                                p_idx, q_idx, m_idx, n_idx, t_idx, u_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[
                                                r_idx, s_idx, p_idx, q_idx, t_idx, u_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[
                                                r_idx, s_idx, p_idx, q_idx, m_idx, n_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[
                                                r_idx, s_idx, t_idx, u_idx, p_idx, q_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[
                                                r_idx, s_idx, t_idx, u_idx, m_idx, n_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[
                                                r_idx, s_idx, m_idx, n_idx, p_idx, q_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[
                                                r_idx, s_idx, m_idx, n_idx, t_idx, u_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[
                                                t_idx, u_idx, p_idx, q_idx, r_idx, s_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[
                                                t_idx, u_idx, p_idx, q_idx, m_idx, n_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[
                                                t_idx, u_idx, r_idx, s_idx, p_idx, q_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[
                                                t_idx, u_idx, r_idx, s_idx, m_idx, n_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[
                                                t_idx, u_idx, m_idx, n_idx, p_idx, q_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[
                                                t_idx, u_idx, m_idx, n_idx, r_idx, s_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[
                                                m_idx, n_idx, p_idx, q_idx, r_idx, s_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[
                                                m_idx, n_idx, p_idx, q_idx, t_idx, u_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[
                                                m_idx, n_idx, r_idx, s_idx, p_idx, q_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[
                                                m_idx, n_idx, r_idx, s_idx, t_idx, u_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[
                                                m_idx, n_idx, t_idx, u_idx, p_idx, q_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[
                                                m_idx, n_idx, t_idx, u_idx, r_idx, s_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[
                                                q_idx, p_idx, s_idx, r_idx, u_idx, t_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[
                                                q_idx, p_idx, s_idx, r_idx, n_idx, m_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[
                                                q_idx, p_idx, u_idx, t_idx, s_idx, r_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[
                                                q_idx, p_idx, u_idx, t_idx, n_idx, m_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[
                                                q_idx, p_idx, n_idx, m_idx, s_idx, r_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[
                                                q_idx, p_idx, n_idx, m_idx, u_idx, t_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[
                                                s_idx, r_idx, q_idx, p_idx, u_idx, t_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[
                                                s_idx, r_idx, q_idx, p_idx, n_idx, m_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[
                                                s_idx, r_idx, u_idx, t_idx, q_idx, p_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[
                                                s_idx, r_idx, u_idx, t_idx, n_idx, m_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[
                                                s_idx, r_idx, n_idx, m_idx, q_idx, p_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[
                                                s_idx, r_idx, n_idx, m_idx, u_idx, t_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[
                                                u_idx, t_idx, q_idx, p_idx, s_idx, r_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[
                                                u_idx, t_idx, q_idx, p_idx, n_idx, m_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[
                                                u_idx, t_idx, s_idx, r_idx, q_idx, p_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[
                                                u_idx, t_idx, s_idx, r_idx, n_idx, m_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[
                                                u_idx, t_idx, n_idx, m_idx, q_idx, p_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[
                                                u_idx, t_idx, n_idx, m_idx, s_idx, r_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[
                                                n_idx, m_idx, q_idx, p_idx, s_idx, r_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[
                                                n_idx, m_idx, q_idx, p_idx, u_idx, t_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[
                                                n_idx, m_idx, s_idx, r_idx, q_idx, p_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[
                                                n_idx, m_idx, s_idx, r_idx, u_idx, t_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[
                                                n_idx, m_idx, u_idx, t_idx, q_idx, p_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[
                                                n_idx, m_idx, u_idx, t_idx, s_idx, r_idx, q_idx, p_idx
                                            ] = val
        return self._rdm4

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

    def run_ucc(
        self,
        excitations: str,
        orbital_optimization: bool = False,
        is_silent: bool = False,
        convergence_threshold: float = 10**-10,
        maxiter: int = 10000,
    ) -> None:
        """Run optimization of UCC wave function.

        Args:
            excitations: Excitation orders to include.
            orbital_optimization: Do orbital optimization.
            is_silent: Do not print any output.
            convergence_threshold: Energy threshold for convergence.
            maxiter: Maximum number of iterations.
        """
        excitations = excitations.lower()
        self._excitations = excitations
        e_tot = partial(
            energy_ucc,
            excitations=excitations,
            orbital_optimized=orbital_optimization,
            wf=self,
        )
        parameter_gradient = partial(
            gradient_ucc,
            excitations=excitations,
            orbital_optimized=orbital_optimization,
            wf=self,
        )
        global iteration
        global start
        iteration = 0  # type: ignore
        start = time.time()  # type: ignore

        def print_progress(x: Sequence[float]) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration
            global start
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{e_tot(x):3.12f}"
            print(f"{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            if iteration > 500:
                raise ValueError("Did not converge in 500 iterations in energy minimization.")
            start = time.time()  # type: ignore

        def silent_progress(x: Sequence[float]) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration
            iteration += 1  # type: ignore
            if iteration > 500:
                raise ValueError("Did not converge in 500 iterations in energy minimization.")

        parameters: list[float] = []
        num_kappa = 0
        num_theta1 = 0
        num_theta2 = 0
        num_theta3 = 0
        num_theta4 = 0
        num_theta5 = 0
        num_theta6 = 0
        if orbital_optimization:
            parameters += self.kappa
            num_kappa += len(self.kappa)
        if "s" in excitations:
            for idx, _, _, _ in self.singlet_excitation_operator_generator.get_t1_generator_sa(0):
                parameters += [self.theta1[idx]]
                num_theta1 += 1
        if "d" in excitations:
            for idx, _, _, _, _, _ in self.singlet_excitation_operator_generator.get_t2_generator_sa(0):
                parameters += [self.theta2[idx]]
                num_theta2 += 1
        if "t" in excitations:
            for idx, _, _, _, _, _, _, _ in self.singlet_excitation_operator_generator.get_t3_generator(0):
                parameters += [self.theta3[idx]]
                num_theta3 += 1
        if "q" in excitations:
            for idx, _, _, _, _, _, _, _, _, _ in self.singlet_excitation_operator_generator.get_t4_generator(
                0
            ):
                parameters += [self.theta4[idx]]
                num_theta4 += 1
        if "5" in excitations:
            for (
                idx,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) in self.singlet_excitation_operator_generator.get_t5_generator(0):
                parameters += [self.theta5[idx]]
                num_theta5 += 1
        if "6" in excitations:
            for (
                idx,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) in self.singlet_excitation_operator_generator.get_t6_generator(0):
                parameters += [self.theta6[idx]]
                num_theta6 += 1
        if is_silent:
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=silent_progress,
                method="SLSQP",
                jac=parameter_gradient,
            )
        else:
            print("### Parameters information:")
            print(f"### Number kappa: {num_kappa}")
            print(f"### Number theta1: {num_theta1}")
            print(f"### Number theta2: {num_theta2}")
            print(f"### Number theta3: {num_theta3}")
            print(f"### Number theta4: {num_theta4}")
            print(f"### Number theta5: {num_theta5}")
            print(f"### Number theta6: {num_theta6}")
            print(
                f"### Total parameters: {num_kappa+num_theta1+num_theta2+num_theta3+num_theta4+num_theta5+num_theta6}\n"
            )
            print("Iteration # | Iteration time [s] | Electronic energy [Hartree]")
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=print_progress,
                method="SLSQP",
                jac=parameter_gradient,
                options={"maxiter": maxiter},
            )
        self.energy_elec = res["fun"]
        param_idx = 0
        if orbital_optimization:
            param_idx += len(self.kappa)
            for i in range(len(self.kappa)):
                self.kappa[i] = 0
                self._kappa_old[i] = 0
            for i in range(len(self.kappa_redundant)):
                self.kappa_redundant[i] = 0
                self._kappa_redundant_old[i] = 0
        if "s" in excitations:
            thetas = res["x"][param_idx : num_theta1 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _ in self.singlet_excitation_operator_generator.get_t1_generator_sa(0):
                self.theta1[idx] = thetas[counter]
                counter += 1
        if "d" in excitations:
            thetas = res["x"][param_idx : num_theta2 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _ in self.singlet_excitation_operator_generator.get_t2_generator_sa(0):
                self.theta2[idx] = thetas[counter]
                counter += 1
        if "t" in excitations:
            thetas = res["x"][param_idx : num_theta3 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _ in self.singlet_excitation_operator_generator.get_t3_generator(0):
                self.theta3[idx] = thetas[counter]
                counter += 1
        if "q" in excitations:
            thetas = res["x"][param_idx : num_theta4 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _, _, _ in self.singlet_excitation_operator_generator.get_t4_generator(
                0
            ):
                self.theta4[idx] = thetas[counter]
                counter += 1
        if "5" in excitations:
            thetas = res["x"][param_idx : num_theta5 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for (
                idx,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) in self.singlet_excitation_operator_generator.get_t5_generator(0):
                self.theta5[idx] = thetas[counter]
                counter += 1
        if "6" in excitations:
            thetas = res["x"][param_idx : num_theta6 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for (
                idx,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) in self.singlet_excitation_operator_generator.get_t6_generator(0):
                self.theta6[idx] = thetas[counter]
                counter += 1


def energy_ucc(
    parameters: Sequence[float],
    excitations: str,
    orbital_optimized: bool,
    wf: WaveFunctionUCC,
) -> float:
    r"""Calculate electronic energy of UCC wave function.

    .. math::
        E = \left<0\left|\hat{H}\right|0\right>

    Args:
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space singles, active-space doubles, ...
        excitations: Excitation orders to consider.
        orbital_optimized: Do orbital optimization.
        wf: Wave function object.

    Returns:
        Electronic energy.
    """
    kappa = []
    theta1 = []
    theta2 = []
    theta3 = []
    theta4 = []
    theta5 = []
    theta6 = []
    idx_counter = 0
    if orbital_optimized:
        for _ in range(len(wf.kappa_idx)):
            kappa.append(parameters[idx_counter])
            idx_counter += 1
    if "s" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t1_generator_sa(0):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if "d" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t2_generator_sa(
            wf.num_inactive_spin_orbs + wf.num_active_spin_orbs + wf.num_virtual_spin_orbs
        ):
            theta2.append(parameters[idx_counter])
            idx_counter += 1
    if "t" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t3_generator(0):
            theta3.append(parameters[idx_counter])
            idx_counter += 1
    if "q" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t4_generator(0):
            theta4.append(parameters[idx_counter])
            idx_counter += 1
    if "5" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t5_generator(0):
            theta5.append(parameters[idx_counter])
            idx_counter += 1
    if "6" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t6_generator(0):
            theta6.append(parameters[idx_counter])
            idx_counter += 1
    assert len(parameters) == len(kappa) + len(theta1) + len(theta2) + len(theta3) + len(theta4) + len(
        theta5
    ) + len(theta6)

    kappa_mat = np.zeros_like(wf.c_orthonormal)
    if orbital_optimized:
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
    if orbital_optimized:
        wf._kappa_old = kappa.copy()
        wf._kappa_redundant_old = wf.kappa_redundant.copy()
    # Moving expansion point of kappa
    wf.c_orthonormal = c_trans
    # Add thetas
    theta_dict = {}
    if "s" in excitations:
        theta_dict["theta1"] = theta1
    if "d" in excitations:
        theta_dict["theta2"] = theta2
    if "t" in excitations:
        theta_dict["theta3"] = theta3
    if "q" in excitations:
        theta_dict["theta4"] = theta4
    if "5" in excitations:
        theta_dict["theta5"] = theta5
    if "6" in excitations:
        theta_dict["theta6"] = theta6
    wf.add_multiple_theta(theta_dict, excitations)
    return expectation_value_pauli(
        wf.state_vector,
        hamiltonian_pauli_0i_0a(
            wf.h_mo,
            wf.g_mo,
            wf.num_inactive_orbs,
            wf.num_active_orbs,
            wf.num_virtual_orbs,
        ),
        wf.state_vector,
    )


def gradient_ucc(
    parameters: Sequence[float],
    excitations: str,
    orbital_optimized: bool,
    wf: WaveFunctionUCC,
) -> np.ndarray:
    """Calcuate electronic gradient.

    Args:
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space singles, active-space doubles, ...
        excitations: Excitation orders to consider.
        orbital_optimized: Do orbital optimization.
        wf: Wave function object.

    Returns:
        Electronic gradient.
    """
    number_kappas = 0
    if orbital_optimized:
        number_kappas = len(wf.kappa_idx)
    gradient = np.zeros_like(parameters)
    if orbital_optimized:
        gradient[:number_kappas] = orbital_rotation_gradient(
            wf,
        )
    gradient[number_kappas:] = active_space_parameter_gradient(
        wf,
        parameters,
        excitations,
        orbital_optimized,
    )
    return gradient


def orbital_rotation_gradient(
    wf: WaveFunctionUCC,
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


def active_space_parameter_gradient(
    wf: WaveFunctionUCC,
    parameters: Sequence[float],
    excitations: str,
    orbital_optimized: bool,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to active space parameters.

    Args:
        wf: Wave function object.
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space singles, active-space doubles, ...
        excitations: Excitation orders to consider.
        orbital_optimized: Do orbital optimization.

    Returns:
        Electronic gradient with respect to active spae parameters.
    """
    kappa = []
    theta1 = []
    theta2 = []
    theta3 = []
    theta4 = []
    theta5 = []
    theta6 = []
    idx_counter = 0
    if orbital_optimized:
        for _ in range(len(wf.kappa_idx)):
            kappa.append(0 * parameters[idx_counter])
            idx_counter += 1
    if "s" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t1_generator_sa(0):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if "d" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t2_generator_sa(
            wf.num_inactive_spin_orbs + wf.num_active_spin_orbs + wf.num_virtual_spin_orbs
        ):
            theta2.append(parameters[idx_counter])
            idx_counter += 1
    if "t" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t3_generator(0):
            theta3.append(parameters[idx_counter])
            idx_counter += 1
    if "q" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t4_generator(0):
            theta4.append(parameters[idx_counter])
            idx_counter += 1
    if "5" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t5_generator(0):
            theta5.append(parameters[idx_counter])
            idx_counter += 1
    if "6" in excitations:
        for _ in wf.singlet_excitation_operator_generator.get_t6_generator(0):
            theta6.append(parameters[idx_counter])
            idx_counter += 1
    assert len(parameters) == len(kappa) + len(theta1) + len(theta2) + len(theta3) + len(theta4) + len(
        theta5
    ) + len(theta6)

    Hamiltonian = hamiltonian_hybrid_0i_0a(
        wf.h_mo,
        wf.g_mo,
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_virtual_orbs,
    )

    theta_params = theta1 + theta2 + theta3 + theta4 + theta5 + theta6
    gradient_theta = np.zeros_like(theta_params)
    eps = np.finfo(np.float64).eps ** (1 / 2)
    E = expectation_value_hybrid(wf.state_vector, Hamiltonian, wf.state_vector)
    for i, _ in enumerate(theta_params):
        sign_step = (theta_params[i] >= 0).astype(float) * 2 - 1
        step_size = eps * sign_step * max(1, abs(theta_params[i]))
        theta_params[i] += step_size
        theta_dict = {}
        idx = 0
        if "s" in excitations:
            theta_dict["theta1"] = theta_params[idx : idx + len(theta1)]
            idx += len(theta1)
        if "d" in excitations:
            theta_dict["theta2"] = theta_params[idx : idx + len(theta2)]
            idx += len(theta2)
        if "t" in excitations:
            theta_dict["theta3"] = theta_params[idx : idx + len(theta3)]
            idx += len(theta3)
        if "q" in excitations:
            theta_dict["theta4"] = theta_params[idx : idx + len(theta4)]
            idx += len(theta4)
        if "5" in excitations:
            theta_dict["theta5"] = theta_params[idx : idx + len(theta5)]
            idx += len(theta5)
        if "6" in excitations:
            theta_dict["theta6"] = theta_params[idx : idx + len(theta6)]
            idx += len(theta6)
        wf.add_multiple_theta(theta_dict, excitations)
        E_plus = expectation_value_hybrid(wf.state_vector, Hamiltonian, wf.state_vector)
        theta_params[i] -= step_size
        theta_dict = {}
        idx = 0
        if "s" in excitations:
            theta_dict["theta1"] = theta_params[idx : idx + len(theta1)]
            idx += len(theta1)
        if "d" in excitations:
            theta_dict["theta2"] = theta_params[idx : idx + len(theta2)]
            idx += len(theta2)
        if "t" in excitations:
            theta_dict["theta3"] = theta_params[idx : idx + len(theta3)]
            idx += len(theta3)
        if "q" in excitations:
            theta_dict["theta4"] = theta_params[idx : idx + len(theta4)]
            idx += len(theta4)
        if "5" in excitations:
            theta_dict["theta5"] = theta_params[idx : idx + len(theta5)]
            idx += len(theta5)
        if "6" in excitations:
            theta_dict["theta6"] = theta_params[idx : idx + len(theta6)]
            idx += len(theta6)
        gradient_theta[i] = (E_plus - E) / step_size
    return gradient_theta


def load_wavefunction(filename: str) -> WaveFunctionUCC:
    """Load wave function from a compressed NumPy object.

    Args:
        filename: Filename of compressed NumPy object without file extension.

    Returns:
        Wave function object.
    """
    dat = np.load(f"{filename}.npz")
    wf = WaveFunctionUCC(
        int(dat["num_spin_orbs"]),
        int(dat["num_elec"]),
        (int(dat["num_active_elec"]), int(dat["num_active_orbs"])),
        dat["c_trans"],
        dat["h_ao"],
        dat["g_ao"],
        bool(dat["include_active_kappa"]),
    )
    excitations = ""
    if len(dat["theta1"]) > 0:
        if np.max(np.abs(dat["theta1"])) > 10**-6:
            excitations += "s"
    if len(dat["theta2"]) > 0:
        if np.max(np.abs(dat["theta2"])) > 10**-6:
            excitations += "d"
    if len(dat["theta3"]) > 0:
        if np.max(np.abs(dat["theta3"])) > 10**-6:
            excitations += "t"
    if len(dat["theta4"]) > 0:
        if np.max(np.abs(dat["theta4"])) > 10**-6:
            excitations += "q"
    if len(dat["theta5"]) > 0:
        if np.max(np.abs(dat["theta5"])) > 10**-6:
            excitations += "5"
    if len(dat["theta6"]) > 0:
        if np.max(np.abs(dat["theta6"])) > 10**-6:
            excitations += "6"
    wf._excitations = excitations
    wf.add_multiple_theta(
        {
            "theta1": list(dat["theta1"]),
            "theta2": list(dat["theta2"]),
            "theta3": list(dat["theta3"]),
            "theta4": list(dat["theta4"]),
            "theta5": list(dat["theta5"]),
            "theta6": list(dat["theta6"]),
        },
        wf._excitations,
    )
    thetas = []
    if "s" in wf._excitations:
        thetas += wf.theta1
    if "d" in wf._excitations:
        thetas += wf.theta2
    if "t" in wf._excitations:
        thetas += wf.theta3
    if "q" in wf._excitations:
        thetas += wf.theta4
    if "5" in wf._excitations:
        thetas += wf.theta5
    if "6" in wf._excitations:
        thetas += wf.theta6
    energy = energy_ucc(thetas, wf._excitations, False, wf)
    if abs(energy - float(dat["energy_elec"])) > 10**-6:
        raise ValueError(
            f'Calculate energy is different from saved energy: {energy} and {float(dat["energy_elec"])}.'
        )
    wf.energy_elec = energy
    return wf
