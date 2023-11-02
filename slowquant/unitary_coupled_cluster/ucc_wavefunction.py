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
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    energy_hamiltonian_pauli,
    epq_pauli,
    epqrs_pauli,
    expectation_value_pauli,
)
from slowquant.unitary_coupled_cluster.util import ThetaPicker, construct_ucc_u


class WaveFunctionUCC:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        is_generalized: bool = False,
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
            is_generalized: Do generalized UCC.
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
        self.num_orbs = num_spin_orbs//2
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
        self.num_inactive_orbs = self.num_inactive_spin_orbs//2
        self.num_active_orbs = self.num_active_spin_orbs//2
        self.num_virtual_orbs = self.num_virtual_spin_orbs//2
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
        self.kappa_redundant = []
        self.kappa_redundant_idx = []
        self._kappa_old = []
        self._kappa_redundant_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_spin_orbs // 2):
            for q in range(p + 1, self.num_spin_orbs // 2):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self.kappa_redundant.append(0)
                    self._kappa_redundant_old.append(0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self.kappa_redundant.append(0)
                    self._kappa_redundant_old.append(0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self.kappa_redundant.append(0)
                        self._kappa_redundant_old.append(0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                if include_active_kappa:
                    if p in self.active_occ_idx and q in self.active_occ_idx:
                        self.kappa_redundant.append(0)
                        self._kappa_redundant_old.append(0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                    if p in self.active_unocc_idx and q in self.active_unocc_idx:
                        self.kappa_redundant.append(0)
                        self._kappa_redundant_old.append(0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                self.kappa.append(0)
                self._kappa_old.append(0)
                self.kappa_idx.append([p, q])
        # HF like orbital rotation indecies
        self.kappa_hf_like_idx = []
        for p in range(0, self.num_spin_orbs // 2):
            for q in range(p + 1, self.num_spin_orbs // 2):
                if p in self.inactive_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append([p, q])
                elif p in self.inactive_idx and q in self.active_unocc_idx:
                    self.kappa_hf_like_idx.append([p, q])
                elif p in self.active_occ_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append([p, q])
        # Construct theta1
        self.theta_picker = ThetaPicker(
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            is_spin_conserving=True,
            is_generalized=is_generalized,
        )
        self.theta_picker_full = ThetaPicker(
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            is_spin_conserving=False,
            is_generalized=is_generalized,
        )
        self._theta1 = []
        for _ in self.theta_picker_full.get_t1_generator_sa(0, 0):
            self._theta1.append(0)
        # Construct theta2
        self._theta2 = []
        for _ in self.theta_picker_full.get_t2_generator_sa(0, 0):
            self._theta2.append(0)
        # Construct theta3
        self._theta3 = []
        for _ in self.theta_picker_full.get_t3_generator(0, 0):
            self._theta3.append(0)
        # Construct theta4
        self._theta4 = []
        for _ in self.theta_picker_full.get_t4_generator(0, 0):
            self._theta4.append(0)
        # Construct theta5
        self._theta5 = []
        for _ in self.theta_picker_full.get_t5_generator(0, 0):
            self._theta5.append(0)
        # Construct theta6
        self._theta6 = []
        for _ in self.theta_picker_full.get_t6_generator(0, 0):
            self._theta6.append(0)

    @property
    def c_orthonormal(self) -> np.ndarray:
        return self._c_orthonormal

    @c_orthonormal.setter
    def c_orthonormal(self, c: np.ndarray) -> None:
        self._h_mo = None
        self._g_mo = None
        self._c_orthonormal = c

    @property
    def theta1(self) -> list[float]:
        return self._theta1

    @theta1.setter
    def theta1(self, theta: list[float]) -> None:
        if len(theta) != len(self._theta1):
            raise ValueError(f'Expected {len(self._theta1)} theta1 values got {len(theta)}')
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._theta1 = theta.copy()
        U = construct_ucc_u(
            self.num_active_spin_orbs,
            self.num_active_elec,
            self.theta1 + self.theta2 + self.theta3 + self.theta4 + self.theta5 + self.theta6,
            self.theta_picker,
            self._excitations,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )
        self.state_vector.new_u(U, allowed_states=self.state_vector.allowed_active_states_number_spin_conserving)

    @property
    def theta2(self) -> list[float]:
        return self._theta2

    @theta2.setter
    def theta2(self, theta: list[float]) -> None:
        if len(theta) != len(self._theta2):
            raise ValueError(f'Expected {len(self._theta2)} theta2 values got {len(theta)}')
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._theta2 = theta.copy()
        U = construct_ucc_u(
            self.num_active_spin_orbs,
            self.num_active_elec,
            self.theta1 + self.theta2 + self.theta3 + self.theta4 + self.theta5 + self.theta6,
            self.theta_picker,
            self._excitations,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )
        self.state_vector.new_u(U, allowed_states=self.state_vector.allowed_active_states_number_spin_conserving)

    @property
    def theta3(self) -> list[float]:
        return self._theta3

    @theta3.setter
    def theta3(self, theta: list[float]) -> None:
        if len(theta) != len(self._theta3):
            raise ValueError(f'Expected {len(self._theta3)} theta3 values got {len(theta)}')
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._theta3 = theta.copy()
        U = construct_ucc_u(
            self.num_active_spin_orbs,
            self.num_active_elec,
            self.theta1 + self.theta2 + self.theta3 + self.theta4 + self.theta5 + self.theta6,
            self.theta_picker,
            self._excitations,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )
        self.state_vector.new_u(U, allowed_states=self.state_vector.allowed_active_states_number_spin_conserving)

    @property
    def theta4(self) -> list[float]:
        return self._theta4

    @theta4.setter
    def theta4(self, theta: list[float]) -> None:
        if len(theta) != len(self._theta4):
            raise ValueError(f'Expected {len(self._theta4)} theta4 values got {len(theta)}')
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._theta4 = theta.copy()
        U = construct_ucc_u(
            self.num_active_spin_orbs,
            self.num_active_elec,
            self.theta1 + self.theta2 + self.theta3 + self.theta4 + self.theta5 + self.theta6,
            self.theta_picker,
            self._excitations,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )
        self.state_vector.new_u(U, allowed_states=self.state_vector.allowed_active_states_number_spin_conserving)

    @property
    def theta5(self) -> list[float]:
        return self._theta5

    @theta5.setter
    def theta5(self, theta: list[float]) -> None:
        if len(theta) != len(self._theta5):
            raise ValueError(f'Expected {len(self._theta5)} theta5 values got {len(theta)}')
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._theta5 = theta.copy()
        U = construct_ucc_u(
            self.num_active_spin_orbs,
            self.num_active_elec,
            self.theta1 + self.theta2 + self.theta3 + self.theta4 + self.theta5 + self.theta6,
            self.theta_picker,
            self._excitations,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )
        self.state_vector.new_u(U, allowed_states=self.state_vector.allowed_active_states_number_spin_conserving)

    @property
    def theta6(self) -> list[float]:
        return self._theta6

    @theta6.setter
    def theta6(self, theta: list[float]) -> None:
        if len(theta) != len(self._theta6):
            raise ValueError(f'Expected {len(self._theta6)} theta6 values got {len(theta)}')
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._theta6 = theta.copy()
        U = construct_ucc_u(
            self.num_active_spin_orbs,
            self.num_active_elec,
            self.theta1 + self.theta2 + self.theta3 + self.theta4 + self.theta5 + self.theta6,
            self.theta_picker,
            self._excitations,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )
        self.state_vector.new_u(U, allowed_states=self.state_vector.allowed_active_states_number_spin_conserving)

    def add_multiple_theta(self, theta: dict[str, list[float]], excitations: str) -> None:
        if 's' in excitations:
            self._theta1 = theta['theta1'].copy()
        if 'd' in excitations:
            self._theta2 = theta['theta2'].copy()
        if 't' in excitations:
            self._theta3 = theta['theta3'].copy()
        if 'q' in excitations:
            self._theta4 = theta['theta4'].copy()
        if '5' in excitations:
            self._theta5 = theta['theta5'].copy()
        if '6' in excitations:
            self._theta6 = theta['theta6'].copy()
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        U = construct_ucc_u(
            self.num_active_spin_orbs,
            self.num_active_elec,
            self.theta1 + self.theta2 + self.theta3 + self.theta4 + self.theta5 + self.theta6,
            self.theta_picker,
            self._excitations,
            allowed_states=self.state_vector.allowed_active_states_number_spin_conserving,
        )
        self.state_vector.new_u(U, allowed_states=self.state_vector.allowed_active_states_number_spin_conserving)

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
        if self._h_mo is None:
            self._h_mo = one_electron_integral_transform(self.c_trans, self.h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_trans, self.g_ao)
        return self._g_mo

    @property
    def rdm1(self) -> np.ndarray:
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
            for p in range(
                self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_spin_orbs
            ):
                for q in range(
                    self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs
                ):
                    self._rdm1[p - self.num_inactive_orbs, q - self.num_inactive_orbs] = expectation_value_pauli(
                        self.state_vector, epq_pauli(p, q, self.num_spin_orbs, self.num_elec), self.state_vector
                    )
        return self._rdm1
    
    @property
    def rdm2(self) -> np.ndarray:
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                )
            )
            for p in range(
                self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_spin_orbs
            ):
                for q in range(
                    self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_spin_orbs
                ):
                    for r in range(
                        self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_spin_orbs
                    ):
                        for s in range(
                            self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_spin_orbs
                        ):
                            self._rdm2[
                                p - self.num_inactive_orbs, q - self.num_inactive_orbs, r - self.num_inactive_orbs, s - self.num_inactive_orbs
                            ] = expectation_value_pauli(
                                self.state_vector,
                                epqrs_pauli(p, q, r, s, self.num_spin_orbs, self.num_elec),
                                self.state_vector)
        return self._rdm2

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
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
    ) -> None:
        """Run optimization of UCC wave function.

        Args:
            excitations: Excitation orders to include.
            orbital_optimization: Do orbital optimization.
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

        def print_progress(X: Sequence[float]) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                X: Wave function parameters.
            """
            global iteration
            global start
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{e_tot(X):3.12f}"
            print(f"{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            if iteration > 500:
                raise ValueError("Did not converge in 500 iterations in energy minimization.")
            start = time.time()  # type: ignore

        def silent_progress(X: Sequence[float]) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                X: Wave function parameters.
            """
            global iteration
            iteration += 1  # type: ignore
            if iteration > 500:
                raise ValueError("Did not converge in 500 iterations in energy minimization.")

        parameters = []
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
            for idx, _, _, _ in self.theta_picker.get_t1_generator_sa(0, 0):
                parameters += [self.theta1[idx]]
                num_theta1 += 1
        if "d" in excitations:
            for idx, _, _, _, _, _ in self.theta_picker.get_t2_generator_sa(0, 0):
                parameters += [self.theta2[idx]]
                num_theta2 += 1
        if "t" in excitations:
            for idx, _, _, _, _, _, _, _ in self.theta_picker.get_t3_generator(0, 0):
                parameters += [self.theta3[idx]]
                num_theta3 += 1
        if "q" in excitations:
            for idx, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_t4_generator(0, 0):
                parameters += [self.theta4[idx]]
                num_theta4 += 1
        if "5" in excitations:
            for idx, _, _, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_t5_generator(0, 0):
                parameters += [self.theta5[idx]]
                num_theta5 += 1
        if "6" in excitations:
            for idx, _, _, _, _, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_t6_generator(0, 0):
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
            for idx, _, _, _ in self.theta_picker.get_t1_generator_sa(0, 0):
                self.theta1[idx] = thetas[counter]
                counter += 1
        if "d" in excitations:
            thetas = res["x"][param_idx : num_theta2 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _ in self.theta_picker.get_t2_generator_sa(0, 0):
                self.theta2[idx] = thetas[counter]
                counter += 1
        if "t" in excitations:
            thetas = res["x"][param_idx : num_theta3 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _ in self.theta_picker.get_t3_generator(0, 0):
                self.theta3[idx] = thetas[counter]
                counter += 1
        if "q" in excitations:
            thetas = res["x"][param_idx : num_theta4 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_t4_generator(0, 0):
                self.theta4[idx] = thetas[counter]
                counter += 1
        if "5" in excitations:
            thetas = res["x"][param_idx : num_theta5 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_t5_generator(0, 0):
                self.theta5[idx] = thetas[counter]
                counter += 1
        if "6" in excitations:
            thetas = res["x"][param_idx : num_theta6 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_t6_generator(0, 0):
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
        num_inactive_spin_orbs: Number of inactive spin orbitals.
        num_active_spin_orbs: Number of active spin orbitals.
        num_virtual_spin_orbs: Number of virtual spin orbitals.
        num_elec: Number of electrons.
        num_active_elec: Number of electrons in active-space.
        state_vector: State vector object.
        c_othonormal: Orbital coefficients.
        h_core: Core Hamiltonian integrals in AO.
        g_eri: Two-electron integrals in AO.
        theta_picker: Cluster operator generator object.
        excitations: Excitation orders to consider.
        orbital_optimized: Do orbital optimization.
        kappa_idx: Indicies of non-redundant orbital rotations.

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
        for _ in wf.theta_picker.get_t1_generator_sa(0, 0):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if "d" in excitations:
        for _ in wf.theta_picker.get_t2_generator_sa(
            wf.num_inactive_spin_orbs + wf.num_active_spin_orbs + wf.num_virtual_spin_orbs, wf.num_elec
        ):
            theta2.append(parameters[idx_counter])
            idx_counter += 1
    if "t" in excitations:
        for _ in wf.theta_picker.get_t3_generator(0, 0):
            theta3.append(parameters[idx_counter])
            idx_counter += 1
    if "q" in excitations:
        for _ in wf.theta_picker.get_t4_generator(0, 0):
            theta4.append(parameters[idx_counter])
            idx_counter += 1
    if "5" in excitations:
        for _ in wf.theta_picker.get_t5_generator(0, 0):
            theta5.append(parameters[idx_counter])
            idx_counter += 1
    if "6" in excitations:
        for _ in wf.theta_picker.get_t6_generator(0, 0):
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
    if 's' in excitations:
        theta_dict['theta1'] = theta1
    if 'd' in excitations:
        theta_dict['theta2'] = theta2
    if 't' in excitations:
        theta_dict['theta3'] = theta3
    if 'q' in excitations:
        theta_dict['theta4'] = theta4
    if '5' in excitations:
        theta_dict['theta5'] = theta5
    if '6' in excitations:
        theta_dict['theta6'] = theta6
    wf.add_multiple_theta(theta_dict, excitations)
    return expectation_value_pauli(
        wf.state_vector,
        energy_hamiltonian_pauli(
            wf.h_ao,
            wf.g_ao,
            c_trans,
            wf.num_inactive_spin_orbs,
            wf.num_active_spin_orbs,
            wf.num_virtual_spin_orbs,
            wf.num_elec,
        ),
        wf.state_vector,
    )


def gradient_ucc(
    parameters: Sequence[float],
    excitations: str,
    orbital_optimized: bool,
    wf: WaveFunctionUCC,
) -> np.ndarray:
    """ """
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
    """ """
    rdms = ReducedDenstiyMatrix(
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_active_orbs,
        rdm1=wf.rdm1,
        rdm2=wf.rdm2,
    )
    gradient = get_orbital_gradient(
        rdms,
        wf.h_ao,
        wf.g_ao,
        wf.c_trans,
        wf.kappa_idx,
        wf.num_inactive_orbs,
        wf.num_active_orbs,
    )
    return gradient


def active_space_parameter_gradient(
    wf: WaveFunctionUCC, 
    parameters: Sequence[float],
    excitations: str,
    orbital_optimized: bool,
    finite_diff_type: str = "forward",
) -> np.ndarray:
    """ """
    if finite_diff_type not in ("central", "forward"):
        raise ValueError(f"finite_diff_type must be central or forward, got {finite_diff_type}")
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
        for _ in wf.theta_picker.get_t1_generator_sa(0, 0):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if "d" in excitations:
        for _ in wf.theta_picker.get_t2_generator_sa(
            wf.num_inactive_spin_orbs + wf.num_active_spin_orbs + wf.num_virtual_spin_orbs, wf.num_elec
        ):
            theta2.append(parameters[idx_counter])
            idx_counter += 1
    if "t" in excitations:
        for _ in wf.theta_picker.get_t3_generator(0, 0):
            theta3.append(parameters[idx_counter])
            idx_counter += 1
    if "q" in excitations:
        for _ in wf.theta_picker.get_t4_generator(0, 0):
            theta4.append(parameters[idx_counter])
            idx_counter += 1
    if "5" in excitations:
        for _ in wf.theta_picker.get_t5_generator(0, 0):
            theta5.append(parameters[idx_counter])
            idx_counter += 1
    if "6" in excitations:
        for _ in wf.theta_picker.get_t6_generator(0, 0):
            theta6.append(parameters[idx_counter])
            idx_counter += 1
    assert len(parameters) == len(kappa) + len(theta1) + len(theta2) + len(theta3) + len(theta4) + len(
        theta5
    ) + len(theta6)

    kappa_mat = np.zeros_like(wf.c_orthonormal)
    if len(kappa) != 0:
        if np.max(np.abs(kappa)) > 0.0:
            for kappa_val, (p, q) in zip(kappa, wf.kappa_idx):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    if len(wf.kappa_redundant) != 0:
        if np.max(np.abs(wf.kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(wf.kappa_redundant, wf.kappa_redundant_idx):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val

    c_trans = np.matmul(wf.c_orthonormal, scipy.linalg.expm(-kappa_mat))
    Hamiltonian = convert_pauli_to_hybrid_form(
        energy_hamiltonian_pauli(
            wf.h_ao,
            wf.g_ao,
            c_trans,
            wf.num_inactive_spin_orbs,
            wf.num_active_spin_orbs,
            wf.num_virtual_spin_orbs,
            wf.num_elec,
        ),
        wf.num_inactive_spin_orbs,
        wf.num_active_spin_orbs,
        wf.num_virtual_spin_orbs,
    )

    theta_params = theta1 + theta2 + theta3 + theta4 + theta5 + theta6
    gradient_theta = np.zeros_like(theta_params)
    if finite_diff_type == "central":
        eps = np.finfo(np.float64).eps ** (1 / 3)
    if finite_diff_type == "forward":
        eps = np.finfo(np.float64).eps ** (1 / 2)
        E = expectation_value_hybrid(wf.state_vector, Hamiltonian, wf.state_vector)
    for i, _ in enumerate(theta_params):
        sign_step = (theta_params[i] >= 0).astype(float) * 2 - 1
        step_size = eps * sign_step * max(1, abs(theta_params[i]))
        theta_params[i] += step_size
        theta_dict = {}
        idx = 0
        if 's' in excitations:
            theta_dict['theta1'] = theta_params[idx:idx+len(theta1)]
            idx += len(theta1)
        if 'd' in excitations:
            theta_dict['theta2'] = theta_params[idx:idx+len(theta2)]
            idx += len(theta2)
        if 't' in excitations:
            theta_dict['theta3'] = theta_params[idx:idx+len(theta3)]
            idx += len(theta3)
        if 'q' in excitations:
            theta_dict['theta4'] = theta_params[idx:idx+len(theta4)]
            idx += len(theta4)
        if '5' in excitations:
            theta_dict['theta5'] = theta_params[idx:idx+len(theta5)]
            idx += len(theta5)
        if '6' in excitations:
            theta_dict['theta6'] = theta_params[idx:idx+len(theta6)]
            idx += len(theta6)
        wf.add_multiple_theta(theta_dict, excitations)
        E_plus = expectation_value_hybrid(wf.state_vector, Hamiltonian, wf.state_vector)
        theta_params[i] -= step_size
        theta_dict = {}
        idx = 0
        if 's' in excitations:
            theta_dict['theta1'] = theta_params[idx:idx+len(theta1)]
            idx += len(theta1)
        if 'd' in excitations:
            theta_dict['theta2'] = theta_params[idx:idx+len(theta2)]
            idx += len(theta2)
        if 't' in excitations:
            theta_dict['theta3'] = theta_params[idx:idx+len(theta3)]
            idx += len(theta3)
        if 'q' in excitations:
            theta_dict['theta4'] = theta_params[idx:idx+len(theta4)]
            idx += len(theta4)
        if '5' in excitations:
            theta_dict['theta5'] = theta_params[idx:idx+len(theta5)]
            idx += len(theta5)
        if '6' in excitations:
            theta_dict['theta6'] = theta_params[idx:idx+len(theta6)]
            idx += len(theta6)
        if finite_diff_type == "central":
            theta_params[i] -= step_size
            theta_dict = {}
            idx = 0
            if 's' in excitations:
                theta_dict['theta1'] = theta_params[idx:idx+len(theta1)]
                idx += len(theta1)
            if 'd' in excitations:
                theta_dict['theta2'] = theta_params[idx:idx+len(theta2)]
                idx += len(theta2)
            if 't' in excitations:
                theta_dict['theta3'] = theta_params[idx:idx+len(theta3)]
                idx += len(theta3)
            if 'q' in excitations:
                theta_dict['theta4'] = theta_params[idx:idx+len(theta4)]
                idx += len(theta4)
            if '5' in excitations:
                theta_dict['theta5'] = theta_params[idx:idx+len(theta5)]
                idx += len(theta5)
            if '6' in excitations:
                theta_dict['theta6'] = theta_params[idx:idx+len(theta6)]
                idx += len(theta6)
            E_minus = expectation_value_hybrid(wf.state_vector, Hamiltonian, wf.state_vector)
            gradient_theta[i] = (E_plus - E_minus) / (2 * step_size)
            theta_params[i] += step_size
            theta_dict = {}
            idx = 0
            if 's' in excitations:
                theta_dict['theta1'] = theta_params[idx:idx+len(theta1)]
                idx += len(theta1)
            if 'd' in excitations:
                theta_dict['theta2'] = theta_params[idx:idx+len(theta2)]
                idx += len(theta2)
            if 't' in excitations:
                theta_dict['theta3'] = theta_params[idx:idx+len(theta3)]
                idx += len(theta3)
            if 'q' in excitations:
                theta_dict['theta4'] = theta_params[idx:idx+len(theta4)]
                idx += len(theta4)
            if '5' in excitations:
                theta_dict['theta5'] = theta_params[idx:idx+len(theta5)]
                idx += len(theta5)
            if '6' in excitations:
                theta_dict['theta6'] = theta_params[idx:idx+len(theta6)]
                idx += len(theta6)
        if finite_diff_type == "forward":
            gradient_theta[i] = (E_plus - E) / step_size
    return gradient_theta
