import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import scipy
import scipy.optimize

from slowquant.unitary_coupled_cluster.base import (
    Hamiltonian_energy_only,
    StateVector,
    expectation_value,
)
from slowquant.unitary_coupled_cluster.util import (
    ThetaPicker,
    construct_integral_trans_mat,
    construct_UCC_U,
)


class WaveFunctionUCC:
    def __init__(
        self,
        number_spin_orbitals: int,
        number_electrons: int,
        cas: Sequence[int, int],
        c_orthonormal: np.ndarray,
        h_core: np.ndarray,
        g_eri: np.ndarray,
        is_generalized: bool = False,
        include_active_kappa: bool = False,
    ) -> None:
        o = np.array([0, 1])
        z = np.array([1, 0])
        self.c_orthonormal = c_orthonormal
        self.h_core = h_core
        self.g_eri = g_eri
        self.inactive = []
        self.virtual = []
        self.active = []
        self.active_occ = []
        self.active_unocc = []
        self.num_elec = number_electrons
        self.num_spin_orbs = number_spin_orbitals
        inactive_on_vector = []
        active_on_vector = []
        virtual_on_vector = []
        self.num_active_elec = 0
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
        self.num_virtual_spin_orbs = 0
        active_space = []
        orbital_counter = 0
        for i in range(number_electrons - cas[0], number_electrons):
            active_space.append(i)
            orbital_counter += 1
        for i in range(number_electrons, number_electrons + 2 * cas[1] - orbital_counter):
            active_space.append(i)
        for i in range(number_electrons):
            if i in active_space:
                self.active.append(i)
                self.active_occ.append(i)
                active_on_vector.append(o)
                self.num_active_spin_orbs += 1
                self.num_active_elec += 1
            else:
                self.inactive.append(i)
                inactive_on_vector.append(o)
                self.num_inactive_spin_orbs += 1
        for i in range(number_electrons, number_spin_orbitals):
            if i in active_space:
                self.active.append(i)
                self.active_unocc.append(i)
                active_on_vector.append(z)
                self.num_active_spin_orbs += 1
            else:
                self.virtual.append(i)
                virtual_on_vector.append(z)
                self.num_virtual_spin_orbs += 1
        if len(self.active) != 0:
            active_shift = np.min(self.active)
            for i in range(len(self.active_occ)):
                self.active_occ[i] -= active_shift
            for i in range(len(self.active_unocc)):
                self.active_unocc[i] -= active_shift
        self.state_vector = StateVector(inactive_on_vector, active_on_vector, virtual_on_vector)
        # Find non-redundant kappas
        self.kappa = []
        self.kappa_idx = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_spin_orbs, 2):
            for q in range(p + 2, self.num_spin_orbs, 2):
                if p in self.inactive and q in self.inactive:
                    continue
                elif p in self.virtual and q in self.virtual:
                    continue
                elif not include_active_kappa:
                    if p in self.active and q in self.active:
                        continue
                self.kappa.append(0)
                self.kappa_idx.append([p // 2, q // 2])
        # Construct theta1
        self.theta_picker = ThetaPicker(
            self.active_occ, self.active_unocc, is_spin_conserving=True, is_generalized=is_generalized
        )
        self.theta_picker_full = ThetaPicker(
            self.active_occ, self.active_unocc, is_spin_conserving=False, is_generalized=is_generalized
        )
        self.theta1 = []
        for _ in self.theta_picker_full.get_T1_generator_SA(0, 0):
            self.theta1.append(0)
        # Construct theta2
        self.theta2 = []
        for _ in self.theta_picker_full.get_T2_generator_SA(0, 0):
            self.theta2.append(0)
        # Construct theta3
        self.theta3 = []
        for _ in self.theta_picker_full.get_T3_generator(0, 0):
            self.theta3.append(0)
        # Construct theta4
        self.theta4 = []
        for _ in self.theta_picker_full.get_T4_generator(0, 0):
            self.theta4.append(0)

    @property
    def c_trans(self) -> np.ndarray:
        kappa_mat = np.zeros_like(self.c_orthonormal)
        for kappa_val, (p, q) in zip(self.kappa, self.kappa_idx):
            kappa_mat[p, q] = kappa_val
            kappa_mat[q, p] = -kappa_val
        return np.matmul(self.c_orthonormal, scipy.linalg.expm(-kappa_mat))

    @property
    def update_state_vector(self) -> None:
        U = construct_UCC_U(
            self.num_active_spin_orbs,
            self.num_active_elec,
            self.theta1 + self.theta2 + self.theta3 + self.theta4,
            "sdtq",
            self.active_occ,
            self.active_unocc,
        )
        self.state_vector.new_U(U)

    def run_UCC(self, excitations: str, orbital_optimization: bool = False) -> None:
        excitations = excitations.lower()
        self._excitations = excitations
        if orbital_optimization:
            e_tot = partial(
                energy_UCC,
                num_inactive_spin_orbs=self.num_inactive_spin_orbs,
                num_active_spin_orbs=self.num_active_spin_orbs,
                num_virtual_spin_orbs=self.num_virtual_spin_orbs,
                num_elec=self.num_elec,
                num_active_elec=self.num_active_elec,
                state_vector=self.state_vector,
                c_orthonormal=self.c_orthonormal,
                h_core=self.h_core,
                g_eri=self.g_eri,
                theta_picker=self.theta_picker,
                excitations=excitations,
                orbital_optimized=True,
                kappa_idx=self.kappa_idx,
            )
        else:
            e_tot = partial(
                energy_UCC,
                num_inactive_spin_orbs=self.num_inactive_spin_orbs,
                num_active_spin_orbs=self.num_active_spin_orbs,
                num_virtual_spin_orbs=self.num_virtual_spin_orbs,
                num_elec=self.num_elec,
                num_active_elec=self.num_active_elec,
                state_vector=self.state_vector,
                c_orthonormal=construct_integral_trans_mat(self.c_orthonormal, self.kappa, self.kappa_idx),
                h_core=self.h_core,
                g_eri=self.g_eri,
                theta_picker=self.theta_picker,
                excitations=excitations,
                orbital_optimized=False,
                kappa_idx=[],
            )
        global iteration
        global start
        iteration = 0
        start = time.time()

        def print_progress(X: list[float]) -> None:
            global iteration
            global start
            time_str = f'{time.time() - start:7.2f}'
            e_str = f'{e_tot(X):3.6f}'
            print(f'{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}')
            iteration += 1
            start = time.time()

        parameters = []
        num_kappa = 0
        num_theta1 = 0
        num_theta2 = 0
        num_theta3 = 0
        num_theta4 = 0
        if orbital_optimization:
            parameters += self.kappa
            num_kappa += len(self.kappa)
        if "s" in excitations:
            for idx, _, _, _ in self.theta_picker.get_T1_generator_SA(0, 0):
                parameters += [self.theta1[idx]]
                num_theta1 += 1
        if "d" in excitations:
            for idx, _, _, _, _, _ in self.theta_picker.get_T2_generator_SA(0, 0):
                parameters += [self.theta2[idx]]
                num_theta2 += 1
        if "t" in excitations:
            for idx, _, _, _, _, _, _, _ in self.theta_picker.get_T3_generator(0, 0):
                parameters += [self.theta3[idx]]
                num_theta3 += 1
        if "q" in excitations:
            for idx, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_T4_generator(0, 0):
                parameters += [self.theta4[idx]]
                num_theta4 += 1
        print("### Parameters information:")
        print(f"### Number kappa: {num_kappa}")
        print(f"### Number theta1: {num_theta1}")
        print(f"### Number theta2: {num_theta2}")
        print(f"### Number theta3: {num_theta3}")
        print(f"### Number theta4: {num_theta4}")
        print(f"### Total parameters: {num_kappa+num_theta1+num_theta2+num_theta3+num_theta4}\n")
        print('Iteration # | Iteration time [s] | Electronic energy [Hartree]')
        res = scipy.optimize.minimize(e_tot, parameters, tol=1e-8, callback=print_progress, method="SLSQP")
        self.ucc_energy = res["fun"]
        param_idx = 0
        if orbital_optimization:
            self.kappa = res["x"][param_idx : len(self.kappa) + param_idx].tolist()
            param_idx += len(self.kappa)
        if "s" in excitations:
            thetas = res["x"][param_idx : num_theta1 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _ in self.theta_picker.get_T1_generator_SA(0, 0):
                self.theta1[idx] = thetas[counter]
                counter += 1
        if "d" in excitations:
            thetas = res["x"][param_idx : num_theta2 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _ in self.theta_picker.get_T2_generator_SA(0, 0):
                self.theta2[idx] = thetas[counter]
                counter += 1
        if "t" in excitations:
            thetas = res["x"][param_idx : num_theta3 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _ in self.theta_picker.get_T3_generator(0, 0):
                self.theta3[idx] = thetas[counter]
                counter += 1
        if "q" in excitations:
            thetas = res["x"][param_idx : num_theta4 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_T4_generator(0, 0):
                self.theta4[idx] = thetas[counter]
                counter += 1


def energy_UCC(
    parameters: list[float],
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    num_virtual_spin_orbs: int,
    num_elec: int,
    num_active_elec: int,
    state_vector: StateVector,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
    theta_picker: ThetaPicker,
    excitations: str,
    orbital_optimized: bool,
    kappa_idx: list[list[int, int]],
) -> float:
    start = time.time()
    kappa = []
    theta1 = []
    theta2 = []
    theta3 = []
    theta4 = []
    idx_counter = 0
    for i in range(len(kappa_idx)):
        kappa.append(parameters[idx_counter])
        idx_counter += 1
    if "s" in excitations:
        for _ in theta_picker.get_T1_generator_SA(0, 0):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if "d" in excitations:
        for _ in theta_picker.get_T2_generator_SA(
            num_inactive_spin_orbs + num_active_spin_orbs + num_virtual_spin_orbs, num_elec
        ):
            theta2.append(parameters[idx_counter])
            idx_counter += 1
    if "t" in excitations:
        for _ in theta_picker.get_T3_generator(0, 0):
            theta3.append(parameters[idx_counter])
            idx_counter += 1
    if "q" in excitations:
        for _ in theta_picker.get_T4_generator(0, 0):
            theta4.append(parameters[idx_counter])
            idx_counter += 1
    assert len(parameters) == len(kappa) + len(theta1) + len(theta2) + len(theta3) + len(theta4)

    if orbital_optimized:
        kappa_mat = np.zeros_like(c_orthonormal)
        for kappa_val, (p, q) in zip(kappa, kappa_idx):
            kappa_mat[p, q] = kappa_val
            kappa_mat[q, p] = -kappa_val
        c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    else:
        c_trans = c_orthonormal

    U = construct_UCC_U(
        num_active_spin_orbs,
        num_active_elec,
        theta1 + theta2 + theta3 + theta4,
        theta_picker,
        excitations,
        allowed_states=state_vector.allowed_active_states_number_spin_conserving,
    )
    state_vector.new_U(U, allowed_states=state_vector.allowed_active_states_number_spin_conserving)
    A = expectation_value(
        state_vector,
        Hamiltonian_energy_only(
            h_core,
            g_eri,
            c_trans,
            num_inactive_spin_orbs,
            num_active_spin_orbs,
            num_virtual_spin_orbs,
            num_elec,
        ),
        state_vector,
    )
    return A
