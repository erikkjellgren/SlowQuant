import numpy as np
import scipy
import scipy.optimize
from slowquant.unitary_coupled_cluster.base import Hamiltonian, expectation_value, StateVector
from functools import partial
import time
from slowquant.unitary_coupled_cluster.util import construct_integral_trans_mat, iterate_T1, iterate_T2, construct_UCC_U

class WaveFunctionUCC:
    def __init__(
        self,
        number_spin_orbitals: int,
        number_electrons: int,
        active_space: list[int],
        c_orthonormal: np.ndarray,
        h_core: np.ndarray,
        g_eri: np.ndarray,
        include_active_kappa=False,
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
        self.num_active_elec = 0
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
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
        if len(self.active) != 0:
            active_shift = np.min(self.active)
            for i in range(len(self.active_occ)):
                self.active_occ[i] -= active_shift
            for i in range(len(self.active_unocc)):
                self.active_unocc[i] -= active_shift
        self.state_vector = StateVector(inactive_on_vector, active_on_vector)
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
        self.theta1 = []
        for _ in iterate_T1(self.active_occ, self.active_unocc):
            self.theta1.append(0)
        # Construct theta2
        self.theta2 = []
        for _ in iterate_T2(self.active_occ, self.active_unocc):
            self.theta2.append(0)

    def run_HF(self) -> None:
        e_tot = partial(
            energy_HF,
            kappa_idx=self.kappa_idx,
            num_spin_orbs=self.num_inactive_spin_orbs,
            num_elec=self.num_elec,
            state_vector=self.state_vector,
            c_orthonormal=self.c_orthonormal,
            h_core=self.h_core,
            g_eri=self.g_eri,
        )
        global iteration
        global start
        iteration = 0
        start = time.time()

        def print_progress(X: list[float]) -> None:
            global iteration
            global start
            print(iteration, time.time() - start, e_tot(X))
            iteration += 1
            start = time.time()

        res = scipy.optimize.minimize(e_tot, self.kappa, tol=1e-6, callback=print_progress)
        self.hf_energy = res["fun"]
        self.kappa = res["x"]

    def run_UCC(self, excitations: str, orbital_optimization: bool = False) -> None:
        excitations = excitations.lower()
        if orbital_optimization:
            e_tot = partial(
                energy_UCC,
                num_spin_orbs=self.num_inactive_spin_orbs + self.num_active_spin_orbs,
                num_active_spin_orbs=self.num_active_spin_orbs,
                num_elec=self.num_elec,
                num_active_elec=self.num_active_elec,
                state_vector=self.state_vector,
                c_orthonormal=self.c_orthonormal,
                h_core=self.h_core,
                g_eri=self.g_eri,
                active_occ=self.active_occ,
                active_unocc=self.active_unocc,
                excitations=excitations,
                orbital_optimized = True,
                kappa_idx=self.kappa_idx
            )
        else:
            e_tot = partial(
                energy_UCC,
                num_spin_orbs=self.num_inactive_spin_orbs + self.num_active_spin_orbs,
                num_active_spin_orbs=self.num_active_spin_orbs,
                num_elec=self.num_elec,
                num_active_elec=self.num_active_elec,
                state_vector=self.state_vector,
                c_orthonormal=construct_integral_trans_mat(self.c_orthonormal, self.kappa, self.kappa_idx),
                h_core=self.h_core,
                g_eri=self.g_eri,
                active_occ=self.active_occ,
                active_unocc=self.active_unocc,
                excitations=excitations,
                orbital_optimized = False,
                kappa_idx = []
            )
        global iteration
        global start
        iteration = 0
        start = time.time()

        def print_progress(X: list[float]) -> None:
            global iteration
            global start
            print(iteration, time.time() - start, e_tot(X))
            iteration += 1
            start = time.time()

        parameters = []
        if orbital_optimization:
            parameters += self.kappa
        if "s" in excitations:
            parameters += self.theta1
        if "d" in excitations:
            parameters += self.theta2
        res = scipy.optimize.minimize(e_tot, parameters, tol=1e-6, callback=print_progress)
        self.ucc_energy = res["fun"]
        param_idx = 0
        if orbital_optimization:
            self.kappa = res["x"][param_idx:len(self.kappa)+param_idx].tolist()
            param_idx += len(self.kappa)
        if "s" in excitations:
            self.theta1 = res["x"][param_idx:len(self.theta1)+param_idx].tolist()
            param_idx += len(self.theta1)
        if "d" in excitations:
            self.theta2 = res["x"][param_idx:len(self.theta2)+param_idx].tolist()
            param_idx += len(self.theta2)

def energy_HF(
    kappa: list[float],
    kappa_idx: list[list[int, int]],
    num_spin_orbs: int,
    num_elec: int,
    state_vector: StateVector,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
) -> float:
    c_trans = construct_integral_trans_mat(c_orthonormal, kappa, kappa_idx)
    return expectation_value(state_vector, Hamiltonian(h_core, g_eri, c_trans, num_spin_orbs, num_elec), state_vector)

def energy_UCC(
    parameters: list[float],
    num_spin_orbs: int,
    num_active_spin_orbs: int,
    num_elec: int,
    num_active_elec: int,
    state_vector: StateVector,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
    active_occ: list[int],
    active_unocc: list[int],
    excitations: str,
    orbital_optimized: bool,
    kappa_idx: list[list[int, int]],
) -> float:
    kappa = []
    theta1 = []
    theta2 = []
    idx_counter = 0
    for i in range(len(kappa_idx)):
        kappa.append(parameters[idx_counter])
        idx_counter += 1
    if "s" in excitations:
        for _ in iterate_T1(active_occ, active_unocc):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if "d" in excitations:
        for _ in iterate_T2(active_occ, active_unocc):
            theta2.append(parameters[idx_counter])
            idx_counter += 1

    if orbital_optimized:
        kappa_mat = np.zeros_like(c_orthonormal)
        for kappa_val, (p, q) in zip(kappa, kappa_idx):
            kappa_mat[p, q] = kappa_val
            kappa_mat[q, p] = -kappa_val
        c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    else:
        c_trans = c_orthonormal

    U = construct_UCC_U(num_active_spin_orbs, num_active_elec, theta1+theta2, excitations, active_occ, active_unocc)
    state_vector.U = U
    return expectation_value(state_vector, Hamiltonian(h_core, g_eri, c_trans, num_spin_orbs, num_elec), state_vector)
