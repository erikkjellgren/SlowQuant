import numpy as np
import scipy
import scipy.optimize
from slowquant.second_quantization_matrix.second_quant_mat_base import Hamiltonian
from functools import partial
import time
from scipy.sparse import csr_matrix
from slowquant.second_quantization_matrix.second_quant_mat_util import construct_integral_trans_mat

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
        self.on_vector = [o] * number_electrons + [z] * (number_spin_orbitals - number_electrons)
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
        for i in range(number_electrons):
            if i in active_space:
                self.active.append(i)
                self.active_occ.append(i)
            else:
                self.inactive.append(i)
        for i in range(number_electrons, number_spin_orbitals):
            if i in active_space:
                self.active.append(i)
                self.active_unocc.append(i)
            else:
                self.virtual.append(i)
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

    def run_HF(self) -> None:
        e_tot = partial(
            total_energy_HF,
            kappa_idx=self.kappa_idx,
            num_spin_orbs=self.num_spin_orbs,
            num_elec=self.num_elec,
            on_vector=np.array(self.on_vector),
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

def total_energy_HF(
    kappa: list[float],
    kappa_idx: list[list[int, int]],
    num_spin_orbs: int,
    num_elec: int,
    on_vector: csr_matrix,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
) -> float:
    c_trans = construct_integral_trans_mat(c_orthonormal, kappa, kappa_idx)
    HF_ket = on_vector.transpose()
    HF_bra = np.conj(HF_ket).transpose()
    return Hamiltonian(h_core, g_eri, c_trans, num_spin_orbs, num_elec, HF_bra, HF_ket)
