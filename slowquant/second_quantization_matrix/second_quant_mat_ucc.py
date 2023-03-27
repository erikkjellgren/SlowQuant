import numpy as np
import scipy
import scipy.optimize
from slowquant.second_quantization_matrix.second_quant_mat_base import H, kronecker_product, a_op_spin
from functools import partial
import time
from scipy.sparse import csr_matrix


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
        on_vector = [o] * number_electrons + [z] * (number_spin_orbitals - number_electrons)
        self.on_vector = kronecker_product(on_vector)
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
                elif not include_active_kappa:
                    if p in self.active and q in self.active:
                        continue
                elif p in self.virtual and q in self.virtual:
                    continue
                self.kappa.append(0)
                self.kappa_idx.append([p // 2, q // 2])
        # Construct theta1
        self.theta1 = []
        for a in self.active_unocc:
            for i in self.active_occ:
                self.theta1.append(0)
        # Construct theta2
        self.theta2 = []
        for a in self.active_unocc:
            for b in self.active_unocc:
                if a >= b:
                    continue
                for i in self.active_occ:
                    for j in self.active_occ:
                        if i >= j:
                            continue
                        self.theta2.append(0)

    def run_HF(self) -> None:
        e_tot = partial(
            total_energy_HF,
            kappa_idx=self.kappa_idx,
            num_spin_orbs=self.num_spin_orbs,
            num_elec=self.num_elec,
            on_vector=self.on_vector,
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

    def run_UCCS(self, orbital_optimization: bool = False) -> None:
        if orbital_optimization:
            e_tot = partial(
                total_energy_UCC,
                num_spin_orbs=self.num_spin_orbs,
                num_elec=self.num_elec,
                on_vector=self.on_vector,
                c_orthonormal=self.c_orthonormal,
                h_core=self.h_core,
                g_eri=self.g_eri,
                active_occ=self.active_occ,
                active_unocc=self.active_unocc,
                excitations="S",
                orbital_optimized =  True,
                kappa_idx=self.kappa_idx
            )
        else:
            e_tot = partial(
                total_energy_UCC,
                num_spin_orbs=self.num_spin_orbs,
                num_elec=self.num_elec,
                on_vector=self.on_vector,
                c_orthonormal=self.c_orthonormal,
                h_core=self.h_core,
                g_eri=self.g_eri,
                active_occ=self.active_occ,
                active_unocc=self.active_unocc,
                excitations="S",
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

        if orbital_optimization:
            res = scipy.optimize.minimize(e_tot, self.kappa+self.theta1, tol=1e-6, callback=print_progress)
            self.uccs_energy = res["fun"]
            self.kappa = res["x"][:len(self.kappa)]
            self.theta1 = res["x"][len(self.kappa):]
        else:
            res = scipy.optimize.minimize(e_tot, self.theta1, tol=1e-6, callback=print_progress)
            self.uccs_energy = res["fun"]
            self.theta1 = res["x"]

    def run_UCCD(self, orbutal_optimization: bool = False) -> None:
        None

    def run_UCCSD(self, orbutal_optimization: bool = False) -> None:
        None


def total_energy_HF(
    kappa: list[float],
    kappa_idx: list[list[int, int]],
    num_spin_orbs: int,
    num_elec: int,
    on_vector: np.ndarray,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
) -> float:
    kappa_mat = np.zeros_like(c_orthonormal)
    for kappa_val, (p, q) in zip(kappa, kappa_idx):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    HF_ket = on_vector.transpose()
    HF_bra = np.conj(HF_ket).transpose()
    return HF_bra.dot(H(h_core, g_eri, c_trans, num_spin_orbs, num_elec).dot(HF_ket)).toarray()[0,0]

def total_energy_UCC(
    parameters: list[float],
    num_spin_orbs: int,
    num_elec: int,
    on_vector: np.ndarray,
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
    if "S" in excitations:
        for a in range(num_spin_orbs):
            for i in range(num_spin_orbs):
                if i in active_occ and a in active_unocc:
                    theta1.append(parameters[idx_counter])
                    idx_counter += 1
    if "D" in excitations:
        for a in range(num_spin_orbs):
            for b in range(num_spin_orbs):
                if a >= b:
                    continue
                for i in range(num_spin_orbs):
                    for j in range(num_spin_orbs):
                        if i >= j:
                            continue
                        if i in active_occ and j in active_occ and a in active_unocc and b in active_unocc:
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
    
    t = np.zeros((2**num_spin_orbs,2**num_spin_orbs))
    if "S" in excitations:
        counter = 0
        for a in range(num_spin_orbs):
            for i in range(num_spin_orbs):
                if i in active_occ and a in active_unocc:
                    tmp = a_op_spin(a, True, num_spin_orbs, num_elec).matrix_form.dot(a_op_spin(i, False, num_spin_orbs, num_elec).matrix_form)
                    t += theta1[counter]*tmp
                    counter += 1
    
    if "D" in excitations:
        counter = 0
        for a in range(num_spin_orbs):
            for b in range(num_spin_orbs):
                if a >= b:
                    continue
                for i in range(num_spin_orbs):
                    for j in range(num_spin_orbs):
                        if i >= j:
                            continue
                        if i in active_occ and j in active_occ and a in active_unocc and b in active_unocc:
                            tmp = a_op_spin(a, True, num_spin_orbs, num_elec).matrix_form.dot(a_op_spin(b, True, num_spin_orbs, num_elec).matrix_form)
                            tmp = tmp.dot(a_op_spin(j, False, num_spin_orbs, num_elec).matrix_form)
                            tmp = tmp.dot(a_op_spin(i, False, num_spin_orbs, num_elec).matrix_form)
                            t += theta2[counter]*tmp
                            counter += 1

    T = t - np.conj(t).transpose()
    U = csr_matrix(scipy.linalg.expm(T))
    UCC_ket = U.dot(on_vector.transpose())
    UCC_bra = np.conj(UCC_ket).transpose()
    return UCC_bra.dot(H(h_core, g_eri, c_trans, num_spin_orbs, num_elec).dot(UCC_ket)).toarray()[0,0]
