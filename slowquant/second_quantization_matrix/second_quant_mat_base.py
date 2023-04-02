from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
import numpy as np
from scipy.sparse import csr_matrix
from functools import lru_cache

Z_mat = np.array([[1, 0], [0, -1]])
I_mat = np.array([[1, 0], [0, 1]])
a_mat = np.array([[0, 1], [0, 0]])
a_mat_dagger = np.array([[0, 0], [1, 0]])

class a_op:
    def __init__(
        self, spinless_idx: int, spin: str, dagger: bool, number_spin_orbitals: int, number_of_electrons: int, bra, ket
    ) -> None:
        """Initialize fermionic annihilation operator.

        Args:
            spinless_idx: Spatial orbital index.
            spin: Alpha or beta spin.
            dagger: If creation operator.
        """
        self.spinless_idx = spinless_idx
        self.idx = 2 * self.spinless_idx
        self.dagger = dagger
        self.spin = spin
        if self.spin == "beta":
            self.idx += 1
        self.operators = []
        for i in range(number_spin_orbitals):
            if i == self.idx:
                if self.dagger:
                    self.operators.append(a_mat_dagger)
                else:
                    self.operators.append(a_mat)
            elif i <= number_of_electrons and i < self.idx:
                self.operators.append(Z_mat)
            else:
                self.operators.append(I_mat)
        self.matrix_form = kronecker_product_braket_contracted(bra, self.operators, ket)

def kronecker_product(A: list[a_op]) -> csr_matrix:
    """Does the P x P x P ..."""
    total = np.kron(A[0], A[1])
    for operator in A[2:]:
        total = np.kron(total, operator)
    return csr_matrix(total)

def kronecker_product_ket_contracted(A: list[a_op], ket) -> csr_matrix:
    """Does the P x P x P ..."""
    total = np.matmul(A[0], ket[0])
    for operator, ket_ in zip(A[1:], ket[1:]):
        next_term = np.matmul(bra_, np.matmul(operator, ket_)) 
        total = np.kron(total, next_term)
    return csr_matrix(total)

def kronecker_product_braket_contracted(bra, A: list[a_op], ket) -> float:
    """Does the P x P x P ..."""
    total = np.matmul(bra[0], np.matmul(A[0], ket[:,0]))
    for i in range(1, len(A)):
        next_term = np.matmul(bra[i], np.matmul(A[i], ket[:,i])) 
        total += next_term
    return total


def Epq(p: int, q: int, num_spin_orbs: int, num_elec: int, bra, ket) -> np.ndarray:
    E = a_op(p, "alpha", True, num_spin_orbs, num_elec, bra, ket).matrix_form * a_op(q, "alpha", False, num_spin_orbs, num_elec, bra, ket).matrix_form
    E += a_op(p, "beta", True, num_spin_orbs, num_elec, bra, ket).matrix_form * a_op(q, "beta", False, num_spin_orbs, num_elec, bra, ket).matrix_form
    return E


def epqrs(p: int, q: int, r: int, s: int, num_spin_orbs: int, num_elec: int, bra, ket) -> np.ndarray:
    if q == r:
        return Epq(p, q, num_spin_orbs, num_elec, bra, ket) * Epq(r, s, num_spin_orbs, num_elec, bra, ket) - Epq(
            p, s, num_spin_orbs, num_elec, bra, ket
        )
    return Epq(p, q, num_spin_orbs, num_elec, bra, ket) * Epq(r, s, num_spin_orbs, num_elec, bra, ket)


def Eminuspq(p: int, q: int, num_spin_orbs: int, num_elec: int, bra, ket) -> np.ndarray:
    return Epq(p, q, num_spin_orbs, num_elec, bra, ket) - Epq(q, p, num_spin_orbs, num_elec, bra, ket)


def Hamiltonian(h: np.ndarray, g: np.ndarray, c_mo: np.ndarray, num_spin_orbs: int, num_elec: int, bra, ket) -> np.ndarray:
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_spatial_orbs = num_spin_orbs//2
    H_expectation = 0
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            H_expectation += h_mo[p, q] * Epq(p, q, num_spin_orbs, num_elec, bra, ket)
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            for r in range(num_spatial_orbs):
                for s in range(num_spatial_orbs):
                    H_expectation += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s, num_spin_orbs, num_elec, bra, ket)
    return H_expectation
