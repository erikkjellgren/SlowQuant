from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
import numpy as np

Z_mat = np.array([[1, 0], [0, -1]])
I_mat = np.array([[1, 0], [0, 1]])
a_mat = np.array([[0, 1], [0, 0]])
a_mat_dagger = np.array([[0, 0], [1, 0]])


class a_op:
    def __init__(
        self, spinless_idx: int, spin: str, dagger: bool, number_spin_orbitals: int, number_of_electrons: int
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
        self.matrix_form = kronecker_product(self.operators)


class a_op_spin:
    def __init__(self, idx: int, dagger: bool, number_spin_orbitals: int, number_of_electrons: int) -> None:
        """Initialize fermionic annihilation operator.

        Args:
            idx: Spin orbital index.
            dagger: If creation operator.
        """
        self.dagger = dagger
        self.operators = []
        self.idx = idx
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
        self.matrix_form = kronecker_product(self.operators)


def kronecker_product(A: list[a_op] | list[a_op_spin]) -> np.ndarray:
    """Does the P x P x P ..."""
    total = np.kron(A[0], A[1])
    for operator in A[2:]:
        total = np.kron(total, operator)
    return total


def Epq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> np.ndarray:
    E = np.matmul(
        a_op(p, "alpha", True, num_spin_orbs, num_elec).matrix_form,
        a_op(q, "alpha", False, num_spin_orbs, num_elec).matrix_form,
    )
    E += np.matmul(
        a_op(p, "beta", True, num_spin_orbs, num_elec).matrix_form,
        a_op(q, "beta", False, num_spin_orbs, num_elec).matrix_form,
    )
    return E


def epqrs(p: int, q: int, r: int, s: int, num_spin_orbs: int, num_elec: int) -> np.ndarray:
    if q == r:
        return np.matmul(Epq(p, q, num_spin_orbs, num_elec), Epq(r, s, num_spin_orbs, num_elec)) - Epq(
            p, s, num_spin_orbs, num_elec
        )
    return np.matmul(Epq(p, q, num_spin_orbs, num_elec), Epq(r, s, num_spin_orbs, num_elec))


def Eminuspq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> np.ndarray:
    return Epq(p, q, num_spin_orbs, num_elec) - Epq(q, p, num_spin_orbs, num_elec)


def H(h: np.ndarray, g: np.ndarray, c_mo: np.ndarray, num_spin_orbs: int, num_elec: int) -> np.ndarray:
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_bf = len(c_mo)
    H_operator = np.zeros((2**num_spin_orbs, 2**num_spin_orbs))
    for p in range(num_bf):
        for q in range(num_bf):
            H_operator += h_mo[p, q] * Epq(p, q, num_spin_orbs, num_elec)
    for p in range(num_bf):
        for q in range(num_bf):
            for r in range(num_bf):
                for s in range(num_bf):
                    H_operator += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s, num_spin_orbs, num_elec)
    return H_operator
