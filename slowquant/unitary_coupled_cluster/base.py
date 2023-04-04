from __future__ import annotations

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
import numpy as np
import copy

Z_mat = np.array([[1, 0], [0, -1]], dtype=float)
I_mat = np.array([[1, 0], [0, 1]], dtype=float)
a_mat = np.array([[0, 1], [0, 0]], dtype=float)
a_mat_dagger = np.array([[0, 0], [1, 0]], dtype=float)


from functools import cache

@cache
def a_op_spin(idx: int, dagger: bool, number_spin_orbitals: int, number_of_electrons: int):
    return a_op_spin_(idx, dagger, number_spin_orbitals, number_of_electrons)

class a_op_spin_:
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


@cache
def kronecker_product_cached(num_prior, num_after, val00, val01, val10, val11):
    """Does the P x P x P ..."""
    I1 = np.identity(int(2**num_prior))
    I2 = np.identity(int(2**num_after))
    mat = np.array([[val00, val01], [val10, val11]])
    return np.kron(I1, np.kron(mat, I2))

def kronecker_product_notcached(num_prior, num_after, operator):
    """Does the P x P x P ..."""
    I1 = np.identity(int(2**num_prior))
    I2 = np.identity(int(2**num_after))
    return np.kron(I1, np.kron(operator, I2))


def kronecker_product(A: list[a_op] | list[a_op_spin]) -> np.ndarray:
    """Does the P x P x P ..."""
    if len(A) < 2:
        return A
    total = np.kron(A[0], A[1])
    for operator in A[2:]:
        total = np.kron(total, operator)
    return total


class StateVector:
    def __init__(self, inactive, active) -> None:
        self.inactive = np.transpose(inactive)
        self._active_onvector = active
        self._active = np.transpose(kronecker_product(active))

    @property
    def bra_inactive(self):
        return np.conj(self.inactive).transpose()

    @property
    def ket_inactive(self):
        return self.inactive
    
    @property
    def bra_active(self):
        return np.conj(self.active).transpose()

    @property
    def ket_inactive(self):
        return self.inactive

    @property
    def U(self):
        return self.U_

    @U.setter
    def U(self, u):
        self.active = np.matmul(u, self._active)
        self.U_ = u


def a_op(
        spinless_idx: int, spin: str, dagger: bool, number_spin_orbitals: int, number_of_electrons: int
    ) -> list[np.ndarray]:
    idx = 2 * spinless_idx
    if spin == "beta":
        idx += 1
    operators = []
    for i in range(number_spin_orbitals):
        if i == idx:
            if dagger:
                operators.append(a_mat_dagger)
            else:
                operators.append(a_mat)
        elif i <= number_of_electrons and i < idx:
            operators.append(Z_mat)
        else:
            operators.append(I_mat)
    return operators


def expectation_value(bra: StateVector, fermiop: FermionicOperator, ket: StateVector) -> float:
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError('Bra and Ket does not have same number of inactive orbitals')
    if len(bra._active) != len(ket._active):
        raise ValueError('Bra and Ket does not have same number of active orbitals')
    total = 0
    for op in fermiop.operators:
        tmp = 1
        for i in range(len(bra.inactive)):
            tmp *= np.matmul(bra.bra_inactive[i], np.matmul(op[i], ket.ket_inactive[:,i]))
        number_active_orbitals = len(bra._active_onvector)
        active_start = len(bra.inactive)
        active_end = active_start + number_active_orbitals
        if number_active_orbitals != 0:
            operator = np.identity(2**number_active_orbitals)
            for op_element_idx, op_element in enumerate(op[active_start:active_end]):
                prior = op_element_idx
                after = number_active_orbitals - op_element_idx - 1
                if abs(op_element[0,0]) not in [0, 1] or abs(op_element[0,1]) not in [0, 1] or abs(op_element[1,0]) not in [0, 1] or abs(op_element[1,1]) not in [0, 1]:
                    operator = np.matmul(operator, kronecker_product_notcached(prior, after, op_element))
                else:
                    operator = np.matmul(operator, kronecker_product_cached(prior, after, op_element[0,0], op_element[0,1], op_element[1,0], op_element[1,1]))
            tmp *= np.matmul(bra.active, np.matmul(operator, ket.active))
        total += tmp
    return total


class FermionicOperator:
    def __init__(self, operator: list[np.ndarray]) -> None:
        if not isinstance(operator[0], list):
            self.operators = [operator]
        else:
            self.operators = operator

    def __add__(self, fermop: FermionicOperator) -> FermionicOperator:
        operators_new = self.operators + fermop.operators
        return FermionicOperator(operators_new)

    def __sub__(self, fermiop: FermionicOperator) -> FermionicOperator:
        operators_new = self.operators + (-1*fermiop).operators
        return FermionicOperator(operators_new)

    def __mul__(self, fermop: FermionicOperator) -> FermionicOperator:
        operators_new = []
        for op1 in self.operators:
            for op2 in fermop.operators:
                new_op = copy.copy(op1)
                for i in range(len(new_op)):
                    new_op[i] = np.matmul(new_op[i], op2[i])
                operators_new.append(new_op)
        return FermionicOperator(operators_new)

    def __rmul__(self, number: float) -> FermionicOperator:
        operators_new = copy.copy(self.operators)
        for i in range(len(operators_new)):
            operators_new[i][0] *= number
        return FermionicOperator(operators_new)


def Epq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> np.ndarray:
    E = FermionicOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec)) * FermionicOperator(a_op(q, "alpha", False, num_spin_orbs, num_elec))
    E += FermionicOperator(a_op(p, "beta", True, num_spin_orbs, num_elec)) * FermionicOperator(a_op(q, "beta", False, num_spin_orbs, num_elec))
    return E


def epqrs(p: int, q: int, r: int, s: int, num_spin_orbs: int, num_elec: int) -> np.ndarray:
    if q == r:
        return Epq(p, q, num_spin_orbs, num_elec) * Epq(r, s, num_spin_orbs, num_elec) - Epq(
            p, s, num_spin_orbs, num_elec
        )
    return Epq(p, q, num_spin_orbs, num_elec) * Epq(r, s, num_spin_orbs, num_elec)


def Eminuspq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> np.ndarray:
    return Epq(p, q, num_spin_orbs, num_elec) - Epq(q, p, num_spin_orbs, num_elec)


def Hamiltonian(h: np.ndarray, g: np.ndarray, c_mo: np.ndarray, num_spin_orbs: int, num_elec: int) -> np.ndarray:
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_spatial_orbs = num_spin_orbs//2
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            if p == 0 and q == 0:
                H_expectation = h_mo[p, q] * Epq(p, q, num_spin_orbs, num_elec)
            else:
                H_expectation += h_mo[p, q] * Epq(p, q, num_spin_orbs, num_elec)
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            for r in range(num_spatial_orbs):
                for s in range(num_spatial_orbs):
                    H_expectation += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s, num_spin_orbs, num_elec)
    return H_expectation
