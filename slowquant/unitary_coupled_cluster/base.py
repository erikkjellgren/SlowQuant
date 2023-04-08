from __future__ import annotations

import copy
import time
from functools import cache

import numpy as np
import scipy.sparse as ss

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)

Z_mat = np.array([[1, 0], [0, -1]], dtype=float)
I_mat = np.array([[1, 0], [0, 1]], dtype=float)
a_mat = np.array([[0, 1], [0, 0]], dtype=float)
a_mat_dagger = np.array([[0, 0], [1, 0]], dtype=float)


@cache
def a_op_spin_matrix(idx: int, dagger: bool, number_spin_orbitals: int, number_electrons: int) -> np.ndarray:
    r"""Get matrix representation of fermionic operator.
    This is the matrix form that depends on number of electrons i.e.:

    .. math::
        Z x Z x .. x a x .. I x I

    Args:
        idx: Spin orbital index.
        dagger: Creation or annihilation operator.
        number_spin_orbitals: Total number of spin orbitals.
        number_electrons: Total number of electrons.

    Returns:
        Matrix representation of ferminonic annihilation or creation operator.
    """
    operators = []
    for i in range(number_spin_orbitals):
        if i == idx:
            if dagger:
                operators.append(a_mat_dagger)
            else:
                operators.append(a_mat)
        elif i <= number_electrons and i < idx:
            operators.append(Z_mat)
        else:
            operators.append(I_mat)
    return kronecker_product(operators)


def a_op_spin(idx: int, dagger: bool, number_spin_orbitals: int, number_electrons: int) -> np.ndarray:
    operators = []
    for i in range(number_spin_orbitals):
        if i == idx:
            if dagger:
                operators.append(a_mat_dagger)
            else:
                operators.append(a_mat)
        elif i <= number_electrons and i < idx:
            operators.append(Z_mat)
        else:
            operators.append(I_mat)
    return operators


@cache
def kronecker_product_cached(
    num_prior: int, num_after: int, val00: int, val01: int, val10: int, val11: int, is_csr: bool
) -> np.ndarray | ss.csr_matrix:
    r"""Get operator in matrix form.
    The operator is returned in the form:

    .. math::
        I x I x .. o .. x I x I

    Args:
       num_prior: Number of left-hand side identity matrices.
       num_after: Number of right-hand side identity matrices.
       val00: First value of operator matrix.
       val10: Second value of operator matrix.
       val01: Third value of operator matrix.
       val11: Fourth value of operator matrix.
       is_csr: If the resulting matrix representation should be a sparse matrix.

    Returns:
       Matrix representation ofi an operator.
    """
    if (
        val00 not in [-1, 0, 1]
        or val01 not in [-1, 0, 1]
        or val10 not in [-1, 0, 1]
        or val11 not in [-1, 0, 1]
    ):
        print(
            f"WARNING: Unexpected element values in cahced kronecker product: {val00} {val01} {val10} {val11}"
        )
    if is_csr:
        I1 = ss.identity(int(2**num_prior))
        I2 = ss.identity(int(2**num_after))
        mat = ss.csr_matrix(np.array([[val00, val01], [val10, val11]]))
        return ss.kron(I1, ss.kron(mat, I2))
    else:
        I1 = np.identity(int(2**num_prior))
        I2 = np.identity(int(2**num_after))
        mat = np.array([[val00, val01], [val10, val11]])
        return np.kron(I1, np.kron(mat, I2))


def kronecker_product(A: list[np.ndarray]) -> np.ndarray:
    r"""Get Kronecker product of a list of matricies
    Does:

    .. math::
        P x P x P ...

    Args:
       A: List of matrices.

    Returns:
       Kronecker product of matrices.
    """
    if len(A) < 2:
        return A
    total = np.kron(A[0], A[1])
    for operator in A[2:]:
        total = np.kron(total, operator)
    return total


class StateVector:
    """State vector."""

    def __init__(self, inactive: list(np.ndarray), active: list(np.ndarray)) -> None:
        """Initialize state vector.

        Args:
            inactive: Kronecker representation of inactive orbitals (reference).
            active: Kronecker representation of active orbitals (refernce).
        """
        self.inactive = np.transpose(inactive)
        self._active_onvector = active
        self._active = np.transpose(kronecker_product(active))

    @property
    def bra_inactive(self) -> list(np.ndarray):
        return np.conj(self.inactive).transpose()

    @property
    def ket_inactive(self) -> list(np.ndarray):
        return self.inactive

    @property
    def bra_active(self) -> np.ndarray:
        return np.conj(self.active).transpose()

    @property
    def ket_active(self) -> np.ndarray:
        return self.active

    @property
    def bra_active_csr(self) -> ss.csr_matrix:
        return ss.csr_matrix(np.conj(self.active).transpose())

    @property
    def ket_active_csr(self) -> ss.csr_matrix:
        return ss.csr_matrix(self.active).transpose()

    @property
    def U(self) -> np.ndarray:
        return self.U_

    @U.setter
    def U(self, u: np.ndarray) -> None:
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


def expectation_value(
    bra: StateVector, fermiop: FermionicOperator, ket: StateVector, use_csr: int = 8
) -> float:
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError("Bra and Ket does not have same number of inactive orbitals")
    if len(bra._active) != len(ket._active):
        raise ValueError("Bra and Ket does not have same number of active orbitals")
    if fermiop.operators == [[]]:
        return 0
    total = 0
    start = time.time()
    for op in copy.copy(fermiop.operators):
        tmp = 1
        for i in range(len(bra.inactive)):
            tmp *= np.matmul(bra.bra_inactive[i], np.matmul(op[i], ket.ket_inactive[:, i]))
        number_active_orbitals = len(bra._active_onvector)
        active_start = len(bra.inactive)
        active_end = active_start + number_active_orbitals
        if number_active_orbitals != 0:
            if number_active_orbitals >= use_csr:
                operator = copy.copy(ket.ket_active_csr)
            else:
                operator = copy.copy(ket.ket_active)
            for op_element_idx, op_element in enumerate(op[active_start:active_end][::-1]):
                prior = op_element_idx
                after = number_active_orbitals - op_element_idx - 1
                factor = 1
                if abs(op_element[0, 0]) not in [0, 1]:
                    factor *= op_element[0, 0]
                    op_element = op_element / factor
                elif abs(op_element[0, 1]) not in [0, 1]:
                    factor *= op_element[0, 1]
                    op_element = op_element / factor
                elif abs(op_element[1, 0]) not in [0, 1]:
                    factor *= op_element[1, 0]
                    op_element = op_element / factor
                elif abs(op_element[1, 1]) not in [0, 1]:
                    factor *= op_element[1, 1]
                    op_element = op_element / factor
                if (
                    op_element[0, 0] == 1
                    and op_element[0, 1] == 0
                    and op_element[1, 0] == 0
                    and op_element[1, 1] == 1
                    and factor == 1
                ):
                    continue
                if number_active_orbitals >= use_csr:
                    operator = factor * kronecker_product_cached(
                        prior,
                        after,
                        op_element[0, 0],
                        op_element[0, 1],
                        op_element[1, 0],
                        op_element[1, 1],
                        True,
                    ).dot(operator)
                    if operator.nnz == 0:
                        break
                else:
                    operator = factor * np.matmul(
                        kronecker_product_cached(
                            prior,
                            after,
                            op_element[0, 0],
                            op_element[0, 1],
                            op_element[1, 0],
                            op_element[1, 1],
                            False,
                        ),
                        operator,
                    )
            if number_active_orbitals >= use_csr:
                tmp *= bra.bra_active_csr.dot(operator).toarray()[0, 0]
            else:
                tmp *= np.matmul(bra.bra_active, operator)
        total += tmp
    #print(f"Expectation value: {time.time() - start}")
    return total


class FermionicOperator:
    def __init__(self, operator: list[np.ndarray]) -> None:
        if not isinstance(operator[0], list):
            self.operators = [operator]
        else:
            self.operators = operator

    def __add__(self, fermiop: FermionicOperator) -> FermionicOperator:
        if self.operators == [[]]:
            operators_new = fermiop.operators
        elif fermiop.operators == [[]]:
            operators_new = self.operators
        else:
            operators_new = self.operators + fermiop.operators
        return FermionicOperator(operators_new)

    def __sub__(self, fermiop: FermionicOperator) -> FermionicOperator:
        if self.operators == [[]] and fermiop.operators == [[]]:
            return FermionicOperator([[]])
        elif self.operators == [[]]:
            other_op = copy.deepcopy(fermiop)
            operators_new = (-1 * other_op).operators
        elif fermiop.operators == [[]]:
            operators_new = self.operators
        else:
            other_op = copy.deepcopy(fermiop)
            operators_new = self.operators + (-1 * other_op).operators
        return FermionicOperator(operators_new)

    def __mul__(self, fermiop: FermionicOperator) -> FermionicOperator:
        operators_new = []
        for op1 in self.operators:
            if op1 == []:
                continue
            for op2 in fermiop.operators:
                if op2 == []:
                    continue
                new_op = copy.deepcopy(op1)
                is_zero = False
                for i in range(len(new_op)):
                    new_op[i] = np.matmul(new_op[i], op2[i])
                    if abs(new_op[i][0,0]) < 10**-12 and abs(new_op[i][0,1]) < 10**-12 and abs(new_op[i][1,0]) < 10**-12 and abs(new_op[i][1,1]) < 10**-12:
                        is_zero = True
                        break
                if not is_zero:
                    operators_new.append(new_op)
        if len(operators_new) == 0:
           return FermionicOperator([[]])  
        return FermionicOperator(operators_new)

    def __rmul__(self, number: float) -> FermionicOperator:
        operators_new = []
        for op in self.operators:
            op_new = copy.deepcopy(op)
            op_new[0] *= number
            if np.sum(np.abs(op_new[0])) > 10**-12:
                operators_new.append(op_new)
        if len(operators_new) == 0:
            return FermionicOperator([[]])
        return FermionicOperator(operators_new)

    @property
    def dagger(self):
        operators_new = []
        for op in self.operators:
            new_op = []
            for mat in op:
                new_op.append(np.conj(mat).transpose())
            operators_new.append(new_op)
        return FermionicOperator(operators_new)

    @property
    def matrix_form(self):
        num_spin_orbs = len(self.operators[0])
        op_matrix = np.zeros((2**num_spin_orbs, 2**num_spin_orbs))
        for op in self.operators:
            op_matrix += kronecker_product(op)
        return op_matrix


def Epq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> FermionicOperator:
    E = FermionicOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec)) * FermionicOperator(
        a_op(q, "alpha", False, num_spin_orbs, num_elec)
    )
    E += FermionicOperator(a_op(p, "beta", True, num_spin_orbs, num_elec)) * FermionicOperator(
        a_op(q, "beta", False, num_spin_orbs, num_elec)
    )
    return E


def epqrs(p: int, q: int, r: int, s: int, num_spin_orbs: int, num_elec: int) -> FermionicOperator:
    if q == r:
        return Epq(p, q, num_spin_orbs, num_elec) * Epq(r, s, num_spin_orbs, num_elec) - Epq(
            p, s, num_spin_orbs, num_elec
        )
    return Epq(p, q, num_spin_orbs, num_elec) * Epq(r, s, num_spin_orbs, num_elec)


def Eminuspq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> FermionicOperator:
    return Epq(p, q, num_spin_orbs, num_elec) - Epq(q, p, num_spin_orbs, num_elec)


def Hamiltonian(
    h: np.ndarray, g: np.ndarray, c_mo: np.ndarray, num_spin_orbs: int, num_elec: int
) -> FermionicOperator:
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_spatial_orbs = num_spin_orbs // 2
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


def commutator(A: FermionicOperator, B: FermionicOperator) -> FermionicOperator:
    return A * B - B * A
