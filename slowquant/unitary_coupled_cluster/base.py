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


@cache
def a_op_spin(idx: int, dagger: bool, number_spin_orbitals: int, number_electrons: int) -> np.ndarray:
    if idx % 2 == 0:
        return a_op(idx // 2, "alpha", dagger, number_spin_orbitals, number_electrons)
    else:
        return a_op((idx - 1) // 2, "beta", dagger, number_spin_orbitals, number_electrons)


@cache
def a_op(
    spinless_idx: int, spin: str, dagger: bool, number_spin_orbitals: int, number_of_electrons: int
) -> list[np.ndarray]:
    idx = 2 * spinless_idx
    if spin == "beta":
        idx += 1
    operators = {}
    op1 = ""
    op2 = ""
    fac1 = 1
    fac2 = 1
    for i in range(number_spin_orbitals):
        if i == idx:
            if dagger:
                op1 += "X"
                fac1 *= 0.5
                op2 += "Y"
                fac2 *= -0.5j
            else:
                op1 += "X"
                fac1 *= 0.5
                op2 += "Y"
                fac2 *= 0.5j
        elif i <= number_of_electrons and i < idx:
            op1 += "Z"
            op2 += "Z"
        else:
            op1 += "I"
            op2 += "I"
    operators[op1] = fac1
    operators[op2] = fac2
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


@cache
def pauli_to_mat(pauli: str) -> np.ndarray:
    if pauli == "I":
        return np.array([[1, 0], [0, 1]], dtype=float)
    elif pauli == "Z":
        return np.array([[1, 0], [0, -1]], dtype=float)
    elif pauli == "X":
        return np.array([[0, 1], [1, 0]], dtype=float)
    elif pauli == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)


def expectation_value(
    bra: StateVector, fermiop: FermionicOperator, ket: StateVector, use_csr: int = 8
) -> float:
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError("Bra and Ket does not have same number of inactive orbitals")
    if len(bra._active) != len(ket._active):
        raise ValueError("Bra and Ket does not have same number of active orbitals")
    total = 0
    start = time.time()
    print(len(fermiop.operators.keys()))
    for op, fac in fermiop.operators.items():
        if abs(fac) < 10**-12:
            continue
        tmp = 1
        for i in range(len(bra.bra_inactive)):
            tmp *= np.matmul(bra.bra_inactive[i], np.matmul(pauli_to_mat(op[i]), ket.ket_inactive[:, i]))
        if abs(tmp) < 10**-12:
            continue
        number_active_orbitals = len(bra._active_onvector)
        active_start = len(bra.inactive)
        active_end = active_start + number_active_orbitals
        if number_active_orbitals != 0:
            if number_active_orbitals >= use_csr:
                operator = copy.copy(ket.ket_active_csr)
            else:
                operator = copy.copy(ket.ket_active)
            for pauli_mat_idx, pauli_mat_symbol in enumerate(op[active_start:active_end]):
                pauli_mat = pauli_to_mat(pauli_mat_symbol)
                prior = pauli_mat_idx
                after = number_active_orbitals - pauli_mat_idx - 1
                if pauli_mat_symbol == "I":
                    continue
                if number_active_orbitals >= use_csr:
                    operator = kronecker_product_cached(
                        prior,
                        after,
                        pauli_mat[0, 0],
                        pauli_mat[0, 1],
                        pauli_mat[1, 0],
                        pauli_mat[1, 1],
                        True,
                    ).dot(operator)
                    if operator.nnz == 0:
                        break
                else:
                    operator = np.matmul(
                        kronecker_product_cached(
                            prior,
                            after,
                            pauli_mat[0, 0],
                            pauli_mat[0, 1],
                            pauli_mat[1, 0],
                            pauli_mat[1, 1],
                            False,
                        ),
                        operator,
                    )
            if number_active_orbitals >= use_csr:
                tmp *= bra.bra_active_csr.dot(operator).toarray()[0, 0]
            else:
                tmp *= np.matmul(bra.bra_active, operator)
        total += fac * tmp
    print(f"Expectation value: {time.time() - start}")
    return total.real


@cache
def pauli_mul(pauli1: str, pauli2: str) -> tuple[float, str]:
    if pauli1 == "I":
        return 1, pauli2
    elif pauli2 == "I":
        return 1, pauli1
    elif pauli1 == pauli2:
        return 1, "I"
    elif pauli1 == "X" and pauli2 == "Y":
        return 1j, "Z"
    elif pauli1 == "X" and pauli2 == "Z":
        return -1j, "Y"
    elif pauli1 == "Y" and pauli2 == "X":
        return -1j, "Z"
    elif pauli1 == "Y" and pauli2 == "Z":
        return 1j, "X"
    elif pauli1 == "Z" and pauli2 == "X":
        return 1j, "Y"
    elif pauli1 == "Z" and pauli2 == "Y":
        return -1j, "X"


class PauliOperator:
    def __init__(self, operator: dict[str, float]) -> None:
        self.operators = operator
        self.screen_zero = True

    def __add__(self, pauliop: PauliOperator) -> PauliOperator:
        new_operators = self.operators.copy()
        for op, fac in pauliop.operators.items():
            if op in new_operators:
                new_operators[op] += fac
                if self.screen_zero:
                    if abs(new_operators[op]) < 10**-12:
                        del new_operators[op]
            else:
                new_operators[op] = fac
        return PauliOperator(new_operators)

    def __sub__(self, pauliop: PauliOperator) -> PauliOperator:
        new_operators = self.operators.copy()
        for op, fac in pauliop.operators.items():
            if op in new_operators:
                new_operators[op] -= fac
                if self.screen_zero:
                    if abs(new_operators[op]) < 10**-12:
                        del new_operators[op]
            else:
                new_operators[op] = -fac
        return PauliOperator(new_operators)

    def __mul__(self, pauliop: PauliOperator) -> PauliOperator:
        new_operators = {}
        for op1, val1 in self.operators.items():
            for op2, val2 in pauliop.operators.items():
                new_op = ""
                fac = val1 * val2
                for pauli1, pauli2 in zip(op1, op2):
                    if pauli1 == "I":
                        new_op += pauli2
                    elif pauli2 == "I":
                        new_op += pauli1
                    elif pauli1 == pauli2:
                        new_op += "I"
                    elif pauli1 == "X" and pauli2 == "Y":
                        new_op += "Z"
                        fac *= 1j
                    elif pauli1 == "X" and pauli2 == "Z":
                        new_op += "Y"
                        fac *= -1j
                    elif pauli1 == "Y" and pauli2 == "X":
                        new_op += "Z"
                        fac *= -1j
                    elif pauli1 == "Y" and pauli2 == "Z":
                        new_op += "X"
                        fac *= 1j
                    elif pauli1 == "Z" and pauli2 == "X":
                        new_op += "Y"
                        fac *= 1j
                    elif pauli1 == "Z" and pauli2 == "Y":
                        new_op += "X"
                        fac *= -1j
                if new_op in new_operators:
                    new_operators[new_op] += fac
                else:
                    new_operators[new_op] = fac
                if self.screen_zero:
                    if abs(new_operators[new_op]) < 10**-12:
                        del new_operators[new_op]
        return PauliOperator(new_operators)

    def __rmul__(self, number: float) -> PauliOperator:
        new_operators = self.operators.copy()
        for op in self.operators:
            new_operators[op] *= number
            if self.screen_zero:
                if abs(new_operators[op]) < 10**-12:
                    del new_operators[op]
        return PauliOperator(new_operators)

    @property
    def dagger(self) -> PauliOperator:
        new_operators = {}
        for op, fac in self.operators.items():
            Y_fac = op.count("Y")
            new_operators[op] = fac * (-1) ** Y_fac
        return PauliOperator(new_operators)


class FermionicOperator:
    def __init__(self, A):
        None


def Epq(p: int, q: int, num_spin_orbs: int, num_elec: int) -> FermionicOperator:
    E = PauliOperator(a_op(p, "alpha", True, num_spin_orbs, num_elec)) * PauliOperator(
        a_op(q, "alpha", False, num_spin_orbs, num_elec)
    )
    E += PauliOperator(a_op(p, "beta", True, num_spin_orbs, num_elec)) * PauliOperator(
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


def Hamiltonian_energy_only(
    h: np.ndarray, g: np.ndarray, c_mo: np.ndarray, num_inactive_spin_orbs: int, num_active_spin_orbs: int, num_active_elec: int
) -> FermionicOperator:
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_inactive_spatial_orbs = num_inactive_spin_orbs // 2
    num_inactive_elec = len(num_inactive_spin_orbs)
    num_active_spatial_orbs = num_active_spin_orbs // 2
    # Inactive one-electron
    for p in range(num_inactive_spatial_orbs):
        if p == 0:
            H_expectation = h_mo[p, p] * Epq(p, p, num_inactive_spin_orbs, num_inactive_elec)
        else:
            H_expectation += h_mo[p, p] * Epq(p, p, num_inactive_spin_orbs, num_inactive_elec)
    # Active one-electron
    for p in range(num_inactive_spatial_orbs, num_inactive_spatial_orbs + num_active_spatial_orbs):
        for q in range(num_spatial_orbs):
            H_expectation += h_mo[p, q] * Epq(p, q, num_active_spin_orbs, num_active_elec)
    # Inactive two-electron
    # 
    for p in range(num_spatial_orbs):
        for q in range(num_spatial_orbs):
            for r in range(num_spatial_orbs):
                for s in range(num_spatial_orbs):
                    H_expectation += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s, num_spin_orbs, num_elec)
    return H_expectation
