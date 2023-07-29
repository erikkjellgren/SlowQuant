from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.unitary_coupled_cluster.base import StateVector
from slowquant.unitary_coupled_cluster.operator_hybrid import OperatorHybrid


def pauli_mul(pauli1: str, pauli2: str) -> tuple[str, complex]:
    """Multiplication of Pauli operators.

    Args:
        pauli1: Pauli operator.
        pauli2: Pauli operator.

    Returns:
        New Pauli operator and factor.
    """
    new_pauli = ''
    fac: complex = 1
    if pauli1 == 'I':
        new_pauli = pauli2
    elif pauli2 == 'I':
        new_pauli = pauli1
    elif pauli1 == pauli2:
        new_pauli = 'I'
    elif pauli1 == 'X' and pauli2 == 'Y':
        new_pauli = 'Z'
        fac *= 1j
    elif pauli1 == 'X' and pauli2 == 'Z':
        new_pauli = 'Y'
        fac *= -1j
    elif pauli1 == 'Y' and pauli2 == 'X':
        new_pauli = 'Z'
        fac *= -1j
    elif pauli1 == 'Y' and pauli2 == 'Z':
        new_pauli = 'X'
        fac *= 1j
    elif pauli1 == 'Z' and pauli2 == 'X':
        new_pauli = 'Y'
        fac *= 1j
    elif pauli1 == 'Z' and pauli2 == 'Y':
        new_pauli = 'X'
        fac *= -1j
    return new_pauli, fac


def paulistring_mul(paulis: Sequence[str]) -> tuple[str, complex]:
    """Multiplication of all Pauli operators in Pauli-string.

    Args:
        Paulis: Pauli-string.

    Returns:
        New Pauli operator and factor.
    """
    fac: complex = 1
    current_pauli = paulis[0]
    for pauli in paulis[1:]:
        new_pauli, new_fac = pauli_mul(current_pauli, pauli)
        current_pauli = new_pauli
        fac *= new_fac
    return current_pauli, fac


class OperatorContracted:
    def __init__(self, operator: np.ndarray | ss.csr_matrix) -> None:
        """Initialize active-space contracted operator.

        Args:
            operator: Operator in matrix form.
        """
        self.operators = operator

    def __add__(self, contracted_op: OperatorContracted) -> OperatorContracted:
        """Overload addition operator.

        Args:
            contracted_op: Contracted operator.

        Returns:
            New contracted operator.
        """
        return OperatorContracted(self.operators + contracted_op.operators)

    def __sub__(self, contracted_op: OperatorContracted) -> OperatorContracted:
        """Overload subtraction operator.

        Args:
            contracted_op: Contracted operator.

        Returns:
            New contracted operator.
        """
        return OperatorContracted(self.operators - contracted_op.operators)

    def __mul__(self, contracted_op: OperatorContracted) -> None:
        """Overload multiplication operator.

        Args:
            contracted_op: Contracted operator.
        """
        raise ArithmeticError('Cannot mulitply operators with inactive and virtual orbitals contracted.')

    def __rmul__(self, number: float) -> OperatorContracted:
        """Overload right multiplication operator.

        Args:
            number: Scalar value.

        Returns:
            New contracted operator.
        """
        return OperatorContracted(number * self.operators)

    @property
    def dagger(self) -> OperatorContracted:
        """Complex conjugate of operator.

        Returns:
            Return complex conjugated operator.
        """
        return OperatorContracted(self.operators.conj().transpose())

    def apply_u_from_right(self, U: np.ndarray | ss.csr_matrix) -> OperatorContracted:
        """Multiply with transformation matrix from the right side.

        Args:
            U: Transformation matrix.

        Returns:
            Transformed operator.
        """
        return OperatorContracted(lw.matmul(self.operators, U))

    def apply_u_from_left(self, U: np.ndarray | ss.csr_matrix) -> OperatorContracted:
        """Multiply with transformation matrix from the left side.

        Args:
            U: Transformation matrix.

        Returns:
            Transformed operator.
        """
        return OperatorContracted(lw.matmul(U, self.operators))


def operatormul_contract(A: OperatorHybrid, B: OperatorHybrid) -> OperatorContracted:
    """Multiply two operators and contract the inactive and virtual orbitals.

    Args:
        A: Uncontracted operator.
        B: Uncontracted operator.

    Returns:
        Contracted operator.
    """
    key = list(A.operators.keys())[0]
    if not isinstance(A.operators[key].active_matrix, (np.ndarray, ss.csr_matrix, ss.csc_matrix)):
        raise TypeError(f'Unknown type: {type(A.operators[key].active_matrix)}')
    new_operators = lw.zeros_like(A.operators[key].active_matrix)
    for _, op1 in A.operators.items():
        for _, op2 in B.operators.items():
            new_inactive = ''
            new_virtual = ''
            fac: complex = 1
            is_zero = False
            for pauli1, pauli2 in zip(op1.inactive_pauli, op2.inactive_pauli):
                new_pauli, new_fac = paulistring_mul((pauli1, pauli2))
                if new_pauli in ('Y', 'X'):
                    is_zero = True
                    break
                new_inactive += new_pauli
                fac *= new_fac
            if is_zero:
                continue
            for pauli1, pauli2 in zip(op1.virtual_pauli, op2.virtual_pauli):
                new_pauli, new_fac = paulistring_mul((pauli1, pauli2))
                if new_pauli in ('Y', 'X'):
                    is_zero = True
                    break
                new_virtual += new_pauli
                fac *= new_fac
            if is_zero:
                continue
            # This should depend on state vector.
            # I.e. (-1)**(#Z_occupied)
            fac *= (-1) ** (new_inactive.count('Z'))
            new_operators += fac * lw.matmul(op1.active_matrix, op2.active_matrix)
    return OperatorContracted(new_operators)


def operatormul3_contract(A: OperatorHybrid, B: OperatorHybrid, C: OperatorHybrid) -> OperatorContracted:
    """Multiply three operators and contract the inactive and virtual orbitals.

    Args:
        A: Uncontracted operator.
        B: Uncontracted operator.
        C: Uncontracted operator.

    Returns:
        Contracted operator.
    """
    key = list(A.operators.keys())[0]
    if not isinstance(A.operators[key].active_matrix, (np.ndarray, ss.csr_matrix, ss.csc_matrix)):
        raise TypeError(f'Unknown type: {type(A.operators[key].active_matrix)}')
    new_operators = lw.zeros_like(A.operators[key].active_matrix)
    for _, op1 in A.operators.items():
        for _, op2 in B.operators.items():
            for _, op3 in C.operators.items():
                new_inactive = ''
                new_virtual = ''
                fac: complex = 1
                is_zero = False
                for pauli1, pauli2, pauli3 in zip(op1.inactive_pauli, op2.inactive_pauli, op3.inactive_pauli):
                    new_pauli, new_fac = paulistring_mul((pauli1, pauli2, pauli3))
                    if new_pauli in ('X', 'Y'):
                        is_zero = True
                        break
                    new_inactive += new_pauli
                    fac *= new_fac
                if is_zero:
                    continue
                for pauli1, pauli2, pauli3 in zip(op1.virtual_pauli, op2.virtual_pauli, op3.virtual_pauli):
                    new_pauli, new_fac = paulistring_mul((pauli1, pauli2, pauli3))
                    if new_pauli in ('X', 'Y'):
                        is_zero = True
                        break
                    new_virtual += new_pauli
                    fac *= new_fac
                if is_zero:
                    continue
                # This should depend on state vector.
                # I.e. (-1)**(#Z_occupied)
                fac *= (-1) ** (new_inactive.count('Z'))
                new_operators += fac * lw.matmul(
                    op1.active_matrix, lw.matmul(op2.active_matrix, op3.active_matrix)
                )
    return OperatorContracted(new_operators)


def operatormul4_contract(
    A: OperatorHybrid,
    B: OperatorHybrid,
    C: OperatorHybrid,
    D: OperatorHybrid,
) -> OperatorContracted:
    """Multiply four operators and contract the inactive and virtual orbitals.

    Args:
        A: Uncontracted operator.
        B: Uncontracted operator.
        C: Uncontracted operator.
        D: Uncontracted operator.

    Returns:
        Contracted operator.
    """
    key = list(A.operators.keys())[0]
    if not isinstance(A.operators[key].active_matrix, (np.ndarray, ss.csr_matrix, ss.csc_matrix)):
        raise TypeError(f'Unknown type: {type(A.operators[key].active_matrix)}')
    new_operators = lw.zeros_like(A.operators[key].active_matrix)
    for _, op1 in A.operators.items():
        for _, op2 in B.operators.items():
            for _, op3 in C.operators.items():
                for _, op4 in D.operators.items():
                    new_inactive = ''
                    new_virtual = ''
                    fac: complex = 1
                    is_zero = False
                    for pauli1, pauli2, pauli3, pauli4 in zip(
                        op1.inactive_pauli, op2.inactive_pauli, op3.inactive_pauli, op4.inactive_pauli
                    ):
                        new_pauli, new_fac = paulistring_mul((pauli1, pauli2, pauli3, pauli4))
                        if new_pauli in ('X', 'Y'):
                            is_zero = True
                            break
                        new_inactive += new_pauli
                        fac *= new_fac
                    if is_zero:
                        continue
                    for pauli1, pauli2, pauli3, pauli4 in zip(
                        op1.virtual_pauli, op2.virtual_pauli, op3.virtual_pauli, op4.virtual_pauli
                    ):
                        new_pauli, new_fac = paulistring_mul((pauli1, pauli2, pauli3, pauli4))
                        if new_pauli in ('X', 'Y'):
                            is_zero = True
                            break
                        new_virtual += new_pauli
                        fac *= new_fac
                    if is_zero:
                        continue
                    # This should depend on state vector.
                    # I.e. (-1)**(#Z_occupied)
                    fac *= (-1) ** (new_inactive.count('Z'))
                    new_operators += fac * lw.matmul(
                        op1.active_matrix,
                        lw.matmul(op2.active_matrix, lw.matmul(op3.active_matrix, op4.active_matrix)),
                    )
    return OperatorContracted(new_operators)


def double_commutator_contract(A: OperatorHybrid, B: OperatorHybrid, C: OperatorHybrid) -> OperatorContracted:
    """Make double commutator and contract the inactive and virtual orbitals.

    Double commutator of the form [A, [B, C]] = ABC - ACB - BCA + CBA

    Args:
        A: Uncontracted operator.
        B: Uncontracted operator.
        C: Uncontracted operator.

    Returns:
        Contracted operator.
    """
    return (
        operatormul3_contract(A, B, C)  # pylint: disable=W1114
        - operatormul3_contract(A, C, B)  # pylint: disable=W1114
        - operatormul3_contract(B, C, A)  # pylint: disable=W1114
        + operatormul3_contract(C, B, A)  # pylint: disable=W1114
    )


def commutator_contract(A: OperatorHybrid, B: OperatorHybrid) -> OperatorContracted:
    """Make commutator and contract the inactive and virtual orbitals.

    Args:
        A: Uncontracted operator.
        B: Uncontracted operator.

    Returns:
        Contracted operator.
    """
    return operatormul_contract(A, B) - operatormul_contract(B, A)


def expectation_value_contracted(
    bra: StateVector, contracted_op: OperatorContracted, ket: StateVector
) -> float:
    """Calculate expectation value of contracted operator.

    Args:
        bra: Bra state-vector.
        contracted_op: Contracted operator.
        ket: Ket state-vector.

    Returns:
        Expectation value of contracted operator.
    """
    if isinstance(contracted_op.operators, np.ndarray):
        return lw.matmul(bra.bra_active, lw.matmul(contracted_op.operators, ket.ket_active)).real
    if isinstance(contracted_op.operators, ss.csr_matrix):
        return lw.matmul(
            bra.bra_active_csr, lw.matmul(contracted_op.operators, ket.ket_active_csr)
        ).real.toarray()[0, 0]
    raise TypeError(f'Unknown type: {type(contracted_op.operators)}')
