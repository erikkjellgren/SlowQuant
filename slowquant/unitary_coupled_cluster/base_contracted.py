from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.base import PauliOperator, StateVector, a_op
from slowquant.unitary_coupled_cluster.base_matrix import (
    PauliOperatorHybridForm,
    convert_pauli_to_hybrid_form,
)


def pauli_mul(pauli1: str, pauli2: str) -> tuple[str, complex]:
    new_pauli = ""
    fac: complex = 1
    if pauli1 == "I":
        new_pauli = pauli2
    elif pauli2 == "I":
        new_pauli = pauli1
    elif pauli1 == pauli2:
        new_pauli = "I"
    elif pauli1 == "X" and pauli2 == "Y":
        new_pauli = "Z"
        fac *= 1j
    elif pauli1 == "X" and pauli2 == "Z":
        new_pauli = "Y"
        fac *= -1j
    elif pauli1 == "Y" and pauli2 == "X":
        new_pauli = "Z"
        fac *= -1j
    elif pauli1 == "Y" and pauli2 == "Z":
        new_pauli = "X"
        fac *= 1j
    elif pauli1 == "Z" and pauli2 == "X":
        new_pauli = "Y"
        fac *= 1j
    elif pauli1 == "Z" and pauli2 == "Y":
        new_pauli = "X"
        fac *= -1j
    return new_pauli, fac


def paulistring_mul(paulis: Sequence[str]) -> tuple[str, complex]:
    fac: complex = 1
    current_pauli = paulis[0]
    for pauli in paulis[1:]:
        new_pauli, new_fac = pauli_mul(current_pauli, pauli)
        current_pauli = new_pauli
        fac *= new_fac
    return current_pauli, fac


class PauliOperatorContracted:
    def __init__(self, operator: np.ndarray | ss.csr_matrix) -> None:
        self.operators = operator

    def __add__(self, hybridop: PauliOperatorContracted) -> PauliOperatorContracted:
        return PauliOperatorContracted(self.operators + hybridop.operators)

    def __sub__(self, hybridop: PauliOperatorContracted) -> PauliOperatorContracted:
        return PauliOperatorContracted(self.operators - hybridop.operators)

    def __mul__(self, pauliop: PauliOperatorContracted) -> None:
        raise ArithmeticError("Cannot mulitply operators with inactive and virtual orbitals contracted")

    def __rmul__(self, number: float) -> PauliOperatorContracted:
        return PauliOperatorContracted(number * self.operators)

    @property
    def dagger(self) -> PauliOperatorContracted:
        return PauliOperatorContracted(self.operators.conj().transpose())

    def apply_U_from_right(self, U: np.ndarray | ss.csr_matrix) -> PauliOperatorContracted:
        return PauliOperatorContracted(lw.matmul(self.operators, U))

    def apply_U_from_left(self, U: np.ndarray | ss.csr_matrix) -> PauliOperatorContracted:
        return PauliOperatorContracted(lw.matmul(U, self.operators))


def operatormul_contract(A: PauliOperatorHybridForm, B: PauliOperatorHybridForm) -> PauliOperatorContracted:
    for _, op1 in A.operators.items():
        if isinstance(op1.active_matrix, np.ndarray):
            new_operators = np.zeros_like(op1.active_matrix)
        elif isinstance(op1.active_matrix, ss.csr_matrix) or isinstance(op1.active_matrix, ss.csc_matrix):
            new_operators = ss.csr_array(op1.active_matrix.shape)
        else:
            raise TypeError(f"Unknown type: {type(op1.active_matrix)}")
        break
    for _, op1 in A.operators.items():
        for _, op2 in B.operators.items():
            new_inactive = ""
            new_virtual = ""
            fac: complex = 1
            do_continue = False
            for pauli1, pauli2 in zip(op1.inactive_pauli, op2.inactive_pauli):
                new_pauli, new_fac = paulistring_mul((pauli1, pauli2))
                if new_pauli == "Y" or new_pauli == "X":
                    do_continue = True
                    break
                new_inactive += new_pauli
                fac *= new_fac
            if do_continue:
                continue
            for pauli1, pauli2 in zip(op1.virtual_pauli, op2.virtual_pauli):
                new_pauli, new_fac = paulistring_mul((pauli1, pauli2))
                if new_pauli == "Y" or new_pauli == "X":
                    do_continue = True
                    break
                new_virtual += new_pauli
                fac *= new_fac
            if do_continue:
                continue
            # This should depend on state vector.
            # I.e. (-1)**(#Z_occupied)
            fac *= (-1) ** (new_inactive.count("Z"))
            new_active = fac * lw.matmul(op1.active_matrix, op2.active_matrix)
            new_operators += new_active
    return PauliOperatorContracted(new_operators)


def operatormul3_contract(
    A: PauliOperatorHybridForm, B: PauliOperatorHybridForm, C: PauliOperatorHybridForm
) -> PauliOperatorContracted:
    for _, op1 in A.operators.items():
        if isinstance(op1.active_matrix, np.ndarray):
            new_operators = np.zeros_like(op1.active_matrix)
        elif isinstance(op1.active_matrix, ss.csr_matrix) or isinstance(op1.active_matrix, ss.csc_matrix):
            new_operators = ss.csr_array(op1.active_matrix.shape)
        else:
            raise TypeError(f"Unknown type: {type(op1.active_matrix)}")
        break
    for _, op1 in A.operators.items():
        for _, op2 in B.operators.items():
            for _, op3 in C.operators.items():
                new_inactive = ""
                new_virtual = ""
                fac: complex = 1
                do_continue = False
                for pauli1, pauli2, pauli3 in zip(op1.inactive_pauli, op2.inactive_pauli, op3.inactive_pauli):
                    new_pauli, new_fac = paulistring_mul((pauli1, pauli2, pauli3))
                    if new_pauli == "Y" or new_pauli == "X":
                        do_continue = True
                        break
                    new_inactive += new_pauli
                    fac *= new_fac
                if do_continue:
                    continue
                for pauli1, pauli2, pauli3 in zip(op1.virtual_pauli, op2.virtual_pauli, op3.virtual_pauli):
                    new_pauli, new_fac = paulistring_mul((pauli1, pauli2, pauli3))
                    if new_pauli == "Y" or new_pauli == "X":
                        do_continue = True
                        break
                    new_virtual += new_pauli
                    fac *= new_fac
                if do_continue:
                    continue
                # This should depend on state vector.
                # I.e. (-1)**(#Z_occupied)
                fac *= (-1) ** (new_inactive.count("Z"))
                new_active = fac * lw.matmul(
                    op1.active_matrix, lw.matmul(op2.active_matrix, op3.active_matrix)
                )
                new_operators += new_active
    return PauliOperatorContracted(new_operators)


def operatormul4_contract(
    A: PauliOperatorHybridForm,
    B: PauliOperatorHybridForm,
    C: PauliOperatorHybridForm,
    D: PauliOperatorHybridForm,
) -> PauliOperatorContracted:
    for _, op1 in A.operators.items():
        if isinstance(op1.active_matrix, np.ndarray):
            new_operators = np.zeros_like(op1.active_matrix)
        elif isinstance(op1.active_matrix, ss.csr_matrix):
            new_operators = ss.csr_array(op1.active_matrix.shape)
        else:
            raise TypeError(f"Unknown type: {type(op1.active_matrix)}")
        break
    for _, op1 in A.operators.items():
        for _, op2 in B.operators.items():
            for _, op3 in C.operators.items():
                for _, op4 in D.operators.items():
                    new_inactive = ""
                    new_virtual = ""
                    fac: complex = 1
                    do_continue = False
                    for pauli1, pauli2, pauli3, pauli4 in zip(
                        op1.inactive_pauli, op2.inactive_pauli, op3.inactive_pauli, op4.inactive_pauli
                    ):
                        new_pauli, new_fac = paulistring_mul((pauli1, pauli2, pauli3, pauli4))
                        if new_pauli == "Y" or new_pauli == "X":
                            do_continue = True
                            break
                        new_inactive += new_pauli
                        fac *= new_fac
                    if do_continue:
                        continue
                    for pauli1, pauli2, pauli3, pauli4 in zip(
                        op1.virtual_pauli, op2.virtual_pauli, op3.virtual_pauli, op4.virtual_pauli
                    ):
                        new_pauli, new_fac = paulistring_mul((pauli1, pauli2, pauli3, pauli4))
                        if new_pauli == "Y" or new_pauli == "X":
                            do_continue = True
                            break
                        new_virtual += new_pauli
                        fac *= new_fac
                    if do_continue:
                        continue
                    # This should depend on state vector.
                    # I.e. (-1)**(#Z_occupied)
                    fac *= (-1) ** (new_inactive.count("Z"))
                    new_active = fac * lw.matmul(
                        op1.active_matrix,
                        lw.matmul(op2.active_matrix, lw.matmul(op3.active_matrix, op4.active_matrix)),
                    )
                    new_operators += new_active
    return PauliOperatorContracted(new_operators)


def double_commutator_contract(
    A: PauliOperatorHybridForm, B: PauliOperatorHybridForm, C: PauliOperatorHybridForm
) -> PauliOperatorContracted:
    """Double commutator of the form [A, [B, C]] = ABC - ACB - BCA + CBA"""
    return (
        operatormul3_contract(A, B, C)
        - operatormul3_contract(A, C, B)
        - operatormul3_contract(B, C, A)
        + operatormul3_contract(C, B, A)
    )


def commutator_contract(A: PauliOperatorHybridForm, B: PauliOperatorHybridForm) -> PauliOperatorContracted:
    return operatormul_contract(A, B) - operatormul_contract(B, A)


def expectation_value_contracted(
    bra: StateVector, contracted_op: PauliOperatorContracted, ket: StateVector
) -> float:
    if isinstance(contracted_op.operators, np.ndarray):
        return lw.matmul(bra.bra_active, lw.matmul(contracted_op.operators, ket.ket_active)).real
    elif isinstance(contracted_op.operators, ss.csr_matrix):
        return lw.matmul(
            bra.bra_active_csr, lw.matmul(contracted_op.operators, ket.ket_active_csr)
        ).real.toarray()[0, 0]
    else:
        raise TypeError(f"Unknown type: {type(contracted_op.operators)}")
