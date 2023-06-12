from __future__ import annotations

import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw


class PauliOperatorContracted:
    def __init__(self, operator: np.ndarray | ss.csr_matrix) -> None:
        self.operators = operator

    def __add__(self, hybridop: PauliOperatorContracted) -> PauliOperatorContracted:
        return PauliOperatorContracted(self.operators + hybridop.operators)

    def __sub__(self, hybridop: PauliOperatorContracted) -> PauliOperatorContracted:
        return PauliOperatorContracted(self.operators - hybridop.operators)

    def __mul__(self, pauliop: PauliOperatorContracted) -> None:
        print("Cannot multiply space contracted operators")
        exit()

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
        else:
            print(f"Unknown type: {type(op1.active_matrix)}")
            exit()
        break
    for _, op1 in A.operators.items():
        for _, op2 in B.operators.items():
            new_inactive = ""
            new_virtual = ""
            fac = 1
            for pauli1, pauli2 in zip(op1.inactive_pauli, op2.inactive_pauli):
                if pauli1 == "I":
                    new_inactive += pauli2
                elif pauli2 == "I":
                    new_inactive += pauli1
                elif pauli1 == pauli2:
                    new_inactive += "I"
                elif pauli1 == "X" and pauli2 == "Y":
                    new_inactive += "Z"
                    fac *= 1j
                elif pauli1 == "X" and pauli2 == "Z":
                    new_inactive += "Y"
                    fac *= -1j
                elif pauli1 == "Y" and pauli2 == "X":
                    new_inactive += "Z"
                    fac *= -1j
                elif pauli1 == "Y" and pauli2 == "Z":
                    new_inactive += "X"
                    fac *= 1j
                elif pauli1 == "Z" and pauli2 == "X":
                    new_inactive += "Y"
                    fac *= 1j
                elif pauli1 == "Z" and pauli2 == "Y":
                    new_inactive += "X"
                    fac *= -1j
            if "Y" in new_inactive or "X" in new_inactive:
                continue
            for pauli1, pauli2 in zip(op1.virtual_pauli, op2.virtual_pauli):
                if pauli1 == "I":
                    new_virtual += pauli2
                elif pauli2 == "I":
                    new_virtual += pauli1
                elif pauli1 == pauli2:
                    new_virtual += "I"
                elif pauli1 == "X" and pauli2 == "Y":
                    new_virtual += "Z"
                    fac *= 1j
                elif pauli1 == "X" and pauli2 == "Z":
                    new_virtual += "Y"
                    fac *= -1j
                elif pauli1 == "Y" and pauli2 == "X":
                    new_virtual += "Z"
                    fac *= -1j
                elif pauli1 == "Y" and pauli2 == "Z":
                    new_virtual += "X"
                    fac *= 1j
                elif pauli1 == "Z" and pauli2 == "X":
                    new_virtual += "Y"
                    fac *= 1j
                elif pauli1 == "Z" and pauli2 == "Y":
                    new_virtual += "X"
                    fac *= -1j
            if "Y" in new_virtual or "X" in new_virtual:
                continue
            # This should depend on state vector.
            # I.e. (-1)**(#Z_occupied)
            fac *= (-1) ** (new_inactive.count("Z"))
            new_active = fac * np.matmul(op1.active_matrix, op2.active_matrix)
            new_operators += new_active
    return PauliOperatorContracted(new_operators)


def commutator_contract(A: PauliOperatorHybridForm, B: PauliOperatorHybridForm) -> PauliOperatorContracted:
    return operatormul_contract(A, B) - operatormul_contract(B, A)


def expectation_value_contracted(
    bra: StateVector, contracted_op: PauliOperatorContracted, ket: StateVector
) -> float:
    return lw.matmul(bra.bra_active, lw.matmul(contracted_op.operators, ket.ket_active)).real
