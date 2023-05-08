from __future__ import annotations

import copy
import functools
import itertools

import numpy as np
import scipy.sparse as ss

from slowquant.unitary_coupled_cluster.base import PauliOperator, pauli_to_mat

def expectation_value_hybrid2(bra: StateVector, hybridop: PauliOperatorHybridForm, ket: StateVector, use_csr: int = 12) -> float:
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError("Bra and Ket does not have same number of inactive orbitals")
    if len(bra._active) != len(ket._active):
        raise ValueError("Bra and Ket does not have same number of active orbitals")
    total = 0
    for _, op in hybridop.operators.items():
        tmp = 1
        for i in range(len(bra.bra_inactive)):
            tmp *= np.matmul(bra.bra_inactive[i], np.matmul(pauli_to_mat(op.inactive_pauli[i]), ket.ket_inactive[:, i]))
        for i in range(len(bra.bra_virtual)):
            tmp *= np.matmul(bra.bra_virtual[i], np.matmul(pauli_to_mat(op.virtual_pauli[i]), ket.ket_virtual[:, i]))
        if abs(tmp) < 10**-12:
            continue
        number_active_orbitals = len(bra._active_onvector)
        if number_active_orbitals != 0:
            if number_active_orbitals >= use_csr:
                operator = copy.deepcopy(ket.ket_active_csr)
                operator = op.active_matrix.dot(operator)
                tmp *= bra.bra_active_csr.dot(operator).toarray()[0, 0]
            else:
                operator = copy.deepcopy(ket.ket_active)
                operator = np.matmul(op.active_matrix, operator)
                tmp *= np.matmul(bra.bra_active, operator)
        total += tmp
    if abs(total.imag) > 10**-10:
        print(f"WARNING, imaginary value of {total.imag}")
    return total.real

def expectation_value_hybrid(bra: StateVector, hybridop: PauliOperatorHybridForm, ket: StateVector, use_csr: int = 12) -> float:
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError("Bra and Ket does not have same number of inactive orbitals")
    if len(bra._active) != len(ket._active):
        raise ValueError("Bra and Ket does not have same number of active orbitals")
    total = 0
    for _, op in hybridop.operators.items():
        tmp = 1
        for i in range(len(bra.bra_inactive)):
            tmp *= np.matmul(bra.bra_inactive[i], np.matmul(pauli_to_mat(op.inactive_pauli[i]), ket.ket_inactive[:, i]))
        for i in range(len(bra.bra_virtual)):
            tmp *= np.matmul(bra.bra_virtual[i], np.matmul(pauli_to_mat(op.virtual_pauli[i]), ket.ket_virtual[:, i]))
        if abs(tmp) < 10**-12:
            continue
        number_active_orbitals = len(bra._active_onvector)
        if number_active_orbitals != 0:
            if number_active_orbitals >= use_csr:
                operator = copy.deepcopy(ket.ket_active_csr)
                operator = op.active_matrix.dot(operator)
                tmp *= bra.bra_active_csr.dot(operator).toarray()[0, 0]
            else:
                operator = copy.deepcopy(ket.ket_active)
                operator = np.matmul(op.active_matrix, operator)
                tmp *= np.matmul(bra.bra_active, operator)
        total += tmp
    if abs(total.imag) > 10**-10:
        print(f"WARNING, imaginary value of {total.imag}")
    return total.real

def convert_pauli_to_hybrid_form(pauliop: PauliOperator, num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int) -> PauliOperatorHybridForm:
    new_operator = {}
    active_start = num_inactive_orbs
    active_end = num_inactive_orbs+num_active_orbs
    for pauli_string, factor in pauliop.operators.items():
        new_inactive = pauli_string[:active_start]
        new_active = pauli_string[active_start:active_end]
        new_virtual = pauli_string[active_end:]
        active_pauli = PauliOperator({new_active: 1})
        new_active_matrix = factor * active_pauli.matrix_form()
        key = new_inactive + new_virtual
        if key in new_operator:
            new_operator[key].active_matrix += new_active_matrix
        else:
            new_operator[key] = OperatorHybridForm(new_inactive, new_active_matrix, new_virtual)
    return PauliOperatorHybridForm(new_operator) 

class OperatorHybridForm:
    def __init__(self, inactive_pauli: str, active_matrix: np.ndarray | ss.csr_matrix, virtual_pauli: str) -> None:
        self.inactive_pauli = inactive_pauli
        self.active_matrix = active_matrix
        self.virtual_pauli = virtual_pauli

class PauliOperatorHybridForm:
    def __init__(self, operator: dict[str, OperatorHybridForm]) -> None:
        """The key is the Pauli-string of inactive + virtual,
        i.e. the active part does not contribute to the key.
        """
        self.operators = operator

    def __add__(self, hybridop: PauliOperatorHybridForm) -> PauliOperatorHybridForm:
        new_operators = copy.deepcopy(self.operators)
        for key, op in hybridop.operators.items():
            if key in new_operators:
                new_operators[key].active_matrix += op.active_matrix
            else:
                new_operators[key] = OperatorHybridForm(op.inactive_pauli, op.active_matrix, op.virtual_pauli)
        return PauliOperatorHybridForm(new_operators)

    def __sub__(self, hybridop: PauliOperatorHybridForm) -> PauliOperatorHybridForm:
        new_operators = copy.deepcopy(self.operators)
        for key, op in hybridop.operators.items():
            if key in new_operators:
                new_operators[key].active_matrix -= op.active_matrix
            else:
                new_operators[key] = OperatorHybridForm(op.inactive_pauli, -op.active_matrix, op.virtual_pauli)
        return PauliOperatorHybridForm(new_operators)

    def __mul__(self, pauliop: PauliOperatorHybridForm) -> PauliOperatorHybridForm:
        new_operators = {}
        for _, op1 in self.operators.items():
            for _, op2 in pauliop.operators.items():
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
                new_active = fac*np.matmul(op1.active_matrix, op2.active_matrix)
                key = new_inactive+new_virtual
                if key in new_operators:
                    new_operators[key].active_matrix += new_active
                else:
                    new_operators[key] = OperatorHybridForm(new_inactive, new_active, new_virtual)
        return PauliOperatorHybridForm(new_operators)

    def __rmul__(self, number: float) -> PauliOperatorHybridForm:
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix *= number
        return PauliOperatorHybridForm(new_operators)

    @property
    def dagger(self) -> PauliOperatorHybridForm: 
        new_operators = {}
        for key, op in self.operators.items():
            new_operators[key] = OperatorHybridForm(op.inactive_pauli, np.conj(op.active_matrix).transpose(), op.virtual_pauli)
        return PauliOperatorHybridForm(new_operators)

    def apply_U_from_right(self, U: np.ndarray | ss.csr_matrix) -> PauliOperatorHybridForm:
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix = np.matmul(new_operators[key].active_matrix, U)
        return PauliOperatorHybridForm(new_operators)

    def apply_U_from_left(self, U: np.ndarray | ss.csr_matrix) -> PauliOperatorHybridForm:
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix = np.matmul(U, new_operators[key].active_matrix)
        return PauliOperatorHybridForm(new_operators)
