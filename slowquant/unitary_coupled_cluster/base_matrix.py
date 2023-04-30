from __future__ import annotations

import copy
import functools
import itertools

import numpy as np
import scipy.sparse as ss

from slowquant.unitary_coupled_cluster.base import PauliOperator

def convert_pauli_to_hybrid_form(pauliop, num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int) -> PauliOperatorHybridForm:
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
            new_operators[key] = OperatorHybridForm(new_inactive, new_active_matrix, new_virtual)
    return PauliOperatorHybridForm(new_operators) 

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
        self.opetarors = operator

    def __add__(self, hybridop: PauliOperatorHybridForm) -> PauliOperatorHybridForm:
        new_operators = copy.deepcopy(operators)
        for key, op in hybdirdop.operators.items():
            if key in new_operators:
                new_operators.active_matrix += op.active_matrix
            else:
                new_operators[key] = OperatorHybridForm(op.inactive_pauli, op.active_matrix, op.virtual_pauli)
        return PauliOperatorHybridForm(new_operators)

    def __sub__(self, hybridop: PauliOperatorHybridForm) -> PauliOperatorHybridForm:
        new_operators = copy.deepcopy(operators)
        for key, op in hybdirdop.operators.items():
            if key in new_operators:
                new_operators.active_matrix -= op.active_matrix
            else:
                new_operators[key] = OperatorHybridForm(op.inactive_pauli, -op.active_matrix, op.virtual_pauli)
        return PauliOperatorHybridForm(new_operators)

    def __mul__(self, pauliop: PauliOperator) -> PauliOperator:
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

    def __rmul__(self, number: float) -> PauliOperator:
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix *= number
        return PauliOperatorHybridForm(new_operators)
