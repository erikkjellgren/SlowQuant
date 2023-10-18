from __future__ import annotations

import copy
import itertools

import numpy as np
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.unitary_coupled_cluster.base import StateVector, pauli_to_mat
from slowquant.unitary_coupled_cluster.operator_pauli import OperatorPauli

def expectation_value_hybrid(
    bra: StateVector, hybridop: OperatorHybrid, ket: StateVector, use_csr: int = 10
) -> float:
    """Calculate expectation value of hybrid operator.

    Args:
        bra: Bra state-vector.
        hybridop: Hybrid operator.
        ket: Ket state-vector.
        use_csr: Size when to use sparse matrices.

    Returns:
        Expectation value of hybrid operator.
    """
    if len(bra.inactive) != len(ket.inactive):
        raise ValueError('Bra and Ket does not have same number of inactive orbitals')
    if len(bra._active) != len(ket._active):
        raise ValueError('Bra and Ket does not have same number of active orbitals')
    total = 0
    for _, op in hybridop.operators.items():
        tmp = 1
        for i in range(len(bra.bra_inactive)):
            tmp *= np.matmul(
                bra.bra_inactive[i], np.matmul(pauli_to_mat(op.inactive_pauli[i]), ket.ket_inactive[:, i])  # type: ignore
            )
        for i in range(len(bra.bra_virtual)):
            tmp *= np.matmul(
                bra.bra_virtual[i], np.matmul(pauli_to_mat(op.virtual_pauli[i]), ket.ket_virtual[:, i])  # type: ignore
            )
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
        print(f'WARNING, imaginary value of {total.imag}')
    return total.real

class StateVectorOperatorData:
    def __init__(self, inactive_orbs: str, active_space: np.ndarray, virtual_orbs: str) -> None:
        self.inactive_orbs = inactive_orbs
        self.virtual_orbs = virtual_orbs
        self.active_space = active_space


class StateVectorOperator:
    def __init__(self, state_vector: dict[str, StateVectorOperatorData]) -> None:
        self.state_vector = state_vector

    def __mul__(self, hybridop: OperatorHybrid) -> StateVectorOperator:
        if isinstance(hybridop, OperatorHybrid):
            new_state_vector = {}
            for _, vec in self.state_vector.items():
                for _, op in hybridop.operators.items():
                    new_inactive = ''
                    new_virtual = ''
                    fac: complex = 1
                    for pauli, orb in zip(op.inactive_pauli, vec.inactive_orbs):
                        if pauli == 'I':
                            new_inactive += orb
                        elif orb == 'o' and pauli == 'X':
                            new_inactive += "z"
                        elif orb == 'o' and pauli == 'Y':
                            new_inactive += "z"
                            fac *= 1j
                        elif orb == 'o' and pauli == 'Z':
                            new_inactive += "o"
                            fac *= -1
                        elif orb == 'z' and pauli == 'X':
                            new_inactive += "o"
                        elif orb == 'z' and pauli == 'Y':
                            new_inactive += "o"
                            fac *= -1j
                        elif orb == 'z' and pauli == 'Z':
                            new_inactive += "z"
                    for pauli, orb in zip(op.virtual_pauli, vec.virtual_orbs):
                        if pauli == 'I':
                            new_virtual += orb
                        elif orb == 'o' and pauli == 'X':
                            new_virtual += "z"
                        elif orb == 'o' and pauli == 'Y':
                            new_virtual += "z"
                            fac *= 1j
                        elif orb == 'o' and pauli == 'Z':
                            new_virtual += "o"
                            fac *= -1
                        elif orb == 'z' and pauli == 'X':
                            new_virtual += "o"
                        elif orb == 'z' and pauli == 'Y':
                            new_virtual += "o"
                            fac *= -1j
                        elif orb == 'z' and pauli == 'Z':
                            new_virtual += "z"
                    new_active = fac * lw.matmul(vec.active_space, op.active_matrix)
                    key = new_inactive + new_virtual
                    if key in new_state_vector:
                        new_state_vector[key].active_space += new_active
                    else:
                        new_state_vector[key] = StateVectorOperatorData(new_inactive, new_active, new_virtual)
            return StateVectorOperator(new_state_vector)
        overlap = 0
        for _, vec1 in self.state_vector.items():
            for _, vec2 in hybridop.state_vector.items():
                if vec1.inactive_orbs != vec2.inactive_orbs:
                    continue
                if vec1.virtual_orbs != vec2.virtual_orbs:
                    continue
                for val1, val2 in zip(vec1.active_space, vec2.active_space):
                    overlap += val1*val2
        return np.real(overlap)

def expectation_value_hybrid_flow(state_vec: StateVector, operators: list[OperatorHybrid], ref_vec: StateVector) -> float:
    if len(state_vec.inactive) != 0:
        num_inactive_spin_orbs = len(state_vec.inactive[0])
    else:
        num_inactive_spin_orbs = 0
    if len(state_vec.virtual) != 0:
        num_virtual_spin_orbs = len(state_vec.virtual[0])
    else:
        num_virtual_spin_orbs = 0
    state_vector = StateVectorOperator({"o"*num_inactive_spin_orbs + "z"*num_virtual_spin_orbs: StateVectorOperatorData("o"*num_inactive_spin_orbs, state_vec.ket_active, "z"*num_virtual_spin_orbs)})
    ref_vector = StateVectorOperator({"o"*num_inactive_spin_orbs + "z"*num_virtual_spin_orbs: StateVectorOperatorData("o"*num_inactive_spin_orbs, ref_vec.ket_active, "z"*num_virtual_spin_orbs)})
    for operator in operators:
        state_vector = state_vector*operator
    return state_vector*ref_vector

def convert_pauli_to_hybrid_form(
    pauliop: OperatorPauli, num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int
) -> OperatorHybrid:
    """Convert Pauli operator to hybrid operator.

    Args:
        pauliop: Pauli operator.
        num_inactive_orbs: Number of inactive orbitals.
        num_active_orbs: Number of active orbitals.
        num_virtual_orbs: Number of virtual orbitals.

    Returns:
        Hybrid operator.
    """
    new_operator: dict[str, np.ndarray] = {}
    active_start = num_inactive_orbs
    active_end = num_inactive_orbs + num_active_orbs
    for pauli_string, factor in pauliop.operators.items():
        new_inactive = pauli_string[:active_start]
        new_active = pauli_string[active_start:active_end]
        new_virtual = pauli_string[active_end:]
        active_pauli = OperatorPauli({new_active: 1})
        new_active_matrix = factor * active_pauli.matrix_form()
        key = new_inactive + new_virtual
        if key in new_operator:
            new_operator[key].active_matrix += new_active_matrix
        else:
            new_operator[key] = OperatorHybridData(new_inactive, new_active_matrix, new_virtual)
    return OperatorHybrid(new_operator)


class OperatorHybridData:
    def __init__(
        self, inactive_pauli: str, active_matrix: np.ndarray | ss.csr_matrix, virtual_pauli: str
    ) -> None:
        """Initialize data structure of hybrid operators.

        Args:
            inactive_pauli: Pauli string of inactive orbitals.
            active_matrix: Matrix operator of active orbitals.
            virtual_pauli: Pauli string of virtual orbitals.
        """
        self.inactive_pauli = inactive_pauli
        self.active_matrix = active_matrix
        self.virtual_pauli = virtual_pauli


class OperatorHybrid:
    def __init__(self, operator: dict[str, OperatorHybridData]) -> None:
        """Initialize hybrid operator.

        The key is the Pauli-string of inactive + virtual,
        i.e. the active part does not contribute to the key.

        Args:
            operator: Dictonary form of hybrid operator.
        """
        self.operators = operator

    def __add__(self, hybridop: OperatorHybrid) -> OperatorHybrid:
        """Overload addition operator.

        Args:
            hybridop: Hybrid operator.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key, op in hybridop.operators.items():
            if key in new_operators:
                new_operators[key].active_matrix += op.active_matrix
            else:
                new_operators[key] = OperatorHybridData(op.inactive_pauli, op.active_matrix, op.virtual_pauli)
        return OperatorHybrid(new_operators)

    def __sub__(self, hybridop: OperatorHybrid) -> OperatorHybrid:
        """Overload subtraction operator.

        Args:
            hybridop: Hybrid operator.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key, op in hybridop.operators.items():
            if key in new_operators:
                new_operators[key].active_matrix -= op.active_matrix
            else:
                new_operators[key] = OperatorHybridData(
                    op.inactive_pauli, -op.active_matrix, op.virtual_pauli
                )
        return OperatorHybrid(new_operators)

    def __mul__(self, pauliop: OperatorHybrid) -> OperatorHybrid:
        """Overload multiplication operator.

        Args:
            hybridop: Hybrid operator.

        Returns:
            New hybrid operator.
        """
        new_operators: dict[str, np.ndarray] = {}
        for _, op1 in self.operators.items():
            for _, op2 in pauliop.operators.items():
                new_inactive = ''
                new_virtual = ''
                fac: complex = 1
                for pauli1, pauli2 in zip(op1.inactive_pauli, op2.inactive_pauli):
                    if pauli1 == 'I':
                        new_inactive += pauli2
                    elif pauli2 == 'I':
                        new_inactive += pauli1
                    elif pauli1 == pauli2:
                        new_inactive += 'I'
                    elif pauli1 == 'X' and pauli2 == 'Y':
                        new_inactive += 'Z'
                        fac *= 1j
                    elif pauli1 == 'X' and pauli2 == 'Z':
                        new_inactive += 'Y'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'X':
                        new_inactive += 'Z'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'Z':
                        new_inactive += 'X'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'X':
                        new_inactive += 'Y'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'Y':
                        new_inactive += 'X'
                        fac *= -1j
                for pauli1, pauli2 in zip(op1.virtual_pauli, op2.virtual_pauli):
                    if pauli1 == 'I':
                        new_virtual += pauli2
                    elif pauli2 == 'I':
                        new_virtual += pauli1
                    elif pauli1 == pauli2:
                        new_virtual += 'I'
                    elif pauli1 == 'X' and pauli2 == 'Y':
                        new_virtual += 'Z'
                        fac *= 1j
                    elif pauli1 == 'X' and pauli2 == 'Z':
                        new_virtual += 'Y'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'X':
                        new_virtual += 'Z'
                        fac *= -1j
                    elif pauli1 == 'Y' and pauli2 == 'Z':
                        new_virtual += 'X'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'X':
                        new_virtual += 'Y'
                        fac *= 1j
                    elif pauli1 == 'Z' and pauli2 == 'Y':
                        new_virtual += 'X'
                        fac *= -1j
                new_active = fac * lw.matmul(op1.active_matrix, op2.active_matrix)
                key = new_inactive + new_virtual
                if key in new_operators:
                    new_operators[key].active_matrix += new_active
                else:
                    new_operators[key] = OperatorHybridData(new_inactive, new_active, new_virtual)
        return OperatorHybrid(new_operators)

    def __rmul__(self, number: float) -> OperatorHybrid:
        """Overload right multiplication operator.

        Args:
            number: Scalar value.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix *= number
        return OperatorHybrid(new_operators)

    @property
    def dagger(self) -> OperatorHybrid:
        """Do complex conjugate of operator.

        Returns:
            New hybrid operator.
        """
        new_operators = {}
        for key, op in self.operators.items():
            new_operators[key] = OperatorHybridData(
                op.inactive_pauli, np.conj(op.active_matrix).transpose(), op.virtual_pauli
            )
        return OperatorHybrid(new_operators)

    def apply_u_from_right(self, U: np.ndarray | ss.csr_matrix) -> OperatorHybrid:
        """Matrix multiply with transformation matrix from the right.

        Args:
            U: Transformation matrix.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix = lw.matmul(new_operators[key].active_matrix, U)
        return OperatorHybrid(new_operators)

    def apply_u_from_left(self, U: np.ndarray | ss.csr_matrix) -> OperatorHybrid:
        """Matrix multiply with transformation matrix from the left.

        Args:
            U: Transformation matrix.

        Returns:
            New hybrid operator.
        """
        new_operators = copy.deepcopy(self.operators)
        for key in self.operators.keys():
            new_operators[key].active_matrix = lw.matmul(U, new_operators[key].active_matrix)
        return OperatorHybrid(new_operators)


def make_projection_operator(state_vector: StateVector, use_csr: int = 10) -> OperatorHybrid:
    """Create a projection operator, |0><0|, from a state vector.

    Args:
        state_vector: State vector class.

    Returns:
        Projection operator in hybrid form.
    """
    new_operator = {}
    if len(state_vector.inactive) != 0:
        num_inactive_orbs = len(state_vector.inactive[0])
    else:
        num_inactive_orbs = 0
    if len(state_vector.virtual) != 0:
        num_virtual_orbs = len(state_vector.virtual[0])
    else:
        num_virtual_orbs = 0
    if len(state_vector._active_onvector) > use_csr:
        active_matrix = (
            lw.outer(state_vector.ket_active_csr, state_vector.bra_active_csr)
            * 1
            / (2 ** (num_inactive_orbs + num_virtual_orbs))
        )
    else:
        active_matrix = (
            lw.outer(state_vector.ket_active, state_vector.bra_active)
            * 1
            / (2 ** (num_inactive_orbs + num_virtual_orbs))
        )
    for pauli in itertools.product(['Z', 'I'], repeat=num_inactive_orbs + num_virtual_orbs):
        active = active_matrix * (-1) ** (pauli[:num_inactive_orbs].count('Z'))
        hybridop = OperatorHybridData(
            ''.join(pauli[:num_inactive_orbs]), active, ''.join(pauli[num_inactive_orbs:])
        )
        new_operator[''.join(pauli)] = hybridop
    return OperatorHybrid(new_operator)
