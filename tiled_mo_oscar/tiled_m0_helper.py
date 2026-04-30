from qiskit.circuit import (QuantumCircuit, Parameter)
from qiskit_nature.second_q.mappers import JordanWignerMapper
from slowquant.qiskit_interface.operators_circuits import (sa_single_excitation, double_excitation, single_excitation)
from slowquant.qiskit_interface.util import f2q
import numpy as np
from numpy._typing import NDArray
import sys
import pennylane as qml
from pennylane.pauli.utils import pauli_word_to_string
from pennylane.pauli import group_observables
from pennylane.operation import Operator

def ApplyMitigator(
    p : list[float],
    mitigatedQubits : list[int],
    mitigator : NDArray[any]
):
    """
    Applies a mitigation matrix to ``p``, mutating ``p``. This is the error mitigation procedure.

    Args:
        p: The probability vector. The bitstrings that p is made from should follow the tiled M0 bitstring order.
        mitigatedQubits: The virtual qubits belonging to ``mitigator``. The qubits should be sorted in ascending order.
        mitigator: The mitigation matrix. The matrix should be made from bitstrings that follow the tiled M0 bitstring order.
    """
    totQubitCount = int(np.log2(len(p)))
    mitigatedQubitCount = len(mitigatedQubits)

    # Reorder the elements of p so that the mitigated qubits occupy the least significant bits
    # The order in ``p`` is the qiskit order - i.e. the most significant bits in the bitstrings
    # correspond to the most significant pre-transpilation qubits: q7 q6 q5 q4 q3 q2 q1 q0 for
    # an 8-qubit system, for example. If the tile qubits are q3 q2 q1 q0 then no reordering needs
    # to be done (although the implementation here doesn't distinguish this case). If the tile qubits are
    # q7 q6 q3 q2, however, then a reordering to [...] q7 q6 q3 q2 will be done. The order
    # of the four most significant qubits does not matter as long as we revert back to the qiskit
    # order after we have applied the mitigator.
    cycles = list()
    for i in range(mitigatedQubitCount):
        cycles.append([mitigatedQubits[i], i]) # Permutation cycles (2-cycles)

    pReordered = [0 for i in range(len(p))]
    for i in range(0, len(p)):
        state = format(i, 'b')
        state = '0' * (totQubitCount - len(state)) + state
        reorderedState = list(state)
        for cycle in cycles:
            tmp = reorderedState[totQubitCount - cycle[0] - 1]
            reorderedState[totQubitCount - cycle[0] - 1] = reorderedState[totQubitCount - cycle[1] - 1]
            reorderedState[totQubitCount - cycle[1] - 1] = tmp

        reorderedState = ''.join(reorderedState)
        pReordered[int(reorderedState, 2)] = p[i]

    # ``pReordered`` is on a form where we can apply the mitigator without creating an exponentially large matrix.
    for i in range(0, 2**totQubitCount // 2**mitigatedQubitCount):
        mitigatedProbs = np.dot(mitigator, pReordered[i * 2**mitigatedQubitCount : (i + 1) * 2**mitigatedQubitCount])
        for j in range(i * 2**mitigatedQubitCount, (i + 1) * 2**mitigatedQubitCount):
            pReordered[j] = mitigatedProbs[j - (i * 2**mitigatedQubitCount)]

    # Paste the mitigated elements back into ``p`` in the normal order
    for i in range(0, len(p)):
        state = format(i, 'b')
        state = '0' * (totQubitCount - len(state)) + state
        reorderedState = list(state)
        for cycle in cycles:
            tmp = reorderedState[totQubitCount - cycle[0] - 1]
            reorderedState[totQubitCount - cycle[0] - 1] = reorderedState[totQubitCount - cycle[1] - 1]
            reorderedState[totQubitCount - cycle[1] - 1] = tmp

        reorderedState = ''.join(reorderedState)
        p[i] = pReordered[int(reorderedState, 2)]

def CalculateExpectationValueOfPauliString(
    ps : str,
    p : list[float]
)   -> float:
    """
    Estimates the expectation value of the Pauli string ``ps`` from the probability vector ``p``.
    Assumes that the right measurement modifications were made for the quantum circuits yielding
    ``p`` so that we can use the formula for the expectation value for a string of Z's.

    Args:
        ps: The Pauli string to estimate the expectation value of.
        p: The probability vector to use for the expectation value estimation. ``p`` should be made from bitstrings following the tiled M0 order, and the bitstrings should be retrieved from quantum circuits with the necessary measurement modifications.
    
    Returns:
        The estimate of the expectation value.
    """
    expectationValue = 0.0

    for i in range(len(p)):
        basisState = format(i, 'b')
        basisState = '0' * (len(ps) - len(basisState)) + basisState
        parity = 0
        for j in range(len(ps)):
            if ((ps[j] == 'X' or ps[j] == 'Y' or ps[j] == 'Z') and basisState[j] == '1'): # basisState[len(ps) - j - 1] == '1'
                parity ^= 1
        if parity == 0:
            expectationValue += p[i]
        else:
            expectationValue -= p[i]

    return expectationValue

# Restricted tUPS
# def GetTileCircuitsAndQubits(
#     activeSpatOrbCount : int,
#     backendQubitCount : int
# )   -> tuple[list[QuantumCircuit], list[list[int]]]:
#     """
#     Args:
#         activeSpatOrbCount: The number of active spatial orbitals.
#         backendQubitCount: The number of qubits on the backend.

#     Returns:
#         A list of tUPS tiles (quantum circuits, pre-transpilation) for one layer of the ansatz,
#         and a list of lists of the qubits (pre-transpilation) associated with those tiles.
#         Element ``i`` in the list of lists of qubits are the qubits associated with tile ``i``
#         as it is returned in the list of tile circuits. The first element in the list of tile circuits
#         is the first tile in the ansatz i.e. the top left tile in the quantum circuit drawing.
#         The next element is the tile below that and when we'e done with a column we move to the top
#         tile in the next column etc.
#     """
#     tileCircuits = list()
#     tileQubits = list()
#     idx = 0
#     for c in range(0, 2):
#         for p in range(c, activeSpatOrbCount - 1, 2):
#             qc = QuantumCircuit(2 *  activeSpatOrbCount, backendQubitCount)

#             qc = sa_single_excitation(p, p + 1, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
#             idx += 1
#             qc = double_excitation(2 * p, 2 * p + 1, 2 * p + 2, 2 * p + 3, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
#             idx += 1
#             qc = sa_single_excitation(p, p + 1, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
#             idx += 1

#             tileQubits.append([f2q((2 * p), activeSpatOrbCount), f2q((2 * p + 1), activeSpatOrbCount), f2q((2 * p + 2), activeSpatOrbCount), f2q((2 * p + 3), activeSpatOrbCount)])
#             tileCircuits.append(qc.copy())
    
#     return tileCircuits, tileQubits


# Unrestricted tUPS
def GetTileCircuitsAndQubits(
    activeSpatOrbCount : int,
    backendQubitCount : int
)   -> tuple[list[QuantumCircuit], list[list[int]]]:
    """
    Args:
        activeSpatOrbCount: The number of active spatial orbitals.
        backendQubitCount: The number of qubits on the backend.

    Returns:
        A list of tUPS tiles (quantum circuits, pre-transpilation) for one layer of the ansatz,
        and a list of lists of the qubits (pre-transpilation) associated with those tiles.
        Element ``i`` in the list of lists of qubits are the qubits associated with tile ``i``
        as it is returned in the list of tile circuits. The first element in the list of tile circuits
        is the first tile in the ansatz i.e. the top left tile in the quantum circuit drawing.
        The next element is the tile below that and when we'e done with a column we move to the top
        tile in the next column etc.
    """
    tileCircuits = list()
    tileQubits = list()
    idx = 0
    for c in range(0, 2):
        for p in range(c, activeSpatOrbCount - 1, 2):
            qc = QuantumCircuit(2 *  activeSpatOrbCount, backendQubitCount)

            qc = single_excitation(2*p, 2*p + 2, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
            idx += 1
            qc = single_excitation(2*p+1, 2*p + 3, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
            idx += 1
            qc = double_excitation(2 * p, 2 * p + 1, 2 * p + 2, 2 * p + 3, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
            idx += 1
            qc = single_excitation(2*p, 2*p + 2, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
            idx += 1
            qc = single_excitation(2*p+1, 2*p + 3, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
            idx += 1

            tileQubits.append([f2q((2 * p), activeSpatOrbCount), f2q((2 * p + 1), activeSpatOrbCount), f2q((2 * p + 2), activeSpatOrbCount), f2q((2 * p + 3), activeSpatOrbCount)])
            tileCircuits.append(qc.copy())
    
    return tileCircuits, tileQubits

def PauliStrings_To_PennylaneRep(
    operator : dict[str, float],
)   -> list[Operator]:
    """
    Args:
        operator: The operator with the Pauli strings to convert (the keys in the dictionary).
    
    Returns:
        A list of the Pauli strings in the Pennylane representation.
    """
    qubitCount = len(list(operator)[0])
    pauliStringsQml = list()
    
    for ps in operator.keys():
        if ps[0] == 'Z':
            psQml = qml.PauliZ(qubitCount - 1)
        elif ps[0] == 'X':
            psQml = qml.PauliX(qubitCount - 1)
        elif ps[0] == 'Y':
            psQml = qml.PauliY(qubitCount - 1)
        else:
            psQml = qml.I(qubitCount - 1)
        
        for i in range(1, len(ps)):
            if ps[i] == 'Z':
                psQml = psQml @ qml.PauliZ(qubitCount - i - 1)
            elif ps[i] == 'X':
                psQml = psQml @ qml.PauliX(qubitCount - i - 1)
            elif ps[i] == 'Y':
                psQml = psQml @ qml.PauliY(qubitCount - i - 1)
            else:
                psQml = psQml @ qml.I(qubitCount - i - 1)
        
        pauliStringsQml.append(psQml)
    
    return pauliStringsQml

def GroupedPennylanePauliStrings_To_GroupedPauliStrings(
    psGroups_Qml : list[list[Operator]]
)   -> list[list[str]]:
    """
    Converts a list of lists of Pauli strings in the Pennylane representation to a list of lists of strings.

    Args:
        psGroups_Qml: A list of lists with the grouped Pauli strings in the Pennylane representation

    Returns:
        A list of lists with the grouped Pauli strings as strings
    """
    qubitCount = len(psGroups_Qml[0][0])
    psGroups = list()
    for psGroup_Qml in psGroups_Qml:
        psGroups.append([])
        for psQml in psGroup_Qml:
            labels = psQml.wires.labels
            ps = 'I' * (qubitCount - labels[0] - 1) + pauli_word_to_string(psQml)[0]
            for i in range(0, len(labels) - 1):
                ps += 'I' * (labels[i] - labels[i + 1] - 1)
                ps += pauli_word_to_string(psQml)[i + 1]
            ps += 'I' * labels[-1]
            psGroups[-1].append(ps)

    return psGroups

def GroupPauliStringsByQwc(
    operator : dict[str, float]
)   -> list[list[str]]:
    """
    Args:
        operator: The operator as a dictionary.
    
    Returns:
        A list of lists with the Pauli strings in ``operator`` grouped based on qubit-wise commutativity.
    """
    pauliStringsQml = PauliStrings_To_PennylaneRep(operator)
    psGroups_Qml = group_observables(pauliStringsQml, grouping_type = "qwc")
    psGroups = GroupedPennylanePauliStrings_To_GroupedPauliStrings(psGroups_Qml)
    return psGroups

def CalculateAllPsGroupOneNorms(
    psGroups : list[list[str]],
    operator : dict[str, float]
)   -> list[float]:
    """
    Args:
        psGroups: The grouped Pauli strings.
        operator: The operator as a dictionary.
    
    Returns:
        For each group in ``psGroups``, the sum of the absolute values of the weights of the Pauli strings. Ignores the all-identity Pauli string.
    """
    psGroupsOneNorms = list()
    for qwcGroup in psGroups:
        oneNorm = 0
        for pauliString in qwcGroup:
            if (pauliString == 'I' * len(pauliString)):
                continue
            oneNorm += abs(operator[pauliString])
        psGroupsOneNorms.append(oneNorm)

    return psGroupsOneNorms

def ApplyMitigatorVersion2(
    p : list[float],
    mitigatedQubits : list[int],
    mitigator : NDArray[any],
    permutation : list[int]
):
    """
    Another way to apply the mitigator. Keeps track of the permutation of the bit order instead of
    switching back to the tiled M0 order in the end. Because of the extra overhead associated with
    keeping track of the permutations, I don't think this is faster.
    """
    totQubitCount = int(np.log2(len(p)))
    mitigatedQubitCount = len(mitigatedQubits)

    # Reorder the elements of p so that the mitigated qubits occupy the least significant bits
    # The order in ``p`` is the qiskit order - i.e. the most significant bits in the bitstrings
    # correspond to the most significant pre-transpilation qubits: q7 q6 q5 q4 q3 q2 q1 q0 for
    # an 8-qubit system, for example. If the tile qubits are q3 q2 q1 q0 then no reordering needs
    # to be done (although the implementation here doesn't distinguish this case). If the tile qubits are
    # q7 q6 q3 q2, however, then a reordering to [...] q7 q6 q3 q2 will be done. The order
    # of the four most significant qubits does not matter as long as we revert back to the qiskit
    # order after we have applied the mitigator.
    cycles = list()
    for i in range(mitigatedQubitCount):
        cycles.append([mitigatedQubits[i], i]) # Permutation cycles (2-cycles)

    pReordered = [0 for i in range(len(p))]
    
    permutationSaved = permutation.copy()
    for i in range(0, len(p)):
        state = format(i, 'b')
        state = '0' * (totQubitCount - len(state)) + state
        reorderedState = list(state)

        permutation = permutationSaved.copy()
        for cycle in cycles:
            tmp = reorderedState[totQubitCount - permutation[cycle[0]] - 1]
            reorderedState[totQubitCount - permutation[cycle[0]] - 1] = reorderedState[totQubitCount - cycle[1] - 1]
            reorderedState[totQubitCount - cycle[1] - 1] = tmp

            # Update permutation
            pos = 0 # pos = which qubit occupied position ``cycle[1]``
            for j in range(len(permutation)):
                if permutation[j] == cycle[1]:
                    pos = j
                    break
            
            # Swap the position
            tmp = permutation[cycle[0]]
            permutation[cycle[0]] = cycle[1]
            permutation[pos] = tmp

        reorderedState = ''.join(reorderedState)
        pReordered[int(reorderedState, 2)] = p[i]

    # ``pReordered`` is on a form where we can apply the mitigator without creating an exponentially large matrix.
    for i in range(0, 2**totQubitCount // 2**mitigatedQubitCount):
        mitigatedProbs = np.dot(mitigator, pReordered[i * 2**mitigatedQubitCount : (i + 1) * 2**mitigatedQubitCount])
        for j in range(i * 2**mitigatedQubitCount, (i + 1) * 2**mitigatedQubitCount):
            pReordered[j] = mitigatedProbs[j - (i * 2**mitigatedQubitCount)]

    return pReordered, permutation

    """
    CODE TO ADD TO CALCULATEEXPECTATIONVALUE IF THIS FUNCTION IS USED TO DO MITIGATION

    permutation = [j for j in range(self.totQubitCount)]
    for j in range(len(mitigators) - 1, -1, -1):
        p, permutation = ApplyMitigatorVersion2(p, mitigatedQubits[j], mitigators[j], permutation)
    
    pMitigated = [0 for j in range(len(p))]
    
    permutationSaved = permutation.copy()
    for j in range(0, len(p)):
        state = format(j, 'b')
        state = '0' * (self.totQubitCount - len(state)) + state
        reorderedState = list(state)

        permutation = permutationSaved.copy()
        for k in range(len(permutation)):
            tmp = reorderedState[self.totQubitCount - k - 1]
            reorderedState[self.totQubitCount - k - 1] = reorderedState[self.totQubitCount - permutation[k] - 1]
            reorderedState[self.totQubitCount - permutation[k] - 1] = tmp
            
            # Update permutation
            pos = 0 # pos = which qubit occupied position k?
            for l in range(len(permutation)):
                if permutation[l] == k:
                    pos = l
                    break
            
            # Swap the positions
            tmp = permutation[k]
            permutation[k] = k
            permutation[pos] = tmp

        reorderedState = ''.join(reorderedState)
        pMitigated[int(reorderedState, 2)] = p[j]
    
    p = pMitigated
    """