from qiskit.circuit import (QuantumCircuit, Parameter)
from qiskit_nature.second_q.mappers import JordanWignerMapper
from slowquant.qiskit_interface.operators_circuits import (sa_single_excitation, double_excitation)
from slowquant.qiskit_interface.util import f2q
import numpy as np
from numpy._typing import NDArray
import pennylane as qml
from pennylane.pauli.utils import pauli_word_to_string
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

    # tmp = permutation[i]
    # permutation[i] = permutation[mitigatedQubits[i]]
    # permutation[mitigatedQubits[i]] = tmp

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
            if ((ps[j] == 'X' or ps[j] == 'Y' or ps[j] == 'Z') and basisState[j] == '1'):
                parity ^= 1
        if parity == 0:
            expectationValue += p[i]
        else:
            expectationValue -= p[i]

    return expectationValue

def GetTileCircuitsAndQubits_TUPS(
    activeSpatOrbCount : int,
    backendQubitCount : int,
    layerCount : int,
    grad_param_R : dict[str, int]
)   -> tuple[list[QuantumCircuit], list[list[int]]]:
    """
    Args:
        activeSpatOrbCount: The number of active spatial orbitals.
        backendQubitCount: The number of qubits on the backend.
        layerCount: The number of layers
        grad_param_R: For SlowQuant (pass by reference)

    Returns:
        A list of tUPS tiles (quantum circuits, pre-transpilation) for ALL layers of the ansatz,
        and a list of lists of the qubits (pre-transpilation) associated with those tiles (only for the first layer).
        Element ``i`` in the list of lists of qubits are the qubits associated with tile ``i``. 
        as it is returned in the list of tile circuits. The first element in the list of tile circuits
        is the first tile in the ansatz i.e. the top left tile in the quantum circuit drawing.
        The next element is the tile below that and when we'e done with a column we move to the top
        tile in the next column etc.
    """
    tileCircuits = list()
    tileQubits = list()
    idx = 0
    for layerNumber in range(layerCount):
        for c in range(0, 2):
            for p in range(c, activeSpatOrbCount - 1, 2):
                qc = QuantumCircuit(2 *  activeSpatOrbCount, backendQubitCount)

                qc = sa_single_excitation(p, p + 1, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
                qc = double_excitation(2 * p, 2 * p + 1, 2 * p + 2, 2 * p + 3, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
                qc = sa_single_excitation(p, p + 1, activeSpatOrbCount, qc, Parameter(f"p{idx:09d}"), JordanWignerMapper())
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1

                if layerNumber == 0:
                    tileQubits.append([f2q((2 * p), activeSpatOrbCount), f2q((2 * p + 1), activeSpatOrbCount), f2q((2 * p + 2), activeSpatOrbCount), f2q((2 * p + 3), activeSpatOrbCount)])

                tileCircuits.append(qc.copy())
    
    return tileCircuits, tileQubits