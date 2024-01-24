import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit


def ErikD_JW():
    """UCCD(2,2) circuit for JW.

    Gate count, cx: 3, u: 3
    """
    p1 = Parameter("p1")
    qc = QuantumCircuit(4)
    qc.ry(2 * p1, 0)
    qc.x(2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.x(0)
    return qc


def ErikSD_JW():
    """UCCSD(2,2) circuit for JW.

    Gate count, cx: 8, u: 7
    """
    p1 = Parameter("p1")
    p2 = Parameter("p2")
    qc = QuantumCircuit(4)
    qc.ry(np.pi + 2 * p1, 0)
    qc.cry(-np.pi / 2 + 2 * p2, 0, 2)
    qc.x(0)
    qc.h(2)
    qc.cx(2, 0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(1, 3)
    qc.x(1)
    qc.cx(0, 2)
    qc.x(3)
    return qc


def ErikD_Parity():
    """UCCD(2,2) circuit for Parity.

    Gate count, cx: 1, u: 2
    """
    p1 = Parameter("p1")
    qc = QuantumCircuit(2)
    qc.ry(2 * p1, 0)
    qc.cx(0, 1)
    qc.x(0)
    return qc


def ErikSD_Parity():
    """UCCSD(2,2) circuit for Parity.

    Gate count, cx: 3, u: 4
    """
    p1 = Parameter("p1")
    p2 = Parameter("p2")
    qc = QuantumCircuit(2)
    qc.ry(np.pi + 2 * p1, 0)
    qc.cry(np.pi / 2 - 2 * p2, 0, 1)
    qc.h(1)
    qc.cx(1, 0)
    return qc
