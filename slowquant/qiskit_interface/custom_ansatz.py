import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.operators import Epq


def ErikD_JW():
    """UCCD(2,2) circuit for JW.

    Gate count, cx: 3, u: 3

    .. code-block::

             ┌──────────┐               ┌───┐
        q_0: ┤ Ry(2*p1) ├──■────■────■──┤ X ├
             └──────────┘┌─┴─┐  │    │  └───┘
        q_1: ────────────┤ X ├──┼────┼───────
                ┌───┐    └───┘┌─┴─┐  │
        q_2: ───┤ X ├─────────┤ X ├──┼───────
                └───┘         └───┘┌─┴─┐
        q_3: ──────────────────────┤ X ├─────
                                   └───┘
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

    .. code-block::

             ┌──────────────┐                  ┌───┐┌───┐
        q_0: ┤ Ry(2*p1 + π) ├────────■─────────┤ X ├┤ X ├──■───────────────────■───────
             └──────────────┘        │         └───┘└─┬─┘┌─┴─┐                 │  ┌───┐
        q_1: ────────────────────────┼────────────────┼──┤ X ├──■─────────■────┼──┤ X ├
                             ┌───────┴────────┐┌───┐  │  └───┘┌─┴─┐       │  ┌─┴─┐└───┘
        q_2: ────────────────┤ Ry(2*p2 - π/2) ├┤ H ├──■───────┤ X ├──■────┼──┤ X ├─────
                             └────────────────┘└───┘          └───┘┌─┴─┐┌─┴─┐├───┤
        q_3: ──────────────────────────────────────────────────────┤ X ├┤ X ├┤ X ├─────
                                                                   └───┘└───┘└───┘
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

    .. code-block::

             ┌──────────┐     ┌───┐
        q_0: ┤ Ry(2*p1) ├──■──┤ X ├
             └──────────┘┌─┴─┐└───┘
        q_1: ────────────┤ X ├─────
                         └───┘

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

    .. code-block::

             ┌──────────────┐                       ┌───┐
        q_0: ┤ Ry(2*p1 + π) ├────────■──────────────┤ X ├
             └──────────────┘┌───────┴────────┐┌───┐└─┬─┘
        q_1: ────────────────┤ Ry(π/2 - 2*p2) ├┤ H ├──■──
                             └────────────────┘└───┘
    """
    p1 = Parameter("p1")
    p2 = Parameter("p2")
    qc = QuantumCircuit(2)
    qc.ry(np.pi + 2 * p1, 0)
    qc.cry(np.pi / 2 - 2 * p2, 0, 1)
    qc.h(1)
    qc.cx(1, 0)
    return qc


def tUPS(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
    n_layers: int,
    do_pp: bool,
) -> QuantumCircuit:
    r"""tUPS ansatz"""
    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"tUPS only implemented for JW mapper, got: {type(mapper)}")
    if num_orbs % 2 != 0:
        raise ValueError(f"tUPS only implemented for even number of spatial orbitals, got: {num_orbs}")
    if do_pp and np.sum(num_elec) != num_orbs:
        raise ValueError(
            f"pp-tUPS only implemented for number of electrons and number of orbitals being the same, got: ({np.sum(num_elec)}, {num_orbs}), (elec, orbs)"
        )

    num_spin_orbs = 2 * num_orbs
    operators = []
    factors = []
    params = []
    idx = 0
    for _ in range(n_layers):
        for p in range(0, num_orbs - 1, 2):
            print(p)
            epq = Epq(p + 1, p)
            eqp = Epq(p, p + 1)
            # First single
            T = epq - eqp
            op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs))
            operators.append(op_mapped.paulis)
            params.append(Parameter(f"p{idx}"))
            factors.append(op_mapped.coeffs)
            idx += 1
            # Double
            T = epq * epq - eqp * eqp
            op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs))
            operators.append(op_mapped.paulis)
            params.append(Parameter(f"p{idx}"))
            factors.append(op_mapped.coeffs)
            idx += 1
            # Second single
            T = epq - eqp
            op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs))
            operators.append(op_mapped.paulis)
            params.append(Parameter(f"p{idx}"))
            factors.append(op_mapped.coeffs)
            idx += 1
        for p in range(1, num_orbs - 2, 2):
            print(p)
            epq = Epq(p + 1, p)
            eqp = Epq(p, p + 1)
            # First single
            T = epq - eqp
            op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs))
            operators.append(op_mapped.paulis)
            params.append(Parameter(f"p{idx}"))
            factors.append(op_mapped.coeffs)
            idx += 1
            # Double
            T = epq * epq - eqp * eqp
            op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs))
            operators.append(op_mapped.paulis)
            params.append(Parameter(f"p{idx}"))
            factors.append(op_mapped.coeffs)
            idx += 1
            # Second single
            T = epq - eqp
            op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs))
            operators.append(op_mapped.paulis)
            params.append(Parameter(f"p{idx}"))
            factors.append(op_mapped.coeffs)
            idx += 1
    params_long = []
    ops_long = []
    facs_long = []
    for param, paulis, facs in zip(params, operators, factors):
        for pauli, fac in zip(paulis, facs):
            ops_long.append(str(pauli))
            params_long.append(param)
            facs_long.append((-1.0j * (fac)).real)
    num_qubits = num_spin_orbs  # qc.num_qubits
    if do_pp:
        qc = QuantumCircuit(num_qubits)
        for p in range(0, 2*num_orbs):
            if p % 2 == 0:
                qc.x(p)
    else:
        qc = HartreeFock(num_orbs, num_elec, mapper)
    for param, pauli, fac in zip(params_long, ops_long, facs_long):
        qc.append(
            PauliEvolutionGate(Pauli(pauli), fac * param),
            np.linspace(0, num_qubits - 1, num_qubits, dtype=int).tolist(),
        )
    return qc
