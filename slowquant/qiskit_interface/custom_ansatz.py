from collections.abc import Sequence

import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.operators import anni
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2


def smallUCCSD(num_orbs: int, num_elec: Sequence[int], mapper: FermionicMapper) -> QuantumCircuit:
    r"""Create UCCSD ansatz that orders Paulies to minimize circuit.

    The UCCSD ansatz is given as:

    .. math::
        U(\boldsymbol{\theta}) = \exp\left(\sum_i\theta_i\hat{T}_i\right)

    In order to implement this on a quantum device the cluster operator needs to be mapped into Pauli strings:

    .. math::
        U(\boldsymbol{\theta}) = \exp\left(\sum_i\theta_i\hat{P}_i\right)

    And a Trotterization need to be performed:

    .. math::
        U(\boldsymbol{\theta}) \approx \prod_i\exp\left(\theta_i\hat{P}_i\right)

    Since there is no natural ordering of the product of exponentials it can be ordered such that as many gates as possible cancel each other.

    Consider the following ansatz:

    .. code-block::

             ┌────────────────────┐┌────────────────────┐
        q_0: ┤0                   ├┤0                   ├
             │                    ││                    │
        q_1: ┤1                   ├┤1                   ├
             │  exp(-it XXXY)(p2) ││  exp(-it XXXX)(p1) │
        q_2: ┤2                   ├┤2                   ├
             │                    ││                    │
        q_3: ┤3                   ├┤3                   ├
             └────────────────────┘└────────────────────┘

    Naively this gives the circuit:

    .. code-block::

             ┌─────┐┌───┐     ┌───┐┌────────────┐┌───┐┌───┐┌───┐┌───┐               »
        q_0: ┤ Sdg ├┤ H ├─────┤ X ├┤ Rz(2.0*p2) ├┤ X ├┤ H ├┤ S ├┤ H ├───────────────»
             └┬───┬┘└───┘┌───┐└─┬─┘└────────────┘└─┬─┘├───┤├───┤├───┤          ┌───┐»
        q_1: ─┤ H ├──────┤ X ├──■──────────────────■──┤ X ├┤ H ├┤ H ├──────────┤ X ├»
              ├───┤ ┌───┐└─┬─┘                        └─┬─┘├───┤├───┤┌───┐┌───┐└─┬─┘»
        q_2: ─┤ H ├─┤ X ├──■────────────────────────────■──┤ X ├┤ H ├┤ H ├┤ X ├──■──»
              ├───┤ └─┬─┘                                  └─┬─┘├───┤├───┤└─┬─┘     »
        q_3: ─┤ H ├───■──────────────────────────────────────■──┤ H ├┤ H ├──■───────»
              └───┘                                             └───┘└───┘          »
        «     ┌───┐┌────────────┐┌───┐┌───┐
        «q_0: ┤ X ├┤ Rz(2.0*p1) ├┤ X ├┤ H ├──────────
        «     └─┬─┘└────────────┘└─┬─┘├───┤┌───┐
        «q_1: ──■──────────────────■──┤ X ├┤ H ├─────
        «                             └─┬─┘├───┤┌───┐
        «q_2: ──────────────────────────■──┤ X ├┤ H ├
        «                                  └─┬─┘├───┤
        «q_3: ───────────────────────────────■──┤ H ├
        «                                       └───┘

    But since in "XXXY" and "XXXX" the first three Paulis are letters are identical these will cancel each other and give a circuit that is shorter:

    .. code-block::

        global phase: π/4
             ┌───────────────┐          ┌───┐┌────────────┐┌──────────┐┌────────────┐»
        q_0: ┤ U3(π/2,0,π/2) ├──────────┤ X ├┤ Rz(2.0*p2) ├┤0         ├┤ Rz(2.0*p1) ├»
             └─────┬───┬─────┘     ┌───┐└─┬─┘└────────────┘│  Unitary │└────────────┘»
        q_1: ──────┤ H ├───────────┤ X ├──■────────────────┤1         ├──────────────»
                   ├───┤      ┌───┐└─┬─┘                   └──────────┘              »
        q_2: ──────┤ H ├──────┤ X ├──■───────────────────────────────────────────────»
                   ├───┤      └─┬─┘                                                  »
        q_3: ──────┤ H ├────────■────────────────────────────────────────────────────»
                   └───┘                                                             »
        «     ┌───┐┌───┐
        «q_0: ┤ X ├┤ H ├──────────
        «     └─┬─┘├───┤┌───┐
        «q_1: ──■──┤ X ├┤ H ├─────
        «          └─┬─┘├───┤┌───┐
        «q_2: ───────■──┤ X ├┤ H ├
        «               └─┬─┘├───┤
        «q_3: ────────────■──┤ H ├
        «                    └───┘

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of electrons (alpha, beta).
        mapper: Fermionic to qubit mapper.

    Returns:
        Small circuit UCCSD.
    """
    num_spin_orbs = 2 * num_orbs
    occ = []
    unocc = []
    idx = 0
    for _ in range(np.sum(num_elec)):
        occ.append(idx)
        idx += 1
    for _ in range(num_spin_orbs - np.sum(num_elec)):
        unocc.append(idx)
        idx += 1

    params = []
    ops = []
    idx = 0
    for _, a, i, _ in iterate_t1(occ, unocc, 0, True):
        op = anni(i, True) * anni(a, False)
        T = op - op.dagger
        ops.append(mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs)).paulis)
        params.append(Parameter(f"p{idx}"))
        idx += 1
    for _, a, i, b, j, _ in iterate_t2(occ, unocc, 0, True):
        op = anni(j, True) * anni(b, False) * anni(i, True) * anni(a, True)
        T = op - op.dagger
        ops.append(mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs)).paulis)
        params.append(Parameter(f"p{idx}"))
        idx += 1
    params_long = []
    ops_long = []
    for param, paulis in zip(params, ops):
        for pauli in paulis:
            ops_long.append(str(pauli))
            params_long.append(param)
    ops_long = np.array(ops_long)
    params_long = np.array(params_long)
    sort_idx = np.argsort(ops_long)
    ops_long = ops_long[sort_idx]
    params_long = params_long[sort_idx]

    qc = HartreeFock(num_orbs, num_elec, mapper)
    num_qubits = qc.num_qubits
    for param, pauli in zip(params_long, ops_long):
        qc.append(
            PauliEvolutionGate(Pauli(pauli), param),
            np.linspace(0, num_qubits - 1, num_qubits, dtype=int).tolist(),
        )
    return transpile(qc, optimization_level=3)


def ErikD_JW() -> QuantumCircuit:
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


def ErikSD_JW() -> QuantumCircuit:
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


def ErikD_Parity() -> QuantumCircuit:
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


def ErikSD_Parity() -> QuantumCircuit:
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
