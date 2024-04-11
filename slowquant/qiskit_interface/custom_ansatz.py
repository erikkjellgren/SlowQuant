from typing import Any

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from slowquant.qiskit_interface.operators_circuits import (
    double_excitation,
    single_excitation,
    tups_double,
    tups_single,
)
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2


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
    ansatz_options: dict[str, Any],
) -> tuple[QuantumCircuit, dict[str, int]]:
    """tUPS ansatz.

    #. 10.48550/arXiv.2312.09761

    Ansatz Options:
        * n_layers [int]: Number of layers.
        * do_pp [bool]: Do perfect pairing.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermioinc to qubit mapper.
        ansatz_options: Ansatz options.

    Returns:
        tUPS ansatz circuit and R parameters needed for gradients.
    """
    valid_options = ("n_layers", "do_pp")
    for option in ansatz_options:
        if option not in valid_options:
            raise ValueError(f"Got unknown option for tUPS, {option}. Valid options are: {valid_options}")
    if "n_layers" not in ansatz_options.keys():
        raise ValueError("tUPS require the option 'n_layers'")
    n_layers = ansatz_options["n_layers"]
    if "do_pp" in ansatz_options.keys():
        do_pp = ansatz_options["do_pp"]
    else:
        do_pp = False

    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"tUPS only implemented for JW mapper, got: {type(mapper)}")
    if num_orbs % 2 != 0:
        raise ValueError(f"tUPS only implemented for even number of spatial orbitals, got: {num_orbs}")
    if do_pp and np.sum(num_elec) != num_orbs:
        raise ValueError(
            f"pp-tUPS only implemented for number of electrons and number of orbitals being the same, got: ({np.sum(num_elec)}, {num_orbs}), (elec, orbs)"
        )

    num_qubits = 2 * num_orbs  # qc.num_qubits
    if do_pp:
        qc = QuantumCircuit(num_qubits)
        for p in range(0, 2 * num_orbs):
            if p % 2 == 0:
                qc.x(p)
    else:
        qc = HartreeFock(num_orbs, num_elec, mapper)
    grad_param_R = {}
    idx = 0
    for _ in range(n_layers):
        for p in range(0, num_orbs - 1, 2):
            # First single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx}"))
            grad_param_R[f"p{idx}"] = 4
            idx += 1
            # Double
            qc = tups_double(p, num_orbs, qc, Parameter(f"p{idx}"))
            grad_param_R[f"p{idx}"] = 2
            idx += 1
            # Second single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx}"))
            grad_param_R[f"p{idx}"] = 4
            idx += 1
        for p in range(1, num_orbs - 2, 2):
            # First single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx}"))
            grad_param_R[f"p{idx}"] = 4
            idx += 1
            # Double
            qc = tups_double(p, num_orbs, qc, Parameter(f"p{idx}"))
            grad_param_R[f"p{idx}"] = 2
            idx += 1
            # Second single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx}"))
            grad_param_R[f"p{idx}"] = 4
            idx += 1
    return qc, grad_param_R


def efficientUCCSD(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Efficient UCCSD ansatz.

    #. 10.1103/PhysRevA.102.062612

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermioinc to qubit mapper.

    Returns:
        Efficient UCCSD ansatz circuit and R parameters needed for gradients.
    """
    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"efficientUCCSD only implemented for JW mapper, got: {type(mapper)}")
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
    print(occ, unocc)
    qc = HartreeFock(num_orbs, num_elec, mapper)
    grad_param_R = {}
    idx = 0
    for _, a, i, _ in iterate_t1(occ, unocc, 0, True):
        print(a, i)
        qc = single_excitation(a, i, num_orbs, qc, Parameter(f"p{idx}"))
        grad_param_R[f"p{idx}"] = 2
        idx += 1
    for _, a, i, b, j, _ in iterate_t2(occ, unocc, 0, True):
        print(a, b, i, j)
        qc = double_excitation(a, b, i, j, num_orbs, qc, Parameter(f"p{idx}"))
        grad_param_R[f"p{idx}"] = 2
        idx += 1
    return qc, grad_param_R
