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


def fUCCSD(
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
