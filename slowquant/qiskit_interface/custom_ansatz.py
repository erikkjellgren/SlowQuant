from typing import Any

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from slowquant.qiskit_interface.operators_circuits import tups_double, tups_single


def tUPS(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
    ansatz_options: dict[str, Any],
) -> tuple[QuantumCircuit, dict[str, int]]:
    r"""tUPS ansatz.

    #. 10.1103/PhysRevResearch.6.023300

    Ansatz Options:
        * n_layers [int]: Number of layers.
        * do_pp [bool]: Do perfect pairing.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
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
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
            # Double
            qc = tups_double(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            # Second single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
        for p in range(1, num_orbs - 2, 2):
            # First single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
            # Double
            qc = tups_double(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            # Second single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
    return qc, grad_param_R


def QNP(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
    ansatz_options: dict[str, Any],
) -> tuple[QuantumCircuit, dict[str, int]]:
    r"""QNP ansatz.

    #. 10.1088/1367-2630/ac2cb3

    Ansatz Options:
        * n_layers [int]: Number of layers.
        * do_pp [bool]: Do perfect pairing.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        ansatz_options: Ansatz options.

    Returns:
        QNP ansatz circuit and R parameters needed for gradients.
    """
    valid_options = ("n_layers", "do_pp")
    for option in ansatz_options:
        if option not in valid_options:
            raise ValueError(f"Got unknown option for QNP, {option}. Valid options are: {valid_options}")
    if "n_layers" not in ansatz_options.keys():
        raise ValueError("QNP require the option 'n_layers'")
    n_layers = ansatz_options["n_layers"]
    if "do_pp" in ansatz_options.keys():
        do_pp = ansatz_options["do_pp"]
    else:
        do_pp = False

    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"QNP only implemented for JW mapper, got: {type(mapper)}")
    if num_orbs % 2 != 0:
        raise ValueError(f"QNP only implemented for even number of spatial orbitals, got: {num_orbs}")
    if do_pp and np.sum(num_elec) != num_orbs:
        raise ValueError(
            f"pp-QNP only implemented for number of electrons and number of orbitals being the same, got: ({np.sum(num_elec)}, {num_orbs}), (elec, orbs)"
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
            # Double
            qc = tups_double(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            # Single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
        for p in range(1, num_orbs - 2, 2):
            # Double
            qc = tups_double(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            # Single
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
    return qc, grad_param_R
