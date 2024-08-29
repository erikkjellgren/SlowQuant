from typing import Any

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from slowquant.qiskit_interface.operators_circuits import (
    double_excitation,
    single_excitation,
    single_sa_excitation,
    tups_double,
    tups_single,
)
from slowquant.unitary_coupled_cluster.util import (
    iterate_pair_t2_generalized,
    iterate_t1,
    iterate_t1_sa_generalized,
    iterate_t2,
)


def tUPS(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
    ansatz_options: dict[str, Any],
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Create tUPS ansatz.

    #. 10.1103/PhysRevResearch.6.023300
    #. 10.1088/1367-2630/ac2cb3 (QNP)

    Ansatz Options:
        * n_layers [int]: Number of layers.
        * do_pp [bool]: Do perfect pairing. (default: False)
        * do_qnp [bool]: Do QNP tiling. (default: False)
        * skip_last_singles [bool]: Skip last layer of singles operators. (default: False)

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermionic to qubit mapper.
        ansatz_options: Ansatz options.

    Returns:
        tUPS ansatz circuit and R parameters needed for gradients.
    """
    valid_options = ("n_layers", "do_pp", "do_qnp", "skip_last_singles")
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
    if "do_qnp" in ansatz_options.keys():
        do_qnp = ansatz_options["do_qnp"]
    else:
        do_qnp = False

    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"tUPS only implemented for JW mapper, got: {type(mapper)}")
    if do_pp and np.sum(num_elec) != num_orbs:
        raise ValueError(
            f"pp-tUPS only implemented for number of electrons and number of orbitals being the same, got: ({np.sum(num_elec)}, {num_orbs}), (elec, orbs)"
        )
    if "skip_last_singles" in ansatz_options.keys():
        skip_last_singles = ansatz_options["skip_last_singles"]
    else:
        skip_last_singles = False

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
    for n in range(n_layers):
        for p in range(0, num_orbs - 1, 2):
            if not do_qnp:
                # First single
                qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
            # Double
            qc = tups_double(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            # Second single
            if n + 1 == n_layers and skip_last_singles and num_orbs == 2:
                # Special case for two orbitals.
                # Here the layer is only one block, thus,
                # the last single excitation is earlier than expected.
                continue
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
        for p in range(1, num_orbs - 1, 2):
            if not do_qnp:
                # First single
                qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
            # Double
            qc = tups_double(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            # Second single
            if n + 1 == n_layers and skip_last_singles:
                continue
            qc = tups_single(p, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
    return qc, grad_param_R


def fUCC(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
    ansatz_options: dict[str, Any],
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Create factorized UCCSD ansatz.

    #. 10.1103/PhysRevA.102.062612

    Ansatz Options:
        * n_layers [int]: Number of layers.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermioinc to qubit mapper.
        ansatz_options: Ansatz options.

    Returns:
        Factorized UCCSD ansatz circuit and R parameters needed for gradients.
    """
    valid_options = ("n_layers",)
    for option in ansatz_options:
        if option not in valid_options:
            raise ValueError(f"Got unknown option for fUCC, {option}. Valid options are: {valid_options}")
    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"fUCC only implemented for JW mapper, got: {type(mapper)}")
    if "n_layers" not in ansatz_options.keys():
        raise ValueError("kSAdUpCCGSD require the option 'n_layers'")
    n_layers = ansatz_options["n_layers"]
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
    qc = HartreeFock(num_orbs, num_elec, mapper)
    grad_param_R = {}
    idx = 0
    for _ in range(n_layers):
        for a, i in iterate_t1(occ, unocc):
            qc = single_excitation(a, i, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
        for a, i, b, j in iterate_t2(occ, unocc):
            qc = double_excitation(a, b, i, j, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
    return qc, grad_param_R


def kSAfUpCCGSD(
    num_orbs: int, num_elec: tuple[int, int], mapper: FermionicMapper, ansatz_options: dict[str, Any]
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Create modified k-UpCCGSD ansatz.

    The ansatz has been modified to use spin-adapted singlet single excitation operators.

    #. 10.1021/acs.jctc.8b01004

    Ansatz Options:
        * n_layers [int]: Number of layers.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermioinc to qubit mapper.
        ansatz_options: Ansatz options.

    Returns:
        Modified k-UpCCGSD ansatz.
    """
    valid_options = ("n_layers",)
    for option in ansatz_options:
        if option not in valid_options:
            raise ValueError(
                f"Got unknown option for kSAfUpCCGSD, {option}. Valid options are: {valid_options}"
            )
    if "n_layers" not in ansatz_options.keys():
        raise ValueError("kSAfUpCCGSD require the option 'n_layers'")
    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"kSAfUpCCGSD only implemented for JW mapper, got: {type(mapper)}")
    n_layers = ansatz_options["n_layers"]
    qc = HartreeFock(num_orbs, num_elec, mapper)
    grad_param_R = {}
    idx = 0
    for _ in range(n_layers):
        for a, i, _ in iterate_t1_sa_generalized(num_orbs):
            qc = single_sa_excitation(a, i, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
        for a, i, b, j in iterate_pair_t2_generalized(num_orbs):
            qc = double_excitation(a, b, i, j, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
    return qc, grad_param_R


def dUCCSD(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
    ansatz_options: dict[str, Any],
) -> tuple[QuantumCircuit, dict[str, int]]:
    r"""Create disentangled UCCSD ansatz with SDS ordering.

    The operator ordering of this implementation is,

    .. math::
        \boldsymbol{U}\left|\text{CSF}\right> = \prod_{ijab}\exp\left(\theta_{jb}\left(\hat{T}_{jb}-\hat{T}_{jb}^\dagger\right)\right)
        \exp\left(\theta_{ijab}\left(\hat{T}_{ijab}-\hat{T}_{ijab}^\dagger\right)\right)
        \exp\left(\theta_{ia}\left(\hat{T}_{ia}-\hat{T}_{ia}^\dagger\right)\right)\left|\text{CSF}\right>

    #. 10.1063/1.5133059, Eq. 25, Eq. 35

    Ansatz Options:
        * n_layers [int]: Number of layers.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermioinc to qubit mapper.
        ansatz_options: Ansatz options.

    Returns:
        Disentangled UCCSD ansatz circuit and R parameters needed for gradients.
    """
    valid_options = ("n_layers",)
    for option in ansatz_options:
        if option not in valid_options:
            raise ValueError(f"Got unknown option for dUCCSD, {option}. Valid options are: {valid_options}")
    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"dUCCSD only implemented for JW mapper, got: {type(mapper)}")
    if "n_layers" not in ansatz_options.keys():
        raise ValueError("kSAdUpCCGSD require the option 'n_layers'")
    n_layers = ansatz_options["n_layers"]
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
    qc = HartreeFock(num_orbs, num_elec, mapper)
    grad_param_R = {}
    idx = 0
    for _ in range(n_layers):
        for a, i, b, j in iterate_t2(occ, unocc):
            qc = single_excitation(a, i, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            qc = double_excitation(a, b, i, j, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            qc = single_excitation(b, j, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
    return qc, grad_param_R


def kSAdUpCCGSD(
    num_orbs: int, num_elec: tuple[int, int], mapper: FermionicMapper, ansatz_options: dict[str, Any]
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Create modified k-UpCCGSD ansatz.

    The ansatz have been modifed to use spin-adapted singet single excitation operators.
    This ansatz has also been modified to follow the operator ordering as dUCCSD.

    #. 10.1021/acs.jctc.8b01004
    #. 10.1063/1.5133059, Eq. 25, Eq. 35 (for ordering)

    Ansatz Options:
        * n_layers [int]: Number of layers.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermioinc to qubit mapper.
        ansatz_options: Ansatz options.

    Returns:
        Modified k-UpCCGSD ansatz.
    """
    valid_options = ("n_layers",)
    for option in ansatz_options:
        if option not in valid_options:
            raise ValueError(
                f"Got unknown option for kSAdUpCCGSD, {option}. Valid options are: {valid_options}"
            )
    if "n_layers" not in ansatz_options.keys():
        raise ValueError("kSAfUpCCGSD require the option 'n_layers'")
    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(f"kSAdUpCCGSD only implemented for JW mapper, got: {type(mapper)}")
    n_layers = ansatz_options["n_layers"]
    qc = HartreeFock(num_orbs, num_elec, mapper)
    grad_param_R = {}
    idx = 0
    for _ in range(n_layers):
        for a, i, b, j in iterate_pair_t2_generalized(num_orbs):
            qc = single_sa_excitation(a // 2, i // 2, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
            qc = double_excitation(a, b, i, j, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            qc = single_sa_excitation(a // 2, i // 2, num_orbs, qc, Parameter(f"p{idx:09d}"))
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
    return qc, grad_param_R
