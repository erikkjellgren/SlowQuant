from typing import Any

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from slowquant.qiskit_interface.operators_circuits import (
    double_excitation,
    sa_single_excitation,
    single_excitation,
)
from slowquant.unitary_coupled_cluster.util import (
    iterate_pair_t2,
    iterate_pair_t2_generalized,
    iterate_t1,
    iterate_t1_sa,
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

    #. 10.1103/PhysRevResearch.6.023300 (tUPS)
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
    valid_options = (
        "n_layers",
        "do_pp",
        "do_qnp",
        "skip_last_singles",
        "reverse_layering",
        "assume_hf_reference",
    )
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

    if not isinstance(mapper, JordanWignerMapper) and do_pp:
        raise ValueError(f"pp-tUPS only implemented for JW mapper, got: {type(mapper)}")
    if do_pp and np.sum(num_elec) != num_orbs:
        raise ValueError(
            f"pp-tUPS only implemented for number of electrons and number of orbitals being the same, got: ({np.sum(num_elec)}, {num_orbs}), (elec, orbs)"
        )
    if "skip_last_singles" in ansatz_options.keys():
        skip_last_singles = ansatz_options["skip_last_singles"]
    else:
        skip_last_singles = False
    if "reverse_layering" in ansatz_options.keys():
        do_reverse_layering = ansatz_options["reverse_layering"]
    else:
        do_reverse_layering = False
    if "assume_hf_reference" in ansatz_options.keys():
        assume_hf_reference = ansatz_options["assume_hf_reference"]
        if assume_hf_reference:
            if do_pp:
                raise ValueError("Cannot do_pp and assume_hf_reference at the same time.")
            if np.sum(num_elec) % 2 != 0:
                raise ValueError(
                    f"Cannot assume HF reference for uneven number of electrons, got {num_elec} electrons"
                )
            if np.sum(num_elec) % 4 == 0:
                print("Doing reversed layering")
                do_reverse_layering = True
            correlating_orbitals = np.zeros(2 * num_orbs)
            correlating_orbitals[: np.sum(num_elec)] = 1
    else:
        assume_hf_reference = False
    if do_reverse_layering:
        start_first = 1
        start_second = 0
    else:
        start_first = 0
        start_second = 1

    qc = HartreeFock(num_orbs, (0, 0), mapper)  # empty circuit with qubit number based on mapper
    grad_param_R = {}
    idx = 0
    # Layer loop
    for n in range(n_layers):
        for p in range(start_first, num_orbs - 1, 2):  # first column of brick-wall
            if assume_hf_reference:
                if (
                    np.sum(correlating_orbitals[2 * p : 2 * p + 4]) == 0
                    or np.sum(correlating_orbitals[2 * p : 2 * p + 4]) == 4
                ):
                    # All of the orbitals are either unccupied or occupied.
                    # Hence no particle conserving parameterization will change anything
                    continue
                correlating_orbitals[2 * p : 2 * p + 4] = -100
            if not do_qnp:
                # First single
                qc = sa_single_excitation(p, p + 1, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
            # Double
            qc = double_excitation(
                2 * p, 2 * p + 1, 2 * p + 2, 2 * p + 3, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper
            )
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            # Second single
            if n + 1 == n_layers and skip_last_singles and num_orbs == 2:
                # Special case for two orbitals.
                # Here the layer is only one block, thus,
                # the last single excitation is earlier than expected.
                continue
            qc = sa_single_excitation(p, p + 1, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
        for p in range(start_second, num_orbs - 1, 2):  # second column of brick-wall
            if assume_hf_reference:
                if (
                    np.sum(correlating_orbitals[2 * p : 2 * p + 4]) == 0
                    or np.sum(correlating_orbitals[2 * p : 2 * p + 4]) == 4
                ):
                    # All of the orbitals are either unccupied or occupied.
                    # Hence no particle conserving parameterization will change anything
                    continue
                correlating_orbitals[2 * p : 2 * p + 4] = -100
            if not do_qnp:
                # First single
                qc = sa_single_excitation(p, p + 1, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
            # Double
            qc = double_excitation(
                2 * p, 2 * p + 1, 2 * p + 2, 2 * p + 3, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper
            )
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
            # Second single
            if n + 1 == n_layers and skip_last_singles:
                continue
            qc = sa_single_excitation(p, p + 1, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
    return qc, grad_param_R


def fUCC(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
    ansatz_options: dict[str, Any],
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Create factorized UCC ansatz.

    #. 10.1103/PhysRevA.102.062612 (efficient circuits for JW)
    #. 10.1021/acs.jctc.8b01004 (k-UpCCGSD)

    Ansatz Options:
        * n_layers [int]: Number of layers.
        * S [bool]: Add single excitations.
        * SAS [bool]: Add spin-adapted single excitations.
        * SAGS [bool]: Add generalized spin-adapted single excitations.
        * D [bool]: Add double excitations.
        * pD [bool]: Add pair double excitations.
        * GpD [bool]: Add generalized pair double excitations.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermioinc to qubit mapper.
        ansatz_options: Ansatz options.

    Returns:
        Factorized UCC ansatz circuit and R parameters needed for gradients.
    """
    valid_options = ("n_layers", "S", "D", "SAGS", "pD", "GpD", "SAS")
    for option in ansatz_options:
        if option not in valid_options:
            raise ValueError(f"Got unknown option for fUCC, {option}. Valid options are: {valid_options}")
    if "n_layers" not in ansatz_options.keys():
        raise ValueError("fUCC require the option 'n_layers'")
    do_S = False
    do_SAS = False
    do_SAGS = False
    do_D = False
    do_pD = False
    do_GpD = False
    if "S" in ansatz_options.keys():
        if ansatz_options["S"]:
            do_S = True
    if "SAS" in ansatz_options.keys():
        if ansatz_options["SAS"]:
            do_SAS = True
    if "SAGS" in ansatz_options.keys():
        if ansatz_options["SAGS"]:
            do_SAGS = True
    if "D" in ansatz_options.keys():
        if ansatz_options["D"]:
            do_D = True
    if "pD" in ansatz_options.keys():
        if ansatz_options["pD"]:
            do_pD = True
    if "GpD" in ansatz_options.keys():
        if ansatz_options["GpD"]:
            do_GpD = True
    if True not in (do_S, do_SAS, do_SAGS, do_D, do_pD, do_GpD):
        raise ValueError("fUCC requires some excitations got none.")
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
    qc = HartreeFock(num_orbs, (0, 0), mapper)  # empty circuit with qubit number based on mapper
    grad_param_R = {}
    idx = 0
    # Layer loop
    for _ in range(n_layers):
        if do_S:
            for a, i in iterate_t1(occ, unocc):
                qc = single_excitation(i, a, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
        if do_SAS:
            for a, i, _ in iterate_t1_sa(occ, unocc):
                qc = sa_single_excitation(i, a, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
        if do_SAGS:
            for a, i, _ in iterate_t1_sa_generalized(num_orbs):
                qc = sa_single_excitation(i, a, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
        if do_D:
            for a, i, b, j in iterate_t2(occ, unocc):
                qc = double_excitation(i, j, a, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
        if do_pD:
            for a, i, b, j in iterate_pair_t2(occ, unocc):
                qc = double_excitation(i, j, a, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
        if do_GpD:
            for a, i, b, j in iterate_pair_t2_generalized(num_orbs):
                qc = double_excitation(i, j, a, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
    return qc, grad_param_R


def SDSfUCC(
    num_orbs: int,
    num_elec: tuple[int, int],
    mapper: FermionicMapper,
    ansatz_options: dict[str, Any],
) -> tuple[QuantumCircuit, dict[str, int]]:
    r"""Create SDS ordered factorized UCC.

    The operator ordering of this implementation is,

    .. math::
        \boldsymbol{U}\left|\text{CSF}\right> = \prod_{ijab}\exp\left(\theta_{jb}\left(\hat{T}_{jb}-\hat{T}_{jb}^\dagger\right)\right)
        \exp\left(\theta_{ijab}\left(\hat{T}_{ijab}-\hat{T}_{ijab}^\dagger\right)\right)
        \exp\left(\theta_{ia}\left(\hat{T}_{ia}-\hat{T}_{ia}^\dagger\right)\right)\left|\text{CSF}\right>

    #. 10.1063/1.5133059, Eq. 25, Eq. 35 (SDS)
    #. 10.1103/PhysRevA.102.062612 (efficient circuits for JW)
    #. 10.1021/acs.jctc.8b01004 (k-UpCCGSD)

    Ansatz Options:
        * n_layers [int]: Number of layers.
        * D [bool]: Add double excitations.
        * pD [bool]: Add pair double excitations.
        * GpD [bool]: Add generalized pair double excitations.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of alpha and beta electrons.
        mapper: Fermioinc to qubit mapper.
        ansatz_options: Ansatz options.

    Returns:
        SDS ordered fUCC ansatz circuit and R parameters needed for gradients.
    """
    valid_options = ("n_layers", "D", "pD", "GpD")
    for option in ansatz_options:
        if option not in valid_options:
            raise ValueError(f"Got unknown option for SDSfUCC, {option}. Valid options are: {valid_options}")
    if "n_layers" not in ansatz_options.keys():
        raise ValueError("SDSfUCC require the option 'n_layers'")
    do_D = False
    do_pD = False
    do_GpD = False
    if "D" in ansatz_options.keys():
        if ansatz_options["D"]:
            do_D = True
    if "pD" in ansatz_options.keys():
        if ansatz_options["pD"]:
            do_pD = True
    if "GpD" in ansatz_options.keys():
        if ansatz_options["GpD"]:
            do_GpD = True
    if True not in (do_D, do_pD, do_GpD):
        raise ValueError("SDSfUCC requires some excitations got none.")
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
    qc = HartreeFock(num_orbs, (0, 0), mapper)  # empty circuit with qubit number based on mapper
    grad_param_R = {}
    idx = 0
    # Layer loop
    for _ in range(n_layers):
        # Kind of D excitation determines indices for complete SDS block
        if do_D:
            for a, i, b, j in iterate_t2(occ, unocc):
                if i % 2 == a % 2:
                    qc = single_excitation(i, a, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                else:
                    qc = single_excitation(i, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
                qc = double_excitation(i, j, a, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
                if i % 2 == a % 2:
                    qc = single_excitation(j, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                else:
                    qc = single_excitation(j, a, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
        if do_pD:
            for a, i, b, j in iterate_pair_t2(occ, unocc):
                qc = sa_single_excitation(i // 2, a // 2, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
                qc = double_excitation(i, j, a, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
                qc = sa_single_excitation(i // 2, a // 2, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
        if do_GpD:
            for a, i, b, j in iterate_pair_t2_generalized(num_orbs):
                qc = sa_single_excitation(i // 2, a // 2, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
                qc = double_excitation(i, j, a, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 2
                idx += 1
                qc = sa_single_excitation(i // 2, a // 2, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
                grad_param_R[f"p{idx:09d}"] = 4
                idx += 1
    return qc, grad_param_R
