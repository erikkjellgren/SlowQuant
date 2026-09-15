"""Test the hand-written excitation circuits against exact matrix exponentials.

The efficient single and double excitation circuits are hand-derived, and they depend on how the
spin-orbital indices are laid out: the Jordan-Wigner strings run between the indices in sorted
order. Under interleaved ordering a spin-conserving double always came out with both creation
indices above both annihilation indices, but under alpha/beta-blocked ordering the two pairs
interleave instead, so the circuits are exercised on index topologies they were not before.

These tests compare the circuit unitary with the exponential of the Jordan-Wigner mapped
generator, built with Qiskit's own mapper, so they check the circuits rather than restate them.
"""

import itertools

import numpy as np
import pytest
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.operators_circuits import (
    _double_excitation_efficient,
    _single_excitation_efficient,
)
from slowquant.unitary_coupled_cluster.util import UpsStructure

THETA = 0.37

NUM_ACTIVE_ORBS = 4
OCC_IDX, UNOCC_IDX = [0, 1], [2, 3]
OCC_SPIN_IDX, UNOCC_SPIN_IDX = [0, 1, 4, 5], [2, 3, 6, 7]

UPS_ANSATZE = {
    "tUPS": lambda layout: layout.create_tiled(NUM_ACTIVE_ORBS, {"n_layers": 2, "do_tups": True}),
    "QNP": lambda layout: layout.create_tiled(NUM_ACTIVE_ORBS, {"n_layers": 1, "do_qnp": True}),
    "fUCC_SD": lambda layout: layout.create_fUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["S", "D"]},
    ),
    "fUCC_GS_GD": lambda layout: layout.create_fUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["GS", "GD"]},
    ),
    "fUCC_pD_GpD": lambda layout: layout.create_fUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["pD", "GpD"]},
    ),
    "SDSfUCCSD": lambda layout: layout.create_SDSfUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["D"]},
    ),
    "SDSfUCC_GpD": lambda layout: layout.create_SDSfUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["GpD"]},
    ),
}


def exact_excitation(key: str, num_orbs: int, theta: float) -> np.ndarray:
    """Build exp(theta*(T - T^dagger)) for an excitation, via Qiskit's Jordan-Wigner mapper.

    Args:
        key: Fermionic operator string in Qiskit's format.
        num_orbs: Number of spatial orbitals.
        theta: Excitation parameter.

    Returns:
        Unitary matrix.
    """
    op = FermionicOp({key: 1.0}, num_spin_orbitals=2 * num_orbs)
    generator = op - op.adjoint()
    matrix = JordanWignerMapper().map(generator.normal_order()).to_matrix()
    return scipy.linalg.expm(theta * np.asarray(matrix))


def circuit_unitary(builder, indices: tuple[int, ...], num_orbs: int, theta: float) -> np.ndarray:
    """Build the unitary of one of the excitation circuits.

    Args:
        builder: Circuit building function.
        indices: Spin-orbital indices to pass to the builder.
        num_orbs: Number of spatial orbitals.
        theta: Excitation parameter.

    Returns:
        Unitary matrix.
    """
    parameter = Parameter("t")
    qc = QuantumCircuit(2 * num_orbs)
    qc = builder(*indices, num_orbs, qc, parameter)
    return Operator(qc.assign_parameters({parameter: theta})).data


@pytest.mark.parametrize("num_orbs", [2, 3])
def test_single_excitation_circuit(num_orbs: int) -> None:
    """Test the single excitation circuit for every spin-conserving pair.

    Args:
        num_orbs: Number of spatial orbitals.
    """
    checked = 0
    for i, k in itertools.permutations(range(2 * num_orbs), 2):
        if (i < num_orbs) != (k < num_orbs):  # spin conserving only
            continue
        if k <= i:  # the circuit requires the excitation to go upwards
            continue
        computed = circuit_unitary(_single_excitation_efficient, (k, i), num_orbs, THETA)
        reference = exact_excitation(f"+_{k} -_{i}", num_orbs, THETA)
        assert np.allclose(computed, reference, atol=1e-10), f"k={k} i={i}"
        checked += 1
    assert checked > 0


SUPPORTED_TOPOLOGIES = ("aacc", "acac", "ccaa")


def sorted_topology(k: int, l: int, i: int, j: int) -> str:
    """Describe how the creation and annihilation indices interleave when sorted.

    Args:
        k: Creation spin-orbital index.
        l: Creation spin-orbital index.
        i: Annihilation spin-orbital index.
        j: Annihilation spin-orbital index.

    Returns:
        String of "c" and "a" in ascending index order.
    """
    return "".join(kind for _, kind in sorted(((k, "c"), (l, "c"), (i, "a"), (j, "a"))))


@pytest.mark.parametrize("num_orbs", [2, 3])
def test_double_excitation_circuit(num_orbs: int) -> None:
    """Test the double excitation circuit for every supported spin-conserving quadruple.

    Args:
        num_orbs: Number of spatial orbitals.
    """
    seen = set()
    for k, l, i, j in itertools.permutations(range(2 * num_orbs), 4):
        if k >= l or i >= j:  # each pair is unordered, take one representative
            continue
        if sum(1 for p in (k, l, i, j) if p < num_orbs) % 2 != 0:
            continue
        topology = sorted_topology(k, l, i, j)
        if topology not in SUPPORTED_TOPOLOGIES:
            continue
        computed = circuit_unitary(_double_excitation_efficient, (k, l, i, j), num_orbs, THETA)
        reference = exact_excitation(f"+_{k} +_{l} -_{j} -_{i}", num_orbs, THETA)
        assert np.allclose(computed, reference, atol=1e-10), f"k={k} l={l} i={i} j={j}"
        seen.add(topology)
    # "aacc" is what interleaved ordering produced, "acac" is what blocked ordering produces for
    # a double sharing spatial orbitals between the spins. Both must be covered.
    assert {"aacc", "acac"} <= seen, seen


@pytest.mark.parametrize("num_orbs", [3])
def test_double_excitation_circuit_rejects_unsupported(num_orbs: int) -> None:
    """Test that an unsupported index ordering raises rather than building a wrong circuit.

    Args:
        num_orbs: Number of spatial orbitals.
    """
    rejected = set()
    for k, l, i, j in itertools.permutations(range(2 * num_orbs), 4):
        if k >= l or i >= j:
            continue
        if sum(1 for p in (k, l, i, j) if p < num_orbs) % 2 != 0:
            continue
        topology = sorted_topology(k, l, i, j)
        if topology in SUPPORTED_TOPOLOGIES:
            continue
        with pytest.raises(ValueError, match="ordering of creation and annihilation"):
            circuit_unitary(_double_excitation_efficient, (k, l, i, j), num_orbs, THETA)
        rejected.add(topology)
    assert rejected == {"acca", "caac", "caca"}, rejected


@pytest.mark.parametrize("name", sorted(UPS_ANSATZE))
def test_ansatz_doubles_are_supported_topologies(name: str) -> None:
    """Test that no ansatz builder emits a double the circuit cannot represent.

    Args:
        name: Key of the ansatz in UPS_ANSATZE.
    """
    layout = UpsStructure()
    UPS_ANSATZE[name](layout)
    for exc_type, exc_indices in zip(layout.excitation_operator_type, layout.excitation_indices):
        if exc_type != "double":
            continue
        i, j, a, b = exc_indices
        assert sorted_topology(a, b, i, j) in SUPPORTED_TOPOLOGIES, f"{name} {exc_indices}"
