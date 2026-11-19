"""Test the hand-written excitation circuits against exact matrix exponentials.

The efficient single and double excitation circuits are a qubit excitation, a fermionic sign
worked out from the indices, and a parity ladder for each Jordan-Wigner string. The strings run
between the indices in sorted order, so every ordering of the indices, every way the creation
and annihilation indices interleave, and every spin combination is a separate case, and all of
them are checked.

These tests compare the circuit unitary with the exponential of the Jordan-Wigner mapped
generator, built with Qiskit's own mapper, so they check the circuits rather than restate them.
"""

import functools
import itertools

import numpy as np
import pytest
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.operators_circuits import (
    _double_excitation_efficient,
    _single_excitation_efficient,
    double_excitation,
    single_excitation,
    upslayout2circuit,
)
from slowquant.unitary_coupled_cluster.spin_ordering import alpha_idx, beta_idx
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


def exact_excitation(
    key: str, num_orbs: int, theta: float, mapper: FermionicMapper | None = None
) -> np.ndarray:
    """Build exp(theta*(T - T^dagger)) for an excitation, via one of Qiskit's mappers.

    Args:
        key: Fermionic operator string in Qiskit's format.
        num_orbs: Number of spatial orbitals.
        theta: Excitation parameter.
        mapper: Fermionic to qubit mapper, Jordan-Wigner if None.

    Returns:
        Unitary matrix.
    """
    if mapper is None:
        mapper = JordanWignerMapper()
    op = FermionicOp({key: 1.0}, num_spin_orbitals=2 * num_orbs)
    generator = op - op.adjoint()
    matrix = mapper.map(generator.normal_order()).to_matrix()
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


ALL_TOPOLOGIES = {"aacc", "acac", "acca", "caac", "caca", "ccaa"}


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


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    """Count the two-qubit gates in a circuit.

    Args:
        qc: Quantum circuit.

    Returns:
        Number of two-qubit gates.
    """
    return sum(1 for instruction in qc.data if instruction.operation.num_qubits == 2)


@pytest.mark.parametrize("num_orbs", [2, 3, 4])
def test_single_excitation_circuit(num_orbs: int) -> None:
    """Test the single excitation circuit for every ordered pair of spin orbitals.

    Both orders of the two indices are included, and so are the pairs that change the spin, as
    the circuit does not restrict either.

    Args:
        num_orbs: Number of spatial orbitals.
    """
    for k, i in itertools.permutations(range(2 * num_orbs), 2):
        computed = circuit_unitary(_single_excitation_efficient, (k, i), num_orbs, THETA)
        reference = exact_excitation(f"+_{k} -_{i}", num_orbs, THETA)
        assert np.allclose(computed, reference, atol=1e-10), f"k={k} i={i}"


@pytest.mark.parametrize(("num_orbs", "both_pair_orders"), [(2, True), (3, True), (4, False)])
def test_double_excitation_circuit(num_orbs: int, both_pair_orders: bool) -> None:
    """Test the double excitation circuit for every ordered quadruple of spin orbitals.

    Every order of the four indices is included, which covers every way the creation and
    annihilation indices interleave and both orders within each pair, so the sign the circuit
    works out from the indices is checked, and so are the doubles that change the spin. Four
    spatial orbitals are the first size with three qubits inside a Jordan-Wigner string, and to
    keep the test fast only one order within each pair is taken there.

    Args:
        num_orbs: Number of spatial orbitals.
        both_pair_orders: Include both orders of the indices within each pair.
    """
    seen = set()
    for k, l, i, j in itertools.permutations(range(2 * num_orbs), 4):
        if not both_pair_orders and (k > l or i > j):
            continue
        computed = circuit_unitary(_double_excitation_efficient, (k, l, i, j), num_orbs, THETA)
        reference = exact_excitation(f"+_{k} +_{l} -_{j} -_{i}", num_orbs, THETA)
        assert np.allclose(computed, reference, atol=1e-10), f"k={k} l={l} i={i} j={j}"
        seen.add(sorted_topology(k, l, i, j))
    assert seen == ALL_TOPOLOGIES, seen


def test_excitation_circuit_gate_count() -> None:
    """Test the two-qubit gate count of the excitation circuits.

    A single costs 2 and a double 12 for the qubit excitation, plus 2 for each qubit inside a
    Jordan-Wigner string.
    """
    num_orbs = 4
    parameter = Parameter("t")
    for k, i in itertools.permutations(range(2 * num_orbs), 2):
        qc = _single_excitation_efficient(k, i, num_orbs, QuantumCircuit(2 * num_orbs), parameter)
        num_string = abs(k - i) - 1
        assert two_qubit_gate_count(qc) == 2 + 2 * num_string, f"k={k} i={i}"
    for k, l, i, j in itertools.permutations(range(2 * num_orbs), 4):
        qc = _double_excitation_efficient(k, l, i, j, num_orbs, QuantumCircuit(2 * num_orbs), parameter)
        p0, p1, p2, p3 = sorted((k, l, i, j))
        num_string = (p1 - p0 - 1) + (p3 - p2 - 1)
        assert two_qubit_gate_count(qc) == 12 + 2 * num_string, f"k={k} l={l} i={i} j={j}"


def test_excitation_circuits_reject_repeated_indices() -> None:
    """Test that an excitation with a repeated spin orbital raises.

    Such an operator is not an excitation, it is zero or a number operator times an excitation
    of lower rank.
    """
    parameter = Parameter("t")
    with pytest.raises(ValueError, match="distinct"):
        _single_excitation_efficient(1, 1, 2, QuantumCircuit(4), parameter)
    with pytest.raises(ValueError, match="distinct"):
        _double_excitation_efficient(0, 1, 1, 2, 2, QuantumCircuit(4), parameter)


@pytest.mark.parametrize(
    ("name", "mapper_name"),
    [
        pytest.param(
            name,
            mapper_name,
            marks=pytest.mark.xfail(
                raises=ValueError,
                strict=True,
                reason="iterate_t2_generalized yields doubles with i == j or a == b, which are zero; "
                "the Jordan-Wigner circuits reject them, the Rustiq path builds an identity",
            ),
        )
        if name == "fUCC_GS_GD" and mapper_name == "JW"
        else (name, mapper_name)
        for name in sorted(UPS_ANSATZE)
        for mapper_name in ["JW", "Parity"]
    ],
)
def test_ansatz_circuit(name: str, mapper_name: str) -> None:
    """Test that an ansatz circuit is the ordered product of its exact excitation unitaries.

    This also checks that upslayout2circuit hands the indices of each excitation to the
    circuits in the right roles, for the hand-written Jordan-Wigner circuits and for the
    Rustiq-synthesized circuits used for every other mapper.

    Args:
        name: Key of the ansatz in UPS_ANSATZE.
        mapper_name: Fermionic to qubit mapper to build the circuit with.
    """
    mapper = JordanWignerMapper() if mapper_name == "JW" else ParityMapper()
    layout = UpsStructure()
    UPS_ANSATZE[name](layout)
    qc, _ = upslayout2circuit(layout, NUM_ACTIVE_ORBS, mapper)
    # Parameters are named p000000000, p000000001, ..., so their sorted order is the layout order.
    thetas = np.random.default_rng(1234).uniform(-np.pi, np.pi, qc.num_parameters)
    computed = Operator(qc.assign_parameters(dict(zip(qc.parameters, thetas)))).data
    reference = np.eye(2 ** (2 * NUM_ACTIVE_ORBS), dtype=complex)
    for theta, exc_type, exc_indices in zip(
        thetas, layout.excitation_operator_type, layout.excitation_indices
    ):
        if exc_type == "single":
            i, a = exc_indices
            keys = [f"+_{a} -_{i}"]
        elif exc_type == "double":
            i, j, a, b = exc_indices
            keys = [f"+_{a} +_{b} -_{j} -_{i}"]
        elif exc_type == "sa_single":
            i, a = exc_indices
            keys = [
                f"+_{alpha_idx(a, NUM_ACTIVE_ORBS)} -_{alpha_idx(i, NUM_ACTIVE_ORBS)}",
                f"+_{beta_idx(a, NUM_ACTIVE_ORBS)} -_{beta_idx(i, NUM_ACTIVE_ORBS)}",
            ]
        else:
            raise ValueError(f"Unexpected excitation type {exc_type}")
        for key in keys:
            reference = exact_excitation(key, NUM_ACTIVE_ORBS, theta, mapper) @ reference
    assert np.allclose(computed, reference, atol=1e-10), name


@pytest.mark.parametrize("num_orbs", [2, 3])
def test_single_excitation_circuit_parity(num_orbs: int) -> None:
    """Test the Rustiq-synthesized single excitation for every ordered pair of spin orbitals.

    Args:
        num_orbs: Number of spatial orbitals.
    """
    mapper = ParityMapper()
    builder = functools.partial(single_excitation, mapper=mapper)
    for k, i in itertools.permutations(range(2 * num_orbs), 2):
        computed = circuit_unitary(builder, (i, k), num_orbs, THETA)
        reference = exact_excitation(f"+_{k} -_{i}", num_orbs, THETA, mapper)
        assert np.allclose(computed, reference, atol=1e-10), f"k={k} i={i}"


@pytest.mark.parametrize("num_orbs", [2, 3])
def test_double_excitation_circuit_parity(num_orbs: int) -> None:
    """Test the Rustiq-synthesized double excitation for every ordered quadruple of spin orbitals.

    Args:
        num_orbs: Number of spatial orbitals.
    """
    mapper = ParityMapper()
    builder = functools.partial(double_excitation, mapper=mapper)
    for k, l, i, j in itertools.permutations(range(2 * num_orbs), 4):
        computed = circuit_unitary(builder, (i, j, k, l), num_orbs, THETA)
        reference = exact_excitation(f"+_{k} +_{l} -_{j} -_{i}", num_orbs, THETA, mapper)
        assert np.allclose(computed, reference, atol=1e-10), f"k={k} l={l} i={i} j={j}"
