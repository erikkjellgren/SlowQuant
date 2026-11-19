import numpy as np
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.unitary_coupled_cluster.operators import a_op_spin
from slowquant.unitary_coupled_cluster.spin_ordering import alpha_idx, beta_idx
from slowquant.unitary_coupled_cluster.util import UpsStructure


def single_excitation(
    i: int,
    a: int,
    num_orbs: int,
    qc: QuantumCircuit,
    theta: Parameter | ParameterExpression,
    mapper: FermionicMapper,
) -> QuantumCircuit:
    """Get single excitation circuit.

    Args:
        i: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.
        mapper: Fermionic to qubit mapper.

    Returns:
        Single excitation circuit.
    """
    if isinstance(mapper, JordanWignerMapper):
        qc = _single_excitation_efficient(a, i, num_orbs, qc, theta)
    else:
        qc = _single_excitation_trotter(i, a, num_orbs, qc, theta, mapper)
    return qc


def double_excitation(
    i: int,
    j: int,
    a: int,
    b: int,
    num_orbs: int,
    qc: QuantumCircuit,
    theta: Parameter | ParameterExpression,
    mapper: FermionicMapper,
) -> QuantumCircuit:
    """Get double excitation circuit.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.
        mapper: Fermionic to qubit mapper.

    Returns:
        Single excitation circuit.
    """
    if isinstance(mapper, JordanWignerMapper):
        qc = _double_excitation_efficient(a, b, i, j, num_orbs, qc, theta)
    else:
        qc = _double_excitation_trotter(i, j, a, b, num_orbs, qc, theta, mapper)
    return qc


def sa_single_excitation(
    i: int,
    a: int,
    num_orbs: int,
    qc: QuantumCircuit,
    theta: Parameter | ParameterExpression,
    mapper: FermionicMapper,
) -> QuantumCircuit:
    """Get spin-adapted single singlet excitation circuit.

    Args:
        i: Strongly occupied spatial orbital index.
        a: Weakly occupied spatial orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.
        mapper: Fermionic to qubit mapper.

    Returns:
        Spin-adpated singlet single excitation circuit.
    """
    if isinstance(mapper, JordanWignerMapper):
        qc = _sa_single_excitation_efficient(a, i, num_orbs, qc, theta)
    else:
        qc = _sa_single_excitation_trotter(i, a, num_orbs, qc, theta, mapper)
    return qc


def _jordan_wigner_sign(operators: list[tuple[int, bool]]) -> int:
    r"""Get the sign of an excitation operator written as a qubit excitation.

    Under the Jordan-Wigner mapping an excitation operator with distinct spin orbital indices
    becomes a qubit excitation times a Pauli-Z string and a sign,

    .. math::
       \hat{a}^\dagger_{k}\hat{a}^\dagger_{l}\hat{a}_{j}\hat{a}_{i}
       = s\hat{Z}_\text{string}\left|1_k1_l0_i0_j\right>\left<0_k0_l1_i1_j\right|

    The Pauli-Z string is one on the state where only the annihilated spin orbitals are
    occupied, so :math:`s` is the sign the operator picks up on that state. Each operator is
    applied right to left and contributes a minus sign per occupied spin orbital below it.

    Args:
        operators: (spin orbital index, is creation operator) in the written order.

    Returns:
        Sign of the excitation operator.
    """
    occupied = {idx for idx, dagger in operators if not dagger}
    sign = 1
    for idx, dagger in reversed(operators):
        if sum(1 for p in occupied if p < idx) % 2 == 1:
            sign *= -1
        if dagger:
            occupied.add(idx)
        else:
            occupied.remove(idx)
    return sign


def _jordan_wigner_string(low: int, high: int, qc: QuantumCircuit, uncompute: bool) -> QuantumCircuit:
    r"""Get the circuit applying a Jordan-Wigner string as a conditional Pauli-Z.

    The qubits strictly between low and high carry a Pauli-Z string. The parity of those qubits
    is collected on qubit high-1 with a nearest-neighbour CNOT ladder and then applied to qubit
    high with a CZ. When placed on both sides of a qubit excitation that flips qubit high, the
    Pauli-Z on qubit high anticommutes with the generator, so the rotation angle changes sign
    exactly when the parity is odd,

    .. math::
       \exp\left(\theta\hat{Z}_\text{string}\hat{G}\right)
       = \text{CZ}_\text{string}\exp\left(\theta\hat{G}\right)\text{CZ}_\text{string}

    #. 10.1103/PhysRevA.102.062612, Fig. 8 and Fig. 9

    Args:
        low: Lowest spin orbital index of the string.
        high: Highest spin orbital index of the string, must be flipped by the excitation.
        qc: Quantum circuit.
        uncompute: Add the second half, i.e. the CZ followed by the reversed ladder.

    Returns:
        Jordan-Wigner string circuit.
    """
    if high - low < 2:
        # No qubits between, no string.
        return qc
    if not uncompute:
        for t in range(low + 1, high - 1):
            qc.cx(t, t + 1)
        qc.cz(high - 1, high)
    else:
        qc.cz(high - 1, high)
        for t in range(high - 2, low, -1):
            qc.cx(t, t + 1)
    return qc


def _single_qubit_excitation(
    k: int, i: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Get the single qubit excitation circuit.

    Rotation between the two states where exactly one of the qubits is set,

    .. math::
       \boldsymbol{U}\left|1_i0_k\right> = \cos\left(\theta\right)\left|1_i0_k\right>
       + \sin\left(\theta\right)\left|0_i1_k\right>

    and the other two basis states are unchanged.

    #. 10.1103/PhysRevA.102.062612, Fig. 3

    Args:
        k: Qubit that is set after the excitation.
        i: Qubit that is set before the excitation.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Single qubit excitation circuit.
    """
    qc.rz(np.pi / 2, i)
    qc.rx(np.pi / 2, i)
    qc.rx(np.pi / 2, k)
    qc.cx(i, k)
    qc.rx(theta, i)
    qc.rz(theta, k)
    qc.cx(i, k)
    qc.rx(-np.pi / 2, k)
    qc.rx(-np.pi / 2, i)
    qc.rz(-np.pi / 2, i)
    return qc


def _double_qubit_excitation(
    k: int, l: int, i: int, j: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Get the 12-CNOT double qubit excitation circuit.

    Rotation between the state where qubits i and j are set and the one where k and l are,

    .. math::
       \boldsymbol{U}\left|1_i1_j0_k0_l\right> = \cos\left(\theta\right)\left|1_i1_j0_k0_l\right>
       + \sin\left(\theta\right)\left|0_i0_j1_k1_l\right>

    and the other fourteen basis states are unchanged.

    #. 10.48550/arXiv.2608.11733, Eq. 2 and Fig. 5

    Args:
        k: Qubit that is set after the excitation.
        l: Qubit that is set after the excitation.
        i: Qubit that is set before the excitation.
        j: Qubit that is set before the excitation.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Double qubit excitation circuit.
    """
    # Fig. 5 names its qubits (q_i, q_j, q_k, q_l) and rotates |0011> into |1100>,
    # so its first two qubits are the ones set after the excitation, here (k, l, i, j).
    qc.cx(k, i)
    qc.cx(l, j)
    qc.h(k)
    qc.sdg(i)
    qc.cx(k, l)
    qc.sxdg(k)
    qc.sxdg(l)
    qc.rz(theta / 4, k)
    qc.rz(-theta / 4, l)
    qc.cx(j, l)
    qc.rz(theta / 4, l)
    qc.cx(j, k)
    qc.cx(i, l)
    qc.rz(-theta / 4, k)
    qc.rz(-theta / 4, l)
    qc.cx(i, k)
    qc.cx(j, l)
    qc.rz(theta / 4, k)
    qc.cx(j, k)
    qc.rz(-np.pi / 2 - theta / 4, k)
    qc.rz(theta / 4, l)
    qc.h(k)
    qc.cx(k, l)
    qc.sx(k)
    qc.sx(l)
    qc.cx(k, i)
    qc.cx(l, j)
    qc.rz(np.pi / 2, i)
    return qc


def _single_excitation_efficient(
    k: int, i: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Exact circuit for single excitation.

    Implementation of the following operator,

    .. math::
       \boldsymbol{U} = \exp\left(\theta\left(\hat{a}^\dagger_k\hat{a}_i - \hat{a}^\dagger_i\hat{a}_k\right)\right)

    for any two distinct spin orbitals, in either order and of either spin. Under the
    Jordan-Wigner mapping the operator is a single qubit excitation on qubits i and k times the
    Pauli-Z string between them, with a plus sign for either order. The cost is
    :math:`2+2m` two-qubit gates for :math:`m` qubits between i and k.

    #. 10.1103/PhysRevA.102.062612, Fig. 3 and Fig. 8
    #. 10.1038/s42005-021-00730-0, Fig. 1

    Args:
        k: Spin orbital index of the creation operator.
        i: Spin orbital index of the annihilation operator.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Single excitation circuit.
    """
    if k == i:
        raise ValueError(f"Single excitation needs two distinct spin orbitals, got k={k}, i={i}")
    low, high = sorted((k, i))
    qc = _jordan_wigner_string(low, high, qc, uncompute=False)
    qc = _single_qubit_excitation(k, i, qc, theta)
    qc = _jordan_wigner_string(low, high, qc, uncompute=True)
    return qc


def _double_excitation_efficient(
    k: int, l: int, i: int, j: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Exact circuit for double excitation.

    Implementation of the following operator,

    .. math::
       \boldsymbol{U} = \exp\left(\theta\left(\hat{a}^\dagger_k\hat{a}^\dagger_l\hat{a}_j\hat{a}_i
       - \hat{a}^\dagger_i\hat{a}^\dagger_j\hat{a}_l\hat{a}_k\right)\right)

    for any four distinct spin orbitals, in any order and of any spin. Under the Jordan-Wigner
    mapping the operator is a double qubit excitation times a sign and a Pauli-Z string. Writing
    the sorted indices as :math:`p_0<p_1<p_2<p_3`, the string covers the qubits strictly between
    :math:`p_0` and :math:`p_1` and strictly between :math:`p_2` and :math:`p_3`. The cost is
    :math:`12+2m` two-qubit gates for :math:`m` qubits in the string.

    #. 10.48550/arXiv.2608.11733, Fig. 5 (12-CNOT double qubit excitation)
    #. 10.1103/PhysRevA.102.062612, Fig. 9 (Jordan-Wigner string)

    Args:
        k: Spin orbital index of a creation operator.
        l: Spin orbital index of a creation operator.
        i: Spin orbital index of an annihilation operator.
        j: Spin orbital index of an annihilation operator.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Double excitation circuit.
    """
    if len({k, l, i, j}) != 4:
        raise ValueError(
            f"Double excitation needs four distinct spin orbitals, got k={k}, l={l}, i={i}, j={j}"
        )
    sign = _jordan_wigner_sign([(k, True), (l, True), (j, False), (i, False)])
    p0, p1, p2, p3 = sorted((k, l, i, j))
    qc = _jordan_wigner_string(p0, p1, qc, uncompute=False)
    qc = _jordan_wigner_string(p2, p3, qc, uncompute=False)
    qc = _double_qubit_excitation(k, l, i, j, qc, sign * theta)
    qc = _jordan_wigner_string(p2, p3, qc, uncompute=True)
    qc = _jordan_wigner_string(p0, p1, qc, uncompute=True)
    return qc


def _sa_single_excitation_efficient(
    k: int, i: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Exact circuit for spin-adapted singlet single excitation.

    Implementation of the following operator,

    .. math::
       \boldsymbol{U} = \exp\left(\frac{\theta}{\sqrt{2}}\left(\hat{E}_{ki} - \hat{E}_{ik}^\dagger\right)\right)

    Implemented as,

    .. math::
       \boldsymbol{U} = \exp\left(\frac{\theta}{\sqrt{2}}\hat{a}^\dagger_{k,\alpha}\hat{a}_{i,\alpha}\right)
                        \exp\left(\frac{\theta}{\sqrt{2}}\hat{a}^\dagger_{k,\beta}\hat{a}_{i,\beta}\right)

    Args:
        k: Weakly occupied spatial orbital index.
        i: Strongly occupied spatial orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Single singlet spin-adapted excitation circuit.
    """
    qc = _single_excitation_efficient(alpha_idx(k, num_orbs), alpha_idx(i, num_orbs), num_orbs, qc, theta)
    qc = _single_excitation_efficient(beta_idx(k, num_orbs), beta_idx(i, num_orbs), num_orbs, qc, theta)
    return qc


def _single_excitation_trotter(
    i: int,
    a: int,
    num_orbs: int,
    qc: QuantumCircuit,
    theta: Parameter | ParameterExpression,
    mapper: FermionicMapper,
) -> QuantumCircuit:
    """Get single excitation as a trotterized fermionic operator.

    The Pauli string from the mapped fermionic operator are sorted
    lexicographically to make the circuit shorter from gate cancelation.

    Args:
        i: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.
        mapper: Fermionic to qubit mapper.

    Returns:
        Trotterized fermionic single excitation circuit.
    """
    num_spin_orbs = 2 * num_orbs
    op = a_op_spin(a, True) * a_op_spin(i, False)
    T = op - op.dagger
    op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(), num_spin_orbs))
    ops = np.array([str(pauli) for pauli in op_mapped.paulis])
    factors = np.array([(1.0j * x).real for x in op_mapped.coeffs])
    sort_idx = np.argsort(ops)
    ops = ops[sort_idx]
    factors = factors[sort_idx]
    num_qubits = qc.num_qubits
    for pauli, fac in zip(ops, factors):
        qc.append(
            PauliEvolutionGate(Pauli(pauli), fac * theta),
            np.linspace(0, num_qubits - 1, num_qubits, dtype=int).tolist(),
        )
    return qc


def _double_excitation_trotter(
    i: int,
    j: int,
    a: int,
    b: int,
    num_orbs: int,
    qc: QuantumCircuit,
    theta: Parameter | ParameterExpression,
    mapper: FermionicMapper,
) -> QuantumCircuit:
    """Get double excitation as a trotterized fermionic operator.

    The Pauli string from the mapped fermionic operator are sorted
    lexicographically to make the circuit shorter from gate cancellation.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.
        mapper: Fermionic to qubit mapper.

    Returns:
        Trotterized fermionic double excitation circuit.
    """
    num_spin_orbs = 2 * num_orbs
    ops = []
    factors = []
    op = a_op_spin(a, True) * a_op_spin(b, True) * a_op_spin(j, False) * a_op_spin(i, False)
    T = op - op.dagger
    op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(), num_spin_orbs))
    ops = np.array([str(pauli) for pauli in op_mapped.paulis])
    factors = np.array([(1.0j * x).real for x in op_mapped.coeffs])
    sort_idx = np.argsort(ops)
    ops = ops[sort_idx]
    factors = factors[sort_idx]
    num_qubits = qc.num_qubits
    for pauli, fac in zip(ops, factors):
        qc.append(
            PauliEvolutionGate(Pauli(pauli), fac * theta),
            np.linspace(0, num_qubits - 1, num_qubits, dtype=int).tolist(),
        )
    return qc


def _sa_single_excitation_trotter(
    i: int,
    a: int,
    num_orbs: int,
    qc: QuantumCircuit,
    theta: Parameter | ParameterExpression,
    mapper: FermionicMapper,
) -> QuantumCircuit:
    """Get spin-adapted singlet single excitation as a trotterized fermionic operator.

    Args:
        i: Strongly occupied spatial orbital index.
        a: Weakly occupied spatial orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.
        mapper: Fermionic to qubit mapper.

    Returns:
        Trotterized fermionic spin-adapted singlet single excitation circuit.
    """
    qc = _single_excitation_trotter(
        alpha_idx(i, num_orbs), alpha_idx(a, num_orbs), num_orbs, qc, theta, mapper
    )
    qc = _single_excitation_trotter(beta_idx(i, num_orbs), beta_idx(a, num_orbs), num_orbs, qc, theta, mapper)
    return qc


def upslayout2circuit(
    upslayout: UpsStructure, num_orbs: int, mapper: FermionicMapper
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Convert a unitary product state wavefunction to a circuit representation.

    Args:
        upslayout: Unitary product state layout.
        num_orbs: Number of spatial orbitals.
        mapper: Fermionic to qubit mapper.

    Returns:
        Qiskit circuit representation of wavefunction.
    """
    grad_param_R = {}
    idx = 0
    qc = HartreeFock(num_orbs, (0, 0), mapper)  # empty circuit with qubit number based on mapper
    for exc_type, exc_indices in zip(upslayout.excitation_operator_type, upslayout.excitation_indices):
        if exc_type == "single":
            (i, a) = exc_indices
            qc = single_excitation(i, a, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            qc = double_excitation(i, j, a, b, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
            grad_param_R[f"p{idx:09d}"] = 2
            idx += 1
        elif exc_type == "sa_single":
            (i, a) = exc_indices
            qc = sa_single_excitation(i, a, num_orbs, qc, Parameter(f"p{idx:09d}"), mapper)
            grad_param_R[f"p{idx:09d}"] = 4
            idx += 1
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
    return qc, grad_param_R
