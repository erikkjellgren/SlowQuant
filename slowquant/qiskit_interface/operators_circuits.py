import numpy as np
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.util import f2q
from slowquant.unitary_coupled_cluster.operators import anni_spin


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
        qc = single_excitation_efficient(a, i, num_orbs, qc, theta)
    else:
        qc = single_excitation_trotter(i, a, num_orbs, qc, theta, mapper)
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
        qc = double_excitation_efficient(a, b, i, j, num_orbs, qc, theta)
    else:
        qc = double_excitation_trotter(i, j, a, b, num_orbs, qc, theta, mapper)
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
        qc = sa_single_excitation_efficient(a, i, num_orbs, qc, theta)
    else:
        qc = sa_single_excitation_trotter(i, a, num_orbs, qc, theta, mapper)
    return qc


def single_excitation_efficient(
    k: int, i: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Exact circuit for single excitation.

    Implementation of the following operator,

    .. math::
       \boldsymbol{U} = \exp\left(\theta\hat{a}^\dagger_k\hat{a}_i\right)

    #. 10.1103/PhysRevA.102.062612, Fig. 3 and Fig. 8
    #. 10.1038/s42005-021-00730-0, Fig. 1

    Args:
        k: Weakly occupied spin orbital index.
        i: Strongly occupied spin orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Single excitation circuit.
    """
    k = f2q(k, num_orbs)
    i = f2q(i, num_orbs)
    if k <= i:
        raise ValueError(f"k={k}, must be larger than i={i}")
    if k - 1 == i:
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
    else:
        qc.cx(k, i)
        for t in range(k - 2, i, -1):
            qc.cx(t + 1, t)
        qc.cz(i + 1, k)
        qc.ry(theta, k)
        qc.cx(i, k)
        qc.ry(-theta, k)
        qc.cx(i, k)
        qc.cz(i + 1, k)
        for t in range(i + 1, k - 1):
            qc.cx(t + 1, t)
        qc.cx(k, i)
    return qc


def double_excitation_efficient(
    k: int, l: int, i: int, j: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Exact circuit for double excitation.

    Implementation of the following operator,

    .. math::
       \boldsymbol{U} = \exp\left(\theta\hat{a}^\dagger_k\hat{a}_i\hat{a}^\dagger_l\hat{a}_j\right)

    #. 10.1103/PhysRevA.102.062612, Fig. 6, Fig. 7, and, Fig. 9
    #. 10.1038/s42005-021-00730-0, Fig. 2

    Args:
        k: Weakly occupied spin orbital index.
        l: Weakly occupied spin orbital index.
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Double excitation circuit.
    """
    if k < i or k < j:
        raise ValueError(f"Operator only implemented for k, {k}, larger than i, {i}, and j, {j}")
    if l < i or l < j:
        raise ValueError(f"Operator only implemented for l, {l}, larger than i, {i}, and j, {j}")
    n_alpha = 0
    n_beta = 0
    if i % 2 == 0:
        n_alpha += 1
    else:
        n_beta += 1
    if j % 2 == 0:
        n_alpha += 1
    else:
        n_beta += 1
    if k % 2 == 0:
        n_alpha += 1
    else:
        n_beta += 1
    if l % 2 == 0:
        n_alpha += 1
    else:
        n_beta += 1
    if n_alpha % 2 != 0 or n_beta % 2 != 0:
        raise ValueError("Operator only implemented for spin conserving operators.")
    fac = 1
    if k % 2 == l % 2 and k % 2 == 0 and i % 2 != 0:
        fac *= -1
    k = f2q(k, num_orbs)
    l = f2q(l, num_orbs)
    i = f2q(i, num_orbs)
    j = f2q(j, num_orbs)
    if k > l:
        l, k = k, l
        fac *= -1
    if i > j:
        j, i = i, j
        fac *= -1
    if l < j:
        l, j = j, l
        fac *= -1
    if k < i:
        k, i = i, k
        fac *= -1
    # cnot ladder is easier to implement if the indices are sorted.
    i_z, k_z, j_z, l_z = np.sort((k, l, i, j))
    theta = 2 * theta * fac

    qc.cx(l, k)
    qc.cx(j, i)
    qc.cx(l, j)

    if l_z != j_z + 1:
        for t in range(i_z + 1, k_z - 1):
            qc.cx(t, t + 1)
        if i_z + 1 != k_z:  # and j+1 != k and k-1 != j+1:
            qc.cx(k_z - 1, j_z + 1)
        # if j+1 != k:
        for t in range(j_z + 1, l_z - 1):
            qc.cx(t, t + 1)
        qc.cz(l_z, l_z - 1)
    elif i_z != k_z - 1:
        for t in range(i_z + 1, k_z - 1):
            qc.cx(t, t + 1)
        qc.cz(l_z, k_z - 1)
    qc.x(k)
    qc.x(i)

    qc.ry(theta / 8, l)
    qc.h(k)
    qc.cx(l, k)
    qc.ry(-theta / 8, l)
    qc.h(i)
    qc.cx(l, i)
    qc.ry(theta / 8, l)
    qc.cx(l, k)
    qc.ry(-theta / 8, l)
    qc.h(j)
    qc.cx(l, j)
    qc.ry(theta / 8, l)
    qc.cx(l, k)
    qc.ry(-theta / 8, l)
    qc.cx(l, i)
    qc.ry(theta / 8, l)
    qc.h(i)
    qc.cx(l, k)
    qc.ry(-theta / 8, l)
    qc.h(k)
    qc.cx(l, j)
    qc.h(j)

    qc.x(k)
    qc.x(i)
    if l_z != j_z + 1:
        qc.cz(l_z, l_z - 1)
        for t in range(l_z - 1, j_z + 1, -1):
            qc.cx(t - 1, t)
        if i_z + 1 != k_z:
            qc.cx(k_z - 1, j_z + 1)
        for t in range(k_z - 1, i_z + 1, -1):
            qc.cx(t - 1, t)
    elif i_z != k_z - 1:
        qc.cz(l_z, k_z - 1)
        for t in range(k_z - 1, i_z + 1, -1):
            qc.cx(t - 1, t)
    qc.cx(l, j)
    qc.cx(l, k)
    qc.cx(j, i)
    return qc


def sa_single_excitation_efficient(
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
    # qc = single_excitation(2 * k, 2 * i, num_orbs, qc, 2 ** (-1 / 2) * theta)
    # qc = single_excitation(2 * k + 1, 2 * i + 1, num_orbs, qc, 2 ** (-1 / 2) * theta)
    qc = single_excitation_efficient(2 * k, 2 * i, num_orbs, qc, theta)
    qc = single_excitation_efficient(2 * k + 1, 2 * i + 1, num_orbs, qc, theta)
    return qc


def single_excitation_trotter(i, a, num_orbs, qc, theta, mapper) -> QuantumCircuit:
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
    op = anni_spin(a, True) * anni_spin(i, False)
    T = op - op.dagger
    op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs))
    ops = np.array([str(pauli) for pauli in op_mapped.paulis])
    factors = np.array([(-1.0j * x).real for x in op_mapped.coeffs])
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


def double_excitation_trotter(i, j, a, b, num_orbs, qc, theta, mapper) -> QuantumCircuit:
    """Get double excitation as a trotterized fermionic operator.

    The Pauli string from the mapped fermionic operator are sorted
    lexicographically to make the circuit shorter from gate cancelation.

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
    op = anni_spin(a, True) * anni_spin(b, True) * anni_spin(j, False) * anni_spin(i, False)
    T = op - op.dagger
    op_mapped = mapper.map(FermionicOp(T.get_qiskit_form(num_orbs), num_spin_orbs))
    ops = np.array([str(pauli) for pauli in op_mapped.paulis])
    factors = np.array([(-1.0j * x).real for x in op_mapped.coeffs])
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


def sa_single_excitation_trotter(
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
    qc = single_excitation_trotter(2 * i, 2 * a, num_orbs, qc, theta, mapper)
    qc = single_excitation_trotter(2 * i + 1, 2 * a + 1, num_orbs, qc, theta, mapper)
    return qc
