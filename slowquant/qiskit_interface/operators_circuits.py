import numpy as np
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit

from slowquant.qiskit_interface.util import f2q


def tups_single(p: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter) -> QuantumCircuit:
    r"""Spin-adapted single excitation as used in the tUPS ansatz.

    Exact implementation of,

    .. math::
        \boldsymbol{U} = \exp\left(\theta\hat{\kappa}_{p,p+1}\right)

    With, :math:`\hat{\kappa}_{p,p+1}=\hat{E}_{p,p+1}-\hat{E}_{p+1,p}`

    .. code-block:

             ┌─────────┐┌─────────┐     ┌───────┐     ┌──────────┐┌──────────┐
        q_0: ┤ Rz(π/2) ├┤ Rx(π/2) ├──■──┤ Rx(θ) ├──■──┤ Rx(-π/2) ├┤ Rz(-π/2) ├
             ├─────────┤└─────────┘┌─┴─┐├───────┤┌─┴─┐├──────────┤└──────────┘
        q_1: ┤ Rx(π/2) ├───────────┤ X ├┤ Rz(θ) ├┤ X ├┤ Rx(-π/2) ├────────────
             ├─────────┤┌─────────┐└───┘├───────┤└───┘├──────────┤┌──────────┐
        q_2: ┤ Rz(π/2) ├┤ Rx(π/2) ├──■──┤ Rx(θ) ├──■──┤ Rx(-π/2) ├┤ Rz(-π/2) ├
             ├─────────┤└─────────┘┌─┴─┐├───────┤┌─┴─┐├──────────┤└──────────┘
        q_3: ┤ Rx(π/2) ├───────────┤ X ├┤ Rz(θ) ├┤ X ├┤ Rx(-π/2) ├────────────
             └─────────┘           └───┘└───────┘└───┘└──────────┘

    #. 10.48550/arXiv.2312.09761
    #. 10.1038/s42005-021-00730-0, Fig. 1

    Args:
        p: Occupied spatial index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Modified quantum circuit.
    """
    pa = p
    pb = p + num_orbs
    qc.rz(np.pi / 2, pa)
    qc.rx(np.pi / 2, pa)
    qc.rx(np.pi / 2, pa + 1)
    qc.cx(pa, pa + 1)
    qc.rx(theta, pa)
    qc.rz(theta, pa + 1)
    qc.cx(pa, pa + 1)
    qc.rx(-np.pi / 2, pa + 1)
    qc.rx(-np.pi / 2, pa)
    qc.rz(-np.pi / 2, pa)
    qc.rz(np.pi / 2, pb)
    qc.rx(np.pi / 2, pb)
    qc.rx(np.pi / 2, pb + 1)
    qc.cx(pb, pb + 1)
    qc.rx(theta, pb)
    qc.rz(theta, pb + 1)
    qc.cx(pb, pb + 1)
    qc.rx(-np.pi / 2, pb + 1)
    qc.rx(-np.pi / 2, pb)
    qc.rz(-np.pi / 2, pb)
    return qc


def tups_double(p: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter) -> QuantumCircuit:
    r"""Spin-adapted pair double excitation as used in the tUPS ansatz.

    Exact implementation of,

    .. math::
        \boldsymbol{U} = \exp\left(\frac{\theta}{2}\hat{\kappa}_{p,p+1}\right)

    With, :math:`\hat{\kappa}_{p,p+1}=\hat{E}_{p,p+1}^2-\hat{E}_{p+1,p}^2`

    .. code-block::

                            ┌─────────────┐     ┌─────────┐     ┌─────────────┐     »
        q_0: ──■─────────■──┤ Ry(-0.25*θ) ├──■──┤ Ry(θ/4) ├──■──┤ Ry(-0.25*θ) ├──■──»
               │       ┌─┴─┐└────┬───┬────┘  │  └─────────┘  │  └─────────────┘  │  »
        q_1: ──┼────■──┤ X ├─────┤ H ├───────┼───────────────┼───────────────────┼──»
             ┌─┴─┐  │  ├───┤     ├───┤     ┌─┴─┐             │                 ┌─┴─┐»
        q_2: ┤ X ├──┼──┤ X ├─────┤ H ├─────┤ X ├─────────────┼─────────────────┤ X ├»
             └───┘┌─┴─┐├───┤     ├───┤     └───┘           ┌─┴─┐               └───┘»
        q_3: ─────┤ X ├┤ X ├─────┤ H ├─────────────────────┤ X ├────────────────────»
                  └───┘└───┘     └───┘                     └───┘                    »
        «     ┌─────────┐     ┌─────────────┐     ┌─────────┐               »
        «q_0: ┤ Ry(θ/4) ├──■──┤ Ry(-0.25*θ) ├──■──┤ Ry(θ/4) ├────────────■──»
        «     └─────────┘┌─┴─┐└────┬───┬────┘  │  ├─────────┤┌────────┐  │  »
        «q_1: ───────────┤ X ├─────┤ H ├───────┼──┤ Ry(π/2) ├┤ P(π/2) ├──┼──»
        «                └───┘     └───┘     ┌─┴─┐└─────────┘└────────┘  │  »
        «q_2: ───────────────────────────────┤ X ├───────────────────────┼──»
        «                                    └───┘                     ┌─┴─┐»
        «q_3: ─────────────────────────────────────────────────────────┤ X ├»
        «                                                              └───┘»
        «     ┌─────────────┐     ┌─────────┐      ┌────────┐
        «q_0: ┤ Ry(-0.25*θ) ├──■──┤ Ry(θ/4) ├──■───┤ P(π/2) ├──────────────■───────
        «     └─────────────┘  │  └─────────┘┌─┴─┐┌┴────────┤┌──────────┐  │
        «q_1: ─────────────────┼─────────────┤ X ├┤ P(-π/2) ├┤ Ry(-π/2) ├──┼────■──
        «                    ┌─┴─┐   ┌───┐   ├───┤└─────────┘└──────────┘┌─┴─┐  │
        «q_2: ───────────────┤ X ├───┤ H ├───┤ X ├───────────────────────┤ X ├──┼──
        «          ┌───┐     ├───┤   └───┘   └───┘                       └───┘┌─┴─┐
        «q_3: ─────┤ H ├─────┤ X ├────────────────────────────────────────────┤ X ├
        «          └───┘     └───┘                                            └───┘

    #. 10.48550/arXiv.2312.09761
    #. 10.1103/PhysRevA.102.062612, Fig. 7
    #. 10.1038/s42005-021-00730-0, Fig. 2

    Args:
        p: Occupied spatial index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Modified quantum circuit.
    """
    l = p
    j = p + 1
    k = p + num_orbs
    i = p + num_orbs + 1
    qc.cx(l, k)
    qc.cx(j, i)
    qc.cx(l, j)
    qc.x(k)
    qc.x(i)
    qc.ry(-theta / 4, l)
    qc.h(k)
    qc.cx(l, k)
    qc.ry(theta / 4, l)
    qc.h(i)
    qc.cx(l, i)
    qc.ry(-theta / 4, l)
    qc.cx(l, k)
    qc.ry(theta / 4, l)
    qc.h(j)
    qc.cx(l, j)
    qc.ry(-theta / 4, l)
    qc.cx(l, k)
    qc.ry(theta / 4, l)
    qc.cx(l, i)
    qc.ry(-theta / 4, l)
    qc.h(i)
    qc.cx(l, k)
    qc.ry(theta / 4, l)
    qc.h(k)
    qc.h(j)
    qc.ry(np.pi / 2, j)
    qc.p(np.pi / 2, j)
    qc.cx(l, j)
    qc.p(np.pi / 2, l)
    qc.p(-np.pi / 2, j)
    qc.ry(-np.pi / 2, j)
    qc.x(k)
    qc.x(i)
    qc.cx(j, i)
    qc.cx(l, k)
    return qc


def single_excitation(
    k: int, i: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Exact circuit for single excitation.

    Implementation of the following operator,

    .. math::
       \boldsymbol{U} = \exp\left(\theta\hat{a}^\dagger_k\hat{a}_i\right)

    #. 10.1103/PhysRevA.102.062612, Fig. 3 and Fig. 8

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


def double_excitation(
    k: int, l: int, i: int, j: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter
) -> QuantumCircuit:
    r"""Exact circuit for double excitation.

    Implementation of the following operator,

    .. math::
       \boldsymbol{U} = \exp\left(\theta\hat{a}^\dagger_k\hat{a}_i\hat{a}^\dagger_l\hat{a}_j\right)

    #. 10.1103/PhysRevA.102.062612, Fig. 6 and Fig. 9

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


def single_sa_excitation(
    k: int, i: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter
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
    qc = single_excitation(2 * k, 2 * i, num_orbs, qc, theta)
    qc = single_excitation(2 * k + 1, 2 * i + 1, num_orbs, qc, theta)
    return qc
