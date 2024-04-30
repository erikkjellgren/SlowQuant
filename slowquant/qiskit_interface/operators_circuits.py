import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit


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
