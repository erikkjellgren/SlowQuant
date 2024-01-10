import numpy as np

from slowquant.qiskit_interface.base import FermionicOperator, a_op


def Epq(p: int, q: int) -> FermionicOperator:
    """Construct the singlet one-electron excitation operator.

    .. math::
        \\hat{E}_{pq} = \\hat{a}^\\dagger_{p,\\alpha}\\hat{a}_{q,\\alpha} + \\hat{a}^\\dagger_{p,\\beta}\\hat{a}_{q,\\beta}

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.

    Returns:
        Singlet one-electron excitation operator.
    """
    E = FermionicOperator(a_op(p, "alpha", dagger=True), 1) * FermionicOperator(
        a_op(q, "alpha", dagger=False), 1
    )
    E += FermionicOperator(a_op(p, "beta", dagger=True), 1) * FermionicOperator(
        a_op(q, "beta", dagger=False), 1
    )
    return E


def epqrs(p: int, q: int, r: int, s: int) -> FermionicOperator:
    """Construct the singlet two-electron excitation operator.

    .. math::
        \\hat{e}_{pqrs} = \\hat{E}_{pq}\\hat{E}_{rs} - \\delta_{qr}\\hat{E}_{ps}

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.
        r: Spatial orbital index.
        s: Spatial orbital index.

    Returns:
        Singlet two-electron excitation operator.
    """
    if q == r:
        return Epq(p, q) * Epq(r, s) - Epq(p, s)
    return Epq(p, q) * Epq(r, s)


def Eminuspq(p: int, q: int) -> FermionicOperator:
    """Construct Hermitian singlet one-electron excitation operator.

    .. math::
        \\hat{E}^-_{pq} = \\hat{E}_{pq} - \\hat{E}_{qp}

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.

    Returns:
        Singlet one-electron excitation operator.
    """
    return Epq(p, q) - Epq(q, p)


def hamiltonian_full_space(h_mo: np.ndarray, g_mo: np.ndarray, num_orbs: int) -> FermionicOperator:
    """Construct full-space electronic Hamiltonian.

    .. math::
        \\hat{H} = \\sum_{pq}h_{pq}\\hat{E}_{pq} + \\frac{1}{2}\\sum_{pqrs}g_{pqrs}\\hat{e}_{pqrs}

    Args:
        h_mo: Core one-electron integrals in MO basis.
        g_mo: Two-electron integrals in MO basis.
        num_orbs: Number of spatial orbitals.

    Returns:
        Hamiltonian operator in full-space.
    """
    H_operator = FermionicOperator({}, {})
    for p in range(num_orbs):
        for q in range(num_orbs):
            if abs(h_mo[p, q]) < 10**-14:
                continue
            H_operator += h_mo[p, q] * Epq(p, q)
    for p in range(num_orbs):
        for q in range(num_orbs):
            for r in range(num_orbs):
                for s in range(num_orbs):
                    if abs(g_mo[p, q, r, s]) < 10**-14:
                        continue
                    H_operator += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s)
    return H_operator


def commutator(A: FermionicOperator, B: FermionicOperator) -> FermionicOperator:
    """Construct operator commutator.

    .. math::
        \\left[\\hat{A},\\hat{B}\\right] = \\hat{A}\\hat{B} - \\hat{B}\\hat{A}

    Args:
        A: Fermionic operator.
        B: Fermionic operator.

    Returns:
        Operator from commutator.
    """
    return A * B - B * A


def double_commutator(A: FermionicOperator, B: FermionicOperator, C: FermionicOperator) -> FermionicOperator:
    """Construct operator double commutator.

    .. math::
        \\left[\\hat{A},\\left[\\hat{B},\\hat{C}\\right]\\right] = \\hat{A}\\hat{B}\\hat{C} - \\hat{A}\\hat{C}\\hat{B} - \\hat{B}\\hat{C}\\hat{A} + \\hat{C}\\hat{B}\\hat{A}

    Args:
        A: Fermionic operator.
        B: Fermionic operator.
        C: Fermionic operator.

    Returns:
        Operator from double commutator.
    """
    return A * B * C - A * C * B - B * C * A + C * B * A


def G1(i: int, a: int) -> FermionicOperator:
    """Construct singlet one-electron spin-adapted excitation operator.

    .. math::
        \\hat{G}^{[1]}_{ia} = \\frac{1}{\\sqrt{2}}\\hat{E}_{ai}

    Args:
        i: Spatial orbital index.
        a: Spatial orbital index.

    Returns singlet one-elecetron spin-adapted exciation operator.
    """
    return 2 ** (-1 / 2) * Epq(a, i)


def G2_1(i: int, j: int, a: int, b: int) -> FermionicOperator:
    """Construct first singlet two-electron spin-adapted excitation operator.

    .. math::
        \\hat{G}^{[1]}_{aibj} = \\frac{1}{2\\sqrt{\\left(1+\\delta_{ab}\\right)\\left(1+\\delta_{ij}\\right)}}\\left(\\hat{E}_{ai}\\hat{E}_{bj} + \\hat{E}_{aj}\\hat{E}_{bi}\\right)

    Args:
        i: Spatial orbital index.
        j: Spatial orbital index.
        a: Spatial orbital index.
        b: Spatial orbital index.

    Returns first singlet two-elecetron spin-adapted exciation operator.
    """
    fac = 1
    if a == b:
        fac *= 2
    if i == j:
        fac *= 2
    return 1 / 2 * (fac) ** (-1 / 2) * (Epq(a, i) * Epq(b, j) + Epq(a, j) * Epq(b, i))


def G2_2(i: int, j: int, a: int, b: int) -> FermionicOperator:
    """Construct second singlet two-electron spin-adapted excitation operator.

    .. math::
        \\hat{G}^{[2]}_{aibj} = \\frac{1}{2\\sqrt{3}}\\left(\\hat{E}_{ai}\\hat{E}_{bj} - \\hat{E}_{aj}\\hat{E}_{bi}\\right)

    Args:
        i: Spatial orbital index.
        j: Spatial orbital index.
        a: Spatial orbital index.
        b: Spatial orbital index.

    Returns second singlet two-elecetron spin-adapted exciation operator.
    """
    return 1 / (2 * 3 ** (1 / 2)) * (Epq(a, i) * Epq(b, j) - Epq(a, j) * Epq(b, i))
