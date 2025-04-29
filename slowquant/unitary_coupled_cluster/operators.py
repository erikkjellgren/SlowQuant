import numpy as np

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator,
    a_op,
    a_op_spin,
)


def anni_spin(p: int, dagger: bool) -> FermionicOperator:
    """Construct annihilation/creation operator.

    Args:
        p: Spin orbital index.
        dagger: If creation operator.

    Returns:
        Annihilation/creation operator.
    """
    return FermionicOperator(a_op_spin(p, dagger=dagger), 1)


def Epq(p: int, q: int) -> FermionicOperator:
    r"""Construct the singlet one-electron excitation operator.

    .. math::
        \hat{E}_{pq} = \hat{a}^\dagger_{p,\alpha}\hat{a}_{q,\alpha} + \hat{a}^\dagger_{p,\beta}\hat{a}_{q,\beta}

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
    r"""Construct the singlet two-electron excitation operator.

    .. math::
        \hat{e}_{pqrs} = \hat{E}_{pq}\hat{E}_{rs} - \delta_{qr}\hat{E}_{ps}

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
    r"""Construct Hermitian singlet one-electron excitation operator.

    .. math::
        \hat{E}^-_{pq} = \hat{E}_{pq} - \hat{E}_{qp}

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.

    Returns:
        Singlet one-electron excitation operator.
    """
    return Epq(p, q) - Epq(q, p)


def commutator(A: FermionicOperator, B: FermionicOperator) -> FermionicOperator:
    r"""Construct operator commutator.

    .. math::
        \left[\hat{A},\hat{B}\right] = \hat{A}\hat{B} - \hat{B}\hat{A}

    Args:
        A: Fermionic operator.
        B: Fermionic operator.

    Returns:
        Operator from commutator.
    """
    return A * B - B * A


def double_commutator(
    A: FermionicOperator, B: FermionicOperator, C: FermionicOperator, do_symmetrized: bool = False
) -> FermionicOperator:
    r"""Construct operator double commutator.

    .. math::
        \left[\hat{A},\left[\hat{B},\hat{C}\right]\right] = \hat{A}\hat{B}\hat{C} - \hat{A}\hat{C}\hat{B} - \hat{B}\hat{C}\hat{A} + \hat{C}\hat{B}\hat{A}

    or for the symmetrized version,

    .. math::
        \left[\hat{A},\hat{B},\hat{C}\right] =
        \hat{A}\hat{H}\hat{B} + \hat{B}\hat{H}\hat{A} - \frac{1}{2}\left(\hat{A}\hat{B}\hat{H} + \hat{H}\hat{B}\hat{A} + \hat{B}\hat{A}\hat{H} + \hat{H}\hat{A}\hat{B}\right)

    Args:
        A: Fermionic operator.
        B: Fermionic operator.
        C: Fermionic operator.
        do_symmetrized: Do symmetrized double commutator (default: False).

    Returns:
        Operator from double commutator.
    """
    if do_symmetrized:
        return A * B * C + C * B * A - 1 / 2 * (A * C * B + B * C * A + C * A * B + B * A * C)
    return A * B * C - A * C * B - B * C * A + C * B * A


def G1(i: int, a: int, return_anti_hermitian: bool = False) -> FermionicOperator:
    r"""Construct one-electron excitation operator.

    .. math::
        \hat{G}^{[1]}_{ia} = \hat{a}_{a}^\dagger\hat{a}_i

    Args:
        i: Spin orbital index.
        a: Spin orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger

    Returns:
        One-electron excitation operator.
    """
    op = FermionicOperator(a_op_spin(a, dagger=True), 1) * FermionicOperator(a_op_spin(i, dagger=False), 1)
    if return_anti_hermitian:
        return op - op.dagger
    return op


def G2(i: int, j: int, a: int, b: int, return_anti_hermitian: bool = False) -> FermionicOperator:
    r"""Construct two-electron excitation operator.

    .. math::
        \hat{G}^{[2]}_{ijab} = \hat{a}_{a}^\dagger\hat{a}_{b}^\dagger\hat{a}_j\hat{a}_i

    Args:
        i: Spin orbital index.
        j: Spin orbital index.
        a: Spin orbital index.
        b: Spin orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger.

    Returns:
        Two-electron excitation operator.
    """
    op = (
        FermionicOperator(a_op_spin(a, dagger=True), 1)
        * FermionicOperator(a_op_spin(b, dagger=True), 1)
        * FermionicOperator(a_op_spin(j, dagger=False), 1)
        * FermionicOperator(a_op_spin(i, dagger=False), 1)
    )
    if return_anti_hermitian:
        return op - op.dagger
    return op


def G3(
    i: int, j: int, k: int, a: int, b: int, c: int, return_anti_hermitian: bool = False
) -> FermionicOperator:
    r"""Construct three-electron excitation operator.

    .. math::
        \hat{G}^{[3]}_{ijkabc} = \hat{a}_{a}^\dagger\hat{a}_{b}^\dagger\hat{a}_{c}^\dagger\hat{a}_k\hat{a}_j\hat{a}_i

    Args:
        i: Spin orbital index.
        j: Spin orbital index.
        k: Spin orbital index.
        a: Spin orbital index.
        b: Spin orbital index.
        c: Spin orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger.

    Returns:
        Three-electron excitation operator.
    """
    op = (
        FermionicOperator(a_op_spin(a, dagger=True), 1)
        * FermionicOperator(a_op_spin(b, dagger=True), 1)
        * FermionicOperator(a_op_spin(c, dagger=True), 1)
        * FermionicOperator(a_op_spin(k, dagger=False), 1)
        * FermionicOperator(a_op_spin(j, dagger=False), 1)
        * FermionicOperator(a_op_spin(i, dagger=False), 1)
    )
    if return_anti_hermitian:
        return op - op.dagger
    return op


def G4(
    i: int, j: int, k: int, l: int, a: int, b: int, c: int, d: int, return_anti_hermitian: bool = False
) -> FermionicOperator:
    r"""Construct four-electron excitation operator.

    .. math::
        \hat{G}^{[4]}_{ijklabcd} = \hat{a}_{a}^\dagger\hat{a}_{b}^\dagger\hat{a}_{c}^\dagger\hat{a}_{d}^\dagger\hat{a}_l\hat{a}_k\hat{a}_j\hat{a}_i

    Args:
        i: Spin orbital index.
        j: Spin orbital index.
        k: Spin orbital index.
        l: Spin orbital index.
        a: Spin orbital index.
        b: Spin orbital index.
        c: Spin orbital index.
        d: Spin orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger.

    Returns:
        Four-electron excitation operator.
    """
    op = (
        FermionicOperator(a_op_spin(a, dagger=True), 1)
        * FermionicOperator(a_op_spin(b, dagger=True), 1)
        * FermionicOperator(a_op_spin(c, dagger=True), 1)
        * FermionicOperator(a_op_spin(d, dagger=True), 1)
        * FermionicOperator(a_op_spin(l, dagger=False), 1)
        * FermionicOperator(a_op_spin(k, dagger=False), 1)
        * FermionicOperator(a_op_spin(j, dagger=False), 1)
        * FermionicOperator(a_op_spin(i, dagger=False), 1)
    )
    if return_anti_hermitian:
        return op - op.dagger
    return op


def G5(
    i: int,
    j: int,
    k: int,
    l: int,
    m: int,
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    return_anti_hermitian: bool = False,
) -> FermionicOperator:
    r"""Construct five-electron excitation operator.

    .. math::
        \hat{G}^{[5]}_{ijklmabcde} = \hat{a}_{a}^\dagger\hat{a}_{b}^\dagger\hat{a}_{c}^\dagger\hat{a}_{d}^\dagger\hat{a}_{e}^\dagger\hat{a}_m\hat{a}_l\hat{a}_k\hat{a}_j\hat{a}_i

    Args:
        i: Spin orbital index.
        j: Spin orbital index.
        k: Spin orbital index.
        l: Spin orbital index.
        m: Spin orbital index.
        a: Spin orbital index.
        b: Spin orbital index.
        c: Spin orbital index.
        d: Spin orbital index.
        e: Spin orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger.

    Returns:
        Five-electron excitation operator.
    """
    op = (
        FermionicOperator(a_op_spin(a, dagger=True), 1)
        * FermionicOperator(a_op_spin(b, dagger=True), 1)
        * FermionicOperator(a_op_spin(c, dagger=True), 1)
        * FermionicOperator(a_op_spin(d, dagger=True), 1)
        * FermionicOperator(a_op_spin(e, dagger=True), 1)
        * FermionicOperator(a_op_spin(m, dagger=False), 1)
        * FermionicOperator(a_op_spin(l, dagger=False), 1)
        * FermionicOperator(a_op_spin(k, dagger=False), 1)
        * FermionicOperator(a_op_spin(j, dagger=False), 1)
        * FermionicOperator(a_op_spin(i, dagger=False), 1)
    )
    if return_anti_hermitian:
        return op - op.dagger
    return op


def G6(
    i: int,
    j: int,
    k: int,
    l: int,
    m: int,
    n: int,
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    f: int,
    return_anti_hermitian: bool = False,
) -> FermionicOperator:
    r"""Construct six-electron excitation operator.

    .. math::
        \hat{G}^{[6]}_{ijklmnabcdef} = \hat{a}_{a}^\dagger\hat{a}_{b}^\dagger\hat{a}_{c}^\dagger\hat{a}_{d}^\dagger\hat{a}_{e}^\dagger\hat{a}_{f}^\dagger
        \hat{a}_n\hat{a}_m\hat{a}_l\hat{a}_k\hat{a}_j\hat{a}_i

    Args:
        i: Spin orbital index.
        j: Spin orbital index.
        k: Spin orbital index.
        l: Spin orbital index.
        m: Spin orbital index.
        n: Spin orbital index.
        a: Spin orbital index.
        b: Spin orbital index.
        c: Spin orbital index.
        d: Spin orbital index.
        e: Spin orbital index.
        f: Spin orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger.

    Returns:
        Six-electron excitation operator.
    """
    op = (
        FermionicOperator(a_op_spin(a, dagger=True), 1)
        * FermionicOperator(a_op_spin(b, dagger=True), 1)
        * FermionicOperator(a_op_spin(c, dagger=True), 1)
        * FermionicOperator(a_op_spin(d, dagger=True), 1)
        * FermionicOperator(a_op_spin(e, dagger=True), 1)
        * FermionicOperator(a_op_spin(f, dagger=True), 1)
        * FermionicOperator(a_op_spin(n, dagger=False), 1)
        * FermionicOperator(a_op_spin(m, dagger=False), 1)
        * FermionicOperator(a_op_spin(l, dagger=False), 1)
        * FermionicOperator(a_op_spin(k, dagger=False), 1)
        * FermionicOperator(a_op_spin(j, dagger=False), 1)
        * FermionicOperator(a_op_spin(i, dagger=False), 1)
    )
    if return_anti_hermitian:
        return op - op.dagger
    return op


def G1_sa(i: int, a: int, return_anti_hermitian: bool = False) -> FermionicOperator:
    r"""Construct singlet one-electron spin-adapted excitation operator.

    .. math::
        \hat{G}^{[1]}_{ia} = \frac{1}{\sqrt{2}}\hat{E}_{ai}

    Args:
        i: Spatial orbital index.
        a: Spatial orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger.

    Returns:
        Singlet one-electron spin-adapted excitation operator.
    """
    op = 2 ** (-1 / 2) * Epq(a, i)
    if return_anti_hermitian:
        return op - op.dagger
    return op


def G2_1_sa(i: int, j: int, a: int, b: int, return_anti_hermitian: bool = False) -> FermionicOperator:
    r"""Construct first singlet two-electron spin-adapted excitation operator.

    .. math::
        \hat{G}^{[1]}_{aibj} = \frac{1}{2\sqrt{\left(1+\delta_{ab}\right)\left(1+\delta_{ij}\right)}}\left(\hat{E}_{ai}\hat{E}_{bj} + \hat{E}_{aj}\hat{E}_{bi}\right)

    Args:
        i: Spatial orbital index.
        j: Spatial orbital index.
        a: Spatial orbital index.
        b: Spatial orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger.

    Returns:
        First singlet two-electron spin-adapted excitation operator.
    """
    fac = 1
    if a == b:
        fac *= 2
    if i == j:
        fac *= 2
    op = 1 / 2 * (fac) ** (-1 / 2) * (Epq(a, i) * Epq(b, j) + Epq(a, j) * Epq(b, i))
    if return_anti_hermitian:
        return op - op.dagger
    return op


def G2_2_sa(i: int, j: int, a: int, b: int, return_anti_hermitian: bool = False) -> FermionicOperator:
    r"""Construct second singlet two-electron spin-adapted excitation operator.

    .. math::
        \hat{G}^{[2]}_{aibj} = \frac{1}{2\sqrt{3}}\left(\hat{E}_{ai}\hat{E}_{bj} - \hat{E}_{aj}\hat{E}_{bi}\right)

    Args:
        i: Spatial orbital index.
        j: Spatial orbital index.
        a: Spatial orbital index.
        b: Spatial orbital index.
        return_anti_hermitian: Return anti-hermitian version of operator, i.e. op - op^\dagger.

    Returns:
        Second singlet two-electron spin-adapted excitation operator.
    """
    op = 1 / (2 * 3 ** (1 / 2)) * (Epq(a, i) * Epq(b, j) - Epq(a, j) * Epq(b, i))
    if return_anti_hermitian:
        return op - op.dagger
    return op


def hamiltonian_full_space(h_mo: np.ndarray, g_mo: np.ndarray, num_orbs: int) -> FermionicOperator:
    r"""Construct full-space electronic Hamiltonian.

    .. math::
        \hat{H} = \sum_{pq}h_{pq}\hat{E}_{pq} + \frac{1}{2}\sum_{pqrs}g_{pqrs}\hat{e}_{pqrs}

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


def hamiltonian_0i_0a(
    h_mo: np.ndarray,
    g_mo: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> FermionicOperator:
    """Get energy Hamiltonian operator.

    Args:
        h_mo: One-electron Hamiltonian integrals in MO.
        g_mo: Two-electron Hamiltonian integrals in MO.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Energy Hamiltonian fermionic operator.
    """
    hamiltonian_operator = FermionicOperator({}, {})
    # Inactive one-electron
    for i in range(num_inactive_orbs):
        if abs(h_mo[i, i]) > 10**-14:
            hamiltonian_operator += h_mo[i, i] * Epq(i, i)
    # Active one-electron
    for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            if abs(h_mo[p, q]) > 10**-14:
                hamiltonian_operator += h_mo[p, q] * Epq(p, q)
    # Inactive two-electron
    for i in range(num_inactive_orbs):
        for j in range(num_inactive_orbs):
            if abs(g_mo[i, i, j, j]) > 10**-14:
                hamiltonian_operator += 1 / 2 * g_mo[i, i, j, j] * epqrs(i, i, j, j)
            if i != j and abs(g_mo[j, i, i, j]) > 10**-14:
                hamiltonian_operator += 1 / 2 * g_mo[j, i, i, j] * epqrs(j, i, i, j)
    # Inactive-Active two-electron
    for i in range(num_inactive_orbs):
        for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
                if abs(g_mo[i, i, p, q]) > 10**-14:
                    hamiltonian_operator += 1 / 2 * g_mo[i, i, p, q] * epqrs(i, i, p, q)
                if abs(g_mo[p, q, i, i]) > 10**-14:
                    hamiltonian_operator += 1 / 2 * g_mo[p, q, i, i] * epqrs(p, q, i, i)
                if abs(g_mo[p, i, i, q]) > 10**-14:
                    hamiltonian_operator += 1 / 2 * g_mo[p, i, i, q] * epqrs(p, i, i, q)
                if abs(g_mo[i, p, q, i]) > 10**-14:
                    hamiltonian_operator += 1 / 2 * g_mo[i, p, q, i] * epqrs(i, p, q, i)
    # Active two-electron
    for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            for r in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
                for s in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
                    if abs(g_mo[p, q, r, s]) > 10**-14:
                        hamiltonian_operator += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s)
    return hamiltonian_operator


def hamiltonian_1i_1a(
    h_mo: np.ndarray,
    g_mo: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
) -> FermionicOperator:
    """Get Hamiltonian operator that works together with an extra inactive and an extra virtual index.

    Args:
        h_mo: One-electron Hamiltonian integrals in MO.
        g_mo: Two-electron Hamiltonian integrals in MO.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.
        num_virtual_orbs: Number of virtual orbitals in spatial basis.

    Returns:
        Modified Hamiltonian fermionic operator.
    """
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    hamiltonian_operator = FermionicOperator({}, {})
    virtual_start = num_inactive_orbs + num_active_orbs
    for p in range(num_orbs):
        for q in range(num_orbs):
            if p >= virtual_start and q >= virtual_start:
                continue
            if p < num_inactive_orbs and q < num_inactive_orbs and p != q:
                continue
            if abs(h_mo[p, q]) > 10**-14:
                hamiltonian_operator += h_mo[p, q] * Epq(p, q)
    for p in range(num_orbs):
        for q in range(num_orbs):
            for r in range(num_orbs):
                for s in range(num_orbs):
                    num_virt = 0
                    if p >= virtual_start:
                        num_virt += 1
                    if q >= virtual_start:
                        num_virt += 1
                    if r >= virtual_start:
                        num_virt += 1
                    if s >= virtual_start:
                        num_virt += 1
                    if num_virt > 1:
                        continue
                    num_act = 0
                    if p < num_inactive_orbs:
                        num_act += 1
                    if q < num_inactive_orbs:
                        num_act += 1
                    if r < num_inactive_orbs:
                        num_act += 1
                    if s < num_inactive_orbs:
                        num_act += 1
                    if p < num_inactive_orbs and q < num_inactive_orbs and p == q:
                        num_act -= 2
                    if r < num_inactive_orbs and s < num_inactive_orbs and r == s:
                        num_act -= 2
                    if p < num_inactive_orbs and s < num_inactive_orbs and p == s:
                        num_act -= 2
                    if q < num_inactive_orbs and r < num_inactive_orbs and q == r:
                        num_act -= 2
                    if num_act > 1:
                        continue
                    if abs(g_mo[p, q, r, s]) > 10**-14:
                        hamiltonian_operator += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s)
    return hamiltonian_operator


def hamiltonian_2i_2a(
    h_mo: np.ndarray,
    g_mo: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
) -> FermionicOperator:
    """Get Hamiltonian operator that works together with two extra inactive and two extra virtual index.

    Args:
        h_mo: One-electron Hamiltonian integrals in MO.
        g_mo: Two-electron Hamiltonian integrals in MO.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.
        num_virtual_orbs: Number of virtual orbitals in spatial basis.

    Returns:
        Modified Hamiltonian fermionic operator.
    """
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    hamiltonian_operator = FermionicOperator({}, {})
    virtual_start = num_inactive_orbs + num_active_orbs
    for p in range(num_orbs):
        for q in range(num_orbs):
            if abs(h_mo[p, q]) > 10**-14:
                hamiltonian_operator += h_mo[p, q] * Epq(p, q)
    for p in range(num_orbs):
        for q in range(num_orbs):
            for r in range(num_orbs):
                for s in range(num_orbs):
                    num_virt = 0
                    if p >= virtual_start:
                        num_virt += 1
                    if q >= virtual_start:
                        num_virt += 1
                    if r >= virtual_start:
                        num_virt += 1
                    if s >= virtual_start:
                        num_virt += 1
                    if num_virt > 2:
                        continue
                    num_act = 0
                    if p < num_inactive_orbs:
                        num_act += 1
                    if q < num_inactive_orbs:
                        num_act += 1
                    if r < num_inactive_orbs:
                        num_act += 1
                    if s < num_inactive_orbs:
                        num_act += 1
                    if p < num_inactive_orbs and q < num_inactive_orbs and p == q:
                        num_act -= 2
                    if r < num_inactive_orbs and s < num_inactive_orbs and r == s:
                        num_act -= 2
                    if p < num_inactive_orbs and s < num_inactive_orbs and p == s:
                        num_act -= 2
                    if q < num_inactive_orbs and r < num_inactive_orbs and q == r:
                        num_act -= 2
                    if num_act > 2:
                        continue
                    if abs(g_mo[p, q, r, s]) > 10**-14:
                        hamiltonian_operator += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s)
    return hamiltonian_operator


def one_elec_op_full_space(ints_mo: np.ndarray, num_orbs: int) -> FermionicOperator:
    r"""Construct full-space one-electron operator.

    .. math::
        \hat{O} = \sum_{pq}h_{pq}\hat{E}_{pq}

    Args:
        ints_mo: One-electron integrals for operator in MO basis.
        num_orbs: Number of spatial orbitals.

    Returns:
        One-electron operator in full-space.
    """
    one_elec_op = FermionicOperator({}, {})
    for p in range(num_orbs):
        for q in range(num_orbs):
            if abs(ints_mo[p, q]) > 10**-14:
                one_elec_op += ints_mo[p, q] * Epq(p, q)
    return one_elec_op


def one_elec_op_0i_0a(ints_mo: np.ndarray, num_inactive_orbs: int, num_active_orbs: int) -> FermionicOperator:
    """Create one-electron operator that makes no changes in the inactive and virtual orbitals.

    Args:
        ints_mo: One-electron integrals for operator in MO basis.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        One-electron operator for active-space.
    """
    one_elec_op = FermionicOperator({}, {})
    # Inactive one-electron
    for i in range(num_inactive_orbs):
        if abs(ints_mo[i, i]) > 10**-14:
            one_elec_op += ints_mo[i, i] * Epq(i, i)
    # Active one-electron
    for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            if abs(ints_mo[p, q]) > 10**-14:
                one_elec_op += ints_mo[p, q] * Epq(p, q)
    return one_elec_op


def one_elec_op_1i_1a(
    ints_mo: np.ndarray, num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int
) -> FermionicOperator:
    """Create one-electron operator that makes up to one change in the inactive and virtual orbitals.

    Args:
        ints_mo: One-electron integrals for operator in MO basis.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.
        num_virtual_orbs: Number of virtual orbitals in spatial basis.

    Returns:
        Modified one-electron operator.
    """
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    one_elec_op = FermionicOperator({}, {})
    virtual_start = num_inactive_orbs + num_active_orbs
    for p in range(num_orbs):
        for q in range(num_orbs):
            if p >= virtual_start and q >= virtual_start:
                continue
            if p < num_inactive_orbs and q < num_inactive_orbs and p != q:
                continue
            if abs(ints_mo[p, q]) > 10**-14:
                one_elec_op += ints_mo[p, q] * Epq(p, q)
    return one_elec_op
