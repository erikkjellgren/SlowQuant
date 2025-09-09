import numpy as np

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import a_op


def unrestricted_hamiltonian_full_space(
    haa_mo: np.ndarray,
    hbb_mo: np.ndarray,
    gaaaa_mo: np.ndarray,
    gbbbb_mo: np.ndarray,
    gaabb_mo: np.ndarray,
    gbbaa_mo: np.ndarray,
    num_orbs: int,
) -> FermionicOperator:
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
    H_operator = FermionicOperator({})
    for p in range(num_orbs):
        for q in range(num_orbs):
            if abs(haa_mo[p, q]) < 10**-14 and abs(hbb_mo[p, q]) < 10**-14:
                continue
            H_operator += haa_mo[p, q] * a_op(p, "alpha", True) * a_op(q, "alpha", False)
            H_operator += hbb_mo[p, q] * a_op(p, "beta", True) * a_op(q, "beta", False)
    for p in range(num_orbs):
        for q in range(num_orbs):
            for r in range(num_orbs):
                for s in range(num_orbs):
                    if (
                        abs(gaaaa_mo[p, q, r, s]) < 10**-14
                        and abs(gbbbb_mo[p, q, r, s]) < 10**-14
                        and abs(gaabb_mo[p, q, r, s]) < 10**-14
                        and abs(gbbaa_mo[p, q, r, s]) < 10**-14
                    ):
                        continue
                    H_operator += (
                        1
                        / 2
                        * gaaaa_mo[p, q, r, s]
                        * a_op(p, "alpha", True)
                        * a_op(r, "alpha", True)
                        * a_op(s, "alpha", False)
                        * a_op(q, "alpha", False)
                    )
                    H_operator += (
                        1
                        / 2
                        * gbbbb_mo[p, q, r, s]
                        * a_op(p, "beta", True)
                        * a_op(r, "beta", True)
                        * a_op(s, "beta", False)
                        * a_op(q, "beta", False)
                    )
                    H_operator += (
                        1
                        / 2
                        * (
                            gaabb_mo[p, q, r, s]
                            * a_op(p, "alpha", True)
                            * a_op(r, "beta", True)
                            * a_op(s, "beta", False)
                            * a_op(q, "alpha", False)
                        )
                    )
                    H_operator += (
                        1
                        / 2
                        * (
                            gbbaa_mo[p, q, r, s]
                            * a_op(p, "beta", True)
                            * a_op(r, "alpha", True)
                            * a_op(s, "alpha", False)
                            * a_op(q, "beta", False)
                        )
                    )
    return H_operator


def unrestricted_hamiltonian_1i_1a(
    haa_mo: np.ndarray,
    hbb_mo: np.ndarray,
    gaaaa_mo: np.ndarray,
    gbbbb_mo: np.ndarray,
    gaabb_mo: np.ndarray,
    gbbaa_mo: np.ndarray,
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
    hamiltonian_operator = FermionicOperator({})
    virtual_start = num_inactive_orbs + num_active_orbs
    for p in range(num_orbs):
        for q in range(num_orbs):
            if p >= virtual_start and q >= virtual_start:
                continue
            if p < num_inactive_orbs and q < num_inactive_orbs and p != q:
                continue
            if abs(haa_mo[p, q]) > 10**-14 or abs(hbb_mo[p, q]) > 10**-14:
                hamiltonian_operator += haa_mo[p, q] * a_op(p, "alpha", True) * a_op(q, "alpha", False)
                hamiltonian_operator += hbb_mo[p, q] * a_op(p, "beta", True) * a_op(q, "beta", False)
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
                    if (
                        abs(gaaaa_mo[p, q, r, s]) > 10**-14
                        or abs(gbbbb_mo[p, q, r, s]) > 10**-14
                        or abs(gaabb_mo[p, q, r, s]) > 10**-14
                        or abs(gbbaa_mo[p, q, r, s]) > 10**-14
                    ):
                        hamiltonian_operator += (
                            1
                            / 2
                            * gaaaa_mo[p, q, r, s]
                            * a_op(p, "alpha", True)
                            * a_op(r, "alpha", True)
                            * a_op(s, "alpha", False)
                            * a_op(q, "alpha", False)
                        )
                        hamiltonian_operator += (
                            1
                            / 2
                            * gbbbb_mo[p, q, r, s]
                            * a_op(p, "beta", True)
                            * a_op(r, "beta", True)
                            * a_op(s, "beta", False)
                            * a_op(q, "beta", False)
                        )
                        hamiltonian_operator += (
                            1
                            / 2
                            * (
                                gaabb_mo[p, q, r, s]
                                * a_op(p, "alpha", True)
                                * a_op(r, "beta", True)
                                * a_op(s, "beta", False)
                                * a_op(q, "alpha", False)
                            )
                        )
                        hamiltonian_operator += (
                            1
                            / 2
                            * (
                                gbbaa_mo[p, q, r, s]
                                * a_op(p, "beta", True)
                                * a_op(r, "alpha", True)
                                * a_op(s, "alpha", False)
                                * a_op(q, "beta", False)
                            )
                        )
    return hamiltonian_operator


def unrestricted_hamiltonian_0i_0a(
    haa_mo: np.ndarray,
    hbb_mo: np.ndarray,
    gaaaa_mo: np.ndarray,
    gbbbb_mo: np.ndarray,
    gaabb_mo: np.ndarray,
    gbbaa_mo: np.ndarray,
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
        Energy Hamilonian fermionic operator.
    """
    hamiltonian_operator = FermionicOperator({})
    # Inactive one-electron
    for i in range(num_inactive_orbs):
        if abs(haa_mo[i, i]) > 10**-14 or abs(hbb_mo[i, i]) > 10**-14:
            hamiltonian_operator += haa_mo[i, i] * a_op(i, "alpha", True) * a_op(i, "alpha", False)
            hamiltonian_operator += hbb_mo[i, i] * a_op(i, "beta", True) * a_op(i, "beta", False)
    # Active one-electron
    for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            if abs(haa_mo[p, q]) > 10**-14 or abs(hbb_mo[p, q]) > 10**-14:
                hamiltonian_operator += haa_mo[p, q] * a_op(p, "alpha", True) * a_op(q, "alpha", False)
                hamiltonian_operator += hbb_mo[p, q] * a_op(p, "beta", True) * a_op(q, "beta", False)
    # Inactive two-electron
    for i in range(num_inactive_orbs):
        for j in range(num_inactive_orbs):
            if (
                abs(gaaaa_mo[i, i, j, j]) > 10**-14
                or abs(gbbbb_mo[i, i, j, j]) > 10**-14
                or abs(gaabb_mo[i, i, j, j]) > 10**-14
                or abs(gbbaa_mo[i, i, j, j]) > 10**-14
            ):
                hamiltonian_operator += (
                    1
                    / 2
                    * gaaaa_mo[i, i, j, j]
                    * a_op(i, "alpha", True)
                    * a_op(j, "alpha", True)
                    * a_op(j, "alpha", False)
                    * a_op(i, "alpha", False)
                )
                hamiltonian_operator += (
                    1
                    / 2
                    * gbbbb_mo[i, i, j, j]
                    * a_op(i, "beta", True)
                    * a_op(j, "beta", True)
                    * a_op(j, "beta", False)
                    * a_op(i, "beta", False)
                )
                hamiltonian_operator += (
                    1
                    / 2
                    * (
                        gaabb_mo[i, i, j, j]
                        * a_op(i, "alpha", True)
                        * a_op(j, "beta", True)
                        * a_op(j, "beta", False)
                        * a_op(i, "alpha", False)
                    )
                )
                hamiltonian_operator += (
                    1
                    / 2
                    * (
                        gbbaa_mo[i, i, j, j]
                        * a_op(i, "beta", True)
                        * a_op(j, "alpha", True)
                        * a_op(j, "alpha", False)
                        * a_op(i, "beta", False)
                    )
                )
            if i != j and (
                abs(gaaaa_mo[j, i, i, j]) > 10**-14
                or abs(gbbbb_mo[j, i, i, j]) > 10**-14
                or abs(gaabb_mo[j, i, i, j]) > 10**-14
                or abs(gbbaa_mo[j, i, i, j]) > 10**-14
            ):
                hamiltonian_operator += (
                    1
                    / 2
                    * gaaaa_mo[j, i, i, j]
                    * a_op(j, "alpha", True)
                    * a_op(i, "alpha", True)
                    * a_op(j, "alpha", False)
                    * a_op(i, "alpha", False)
                )
                hamiltonian_operator += (
                    1
                    / 2
                    * gbbbb_mo[j, i, i, j]
                    * a_op(j, "beta", True)
                    * a_op(i, "beta", True)
                    * a_op(j, "beta", False)
                    * a_op(i, "beta", False)
                )
                hamiltonian_operator += (
                    1
                    / 2
                    * (
                        gaabb_mo[j, i, i, j]
                        * a_op(j, "alpha", True)
                        * a_op(i, "beta", True)
                        * a_op(j, "beta", False)
                        * a_op(i, "alpha", False)
                    )
                )
                hamiltonian_operator += (
                    1
                    / 2
                    * (
                        gbbaa_mo[j, i, i, j]
                        * a_op(j, "beta", True)
                        * a_op(i, "alpha", True)
                        * a_op(j, "alpha", False)
                        * a_op(i, "beta", False)
                    )
                )
    # Inactive-Active two-electron
    for i in range(num_inactive_orbs):
        for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
                if (
                    abs(gaaaa_mo[i, i, p, q]) > 10**-14
                    or abs(gbbbb_mo[i, i, p, q]) > 10**-14
                    or abs(gaabb_mo[i, i, p, q]) > 10**-14
                    or abs(gbbaa_mo[i, i, p, q]) > 10**-14
                ):
                    hamiltonian_operator += (
                        1
                        / 2
                        * gaaaa_mo[i, i, p, q]
                        * a_op(i, "alpha", True)
                        * a_op(p, "alpha", True)
                        * a_op(q, "alpha", False)
                        * a_op(i, "alpha", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * gbbbb_mo[i, i, p, q]
                        * a_op(i, "beta", True)
                        * a_op(p, "beta", True)
                        * a_op(q, "beta", False)
                        * a_op(i, "beta", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * (
                            gaabb_mo[i, i, p, q]
                            * a_op(i, "alpha", True)
                            * a_op(p, "beta", True)
                            * a_op(q, "beta", False)
                            * a_op(i, "alpha", False)
                        )
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * (
                            gbbaa_mo[i, i, p, q]
                            * a_op(i, "beta", True)
                            * a_op(p, "alpha", True)
                            * a_op(q, "alpha", False)
                            * a_op(i, "beta", False)
                        )
                    )
                if (
                    abs(gaaaa_mo[p, q, i, i]) > 10**-14
                    or abs(gbbbb_mo[p, q, i, i]) > 10**-14
                    or abs(gaabb_mo[p, q, i, i]) > 10**-14
                    or abs(gbbaa_mo[p, q, i, i]) > 10**-14
                ):
                    hamiltonian_operator += (
                        1
                        / 2
                        * gaaaa_mo[p, q, i, i]
                        * a_op(p, "alpha", True)
                        * a_op(i, "alpha", True)
                        * a_op(i, "alpha", False)
                        * a_op(q, "alpha", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * gbbbb_mo[p, q, i, i]
                        * a_op(p, "beta", True)
                        * a_op(i, "beta", True)
                        * a_op(i, "beta", False)
                        * a_op(q, "beta", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * (
                            gaabb_mo[p, q, i, i]
                            * a_op(p, "alpha", True)
                            * a_op(i, "beta", True)
                            * a_op(i, "beta", False)
                            * a_op(q, "alpha", False)
                        )
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * (
                            gbbaa_mo[p, q, i, i]
                            * a_op(p, "beta", True)
                            * a_op(i, "alpha", True)
                            * a_op(i, "alpha", False)
                            * a_op(q, "beta", False)
                        )
                    )
                if abs(gaaaa_mo[p, i, i, q]) > 10**-14 or abs(gbbbb_mo[p, i, i, q]) > 10**-14:
                    hamiltonian_operator += (
                        1
                        / 2
                        * gaaaa_mo[p, i, i, q]
                        * a_op(p, "alpha", True)
                        * a_op(i, "alpha", True)
                        * a_op(q, "alpha", False)
                        * a_op(i, "alpha", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * gbbbb_mo[p, i, i, q]
                        * a_op(p, "beta", True)
                        * a_op(i, "beta", True)
                        * a_op(q, "beta", False)
                        * a_op(i, "beta", False)
                    )
                if abs(gaaaa_mo[i, p, q, i]) > 10**-14 or abs(gbbbb_mo[i, p, q, i]) > 10**-14:
                    hamiltonian_operator += (
                        1
                        / 2
                        * gaaaa_mo[i, p, q, i]
                        * a_op(i, "alpha", True)
                        * a_op(q, "alpha", True)
                        * a_op(i, "alpha", False)
                        * a_op(p, "alpha", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * gbbbb_mo[i, p, q, i]
                        * a_op(i, "beta", True)
                        * a_op(q, "beta", True)
                        * a_op(i, "beta", False)
                        * a_op(p, "beta", False)
                    )
    # Active two-electron
    for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            for r in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
                for s in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
                    if (
                        abs(gaaaa_mo[p, q, r, s]) > 10**-14
                        or abs(gbbbb_mo[p, q, r, s]) > 10**-14
                        or abs(gaabb_mo[p, q, r, s]) > 10**-14
                        or abs(gbbaa_mo[p, q, r, s]) > 10**-14
                    ):
                        hamiltonian_operator += (
                            1
                            / 2
                            * gaaaa_mo[p, q, r, s]
                            * a_op(p, "alpha", True)
                            * a_op(r, "alpha", True)
                            * a_op(s, "alpha", False)
                            * a_op(q, "alpha", False)
                        )
                        hamiltonian_operator += (
                            1
                            / 2
                            * gbbbb_mo[p, q, r, s]
                            * a_op(p, "beta", True)
                            * a_op(r, "beta", True)
                            * a_op(s, "beta", False)
                            * a_op(q, "beta", False)
                        )
                        hamiltonian_operator += (
                            1
                            / 2
                            * (
                                gaabb_mo[p, q, r, s]
                                * a_op(p, "alpha", True)
                                * a_op(r, "beta", True)
                                * a_op(s, "beta", False)
                                * a_op(q, "alpha", False)
                            )
                        )
                        hamiltonian_operator += (
                            1
                            / 2
                            * (
                                gbbaa_mo[p, q, r, s]
                                * a_op(p, "beta", True)
                                * a_op(r, "alpha", True)
                                * a_op(s, "alpha", False)
                                * a_op(q, "beta", False)
                            )
                        )
    return hamiltonian_operator


def unrestricted_one_elec_op_full_space(
    intsaa_mo: np.ndarray, intsbb_mo: np.ndarray, num_orbs: int
) -> FermionicOperator:
    r"""Construct full-space one-electron operator.

    .. math::
        \hat{O} = \sum_{pq}h_{pq}\hat{E}_{pq}

    Args:
        intsaa_mo: One-electron integrals alpha-alpha for operator in MO basis.
        intsbb_mo: One-electron integrals beta-beta for operator in MO basis.
        num_orbs: Number of spatial orbitals.

    Returns:
        One-electron operator in full-space.
    """
    one_elec_op = FermionicOperator({})
    for p in range(num_orbs):
        for q in range(num_orbs):
            if abs(intsaa_mo[p, q]) < 10**-14 and abs(intsbb_mo[p, q]) < 10**-14:
                continue
            one_elec_op += intsaa_mo[p, q] * a_op(p, "alpha", True) * a_op(q, "alpha", False)
            one_elec_op += intsbb_mo[p, q] * a_op(p, "beta", True) * a_op(q, "beta", False)
    return one_elec_op
