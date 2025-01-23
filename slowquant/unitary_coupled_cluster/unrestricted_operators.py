import numpy as np

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import anni


def unrestricted_hamiltonian_full_space(
    haa_mo: np.ndarray,
    hbb_mo: np.ndarray,
    gaaaa_mo: np.ndarray,
    gbbbb_mo: np.ndarray,
    gaabb_mo: np.ndarray,
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
    H_operator = FermionicOperator({}, {})
    for p in range(num_orbs):
        for q in range(num_orbs):
            if abs(haa_mo[p, q]) < 10**-14 and abs(hbb_mo[p, q]) < 10**-14:
                continue
            H_operator += haa_mo[p, q] * anni(p, "alpha", True) * anni(q, "alpha", False)
            H_operator += hbb_mo[p, q] * anni(p, "beta", True) * anni(q, "beta", False)
    for p in range(num_orbs):
        for q in range(num_orbs):
            for r in range(num_orbs):
                for s in range(num_orbs):
                    if (
                        abs(gaaaa_mo[p, q, r, s]) < 10**-14
                        and abs(gbbbb_mo[p, q, r, s]) < 10**-14
                        and abs(gaabb_mo[p, q, r, s]) < 10**-14
                    ):
                        continue
                    H_operator += (
                        1
                        / 2
                        * gaaaa_mo[p, q, r, s]
                        * anni(p, "alpha", True)
                        * anni(r, "alpha", True)
                        * anni(s, "alpha", False)
                        * anni(q, "alpha", False)
                    )
                    H_operator += (
                        1
                        / 2
                        * gbbbb_mo[p, q, r, s]
                        * anni(p, "beta", True)
                        * anni(r, "beta", True)
                        * anni(s, "beta", False)
                        * anni(q, "beta", False)
                    )
                    H_operator += (
                        gaabb_mo[p, q, r, s]
                        * anni(p, "alpha", True)
                        * anni(r, "beta", True)
                        * anni(s, "beta", False)
                        * anni(q, "alpha", False)
                    )
    return H_operator


def unrestricted_hamiltonian_0i_0a(
    haa_mo: np.ndarray,
    hbb_mo: np.ndarray,
    gaaaa_mo: np.ndarray,
    gbbbb_mo: np.ndarray,
    gaabb_mo: np.ndarray,
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
    hamiltonian_operator = FermionicOperator({}, {})
    # Inactive one-electron
    for i in range(num_inactive_orbs):
        if abs(haa_mo[i, i]) > 10**-14 or abs(hbb_mo[i, i]) > 10**-14:
            hamiltonian_operator += haa_mo[i, i] * anni(i, "alpha", True) * anni(i, "alpha", False)
            hamiltonian_operator += hbb_mo[i, i] * anni(i, "beta", True) * anni(i, "beta", False)
    # Active one-electron
    for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            if abs(haa_mo[p, q]) > 10**-14 or abs(hbb_mo[p, q]) > 10**-14:
                hamiltonian_operator += haa_mo[p, q] * anni(p, "alpha", True) * anni(q, "alpha", False)
                hamiltonian_operator += hbb_mo[p, q] * anni(p, "beta", True) * anni(q, "beta", False)
    # Inactive two-electron
    for i in range(num_inactive_orbs):
        for j in range(num_inactive_orbs):
            if (
                abs(gaaaa_mo[i, i, j, j]) > 10**-14
                or abs(gbbbb_mo[i, i, j, j]) > 10**-14
                or abs(gaabb_mo[i, i, j, j]) > 10**-14
            ):
                hamiltonian_operator += (
                    1
                    / 2
                    * gaaaa_mo[i, i, j, j]
                    * anni(i, "alpha", True)
                    * anni(j, "alpha", True)
                    * anni(j, "alpha", False)
                    * anni(i, "alpha", False)
                )
                hamiltonian_operator += (
                    1
                    / 2
                    * gbbbb_mo[i, i, j, j]
                    * anni(i, "beta", True)
                    * anni(j, "beta", True)
                    * anni(j, "beta", False)
                    * anni(i, "beta", False)
                )
                hamiltonian_operator += (
                    gaabb_mo[i, i, j, j]
                    * anni(i, "alpha", True)
                    * anni(j, "beta", True)
                    * anni(j, "beta", False)
                    * anni(i, "alpha", False)
                )
            if i != j and (
                abs(gaaaa_mo[j, i, i, j]) > 10**-14
                or abs(gbbbb_mo[j, i, i, j]) > 10**-14
                or abs(gaabb_mo[j, i, i, j]) > 10**-14
            ):
                hamiltonian_operator += (
                    1
                    / 2
                    * gaaaa_mo[j, i, i, j]
                    * anni(j, "alpha", True)
                    * anni(i, "alpha", True)
                    * anni(j, "alpha", False)
                    * anni(i, "alpha", False)
                )
                hamiltonian_operator += (
                    1
                    / 2
                    * gbbbb_mo[j, i, i, j]
                    * anni(j, "beta", True)
                    * anni(i, "beta", True)
                    * anni(j, "beta", False)
                    * anni(i, "beta", False)
                )
                hamiltonian_operator += (
                    gaabb_mo[j, i, i, j]
                    * anni(j, "alpha", True)
                    * anni(i, "beta", True)
                    * anni(j, "beta", False)
                    * anni(i, "alpha", False)
                )
    # Inactive-Active two-electron
    for i in range(num_inactive_orbs):
        for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
                if (
                    abs(gaaaa_mo[i, i, p, q]) > 10**-14
                    or abs(gbbbb_mo[i, i, p, q]) > 10**-14
                    or abs(gaabb_mo[i, i, p, q]) > 10**-14
                ):
                    hamiltonian_operator += (
                        1
                        / 2
                        * gaaaa_mo[i, i, p, q]
                        * anni(i, "alpha", True)
                        * anni(p, "alpha", True)
                        * anni(q, "alpha", False)
                        * anni(i, "alpha", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * gbbbb_mo[i, i, p, q]
                        * anni(i, "beta", True)
                        * anni(p, "beta", True)
                        * anni(q, "beta", False)
                        * anni(i, "beta", False)
                    )
                    hamiltonian_operator += (
                        gaabb_mo[i, i, p, q]
                        * anni(i, "alpha", True)
                        * anni(p, "beta", True)
                        * anni(q, "beta", False)
                        * anni(i, "alpha", False)
                    )
                if (
                    abs(gaaaa_mo[p, q, i, i]) > 10**-14
                    or abs(gbbbb_mo[p, q, i, i]) > 10**-14
                    or abs(gaabb_mo[p, q, i, i]) > 10**-14
                ):
                    hamiltonian_operator += (
                        1
                        / 2
                        * gaaaa_mo[p, q, i, i]
                        * anni(p, "alpha", True)
                        * anni(i, "alpha", True)
                        * anni(i, "alpha", False)
                        * anni(q, "alpha", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * gbbbb_mo[p, q, i, i]
                        * anni(p, "beta", True)
                        * anni(i, "beta", True)
                        * anni(i, "beta", False)
                        * anni(q, "beta", False)
                    )
                    hamiltonian_operator += (
                        gaabb_mo[p, q, i, i]
                        * anni(p, "alpha", True)
                        * anni(i, "beta", True)
                        * anni(i, "beta", False)
                        * anni(q, "alpha", False)
                    )
                if (
                    abs(gaaaa_mo[p, i, i, q]) > 10**-14
                    or abs(gbbbb_mo[p, i, i, q]) > 10**-14
                    or abs(gaabb_mo[p, i, i, q]) > 10**-14
                ):
                    hamiltonian_operator += (
                        1
                        / 2
                        * gaaaa_mo[p, i, i, q]
                        * anni(p, "alpha", True)
                        * anni(i, "alpha", True)
                        * anni(q, "alpha", False)
                        * anni(i, "alpha", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * gbbbb_mo[p, i, i, q]
                        * anni(p, "beta", True)
                        * anni(i, "beta", True)
                        * anni(q, "beta", False)
                        * anni(i, "beta", False)
                    )
                if (
                    abs(gaaaa_mo[i, p, q, i]) > 10**-14
                    or abs(gbbbb_mo[i, p, q, i]) > 10**-14
                    or abs(gaabb_mo[i, p, q, i]) > 10**-14
                ):
                    hamiltonian_operator += (
                        1
                        / 2
                        * gaaaa_mo[i, p, q, i]
                        * anni(i, "alpha", True)
                        * anni(q, "alpha", True)
                        * anni(i, "alpha", False)
                        * anni(p, "alpha", False)
                    )
                    hamiltonian_operator += (
                        1
                        / 2
                        * gbbbb_mo[i, p, q, i]
                        * anni(i, "beta", True)
                        * anni(q, "beta", True)
                        * anni(i, "beta", False)
                        * anni(p, "beta", False)
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
                    ):
                        hamiltonian_operator += (
                            1
                            / 2
                            * gaaaa_mo[p, q, r, s]
                            * anni(p, "alpha", True)
                            * anni(r, "alpha", True)
                            * anni(s, "alpha", False)
                            * anni(q, "alpha", False)
                        )
                        hamiltonian_operator += (
                            1
                            / 2
                            * gbbbb_mo[p, q, r, s]
                            * anni(p, "beta", True)
                            * anni(r, "beta", True)
                            * anni(s, "beta", False)
                            * anni(q, "beta", False)
                        )
                        hamiltonian_operator += (
                            gaabb_mo[p, q, r, s]
                            * anni(p, "alpha", True)
                            * anni(r, "beta", True)
                            * anni(s, "beta", False)
                            * anni(q, "alpha", False)
                        )
    return hamiltonian_operator

def unrestricted_hamiltonian_0i_0a_1elec(
    haa_mo: np.ndarray,
    hbb_mo: np.ndarray,
    gaaaa_mo: np.ndarray,
    gbbbb_mo: np.ndarray,
    gaabb_mo: np.ndarray,
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
    hamiltonian_operator = FermionicOperator({}, {})
    # Inactive one-electron
    for i in range(num_inactive_orbs):
        if abs(haa_mo[i, i]) > 10**-14 or abs(hbb_mo[i, i]) > 10**-14:
            hamiltonian_operator += haa_mo[i, i] * anni(i, "alpha", True) * anni(i, "alpha", False)
            hamiltonian_operator += hbb_mo[i, i] * anni(i, "beta", True) * anni(i, "beta", False)
    # Active one-electron
    for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            if abs(haa_mo[p, q]) > 10**-14 or abs(hbb_mo[p, q]) > 10**-14:
                hamiltonian_operator += haa_mo[p, q] * anni(p, "alpha", True) * anni(q, "alpha", False)
                hamiltonian_operator += hbb_mo[p, q] * anni(p, "beta", True) * anni(q, "beta", False)
    return hamiltonian_operator
