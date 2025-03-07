from collections.abc import Generator

import numpy as np
from sympy.utilities.iterables import multiset_permutations

from slowquant.unitary_coupled_cluster.operator_matrix import CI_Info


def get_indexing_extended(
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    order: int,
) -> CI_Info:
    r"""Get indexing between index and determinant, extended to include complete active-space on-top of a full space singles or full space singles and doubles.

    Needed for full-space operators (e.g. orbital rotations between spaces) that act on the reference before the unitary ansatz is applied (e.g. $Uq\left|CSF\right>$) .
    This leads to a change in particle number in the active space and precludes the standard indexing formalism that is based on operator folding into the active space.
    Now the determinant basis spans a larger portion of the Fock space made of the complete active space and singles and doubles in virtual and occupied space.

    Args:
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        order: Excitation order the space will be extended with.

    Returns:
        List to map index to determiant and dictionary to map determiant to index.
    """
    if order > 2:
        raise ValueError("Excitation order needs to be <= 2")
    # Obtain additional determinants from single excitations that break active space particle symmetry
    inactive_singles = []
    virtual_singles = []
    for inactive, virtual in generate_singles(num_inactive_orbs, num_virtual_orbs):
        inactive_singles.append(inactive)
        virtual_singles.append(virtual)
    # Obtain additional determinants from double excitations that break active space particle symmetry
    inactive_doubles = []
    virtual_doubles = []
    if order == 2:
        for inactive, virtual in generate_doubles(num_inactive_orbs, num_virtual_orbs):
            inactive_doubles.append(inactive)
            virtual_doubles.append(virtual)
    idx = 0
    idx2det = []
    det2idx = {}
    # Generate 0th space
    # Particle and spin conserving determinants in active space. No excitation in occ and virtual orbs.
    for alpha_string in multiset_permutations(
        [1] * num_active_elec_alpha + [0] * (num_active_orbs - num_active_elec_alpha)
    ):  # active space permutations in alpha
        for beta_string in multiset_permutations(
            [1] * num_active_elec_beta + [0] * (num_active_orbs - num_active_elec_beta)
        ):  # active space permutations in beta
            det_str = ""
            for a, b in zip(
                [1] * num_inactive_orbs + alpha_string + [0] * num_virtual_orbs,
                [1] * num_inactive_orbs + beta_string + [0] * num_virtual_orbs,
            ):
                det_str += str(a) + str(b)
            det = int(det_str, 2)
            if det in idx2det:
                continue
            idx2det.append(det)
            det2idx[det] = idx
            idx += 1
    # Generate 1,2 exc alpha space
    # Loop over occ and virtual particle number breaking determinants (single and double exc) for alpha electrons
    # Beta electrons stay particle number conserving in active space
    for alpha_inactive, alpha_virtual in zip(
        inactive_singles + inactive_doubles, virtual_singles + virtual_doubles
    ):  # singles/doubles inactive and virtual determinants in alpha
        active_alpha_elec = int(
            num_active_elec_alpha - np.sum(alpha_virtual) + num_inactive_orbs - np.sum(alpha_inactive)
        )
        for alpha_string in multiset_permutations(
            [1] * active_alpha_elec + [0] * (num_active_orbs - active_alpha_elec)
        ):  # active space permutations in alpha
            for beta_string in multiset_permutations(
                [1] * num_active_elec_beta + [0] * (num_active_orbs - num_active_elec_beta)
            ):  # active space permutations in alpha
                det_str = ""
                for a, b in zip(
                    alpha_inactive + alpha_string + alpha_virtual,
                    [1] * num_inactive_orbs + beta_string + [0] * num_virtual_orbs,
                ):
                    det_str += str(a) + str(b)
                det = int(det_str, 2)
                if det in idx2det:
                    continue
                idx2det.append(det)
                det2idx[det] = idx
                idx += 1
    # Generate 1,2 exc beta space
    # Loop over occ and virtual particle number breaking determinants (single and double exc) for beta orbs
    # Alpha orbs stay particle number conserving in active space
    for beta_inactive, beta_virtual in zip(
        inactive_singles + inactive_doubles, virtual_singles + virtual_doubles
    ):  # singles/doubles inactive and virtual determinants in beta
        active_beta_elec = int(
            num_active_elec_beta - np.sum(beta_virtual) + num_inactive_orbs - np.sum(beta_inactive)
        )
        for alpha_string in multiset_permutations(
            [1] * num_active_elec_alpha + [0] * (num_active_orbs - num_active_elec_alpha)
        ):  # active space permutations in alpha
            for beta_string in multiset_permutations(
                [1] * active_beta_elec + [0] * (num_active_orbs - active_beta_elec)
            ):  # active space permutations in beta
                det_str = ""
                for a, b in zip(
                    [1] * num_inactive_orbs + alpha_string + [0] * num_virtual_orbs,
                    beta_inactive + beta_string + beta_virtual,
                ):
                    det_str += str(a) + str(b)
                det = int(det_str, 2)
                if det in idx2det:
                    continue
                idx2det.append(det)
                det2idx[det] = idx
                idx += 1
    # Generate 1 exc alpha 1 exc beta space
    # Loop over occ and virtual particle number breaking determinants (single excitation) for alpha and beta orbs
    if order == 2:
        for alpha_inactive, alpha_virtual in zip(inactive_singles, virtual_singles):
            active_alpha_elec = int(
                num_active_elec_alpha - np.sum(alpha_virtual) + num_inactive_orbs - np.sum(alpha_inactive)
            )  # singles inactive and virtual determinants in alpha
            for beta_inactive, beta_virtual in zip(inactive_singles, virtual_singles):
                active_beta_elec = int(
                    num_active_elec_beta - np.sum(beta_virtual) + num_inactive_orbs - np.sum(beta_inactive)
                )  # singles inactive and virtual determinants in beta
                for alpha_string in multiset_permutations(
                    [1] * active_alpha_elec + [0] * (num_active_orbs - active_alpha_elec)
                ):  # active space permutations in alpha
                    for beta_string in multiset_permutations(
                        [1] * active_beta_elec + [0] * (num_active_orbs - active_beta_elec)
                    ):  # active space permutations in beta
                        det_str = ""
                        for a, b in zip(
                            alpha_inactive + alpha_string + alpha_virtual,
                            beta_inactive + beta_string + beta_virtual,
                        ):
                            det_str += str(a) + str(b)
                        det = int(det_str, 2)
                        if det in idx2det:
                            continue
                        idx2det.append(det)
                        det2idx[det] = idx
                        idx += 1
    ci_info = CI_Info(
        0,
        num_inactive_orbs + num_active_orbs + num_virtual_orbs,
        0,
        num_active_elec_alpha + num_inactive_orbs,
        num_active_elec_beta + num_inactive_orbs,
        np.array(idx2det, dtype=int),
        det2idx,
    )
    ci_info.space_extension_offset = num_inactive_orbs
    return ci_info


def generate_singles(
    num_inactive_orbs: int, num_virtual_orbs: int
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Generate single excited determinant in the inactive and virtual space.

    These are generated via single excitation between all three spaces and thus are only particle conserving in the full space.
    It includes single excitations: inactive -> virtual, inactive -> active (no change in virtual), active -> virtual (no change in occ)
    The reference is also included.

    Args:
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.

    Returns:
        Single excited determinants.
    """
    inactive = [1] * num_inactive_orbs
    virtual = [0] * num_virtual_orbs
    # loop over excitations out of all inactive orbs
    # add loop index for not changing inactive orb
    for i in range(num_inactive_orbs + 1):
        if i != num_inactive_orbs:  # excite out
            inactive[i] = 0
        # loop over excitations into virtual orbs
        # add loop index for not changing virtual orb
        for j in range(num_virtual_orbs + 1):
            if j != num_virtual_orbs:  # excite in
                virtual[j] = 1
            yield inactive.copy(), virtual.copy()
            if j != num_virtual_orbs:  # reset
                virtual[j] = 0
        if i != num_inactive_orbs:  # reset
            inactive[i] = 1


def generate_doubles(
    num_inactive_orbs: int, num_virtual_orbs: int
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Generate double excited determinant in the inactive and virtual space.

    These are generated via double excitation between all three spaces and thus are only particle conserving in the full space.
    It includes double excitations: inactive -> virtual, inactive -> active (no change in virtual), active -> virtual (no change in occ)
    The reference is also included.

    Args:
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.

    Returns:
        Double excited determinants.
    """
    inactive = [1] * num_inactive_orbs
    virtual = [0] * num_virtual_orbs
    # loop over excitations out of all inactive orbs
    # add loop index for not changing inactive orb
    for i in range(num_inactive_orbs + 1):
        if i != num_inactive_orbs:
            inactive[i] = 0
        # second orb for excitation out of
        for i2 in range(min(i + 1, num_inactive_orbs), num_inactive_orbs + 1):
            if i2 != num_inactive_orbs:
                inactive[i2] = 0
            # loop over excitations into virtual orbs
            # add loop index for not changing virtual orb
            for j in range(num_virtual_orbs + 1):
                if j != num_virtual_orbs:
                    virtual[j] = 1
                # second orb for excitation into
                for j2 in range(min(j + 1, num_virtual_orbs), num_virtual_orbs + 1):
                    if j2 != num_virtual_orbs:
                        virtual[j2] = 1
                    yield inactive.copy(), virtual.copy()
                    if j2 != num_virtual_orbs:
                        virtual[j2] = 0
                if j != num_virtual_orbs:
                    virtual[j] = 0
            if i2 != num_inactive_orbs:
                inactive[i2] = 1
        if i != num_inactive_orbs:
            inactive[i] = 1
