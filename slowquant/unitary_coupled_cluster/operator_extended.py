import functools
from collections.abc import Generator, Sequence

import numpy as np
import scipy.sparse as ss
from sympy.utilities.iterables import multiset_permutations

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    G3,
    G4,
    G5,
    G6,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
)
from slowquant.unitary_coupled_cluster.util import UccStructure, UpsStructure


def get_indexing_extended(
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    order: int,
) -> tuple[list[int], dict[int, int]]:
    r"""Get indexing between index and determiant, extended to include complete active-space on-top of a full space singles or full space singles and doubles.

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
    return idx2det, det2idx


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
        # add loop index for not changing virtal orb
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
            # add loop index for not changing virtal orb
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


def build_operator_matrix_extended(
    op: FermionicOperator,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_orbs: int,
) -> np.ndarray:
    """Build matrix representation of operator.

    Args:
        op: Fermionic number and spin conserving operator.
        idx2det: Index to determinant.
        det2idx: Determinant to index.
        num_orbs: Number of spatial orbitals.

    Returns:
        Matrix representation of operator.
    """
    num_dets = len(idx2det)  # number of determinants in extended space
    op_mat = np.zeros((num_dets, num_dets))  # basis
    # Create bitstrings for parity check. Key=orbital index. Value=det as int
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_orbs - i] = num
    # loop over all determinants
    for i in range(num_dets):
        # loop over all strings of annihilation operators in FermionicOperator sum
        for fermi_label in op.factors:
            det = idx2det[i]
            phase_changes = 0
            # evaluate how string of annihilation operator change det
            # take care of phases using parity_check
            for fermi_op in op.operators[fermi_label][::-1]:
                orb_idx = fermi_op.idx
                nth_bit = (det >> 2 * num_orbs - 1 - orb_idx) & 1
                if nth_bit == 0 and fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
                elif nth_bit == 1 and fermi_op.dagger:
                    break
                elif nth_bit == 0 and not fermi_op.dagger:
                    break
                elif nth_bit == 1 and not fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
            else:  # nobreak
                if det not in det2idx:
                    continue
                val = op.factors[fermi_label] * (-1) ** phase_changes
                if abs(val) > 10**-14:
                    op_mat[det2idx[det], i] += val
    return op_mat


def propagate_state_extended(
    operators: list[FermionicOperator | str],
    state: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure | UccStructure,
    order: int,
) -> np.ndarray:
    r"""Propagate state by applying operator.

    This operates in the extended space, so no operator folding is performed.

    .. math::
        \left|\tilde{0}\right> = \hat{O}\left|0\right>

    Args:
        operators: List of operators.
        state: State.
        idx2det: Index to determiant mapping.
        det2idx: Determinant to index mapping.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        wf_struct: wave function structure object

    Returns:
        New state.
    """
    if len(operators) == 0:
        return np.copy(state)
    num_dets = len(idx2det)
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    new_state = np.copy(state)
    tmp_state = np.zeros(num_dets)
    # Create bitstrings for parity check. Key=orbital index. Value=det as int
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_orbs - i] = num

    for op in operators[::-1]:
        tmp_state[:] = 0.0
        # Ansatz unitary in operators
        if isinstance(op, str):
            if op not in ("U", "Ud"):
                raise ValueError(f"Unknown str operator, expected ('U', 'Ud') got {op}")
            dagger = False
            if op == "Ud":
                dagger = True
            if isinstance(wf_struct, UpsStructure):
                new_state = construct_ups_state_extended(
                    new_state,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    thetas,
                    wf_struct,
                    order,
                    dagger=dagger,
                )
            elif isinstance(wf_struct, UccStructure):
                new_state = construct_ucc_state_extended(
                    new_state,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    thetas,
                    wf_struct,
                    order,
                    dagger=dagger,
                )
            else:
                raise TypeError(f"Got unknown wave function structure type, {type(wf_struct)}")
        # FermionicOperator in operators
        else:
            # loop over all determinants
            for i in range(num_dets):
                if abs(new_state[i]) < 10**-14:
                    continue
                # loop over all strings of annihilation operators in FermionicOperator sum
                for fermi_label in op.factors:
                    det = idx2det[i]
                    phase_changes = 0
                    # evaluate how string of annihilation operator change det
                    # take care of phases using parity_check
                    for fermi_op in op.operators[fermi_label][::-1]:
                        orb_idx = fermi_op.idx
                        nth_bit = (det >> 2 * num_orbs - 1 - orb_idx) & 1
                        if nth_bit == 0 and fermi_op.dagger:
                            det = det ^ 2 ** (2 * num_orbs - 1 - orb_idx)
                            phase_changes += (det & parity_check[orb_idx]).bit_count()
                        elif nth_bit == 1 and fermi_op.dagger:
                            break
                        elif nth_bit == 0 and not fermi_op.dagger:
                            break
                        elif nth_bit == 1 and not fermi_op.dagger:
                            det = det ^ 2 ** (2 * num_orbs - 1 - orb_idx)
                            phase_changes += (det & parity_check[orb_idx]).bit_count()
                    else:  # nobreak
                        if det not in det2idx:
                            continue
                        val = op.factors[fermi_label] * (-1) ** phase_changes
                        if abs(val) > 10**-14:
                            tmp_state[det2idx[det]] += val * new_state[i]  # Update value
            new_state = np.copy(tmp_state)
    return new_state


def expectation_value_extended(
    bra: np.ndarray,
    operators: list[FermionicOperator | str],
    ket: np.ndarray,
    idx2det: Sequence[int],
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure | UccStructure,
    order: int,
) -> float:
    """Calculate expectation value of operator.

    Args:
        bra: Bra state.
        op: Operator.
        ket: Ket state.
        idx2det: Index to determinant mapping.
        det2idx: Determinant to index mapping.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        wf_struct: wave function structure object

    Returns:
        Expectation value.
    """
    # build state vector of operator on ket
    op_ket = propagate_state_extended(
        operators,
        ket,
        idx2det,
        det2idx,
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_active_elec_alpha,
        num_active_elec_beta,
        thetas,
        wf_struct,
        order,
    )
    return bra @ op_ket


@functools.cache
def T1_sa_extended_matrix(
    i: int,
    a: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T1 spin-adapted cluster operator.

    Args:
        i: Strongly occupied spatial orbital index.
        a: Weakly occupied spatial orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G1_sa(i, a), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_1_sa_extended_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T2 spin-adapted cluster operator - G2_1 part.

    Args:
        i: Strongly occupied spatial orbital index.
        j: Strongly occupied spatial orbital index.
        a: Weakly occupied spatial orbital index.
        b: Weakly occupied spatial orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G2_1_sa(i, j, a, b), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_2_sa_extended_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T2 spin-adapted cluster operator - G2_2 part.

    Args:
        i: Strongly occupied spatial orbital index.
        j: Strongly occupied spatial orbital index.
        a: Weakly occupied spatial orbital index.
        b: Weakly occupied spatial orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G2_2_sa(i, j, a, b), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T1_extended_matrix(
    i: int,
    a: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T1 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G1(i, a), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T2_extended_matrix(
    i: int,
    j: int,
    a: int,
    b: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T2 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G2(i, j, a, b), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T3_extended_matrix(
    i: int,
    j: int,
    k: int,
    a: int,
    b: int,
    c: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T3 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G3(i, j, k, a, b, c), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T4_extended_matrix(
    i: int,
    j: int,
    k: int,
    l: int,
    a: int,
    b: int,
    c: int,
    d: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T4 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G4(i, j, k, l, a, b, c, d), idx2det, det2idx, num_inactive_orbs + num_active_orbs + num_virtual_orbs
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T5_extended_matrix(
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
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T5 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        m: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        e: Weakly occupied spin orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G5(i, j, k, l, m, a, b, c, d, e),
        idx2det,
        det2idx,
        num_inactive_orbs + num_active_orbs + num_virtual_orbs,
    )
    return ss.lil_array(op - op.conjugate().transpose())


@functools.cache
def T6_extended_matrix(
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
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    order: int,
) -> ss.lil_array:
    """Get matrix representation of anti-Hermitian T6 spin-conserving cluster operator.

    Args:
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        k: Strongly occupied spin orbital index.
        l: Strongly occupied spin orbital index.
        m: Strongly occupied spin orbital index.
        n: Strongly occupied spin orbital index.
        a: Weakly occupied spin orbital index.
        b: Weakly occupied spin orbital index.
        c: Weakly occupied spin orbital index.
        d: Weakly occupied spin orbital index.
        e: Weakly occupied spin orbital index.
        f: Weakly occupied spin orbital index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        order: Excitation order of extended space.

    Returns:
        Matrix representation of anti-Hermitian cluster operator.
    """
    idx2det, det2idx = get_indexing_extended(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_elec_alpha,
        num_elec_beta,
        order,
    )
    op = build_operator_matrix_extended(
        G6(i, j, k, l, m, n, a, b, c, d, e, f),
        idx2det,
        det2idx,
        num_inactive_orbs + num_active_orbs + num_virtual_orbs,
    )
    return ss.lil_array(op - op.conjugate().transpose())


def construct_ucc_state_extended(
    state: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    thetas: Sequence[float],
    ucc_struct: UccStructure,
    order: int,
    dagger: bool = False,
) -> np.ndarray:
    """Construct UCC state by applying UCC unitary to reference state.

    Args:
        num_det: Number of determinants.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        theta: Active-space parameters.
               Ordered as (S, D, T, ...).
        theta_picker: Helper class to pick the parameters in the right order.
        excitations: Excitation orders to include.
        order: Excitation order of extended space.
        dagger: If true, do dagger unitaries.

    Returns:
        New state vector with unitaries applied.
    """
    # Build up T matrix based on excitations in ucc_struct and given thetas
    T = np.zeros((len(state), len(state)))
    for exc_type, exc_indices, theta in zip(
        ucc_struct.excitation_operator_type, ucc_struct.excitation_indicies, thetas
    ):
        if abs(theta) < 10**-14:
            continue
        if exc_type == "sa_single":
            (i, a) = exc_indices
            T += (
                theta
                * T1_sa_extended_matrix(
                    i + num_inactive_orbs,
                    a + num_inactive_orbs,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            )
        elif exc_type == "sa_double_1":
            (i, j, a, b) = exc_indices
            T += (
                theta
                * T2_1_sa_extended_matrix(
                    i + num_inactive_orbs,
                    j + num_inactive_orbs,
                    a + num_inactive_orbs,
                    b + num_inactive_orbs,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            )
        elif exc_type == "sa_double_2":
            (i, j, a, b) = exc_indices
            T += (
                theta
                * T2_2_sa_extended_matrix(
                    i + num_inactive_orbs,
                    j + num_inactive_orbs,
                    a + num_inactive_orbs,
                    b + num_inactive_orbs,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            )
        elif exc_type == "triple":
            (i, j, k, a, b, c) = exc_indices
            T += (
                theta
                * T3_extended_matrix(
                    i + 2 * num_inactive_orbs,
                    j + 2 * num_inactive_orbs,
                    k + 2 * num_inactive_orbs,
                    a + 2 * num_inactive_orbs,
                    b + 2 * num_inactive_orbs,
                    c + 2 * num_inactive_orbs,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            )
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = exc_indices
            T += (
                theta
                * T4_extended_matrix(
                    i + 2 * num_inactive_orbs,
                    j + 2 * num_inactive_orbs,
                    k + 2 * num_inactive_orbs,
                    l + 2 * num_inactive_orbs,
                    a + 2 * num_inactive_orbs,
                    b + 2 * num_inactive_orbs,
                    c + 2 * num_inactive_orbs,
                    d + 2 * num_inactive_orbs,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            )
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = exc_indices
            T += (
                theta
                * T5_extended_matrix(
                    i + 2 * num_inactive_orbs,
                    j + 2 * num_inactive_orbs,
                    k + 2 * num_inactive_orbs,
                    l + 2 * num_inactive_orbs,
                    m + 2 * num_inactive_orbs,
                    a + 2 * num_inactive_orbs,
                    b + 2 * num_inactive_orbs,
                    c + 2 * num_inactive_orbs,
                    d + 2 * num_inactive_orbs,
                    e + 2 * num_inactive_orbs,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            )
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = exc_indices
            T += (
                theta
                * T6_extended_matrix(
                    i + 2 * num_inactive_orbs,
                    j + 2 * num_inactive_orbs,
                    k + 2 * num_inactive_orbs,
                    l + 2 * num_inactive_orbs,
                    m + 2 * num_inactive_orbs,
                    n + 2 * num_inactive_orbs,
                    a + 2 * num_inactive_orbs,
                    b + 2 * num_inactive_orbs,
                    c + 2 * num_inactive_orbs,
                    d + 2 * num_inactive_orbs,
                    e + 2 * num_inactive_orbs,
                    f + 2 * num_inactive_orbs,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    if dagger:
        return ss.linalg.expm_multiply(-T, state)
    return ss.linalg.expm_multiply(T, state)


def construct_ups_state_extended(
    state: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
    order: int,
    dagger: bool = False,
) -> np.ndarray:
    r"""Construct unitary product state by applying UPS unitary to reference state.

    .. math::
        \boldsymbol{U}_N...\boldsymbol{U}_0\left|\nu\right> = \left|\tilde\nu\right>

    #. 10.48550/arXiv.2303.10825, Eq. 15

    Args:
        state: Reference state vector.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_betaa: Number of active beta electrons.
        thetas: Ansatz parameters values.
        ups_struct: Unitary product state structure.
        order: Excitation order of extended space.
        dagger: If do dagger unitaries.

    Returns:
        New state vector with unitaries applied.
    """
    tmp = state.copy()
    order = 1
    if dagger:
        order = -1
    # Loop over all excitation in UPSStructure
    for exc_type, exc_indices, theta in zip(
        ups_struct.excitation_operator_type[::order], ups_struct.excitation_indicies[::order], thetas[::order]
    ):
        if abs(theta) < 10**-14:
            continue
        if dagger:
            theta = -theta
        if exc_type in ("sa_single",):
            A = 1  # 2**(-1/2)
            (i, a) = exc_indices
            i += num_inactive_orbs
            a += num_inactive_orbs
            # Create T matrices
            Ta = T1_extended_matrix(
                i * 2,
                a * 2,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_elec_alpha,
                num_elec_beta,
                order,
            ).todense()
            Tb = T1_extended_matrix(
                i * 2 + 1,
                a * 2 + 1,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_elec_alpha,
                num_elec_beta,
                order,
            ).todense()
            # Analytical application on state vector
            tmp = (
                tmp
                + np.sin(A * theta) * np.matmul(Ta, tmp)
                + (1 - np.cos(A * theta)) * np.matmul(Ta, np.matmul(Ta, tmp))
            )
            tmp = (
                tmp
                + np.sin(A * theta) * np.matmul(Tb, tmp)
                + (1 - np.cos(A * theta)) * np.matmul(Tb, np.matmul(Tb, tmp))
            )
        elif exc_type in ("single", "double"):
            # Create T matrix
            if exc_type == "single":
                (i, a) = exc_indices
                i += 2 * num_inactive_orbs
                a += 2 * num_inactive_orbs
                T = T1_extended_matrix(
                    i,
                    a,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            elif exc_type == "double":
                (i, j, a, b) = exc_indices
                i += 2 * num_inactive_orbs
                j += 2 * num_inactive_orbs
                a += 2 * num_inactive_orbs
                b += 2 * num_inactive_orbs
                T = T2_extended_matrix(
                    i,
                    j,
                    a,
                    b,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_elec_alpha,
                    num_elec_beta,
                    order,
                ).todense()
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Analytical application on state vector
            tmp = (
                tmp
                + np.sin(theta) * np.matmul(T, tmp)
                + (1 - np.cos(theta)) * np.matmul(T, np.matmul(T, tmp))
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp
