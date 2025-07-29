import math
from collections.abc import Generator, Sequence

import numba as nb
import numpy as np
import scipy.sparse as ss

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    G3,
    G4,
    G5,
    G6,
    G1_sa,
    G2_sa,
)
from slowquant.unitary_coupled_cluster.util import UccStructure, UpsStructure


def build_operator_matrix(op: FermionicOperator, ci_info: CI_Info) -> np.ndarray:
    """Build matrix representation of operator.

    Args:
        op: Fermionic number and spin conserving operator.
        ci_info: Information about the CI space.

    Returns:
        Matrix representation of operator.
    """
    idx2det = ci_info.idx2det
    det2idx = ci_info.det2idx
    num_active_orbs = ci_info.num_active_orbs
    num_dets = len(idx2det)  # number of spin and particle conserving determinants
    ones = np.ones(
        num_dets
    )  # Used with the determinant generator below as state argument. This ensures that no screening of determinants based on state vector weight is performed.
    op_mat = np.zeros((num_dets, num_dets))  # basis
    # Create bitstrings for parity check. Contains occupied determinant up to orbital index.
    parity_check = np.zeros(2 * num_active_orbs + 1, dtype=int)
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    # loop over all strings of annihilation operators in FermionicOperator sum
    for fermi_label in op.factors:  # get strings as key of op.factors
        # Separate each annihilation operator string in creation and annihilation indices
        anni_idx = []
        create_idx = []
        for fermi_op in op.operators[fermi_label]:
            if fermi_op.dagger:
                create_idx.append(fermi_op.idx)
            else:
                anni_idx.append(fermi_op.idx)
        anni_idx = np.array(anni_idx, dtype=int)
        create_idx = np.array(create_idx, dtype=int)
        # loop over all determinants
        for i, det in get_determinants(idx2det, ones, anni_idx, create_idx, num_active_orbs):
            phase_changes = 0
            # evaluate how string of annihilation operator change det
            for fermi_op in reversed(op.operators[fermi_label]):
                orb_idx = fermi_op.idx
                det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                # take care of phases using parity_check
                phase_changes += (det & parity_check[orb_idx]).bit_count()
            op_mat[det2idx[det], i] += op.factors[fermi_label] * (-1) ** phase_changes
    return op_mat


def propagate_state(
    operators: list[FermionicOperator | str],
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: Sequence[float] | None = None,
    wf_struct: UpsStructure | UccStructure | None = None,
    do_folding: bool = True,
    do_unsafe: bool = False,
) -> np.ndarray:
    r"""Propagate state by applying operators.

    The operators will be folded to only work on the active orbitals.
    The resulting state should not be acted on with another folded operator.
    This would violate the "do not multiply folded operators" rule.

    .. math::
        \left|\tilde{0}\right> = \hat{O}\left|0\right>

    Args:
        operators: List of operators.
        state: State.
        ci_info: Information about the CI space.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        wf_struct: wave function structure object.
        do_folding: Do folding of operator (default: True).
        do_unsafe: Ignore elements that are outside the space defined in ci_info. (default: False)
                If not ignored, getting elements outside the space will stop the calculation.

    Returns:
        New state.
    """
    idx2det = ci_info.idx2det
    det2idx = ci_info.det2idx
    num_inactive_orbs = ci_info.num_inactive_orbs
    num_active_orbs = ci_info.num_active_orbs
    num_virtual_orbs = ci_info.num_virtual_orbs
    if len(operators) == 0:
        return np.copy(state)
    new_state = np.copy(state)
    tmp_state = np.zeros_like(state)
    # Create bitstrings for parity check. Contains occupied determinant up to orbital index.
    parity_check = np.zeros(2 * num_active_orbs + 1, dtype=int)
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    for op in operators[::-1]:
        # Ansatz unitary in operators
        if isinstance(op, str):
            if op not in ("U", "Ud"):
                raise ValueError(f"Unknown str operator, expected ('U', 'Ud') got {op}")
            dagger = False
            if op == "Ud":
                dagger = True
            if isinstance(wf_struct, UpsStructure):
                if thetas is None:
                    raise ValueError("theta must be different from None")
                new_state = construct_ups_state(
                    new_state,
                    ci_info,
                    thetas,
                    wf_struct,
                    dagger=dagger,
                )
            elif isinstance(wf_struct, UccStructure):
                if thetas is None:
                    raise ValueError("theta must be different from None")
                new_state = construct_ucc_state(
                    new_state,
                    ci_info,
                    thetas,
                    wf_struct,
                    dagger=dagger,
                )
            else:
                raise TypeError(f"Got unknown wave function structure type, {type(wf_struct)}")
        # FermionicOperator in operators
        else:
            tmp_state[:] = 0.0
            # Fold operator to only get active contributions
            if do_folding:
                op_folded = op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
            else:
                op_folded = op
            # loop over all strings of annihilation operators in FermionicOperator sum
            for fermi_label in op_folded.factors:
                # Separate each annihilation operator string in creation and annihilation indices
                anni_idx = []
                create_idx = []
                for fermi_op in op_folded.operators[fermi_label]:
                    if fermi_op.dagger:
                        create_idx.append(fermi_op.idx)
                    else:
                        anni_idx.append(fermi_op.idx)
                anni_idx = np.array(anni_idx, dtype=int)
                create_idx = np.array(create_idx, dtype=int)
                # loop over all determinants in new_state
                for i, det in get_determinants(idx2det, new_state, anni_idx, create_idx, num_active_orbs):
                    phase_changes = 0
                    # evaluate how string of annihilation operator change det
                    for fermi_op in reversed(op_folded.operators[fermi_label]):
                        orb_idx = fermi_op.idx
                        det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                        # take care of phases using parity_check
                        phase_changes += (det & parity_check[orb_idx]).bit_count()
                    if do_unsafe:
                        # For some algorithms it is guaranteed that the application of operators will always
                        # keep the new determinants within a pre-defined space (in det2idx and idx2det).
                        # For these algorithms it is a sign of bug if a keyerror when calling det2idx is found.
                        # These algorithms thus does also not need to check for the exsistence of the new determinant
                        # in det2idx.
                        # For other algorithms this 'safety' is not guaranteed, hence the keyword is called 'do_unsafe'.
                        if det not in det2idx:
                            continue
                    tmp_state[det2idx[det]] += (
                        op_folded.factors[fermi_label] * (-1) ** phase_changes * new_state[i]
                    )  # Update value
            new_state = np.copy(tmp_state)
    return new_state


@nb.jit(nopython=True)
def get_determinants(
    det: np.ndarray,
    state: np.ndarray,
    anni_idxs: np.ndarray,
    create_idxs: np.ndarray,
    num_active_orbs: int,
) -> Generator[tuple[int, int], None, None]:
    """Generate relevant determinants.

    This part is factored out for performance - jit.

    This generator yields determinants that does need reach kill-state.
    It is assumed that operators are normal-ordered for the checking of kill-state to work.

    Args:
        det: List of all determinants.
        state: State-vector.
        anni_idxs: Annihilation operator indices.
        create_idxs: Creation operator indices.
        num_active_orbs: Number of active spatial orbitals.

    Returns:
        Index of determinant and determinant.
    """
    for i, val in enumerate(state):
        if abs(val) < 10**-14:
            continue
        for anni_idx in anni_idxs:
            if (det[i] >> 2 * num_active_orbs - 1 - anni_idx) & 1 == 1:
                continue
            # If an annihilation operator works on zero, then we reach kill-state.
            break
        else:  # no-break
            for create_idx in create_idxs:
                if create_idx in anni_idxs:
                    # A creation operator can always act on index that an annihilation operator,
                    # has previously worked on.
                    continue
                if (det[i] >> 2 * num_active_orbs - 1 - create_idx) & 1 == 0:
                    continue
                # If creation operator works on one, then we reach kill-state.
                break
            else:  # no-break
                yield i, det[i]


def propagate_state_SA(
    operators: list[FermionicOperator | str],
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: Sequence[float] | None = None,
    wf_struct: UpsStructure | None = None,
    do_folding: bool = True,
) -> np.ndarray:
    r"""Propagate state by applying operator.

    The operator will be folded to only work on the active orbitals.
    The resulting state should not be acted on with another folded operator.
    This would violate the "do not multiply folded operators" rule.

    .. math::
        \left|\tilde{0}\right> = \hat{O}\left|0\right>

    Args:
        operators: List of operators.
        state: State.
        ci_info: Information about the CI space.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        wf_struct: wave function structure object.
        do_folding: Do folding of operator (default: True).

    Returns:
        New state.
    """
    idx2det = ci_info.idx2det
    det2idx = ci_info.det2idx
    num_inactive_orbs = ci_info.num_inactive_orbs
    num_active_orbs = ci_info.num_active_orbs
    num_virtual_orbs = ci_info.num_virtual_orbs
    if len(operators) == 0:
        return np.copy(state)
    new_state = np.copy(state)
    tmp_state = np.zeros_like(state)
    # Create bitstrings for parity check. Contains occupied determinant up to orbital index.
    parity_check = np.zeros(2 * num_active_orbs + 1, dtype=int)
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    for op in operators[::-1]:
        # Ansatz unitary in operators
        if isinstance(op, str):
            if op not in ("U", "Ud"):
                raise ValueError(f"Unknown str operator, expected ('U', 'Ud') got {op}")
            dagger = False
            if op == "Ud":
                dagger = True
            if isinstance(wf_struct, UpsStructure):
                if thetas is None:
                    raise ValueError("theta must be different from None")
                new_state = construct_ups_state_SA(
                    new_state,
                    ci_info,
                    thetas,
                    wf_struct,
                    dagger=dagger,
                )
            else:
                raise TypeError(f"Got unknown wave function structure type, {type(wf_struct)}")
        # FermionicOperator in operators
        else:
            tmp_state[:, :] = 0.0
            # Fold operator to only get active contributions
            if do_folding:
                op_folded = op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
            else:
                op_folded = op
            # loop over all strings of annihilation operators in FermionicOperator sum
            for fermi_label in op_folded.factors:
                # Separate each annihilation operator string in creation and annihilation indices
                anni_idx = []
                create_idx = []
                for fermi_op in op_folded.operators[fermi_label]:
                    if fermi_op.dagger:
                        create_idx.append(fermi_op.idx)
                    else:
                        anni_idx.append(fermi_op.idx)
                anni_idx = np.array(anni_idx, dtype=int)
                create_idx = np.array(create_idx, dtype=int)
                # loop over all determinants
                for i, det in get_determinants_SA(idx2det, new_state, anni_idx, create_idx, num_active_orbs):
                    phase_changes = 0
                    # evaluate how string of annihilation operator change det
                    for fermi_op in reversed(op_folded.operators[fermi_label]):
                        orb_idx = fermi_op.idx
                        det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                        # take care of phases using parity_check
                        phase_changes += (det & parity_check[orb_idx]).bit_count()
                    val = op_folded.factors[fermi_label] * (-1) ** phase_changes
                    tmp_state[:, det2idx[det]] += val * new_state[:, i]  # Update value
            new_state = np.copy(tmp_state)
    return new_state


@nb.jit(nopython=True)
def get_determinants_SA(
    det: np.ndarray,
    state: np.ndarray,
    anni_idxs: np.ndarray,
    create_idxs: np.ndarray,
    num_active_orbs: int,
) -> Generator[tuple[int, int], None, None]:
    """Generate relevant determinants.

    This part is factored out for performance - jit.

    This generator yields determinants that does need reach kill-state.
    It is assumed that operators are normal-ordered for the checking of kill-state to work.

    Args:
        det: List of all determinants.
        state: State-vector.
        anni_idxs: Annihilation operator indices.
        create_idxs: Creation operator indices.
        num_active_orbs: Number of active spatial orbitals.

    Returns:
        Index of determinant and determinant.
    """
    for i, vals in enumerate(state.T):
        is_non_zero = False
        for val in vals:
            if abs(val) > 10**-14:
                is_non_zero = True
                break
        if not is_non_zero:
            continue
        for anni_idx in anni_idxs:
            if (det[i] >> 2 * num_active_orbs - 1 - anni_idx) & 1 == 1:
                continue
            # If an annihilation operator works on zero, then we reach kill-state.
            break
        else:  # no-break
            for create_idx in create_idxs:
                if create_idx in anni_idxs:
                    # A creation operator can always act on index that an annihilation operator,
                    # has previously worked on.
                    continue
                if (det[i] >> 2 * num_active_orbs - 1 - create_idx) & 1 == 0:
                    continue
                # If creation operator works on one, then we reach kill-state.
                break
            else:  # no-break
                yield i, det[i]


def expectation_value(
    bra: np.ndarray,
    operators: list[FermionicOperator | str],
    ket: np.ndarray,
    ci_info: CI_Info,
    thetas: Sequence[float] | None = None,
    wf_struct: UpsStructure | UccStructure | None = None,
    do_folding: bool = True,
    do_unsafe: bool = False,
) -> float:
    """Calculate expectation value of operator using propagate state.

    Args:
        bra: Bra state.
        operators: Operator.
        ket: Ket state.
        ci_info: Information about the CI space.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        wf_struct: Wave function structure object.
        do_folding: Do folding of operator (default: True).
        do_unsafe: Ignore elements that are outside the space defined in ci_info. (default: False)
                If not ignored, getting elements outside the space will stop the calculation.

    Returns:
        Expectation value.
    """
    # build state vector of operator on ket
    op_ket = propagate_state(
        operators,
        ket,
        ci_info,
        thetas,
        wf_struct,
        do_folding=do_folding,
        do_unsafe=do_unsafe,
    )
    val = bra @ op_ket
    if not isinstance(val, float):
        raise ValueError(f"Calculated expectation value is not a float, got type {type(val)}")
    return val


def expectation_value_SA(
    bra: np.ndarray,
    operators: list[FermionicOperator | str],
    ket: np.ndarray,
    ci_info: CI_Info,
    thetas: Sequence[float] | None = None,
    wf_struct: UpsStructure | None = None,
    do_folding: bool = True,
) -> float:
    """Calculate expectation value of operator with a SA wave function using propagate state.

    Args:
        bra: Bra state.
        operators: Operator.
        ket: Ket state.
        ci_info: Information about the CI space.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        wf_struct: Wave function structure object.
        do_folding: Do folding of operator (default: True).

    Returns:
        Expectation value.
    """
    # build state vector of operator on ket
    op_ket = propagate_state_SA(
        operators,
        ket,
        ci_info,
        thetas,
        wf_struct,
        do_folding=do_folding,
    )
    val = np.einsum("ij,ij->", bra, op_ket)
    if not isinstance(val, float):
        raise ValueError(f"Calculated expectation value is not a float, got type {type(val)}")
    return val / len(bra)


def construct_ucc_state(
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: Sequence[float],
    ucc_struct: UccStructure,
    dagger: bool = False,
) -> np.ndarray:
    """Construct UCC state by applying UCC unitary to reference state.

    Args:
        state: Reference state vector.
        ci_info: Information about the CI space.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        ucc_struct: UCCStructure object.
        dagger: If true, do dagger unitaries.

    Returns:
        New state vector with unitaries applied.
    """
    # Build up T matrix based on excitations in ucc_struct and given thetas
    T = get_ucc_T(thetas, ucc_struct, ci_info.space_extension_offset)
    # Evil matrix construction
    Tmat = build_operator_matrix(T, ci_info)
    if dagger:
        return ss.linalg.expm_multiply(-Tmat, state, traceA=0.0)
    return ss.linalg.expm_multiply(Tmat, state, traceA=0.0)


def get_ucc_T(
    thetas: Sequence[float],
    ucc_struct: UccStructure,
    offset: int = 0,
) -> FermionicOperator:
    """Construct UCC operator.

    Args:
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        ucc_struct: UCCStructure object.
        offset: Offset needed for extended spaces.

    Returns:
        UCC operator.
    """
    # Build up T matrix based on excitations in ucc_struct and given thetas
    T = FermionicOperator({}, {})
    for exc_type, exc_indices, theta in zip(
        ucc_struct.excitation_operator_type, ucc_struct.excitation_indices, thetas
    ):
        if abs(theta) < 10**-14:
            continue
        if exc_type == "sa_single":
            (i, a) = np.array(exc_indices) + offset
            T += theta * G1_sa(i, a, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = np.array(exc_indices) + offset
            T += theta * G2_sa(i, j, a, b, 1, True)
        elif exc_type == "sa_double_2":
            (i, j, a, b) = np.array(exc_indices) + offset
            T += theta * G2_sa(i, j, a, b, 2, True)
        elif exc_type == "sa_double_3":
            (i, j, a, b) = np.array(exc_indices) + offset
            T += theta * G2_sa(i, j, a, b, 3, True)
        elif exc_type == "sa_double_4":
            (i, j, a, b) = np.array(exc_indices) + offset
            T += theta * G2_sa(i, j, a, b, 4, True)
        elif exc_type == "sa_double_5":
            (i, j, a, b) = np.array(exc_indices) + offset
            T += theta * G2_sa(i, j, a, b, 5, True)
        elif exc_type == "single":
            (i, a) = np.array(exc_indices) + 2 * offset
            T += theta * G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = np.array(exc_indices) + 2 * offset
            T += theta * G2(i, j, a, b, True)
        elif exc_type == "triple":
            (i, j, k, a, b, c) = np.array(exc_indices) + 2 * offset
            T += theta * G3(i, j, k, a, b, c, True)
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = np.array(exc_indices) + 2 * offset
            T += theta * G4(i, j, k, l, a, b, c, d, True)
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = np.array(exc_indices) + 2 * offset
            T += theta * G5(i, j, k, l, m, a, b, c, d, e, True)
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = np.array(exc_indices) + 2 * offset
            T += theta * G6(i, j, k, l, m, n, a, b, c, d, e, f, True)
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return T


def construct_ups_state(
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
    dagger: bool = False,
) -> np.ndarray:
    r"""Construct unitary product state by applying UPS unitary to reference state.

    .. math::
        \boldsymbol{U}_N...\boldsymbol{U}_0\left|\nu\right> = \left|\tilde\nu\right>

    #. 10.48550/arXiv.2303.10825, Eq. 15

    Args:
        state: Reference state vector.
        ci_info: Information about the CI space.
        thetas: Ansatz parameters values.
        ups_struct: Unitary product state structure.
        dagger: If true, do dagger unitaries.

    Returns:
        New state vector with unitaries applied.
    """
    out = state.copy()
    order = 1
    offset = ci_info.space_extension_offset
    if dagger:
        order = -1
    # Loop over all excitation in UPSStructure
    for exc_type, exc_indices, theta in zip(
        ups_struct.excitation_operator_type[::order], ups_struct.excitation_indices[::order], thetas[::order]
    ):
        if abs(theta) < 10**-14:
            continue
        if dagger:
            theta = -theta
        if exc_type in ("sa_single",):
            A = 1  # 2**(-1/2)
            (i, a) = np.array(exc_indices) + offset
            # Create T matrix
            Ta = G1(i * 2, a * 2, True)
            Tb = G1(i * 2 + 1, a * 2 + 1, True)
            # Analytical application on state vector
            out = (
                out
                + np.sin(A * theta)
                * propagate_state(
                    [Ta],
                    out,
                    ci_info,
                    do_folding=False,
                )
                + (1 - np.cos(A * theta))
                * propagate_state(
                    [Ta, Ta],
                    out,
                    ci_info,
                    do_folding=False,
                )
            )
            out = (
                out
                + np.sin(A * theta)
                * propagate_state(
                    [Tb],
                    out,
                    ci_info,
                    do_folding=False,
                )
                + (1 - np.cos(A * theta))
                * propagate_state(
                    [Tb, Tb],
                    out,
                    ci_info,
                    do_folding=False,
                )
            )
        elif exc_type in ("single", "double", "sa_double_1"):
            # Create T matrix
            if exc_type == "single":
                (i, a) = np.array(exc_indices) + 2 * offset
                T = G1(i, a, True)
            elif exc_type == "double":
                (i, j, a, b) = np.array(exc_indices) + 2 * offset
                T = G2(i, j, a, b, True)
            elif exc_type == "sa_double_1":
                (i, j, a, b) = np.array(exc_indices) + offset
                T = G2_sa(i, j, a, b, 1, True)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Analytical application on state vector
            out = (
                out
                + np.sin(theta)
                * propagate_state(
                    [T],
                    out,
                    ci_info,
                    do_folding=False,
                )
                + (1 - np.cos(theta))
                * propagate_state(
                    [T, T],
                    out,
                    ci_info,
                    do_folding=False,
                )
            )
        elif exc_type in ("sa_double_2", "sa_double_3"):
            if exc_type == "sa_double_2":
                (i, j, a, b) = np.array(exc_indices) + offset
                T = G2_sa(i, j, a, b, 2, True)
            elif exc_type == "sa_double_3":
                (i, j, a, b) = np.array(exc_indices) + offset
                T = G2_sa(i, j, a, b, 3, True)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            S = (1, math.sqrt(2) / 2)
            k1 = (-1, 2 * math.sqrt(2))
            k3 = (-2, 2 * math.sqrt(2))
            k2 = (1, -4)
            k4 = (2, -4)
            tmp = propagate_state(
                [T],
                out,
                ci_info,
                do_folding=False,
            )
            out += (k1[0] * np.sin(S[0] * theta) + k1[1] * np.sin(S[1] * theta)) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (k2[0] * (np.cos(S[0] * theta) - 1) + k2[1] * (np.cos(S[1] * theta) - 1)) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (k3[0] * np.sin(S[0] * theta) + k3[1] * np.sin(S[1] * theta)) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (k4[0] * (np.cos(S[0] * theta) - 1) + k4[1] * (np.cos(S[1] * theta) - 1)) * tmp
        elif exc_type in ("sa_double_4",):
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 4, True)
            S = (1, math.sqrt(2), math.sqrt(2) / 2, 1 / 2)  # type: ignore
            k1 = (2 / 3, -math.sqrt(2) / 42, -8 * math.sqrt(2) / 3, 128 / 21)  # type: ignore
            k3 = (13 / 3, -math.sqrt(2) / 6, -44 * math.sqrt(2) / 3, 64 / 3)  # type: ignore
            k5 = (22 / 3, -math.sqrt(2) / 3, -52 * math.sqrt(2) / 3, 64 / 3)  # type: ignore
            k7 = (8 / 3, -4 * math.sqrt(2) / 21, -16 * math.sqrt(2) / 3, 128 / 21)  # type: ignore
            k2 = (-2 / 3, 1 / 42, 16 / 3, -256 / 21)  # type: ignore
            k4 = (-13 / 3, 1 / 6, 88 / 3, -128 / 3)  # type: ignore
            k6 = (-22 / 3, 1 / 3, 104 / 3, -128 / 3)  # type: ignore
            k8 = (-8 / 3, 4 / 21, 32 / 3, -256 / 21)  # type: ignore
            tmp = propagate_state(
                [T],
                out,
                ci_info,
                do_folding=False,
            )
            out += (
                k1[0] * np.sin(S[0] * theta)  # type: ignore
                + k1[1] * np.sin(S[1] * theta)  # type: ignore
                + k1[2] * np.sin(S[2] * theta)  # type: ignore
                + k1[3] * np.sin(S[3] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k2[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k2[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k2[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k2[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k3[0] * np.sin(S[0] * theta)  # type: ignore
                + k3[1] * np.sin(S[1] * theta)  # type: ignore
                + k3[2] * np.sin(S[2] * theta)  # type: ignore
                + k3[3] * np.sin(S[3] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k4[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k4[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k4[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k4[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k5[0] * np.sin(S[0] * theta)  # type: ignore
                + k5[1] * np.sin(S[1] * theta)  # type: ignore
                + k5[2] * np.sin(S[2] * theta)  # type: ignore
                + k5[3] * np.sin(S[3] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k6[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k6[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k6[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k6[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k7[0] * np.sin(S[0] * theta)  # type: ignore
                + k7[1] * np.sin(S[1] * theta)  # type: ignore
                + k7[2] * np.sin(S[2] * theta)  # type: ignore
                + k7[3] * np.sin(S[3] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k8[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k8[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k8[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k8[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            ) * tmp
        elif exc_type in ("sa_double_5",):
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 5, True)
            S = (math.sqrt(2), math.sqrt(2) / 2, math.sqrt(3) / 3, math.sqrt(3) / 2, math.sqrt(3) / 6)  # type: ignore
            k1 = (  # type: ignore
                math.sqrt(2) / 1150,
                8 * math.sqrt(2) / 5,
                -54 * math.sqrt(3) / 25,
                -16 * math.sqrt(3) / 75,
                432 * math.sqrt(3) / 115,
            )
            k3 = (  # type: ignore
                11 * math.sqrt(2) / 690,
                404 * math.sqrt(2) / 15,
                -171 * math.sqrt(3) / 5,
                -56 * math.sqrt(3) / 15,
                2952 * math.sqrt(3) / 115,
            )
            k5 = (  # type: ignore
                133 * math.sqrt(2) / 1725,
                308 * math.sqrt(2) / 3,
                -2718 * math.sqrt(3) / 25,
                -1192 * math.sqrt(3) / 75,
                1368 * math.sqrt(3) / 23,
            )
            k7 = (  # type: ignore
                16 * math.sqrt(2) / 115,
                608 * math.sqrt(2) / 5,
                -576 * math.sqrt(3) / 5,
                -112 * math.sqrt(3) / 5,
                6192 * math.sqrt(3) / 115,
            )
            k9 = (  # type: ignore
                48 * math.sqrt(2) / 575,
                192 * math.sqrt(2) / 5,
                -864 * math.sqrt(3) / 25,
                -192 * math.sqrt(3) / 25,
                1728 * math.sqrt(3) / 115,
            )
            k2 = (-1 / 1150, -16 / 5, 162 / 25, 32 / 75, -2592 / 115)  # type: ignore
            k4 = (-11 / 690, -808 / 15, 513 / 5, 112 / 15, -17712 / 115)  # type: ignore
            k6 = (-133 / 1725, -616 / 3, 8154 / 25, 2384 / 75, -8208 / 23)  # type: ignore
            k8 = (-16 / 115, -1216 / 5, 1728 / 5, 224 / 5, -37152 / 115)  # type: ignore
            k10 = (-48 / 575, -384 / 5, 2592 / 25, 384 / 25, -10368 / 115)  # type: ignore
            tmp = propagate_state(
                [T],
                out,
                ci_info,
                do_folding=False,
            )
            out += (
                k1[0] * np.sin(S[0] * theta)  # type: ignore
                + k1[1] * np.sin(S[1] * theta)  # type: ignore
                + k1[2] * np.sin(S[2] * theta)  # type: ignore
                + k1[3] * np.sin(S[3] * theta)  # type: ignore
                + k1[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k2[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k2[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k2[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k2[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k2[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k3[0] * np.sin(S[0] * theta)  # type: ignore
                + k3[1] * np.sin(S[1] * theta)  # type: ignore
                + k3[2] * np.sin(S[2] * theta)  # type: ignore
                + k3[3] * np.sin(S[3] * theta)  # type: ignore
                + k3[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k4[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k4[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k4[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k4[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k4[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k5[0] * np.sin(S[0] * theta)  # type: ignore
                + k5[1] * np.sin(S[1] * theta)  # type: ignore
                + k5[2] * np.sin(S[2] * theta)  # type: ignore
                + k5[3] * np.sin(S[3] * theta)  # type: ignore
                + k5[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k6[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k6[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k6[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k6[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k6[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k7[0] * np.sin(S[0] * theta)  # type: ignore
                + k7[1] * np.sin(S[1] * theta)  # type: ignore
                + k7[2] * np.sin(S[2] * theta)  # type: ignore
                + k7[3] * np.sin(S[3] * theta)  # type: ignore
                + k7[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k8[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k8[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k8[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k8[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k8[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k9[0] * np.sin(S[0] * theta)  # type: ignore
                + k9[1] * np.sin(S[1] * theta)  # type: ignore
                + k9[2] * np.sin(S[2] * theta)  # type: ignore
                + k9[3] * np.sin(S[3] * theta)  # type: ignore
                + k9[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k10[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k10[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k10[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k10[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k10[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return out


def construct_ups_state_SA(
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
    dagger: bool = False,
) -> np.ndarray:
    r"""Construct unitary product state by applying UPS unitary to reference state.

    .. math::
        \boldsymbol{U}_N...\boldsymbol{U}_0\left|\nu\right> = \left|\tilde\nu\right>

    #. 10.48550/arXiv.2303.10825, Eq. 15

    Args:
        state: Reference state vector.
        ci_info: Information about the CI space.
        thetas: Ansatz parameters values.
        ups_struct: Unitary product state structure.
        dagger: If true, do dagger unitaries.

    Returns:
        New state vector with unitaries applied.
    """
    out = state.copy()
    order = 1
    offset = ci_info.space_extension_offset
    if dagger:
        order = -1
    # Loop over all excitation in UPSStructure
    for exc_type, exc_indices, theta in zip(
        ups_struct.excitation_operator_type[::order], ups_struct.excitation_indices[::order], thetas[::order]
    ):
        if abs(theta) < 10**-14:
            continue
        if dagger:
            theta = -theta
        if exc_type in ("sa_single",):
            A = 1  # 2**(-1/2)
            (i, a) = np.array(exc_indices) + offset
            # Create T matrices
            Ta = G1(i * 2, a * 2, True)
            Tb = G1(i * 2 + 1, a * 2 + 1, True)
            # Analytical application on state vector
            out = (
                out
                + np.sin(A * theta)
                * propagate_state_SA(
                    [Ta],
                    out,
                    ci_info,
                    do_folding=False,
                )
                + (1 - np.cos(A * theta))
                * propagate_state_SA(
                    [Ta, Ta],
                    out,
                    ci_info,
                    do_folding=False,
                )
            )
            out = (
                out
                + np.sin(A * theta)
                * propagate_state_SA(
                    [Tb],
                    out,
                    ci_info,
                    do_folding=False,
                )
                + (1 - np.cos(A * theta))
                * propagate_state_SA(
                    [Tb, Tb],
                    out,
                    ci_info,
                    do_folding=False,
                )
            )
        elif exc_type in ("single", "double", "sa_double_1"):
            # Create T matrix
            if exc_type == "single":
                (i, a) = np.array(exc_indices) + 2 * offset
                T = G1(i, a, True)
            elif exc_type == "double":
                (i, j, a, b) = np.array(exc_indices) + 2 * offset
                T = G2(i, j, a, b, True)
            elif exc_type == "sa_double_1":
                (i, j, a, b) = np.array(exc_indices) + offset
                T = G2_sa(i, j, a, b, 1, True)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Analytical application on state vector
            out = (
                out
                + np.sin(theta)
                * propagate_state_SA(
                    [T],
                    out,
                    ci_info,
                    do_folding=False,
                )
                + (1 - np.cos(theta))
                * propagate_state_SA(
                    [T, T],
                    out,
                    ci_info,
                    do_folding=False,
                )
            )
        elif exc_type in ("sa_double_2", "sa_double_3"):
            if exc_type == "sa_double_2":
                (i, j, a, b) = np.array(exc_indices) + offset
                T = G2_sa(i, j, a, b, 2, True)
            elif exc_type == "sa_double_3":
                (i, j, a, b) = np.array(exc_indices) + offset
                T = G2_sa(i, j, a, b, 3, True)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            S = (1, math.sqrt(2) / 2)
            k1 = (-1, 2 * math.sqrt(2))
            k3 = (-2, 2 * math.sqrt(2))
            k2 = (1, -4)
            k4 = (2, -4)
            tmp = propagate_state_SA(
                [T],
                out,
                ci_info,
                do_folding=False,
            )
            out += (k1[0] * np.sin(S[0] * theta) + k1[1] * np.sin(S[1] * theta)) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (k2[0] * (np.cos(S[0] * theta) - 1) + k2[1] * (np.cos(S[1] * theta) - 1)) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (k3[0] * np.sin(S[0] * theta) + k3[1] * np.sin(S[1] * theta)) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (k4[0] * (np.cos(S[0] * theta) - 1) + k4[1] * (np.cos(S[1] * theta) - 1)) * tmp
        elif exc_type in ("sa_double_4",):
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 4, True)
            S = (1, math.sqrt(2), math.sqrt(2) / 2, 1 / 2)  # type: ignore
            k1 = (2 / 3, -math.sqrt(2) / 42, -8 * math.sqrt(2) / 3, 128 / 21)  # type: ignore
            k3 = (13 / 3, -math.sqrt(2) / 6, -44 * math.sqrt(2) / 3, 64 / 3)  # type: ignore
            k5 = (22 / 3, -math.sqrt(2) / 3, -52 * math.sqrt(2) / 3, 64 / 3)  # type: ignore
            k7 = (8 / 3, -4 * math.sqrt(2) / 21, -16 * math.sqrt(2) / 3, 128 / 21)  # type: ignore
            k2 = (-2 / 3, 1 / 42, 16 / 3, -256 / 21)  # type: ignore
            k4 = (-13 / 3, 1 / 6, 88 / 3, -128 / 3)  # type: ignore
            k6 = (-22 / 3, 1 / 3, 104 / 3, -128 / 3)  # type: ignore
            k8 = (-8 / 3, 4 / 21, 32 / 3, -256 / 21)  # type: ignore
            tmp = propagate_state_SA(
                [T],
                out,
                ci_info,
                do_folding=False,
            )
            out += (
                k1[0] * np.sin(S[0] * theta)  # type: ignore
                + k1[1] * np.sin(S[1] * theta)  # type: ignore
                + k1[2] * np.sin(S[2] * theta)  # type: ignore
                + k1[3] * np.sin(S[3] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k2[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k2[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k2[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k2[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k3[0] * np.sin(S[0] * theta)  # type: ignore
                + k3[1] * np.sin(S[1] * theta)  # type: ignore
                + k3[2] * np.sin(S[2] * theta)  # type: ignore
                + k3[3] * np.sin(S[3] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k4[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k4[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k4[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k4[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k5[0] * np.sin(S[0] * theta)  # type: ignore
                + k5[1] * np.sin(S[1] * theta)  # type: ignore
                + k5[2] * np.sin(S[2] * theta)  # type: ignore
                + k5[3] * np.sin(S[3] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k6[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k6[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k6[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k6[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k7[0] * np.sin(S[0] * theta)  # type: ignore
                + k7[1] * np.sin(S[1] * theta)  # type: ignore
                + k7[2] * np.sin(S[2] * theta)  # type: ignore
                + k7[3] * np.sin(S[3] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k8[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k8[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k8[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k8[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            ) * tmp
        elif exc_type in ("sa_double_5",):
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 5, True)
            S = (math.sqrt(2), math.sqrt(2) / 2, math.sqrt(3) / 3, math.sqrt(3) / 2, math.sqrt(3) / 6)  # type: ignore
            k1 = (  # type: ignore
                math.sqrt(2) / 1150,
                8 * math.sqrt(2) / 5,
                -54 * math.sqrt(3) / 25,
                -16 * math.sqrt(3) / 75,
                432 * math.sqrt(3) / 115,
            )
            k3 = (  # type: ignore
                11 * math.sqrt(2) / 690,
                404 * math.sqrt(2) / 15,
                -171 * math.sqrt(3) / 5,
                -56 * math.sqrt(3) / 15,
                2952 * math.sqrt(3) / 115,
            )
            k5 = (  # type: ignore
                133 * math.sqrt(2) / 1725,
                308 * math.sqrt(2) / 3,
                -2718 * math.sqrt(3) / 25,
                -1192 * math.sqrt(3) / 75,
                1368 * math.sqrt(3) / 23,
            )
            k7 = (  # type: ignore
                16 * math.sqrt(2) / 115,
                608 * math.sqrt(2) / 5,
                -576 * math.sqrt(3) / 5,
                -112 * math.sqrt(3) / 5,
                6192 * math.sqrt(3) / 115,
            )
            k9 = (  # type: ignore
                48 * math.sqrt(2) / 575,
                192 * math.sqrt(2) / 5,
                -864 * math.sqrt(3) / 25,
                -192 * math.sqrt(3) / 25,
                1728 * math.sqrt(3) / 115,
            )
            k2 = (-1 / 1150, -16 / 5, 162 / 25, 32 / 75, -2592 / 115)  # type: ignore
            k4 = (-11 / 690, -808 / 15, 513 / 5, 112 / 15, -17712 / 115)  # type: ignore
            k6 = (-133 / 1725, -616 / 3, 8154 / 25, 2384 / 75, -8208 / 23)  # type: ignore
            k8 = (-16 / 115, -1216 / 5, 1728 / 5, 224 / 5, -37152 / 115)  # type: ignore
            k10 = (-48 / 575, -384 / 5, 2592 / 25, 384 / 25, -10368 / 115)  # type: ignore
            tmp = propagate_state_SA(
                [T],
                out,
                ci_info,
                do_folding=False,
            )
            out += (
                k1[0] * np.sin(S[0] * theta)  # type: ignore
                + k1[1] * np.sin(S[1] * theta)  # type: ignore
                + k1[2] * np.sin(S[2] * theta)  # type: ignore
                + k1[3] * np.sin(S[3] * theta)  # type: ignore
                + k1[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k2[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k2[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k2[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k2[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k2[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k3[0] * np.sin(S[0] * theta)  # type: ignore
                + k3[1] * np.sin(S[1] * theta)  # type: ignore
                + k3[2] * np.sin(S[2] * theta)  # type: ignore
                + k3[3] * np.sin(S[3] * theta)  # type: ignore
                + k3[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k4[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k4[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k4[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k4[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k4[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k5[0] * np.sin(S[0] * theta)  # type: ignore
                + k5[1] * np.sin(S[1] * theta)  # type: ignore
                + k5[2] * np.sin(S[2] * theta)  # type: ignore
                + k5[3] * np.sin(S[3] * theta)  # type: ignore
                + k5[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k6[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k6[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k6[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k6[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k6[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k7[0] * np.sin(S[0] * theta)  # type: ignore
                + k7[1] * np.sin(S[1] * theta)  # type: ignore
                + k7[2] * np.sin(S[2] * theta)  # type: ignore
                + k7[3] * np.sin(S[3] * theta)  # type: ignore
                + k7[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k8[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k8[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k8[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k8[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k8[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k9[0] * np.sin(S[0] * theta)  # type: ignore
                + k9[1] * np.sin(S[1] * theta)  # type: ignore
                + k9[2] * np.sin(S[2] * theta)  # type: ignore
                + k9[3] * np.sin(S[3] * theta)  # type: ignore
                + k9[4] * np.sin(S[4] * theta)  # type: ignore
            ) * tmp
            tmp = propagate_state_SA(
                [T],
                tmp,
                ci_info,
                do_folding=False,
            )
            out += (
                k10[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
                + k10[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
                + k10[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
                + k10[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
                + k10[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
            ) * tmp
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return out


def propagate_unitary(
    state: np.ndarray,
    idx: int,
    ci_info: CI_Info,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
) -> np.ndarray:
    """Apply unitary from UPS operator number 'idx' to state.

    Args:
        state: State vector.
        idx: Index of operator in the ups_struct.
        ci_info: Information about the CI space.
        thetas: Values for ansatz parameters.
        ups_struct: UPS structure object.

    Returns:
        State with unitary applied.
    """
    # Select unitary operation based on idx
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indices[idx]
    theta = thetas[idx]
    offset = ci_info.space_extension_offset
    if abs(theta) < 10**-14:
        return np.copy(state)
    if exc_type in ("sa_single",):
        A = 1  # 2**(-1/2)
        (i, a) = np.array(exc_indices) + offset
        # Create T matrix
        Ta = G1(i * 2, a * 2, True)
        Tb = G1(i * 2 + 1, a * 2 + 1, True)
        # Analytical application on state vector
        out = (
            state
            + np.sin(A * theta)
            * propagate_state(
                [Ta],
                state,
                ci_info,
                do_folding=False,
            )
            + (1 - np.cos(A * theta))
            * propagate_state(
                [Ta, Ta],
                state,
                ci_info,
                do_folding=False,
            )
        )
        out = (
            out
            + np.sin(A * theta)
            * propagate_state(
                [Tb],
                out,
                ci_info,
                do_folding=False,
            )
            + (1 - np.cos(A * theta))
            * propagate_state(
                [Tb, Tb],
                out,
                ci_info,
                do_folding=False,
            )
        )
    elif exc_type in ("single", "double", "sa_double_1"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = np.array(exc_indices) + 2 * offset
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = np.array(exc_indices) + 2 * offset
            T = G2(i, j, a, b, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 1, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Analytical application on state vector
        out = (
            state
            + np.sin(theta)
            * propagate_state(
                [T],
                state,
                ci_info,
                do_folding=False,
            )
            + (1 - np.cos(theta))
            * propagate_state(
                [T, T],
                state,
                ci_info,
                do_folding=False,
            )
        )
    elif exc_type in ("sa_double_2", "sa_double_3"):
        if exc_type == "sa_double_2":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 2, True)
        elif exc_type == "sa_double_3":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 3, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        S = (1, math.sqrt(2) / 2)
        k1 = (-1, 2 * math.sqrt(2))
        k3 = (-2, 2 * math.sqrt(2))
        k2 = (1, -4)
        k4 = (2, -4)
        out = np.copy(state)
        tmp = propagate_state(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
        out += (k1[0] * np.sin(S[0] * theta) + k1[1] * np.sin(S[1] * theta)) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (k2[0] * (np.cos(S[0] * theta) - 1) + k2[1] * (np.cos(S[1] * theta) - 1)) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (k3[0] * np.sin(S[0] * theta) + k3[1] * np.sin(S[1] * theta)) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (k4[0] * (np.cos(S[0] * theta) - 1) + k4[1] * (np.cos(S[1] * theta) - 1)) * tmp
    elif exc_type in ("sa_double_4",):
        (i, j, a, b) = np.array(exc_indices) + offset
        T = G2_sa(i, j, a, b, 4, True)
        S = (1, math.sqrt(2), math.sqrt(2) / 2, 1 / 2)  # type: ignore
        k1 = (2 / 3, -math.sqrt(2) / 42, -8 * math.sqrt(2) / 3, 128 / 21)  # type: ignore
        k3 = (13 / 3, -math.sqrt(2) / 6, -44 * math.sqrt(2) / 3, 64 / 3)  # type: ignore
        k5 = (22 / 3, -math.sqrt(2) / 3, -52 * math.sqrt(2) / 3, 64 / 3)  # type: ignore
        k7 = (8 / 3, -4 * math.sqrt(2) / 21, -16 * math.sqrt(2) / 3, 128 / 21)  # type: ignore
        k2 = (-2 / 3, 1 / 42, 16 / 3, -256 / 21)  # type: ignore
        k4 = (-13 / 3, 1 / 6, 88 / 3, -128 / 3)  # type: ignore
        k6 = (-22 / 3, 1 / 3, 104 / 3, -128 / 3)  # type: ignore
        k8 = (-8 / 3, 4 / 21, 32 / 3, -256 / 21)  # type: ignore
        out = np.copy(state)
        tmp = propagate_state(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
        out += (
            k1[0] * np.sin(S[0] * theta)  # type: ignore
            + k1[1] * np.sin(S[1] * theta)  # type: ignore
            + k1[2] * np.sin(S[2] * theta)  # type: ignore
            + k1[3] * np.sin(S[3] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k2[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k2[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k2[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k2[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k3[0] * np.sin(S[0] * theta)  # type: ignore
            + k3[1] * np.sin(S[1] * theta)  # type: ignore
            + k3[2] * np.sin(S[2] * theta)  # type: ignore
            + k3[3] * np.sin(S[3] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k4[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k4[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k4[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k4[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k5[0] * np.sin(S[0] * theta)  # type: ignore
            + k5[1] * np.sin(S[1] * theta)  # type: ignore
            + k5[2] * np.sin(S[2] * theta)  # type: ignore
            + k5[3] * np.sin(S[3] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k6[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k6[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k6[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k6[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k7[0] * np.sin(S[0] * theta)  # type: ignore
            + k7[1] * np.sin(S[1] * theta)  # type: ignore
            + k7[2] * np.sin(S[2] * theta)  # type: ignore
            + k7[3] * np.sin(S[3] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k8[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k8[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k8[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k8[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
        ) * tmp
    elif exc_type in ("sa_double_5",):
        (i, j, a, b) = np.array(exc_indices) + offset
        T = G2_sa(i, j, a, b, 5, True)
        S = (math.sqrt(2), math.sqrt(2) / 2, math.sqrt(3) / 3, math.sqrt(3) / 2, math.sqrt(3) / 6)  # type: ignore
        k1 = (  # type: ignore
            math.sqrt(2) / 1150,
            8 * math.sqrt(2) / 5,
            -54 * math.sqrt(3) / 25,
            -16 * math.sqrt(3) / 75,
            432 * math.sqrt(3) / 115,
        )
        k3 = (  # type: ignore
            11 * math.sqrt(2) / 690,
            404 * math.sqrt(2) / 15,
            -171 * math.sqrt(3) / 5,
            -56 * math.sqrt(3) / 15,
            2952 * math.sqrt(3) / 115,
        )
        k5 = (  # type: ignore
            133 * math.sqrt(2) / 1725,
            308 * math.sqrt(2) / 3,
            -2718 * math.sqrt(3) / 25,
            -1192 * math.sqrt(3) / 75,
            1368 * math.sqrt(3) / 23,
        )
        k7 = (  # type: ignore
            16 * math.sqrt(2) / 115,
            608 * math.sqrt(2) / 5,
            -576 * math.sqrt(3) / 5,
            -112 * math.sqrt(3) / 5,
            6192 * math.sqrt(3) / 115,
        )
        k9 = (  # type: ignore
            48 * math.sqrt(2) / 575,
            192 * math.sqrt(2) / 5,
            -864 * math.sqrt(3) / 25,
            -192 * math.sqrt(3) / 25,
            1728 * math.sqrt(3) / 115,
        )
        k2 = (-1 / 1150, -16 / 5, 162 / 25, 32 / 75, -2592 / 115)  # type: ignore
        k4 = (-11 / 690, -808 / 15, 513 / 5, 112 / 15, -17712 / 115)  # type: ignore
        k6 = (-133 / 1725, -616 / 3, 8154 / 25, 2384 / 75, -8208 / 23)  # type: ignore
        k8 = (-16 / 115, -1216 / 5, 1728 / 5, 224 / 5, -37152 / 115)  # type: ignore
        k10 = (-48 / 575, -384 / 5, 2592 / 25, 384 / 25, -10368 / 115)  # type: ignore
        out = np.copy(state)
        tmp = propagate_state(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
        out += (
            k1[0] * np.sin(S[0] * theta)  # type: ignore
            + k1[1] * np.sin(S[1] * theta)  # type: ignore
            + k1[2] * np.sin(S[2] * theta)  # type: ignore
            + k1[3] * np.sin(S[3] * theta)  # type: ignore
            + k1[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k2[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k2[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k2[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k2[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k2[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k3[0] * np.sin(S[0] * theta)  # type: ignore
            + k3[1] * np.sin(S[1] * theta)  # type: ignore
            + k3[2] * np.sin(S[2] * theta)  # type: ignore
            + k3[3] * np.sin(S[3] * theta)  # type: ignore
            + k3[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k4[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k4[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k4[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k4[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k4[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k5[0] * np.sin(S[0] * theta)  # type: ignore
            + k5[1] * np.sin(S[1] * theta)  # type: ignore
            + k5[2] * np.sin(S[2] * theta)  # type: ignore
            + k5[3] * np.sin(S[3] * theta)  # type: ignore
            + k5[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k6[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k6[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k6[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k6[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k6[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k7[0] * np.sin(S[0] * theta)  # type: ignore
            + k7[1] * np.sin(S[1] * theta)  # type: ignore
            + k7[2] * np.sin(S[2] * theta)  # type: ignore
            + k7[3] * np.sin(S[3] * theta)  # type: ignore
            + k7[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k8[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k8[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k8[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k8[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k8[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k9[0] * np.sin(S[0] * theta)  # type: ignore
            + k9[1] * np.sin(S[1] * theta)  # type: ignore
            + k9[2] * np.sin(S[2] * theta)  # type: ignore
            + k9[3] * np.sin(S[3] * theta)  # type: ignore
            + k9[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k10[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k10[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k10[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k10[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k10[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return out


def propagate_unitary_SA(
    state: np.ndarray,
    idx: int,
    ci_info: CI_Info,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
) -> np.ndarray:
    """Apply unitary from UPS operator number 'idx' to state.

    Args:
        state: State vector.
        idx: Index of operator in the ups_struct.
        ci_info: Information about the CI space.
        thetas: Values for ansatz parameters.
        ups_struct: UPS structure object.

    Returns:
        State with unitary applied.
    """
    # Select unitary operation based on idx
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indices[idx]
    theta = thetas[idx]
    offset = ci_info.space_extension_offset
    if abs(theta) < 10**-14:
        return np.copy(state)
    if exc_type in ("sa_single",):
        A = 1  # 2**(-1/2)
        (i, a) = np.array(exc_indices) + offset
        # Create T matrix
        Ta = G1(i * 2, a * 2, True)
        Tb = G1(i * 2 + 1, a * 2 + 1, True)
        # Analytical application on state vector
        out = (
            state
            + np.sin(A * theta)
            * propagate_state_SA(
                [Ta],
                state,
                ci_info,
                do_folding=False,
            )
            + (1 - np.cos(A * theta))
            * propagate_state_SA(
                [Ta, Ta],
                state,
                ci_info,
                do_folding=False,
            )
        )
        out = (
            out
            + np.sin(A * theta)
            * propagate_state_SA(
                [Tb],
                out,
                ci_info,
                do_folding=False,
            )
            + (1 - np.cos(A * theta))
            * propagate_state_SA(
                [Tb, Tb],
                out,
                ci_info,
                do_folding=False,
            )
        )
    elif exc_type in ("single", "double", "sa_double_1"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = np.array(exc_indices) + 2 * offset
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = np.array(exc_indices) + 2 * offset
            T = G2(i, j, a, b, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 1, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Analytical application on state vector
        out = (
            state
            + np.sin(theta)
            * propagate_state_SA(
                [T],
                state,
                ci_info,
                do_folding=False,
            )
            + (1 - np.cos(theta))
            * propagate_state_SA(
                [T, T],
                state,
                ci_info,
                do_folding=False,
            )
        )
    elif exc_type in ("sa_double_2", "sa_double_3"):
        if exc_type == "sa_double_2":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 2, True)
        elif exc_type == "sa_double_3":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 3, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        S = (1, math.sqrt(2) / 2)
        k1 = (-1, 2 * math.sqrt(2))
        k3 = (-2, 2 * math.sqrt(2))
        k2 = (1, -4)
        k4 = (2, -4)
        out = np.copy(state)
        tmp = propagate_state_SA(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
        out += (k1[0] * np.sin(S[0] * theta) + k1[1] * np.sin(S[1] * theta)) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (k2[0] * (np.cos(S[0] * theta) - 1) + k2[1] * (np.cos(S[1] * theta) - 1)) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (k3[0] * np.sin(S[0] * theta) + k3[1] * np.sin(S[1] * theta)) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (k4[0] * (np.cos(S[0] * theta) - 1) + k4[1] * (np.cos(S[1] * theta) - 1)) * tmp
    elif exc_type in ("sa_double_4",):
        (i, j, a, b) = np.array(exc_indices) + offset
        T = G2_sa(i, j, a, b, 4, True)
        S = (1, math.sqrt(2), math.sqrt(2) / 2, 1 / 2)  # type: ignore
        k1 = (2 / 3, -math.sqrt(2) / 42, -8 * math.sqrt(2) / 3, 128 / 21)  # type: ignore
        k3 = (13 / 3, -math.sqrt(2) / 6, -44 * math.sqrt(2) / 3, 64 / 3)  # type: ignore
        k5 = (22 / 3, -math.sqrt(2) / 3, -52 * math.sqrt(2) / 3, 64 / 3)  # type: ignore
        k7 = (8 / 3, -4 * math.sqrt(2) / 21, -16 * math.sqrt(2) / 3, 128 / 21)  # type: ignore
        k2 = (-2 / 3, 1 / 42, 16 / 3, -256 / 21)  # type: ignore
        k4 = (-13 / 3, 1 / 6, 88 / 3, -128 / 3)  # type: ignore
        k6 = (-22 / 3, 1 / 3, 104 / 3, -128 / 3)  # type: ignore
        k8 = (-8 / 3, 4 / 21, 32 / 3, -256 / 21)  # type: ignore
        out = np.copy(state)
        tmp = propagate_state_SA(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
        out += (
            k1[0] * np.sin(S[0] * theta)  # type: ignore
            + k1[1] * np.sin(S[1] * theta)  # type: ignore
            + k1[2] * np.sin(S[2] * theta)  # type: ignore
            + k1[3] * np.sin(S[3] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k2[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k2[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k2[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k2[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k3[0] * np.sin(S[0] * theta)  # type: ignore
            + k3[1] * np.sin(S[1] * theta)  # type: ignore
            + k3[2] * np.sin(S[2] * theta)  # type: ignore
            + k3[3] * np.sin(S[3] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k4[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k4[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k4[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k4[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k5[0] * np.sin(S[0] * theta)  # type: ignore
            + k5[1] * np.sin(S[1] * theta)  # type: ignore
            + k5[2] * np.sin(S[2] * theta)  # type: ignore
            + k5[3] * np.sin(S[3] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k6[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k6[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k6[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k6[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k7[0] * np.sin(S[0] * theta)  # type: ignore
            + k7[1] * np.sin(S[1] * theta)  # type: ignore
            + k7[2] * np.sin(S[2] * theta)  # type: ignore
            + k7[3] * np.sin(S[3] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k8[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k8[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k8[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k8[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
        ) * tmp
    elif exc_type in ("sa_double_5",):
        (i, j, a, b) = np.array(exc_indices) + offset
        T = G2_sa(i, j, a, b, 5, True)
        S = (math.sqrt(2), math.sqrt(2) / 2, math.sqrt(3) / 3, math.sqrt(3) / 2, math.sqrt(3) / 6)  # type: ignore
        k1 = (  # type: ignore
            math.sqrt(2) / 1150,
            8 * math.sqrt(2) / 5,
            -54 * math.sqrt(3) / 25,
            -16 * math.sqrt(3) / 75,
            432 * math.sqrt(3) / 115,
        )
        k3 = (  # type: ignore
            11 * math.sqrt(2) / 690,
            404 * math.sqrt(2) / 15,
            -171 * math.sqrt(3) / 5,
            -56 * math.sqrt(3) / 15,
            2952 * math.sqrt(3) / 115,
        )
        k5 = (  # type: ignore
            133 * math.sqrt(2) / 1725,
            308 * math.sqrt(2) / 3,
            -2718 * math.sqrt(3) / 25,
            -1192 * math.sqrt(3) / 75,
            1368 * math.sqrt(3) / 23,
        )
        k7 = (  # type: ignore
            16 * math.sqrt(2) / 115,
            608 * math.sqrt(2) / 5,
            -576 * math.sqrt(3) / 5,
            -112 * math.sqrt(3) / 5,
            6192 * math.sqrt(3) / 115,
        )
        k9 = (  # type: ignore
            48 * math.sqrt(2) / 575,
            192 * math.sqrt(2) / 5,
            -864 * math.sqrt(3) / 25,
            -192 * math.sqrt(3) / 25,
            1728 * math.sqrt(3) / 115,
        )
        k2 = (-1 / 1150, -16 / 5, 162 / 25, 32 / 75, -2592 / 115)  # type: ignore
        k4 = (-11 / 690, -808 / 15, 513 / 5, 112 / 15, -17712 / 115)  # type: ignore
        k6 = (-133 / 1725, -616 / 3, 8154 / 25, 2384 / 75, -8208 / 23)  # type: ignore
        k8 = (-16 / 115, -1216 / 5, 1728 / 5, 224 / 5, -37152 / 115)  # type: ignore
        k10 = (-48 / 575, -384 / 5, 2592 / 25, 384 / 25, -10368 / 115)  # type: ignore
        out = np.copy(state)
        tmp = propagate_state_SA(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
        out += (
            k1[0] * np.sin(S[0] * theta)  # type: ignore
            + k1[1] * np.sin(S[1] * theta)  # type: ignore
            + k1[2] * np.sin(S[2] * theta)  # type: ignore
            + k1[3] * np.sin(S[3] * theta)  # type: ignore
            + k1[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k2[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k2[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k2[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k2[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k2[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k3[0] * np.sin(S[0] * theta)  # type: ignore
            + k3[1] * np.sin(S[1] * theta)  # type: ignore
            + k3[2] * np.sin(S[2] * theta)  # type: ignore
            + k3[3] * np.sin(S[3] * theta)  # type: ignore
            + k3[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k4[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k4[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k4[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k4[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k4[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k5[0] * np.sin(S[0] * theta)  # type: ignore
            + k5[1] * np.sin(S[1] * theta)  # type: ignore
            + k5[2] * np.sin(S[2] * theta)  # type: ignore
            + k5[3] * np.sin(S[3] * theta)  # type: ignore
            + k5[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k6[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k6[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k6[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k6[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k6[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k7[0] * np.sin(S[0] * theta)  # type: ignore
            + k7[1] * np.sin(S[1] * theta)  # type: ignore
            + k7[2] * np.sin(S[2] * theta)  # type: ignore
            + k7[3] * np.sin(S[3] * theta)  # type: ignore
            + k7[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k8[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k8[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k8[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k8[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k8[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k9[0] * np.sin(S[0] * theta)  # type: ignore
            + k9[1] * np.sin(S[1] * theta)  # type: ignore
            + k9[2] * np.sin(S[2] * theta)  # type: ignore
            + k9[3] * np.sin(S[3] * theta)  # type: ignore
            + k9[4] * np.sin(S[4] * theta)  # type: ignore
        ) * tmp
        tmp = propagate_state_SA(
            [T],
            tmp,
            ci_info,
            do_folding=False,
        )
        out += (
            k10[0] * (np.cos(S[0] * theta) - 1)  # type: ignore
            + k10[1] * (np.cos(S[1] * theta) - 1)  # type: ignore
            + k10[2] * (np.cos(S[2] * theta) - 1)  # type: ignore
            + k10[3] * (np.cos(S[3] * theta) - 1)  # type: ignore
            + k10[4] * (np.cos(S[4] * theta) - 1)  # type: ignore
        ) * tmp
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return out


def get_grad_action(
    state: np.ndarray,
    idx: int,
    ci_info: CI_Info,
    ups_struct: UpsStructure,
) -> np.ndarray:
    r"""Get effect of differentiation with respect to "idx" operator in the UPS expansion.

    .. math::
        \frac{\partial}{\partial \theta_i}\left(\left<\text{CSF}\right|\boldsymbol{U}(\theta_{i-1})\boldsymbol{U}(\theta_i)\right) =
        \left<\text{CSF}\right|\boldsymbol{U}(\theta_{i-1})\frac{\partial \boldsymbol{U}(\theta_i)}{\partial \theta_i}

    With,

    .. math::
        \begin{align}
        \frac{\partial \boldsymbol{U}(\theta_i)}{\partial \theta_i} &= \frac{\partial}{\partial \theta_i}\exp\left(\theta_i \hat{T}_i\right)\\
                &= \exp\left(\theta_i \hat{T}_i\right)\hat{T}_i
        \end{align}

    This function only applies the $\hat{T}_i$ part to the state.

    #. 10.48550/arXiv.2303.10825, Eq. 20 (appendix - v1)

    Args:
        state: State vector.
        idx: Index of operator in the ups_struct.
        ci_info: Information about the CI space.
        ups_struct: UPS structure object.

    Returns:
        State with derivative of the idx'th unitary applied.
    """
    # Select unitary operation based on idx
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indices[idx]
    offset = ci_info.space_extension_offset
    if exc_type in ("sa_single",):
        # Create T matrix
        A = 1  # 2**(-1/2)
        (i, a) = np.array(exc_indices) + offset
        Ta = G1(i * 2, a * 2, True)
        Tb = G1(i * 2 + 1, a * 2 + 1, True)
        # Apply missing T factor of derivative
        tmp = propagate_state(
            [A * (Ta + Tb)],
            state,
            ci_info,
            do_folding=False,
        )
    elif exc_type in (
        "single",
        "double",
        "sa_double_1",
        "sa_double_2",
        "sa_double_3",
        "sa_double_4",
        "sa_double_5",
    ):
        # Create T matrix
        if exc_type == "single":
            (i, a) = np.array(exc_indices) + 2 * offset
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = np.array(exc_indices) + 2 * offset
            T = G2(i, j, a, b, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 1, True)
        elif exc_type == "sa_double_2":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 2, True)
        elif exc_type == "sa_double_3":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 3, True)
        elif exc_type == "sa_double_4":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 4, True)
        elif exc_type == "sa_double_5":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 5, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Apply missing T factor of derivative
        tmp = propagate_state(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def get_grad_action_SA(
    state: np.ndarray,
    idx: int,
    ci_info: CI_Info,
    ups_struct: UpsStructure,
) -> np.ndarray:
    r"""Get effect of differentiation with respect to "idx" operator in the UPS expansion.

    .. math::
        \frac{\partial}{\partial \theta_i}\left(\left<\text{CSF}\right|\boldsymbol{U}(\theta_{i-1})\boldsymbol{U}(\theta_i)\right) =
        \left<\text{CSF}\right|\boldsymbol{U}(\theta_{i-1})\frac{\partial \boldsymbol{U}(\theta_i)}{\partial \theta_i}

    With,

    .. math::
        \begin{align}
        \frac{\partial \boldsymbol{U}(\theta_i)}{\partial \theta_i} &= \frac{\partial}{\partial \theta_i}\exp\left(\theta_i \hat{T}_i\right)\\
                &= \exp\left(\theta_i \hat{T}_i\right)\hat{T}_i
        \end{align}

    This function only applies the $\hat{T}_i$ part to the state.

    #. 10.48550/arXiv.2303.10825, Eq. 20 (appendix - v1)

    Args:
        state: State vector.
        idx: Index of operator in the ups_struct.
        ci_info: Information about the CI space.
        ups_struct: UPS structure object.

    Returns:
        State with derivative of the idx'th unitary applied.
    """
    # Select unitary operation based on idx
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indices[idx]
    offset = ci_info.space_extension_offset
    if exc_type in ("sa_single",):
        # Create T matrix
        A = 1  # 2**(-1/2)
        (i, a) = np.array(exc_indices) + offset
        Ta = G1(i * 2, a * 2, True)
        Tb = G1(i * 2 + 1, a * 2 + 1, True)
        # Apply missing T factor of derivative
        tmp = propagate_state_SA(
            [A * (Ta + Tb)],
            state,
            ci_info,
            do_folding=False,
        )
    elif exc_type in (
        "single",
        "double",
        "sa_double_1",
        "sa_double_2",
        "sa_double_3",
        "sa_double_4",
        "sa_double_5",
    ):
        # Create T matrix
        if exc_type == "single":
            (i, a) = np.array(exc_indices) + 2 * offset
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = np.array(exc_indices) + 2 * offset
            T = G2(i, j, a, b, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 1, True)
        elif exc_type == "sa_double_2":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 2, True)
        elif exc_type == "sa_double_3":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 3, True)
        elif exc_type == "sa_double_4":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 4, True)
        elif exc_type == "sa_double_5":
            (i, j, a, b) = np.array(exc_indices) + offset
            T = G2_sa(i, j, a, b, 5, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Apply missing T factor of derivative
        tmp = propagate_state_SA(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp
