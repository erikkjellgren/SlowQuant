from collections.abc import Generator, Sequence

import numba as nb
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


def get_indexing(
    num_active_orbs: int, num_active_elec_alpha: int, num_active_elec_beta: int
) -> tuple[np.ndarray, dict[int, int]]:
    """Get relation between index and determinant.

    Args:
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.

    Returns:
        List to map index to determinant and dictionary to map determinant to index.
    """
    idx = 0
    idx2det = []
    det2idx = {}
    # Loop over all possible particle and spin conserving determinant combinations
    for alpha_string in multiset_permutations(
        [1] * num_active_elec_alpha + [0] * (num_active_orbs - num_active_elec_alpha)
    ):
        for beta_string in multiset_permutations(
            [1] * num_active_elec_beta + [0] * (num_active_orbs - num_active_elec_beta)
        ):
            det_str = ""
            for a, b in zip(alpha_string, beta_string):
                det_str += str(a) + str(b)
            det = int(det_str, 2)  # save determinant as int
            idx2det.append(det)  # relate index to determinant
            det2idx[det] = idx  # relate determinant to index
            idx += 1
    return np.array(idx2det, dtype=int), det2idx


def build_operator_matrix(
    op: FermionicOperator,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_active_orbs: int,
) -> np.ndarray:
    """Build matrix representation of operator.

    Args:
        op: Fermionic number and spin conserving operator.
        idx2det: Index to determinant.
        det2idx: Determinant to index.
        num_active_orbs: Number of active spatial orbitals.

    Returns:
        Matrix representation of operator.
    """
    num_dets = len(idx2det)  # number of spin and particle conserving determinants
    ones = np.ones(num_dets)  # Used with the determinant generator to ensure no determinants are screened.
    op_mat = np.zeros((num_dets, num_dets))  # basis
    # Create bitstrings for parity check. Key=orbital index. Value=det as int
    parity_check = np.zeros(2 * num_active_orbs + 1, dtype=int)
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    # loop over all strings of annihilation operators in FermionicOperator sum
    for fermi_label in op.factors:
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
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure | UccStructure,
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
        idx2det: Index to determinant.
        det2idx: Determinant to index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        wf_struct: wave function structure object.
        do_folding: Do folding of operator (default: True).

    Returns:
        New state.
    """
    if len(operators) == 0:
        return np.copy(state)
    new_state = np.copy(state)
    tmp_state = np.zeros_like(state)
    # Create bitstrings for parity check. Key=orbital index. Value=det as int
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
                new_state = construct_ups_state(
                    new_state,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    wf_struct,
                    dagger=dagger,
                )
            elif isinstance(wf_struct, UccStructure):
                new_state = construct_ucc_state(
                    new_state,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
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
                for i, det in get_determinants(idx2det, new_state, anni_idx, create_idx, num_active_orbs):
                    phase_changes = 0
                    # evaluate how string of annihilation operator change det
                    for fermi_op in reversed(op_folded.operators[fermi_label]):
                        orb_idx = fermi_op.idx
                        det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                        # take care of phases using parity_check
                        phase_changes += (det & parity_check[orb_idx]).bit_count()
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
        anni_idxs: Annihilation operator indicies.
        create_idxs: Creation operator indicies.
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
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure,
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
        idx2det: Index to determinant.
        det2idx: Determinant to index.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        wf_struct: wave function structure object.
        do_folding: Do folding of operator (default: True).

    Returns:
        New state.
    """
    if len(operators) == 0:
        return np.copy(state)
    new_state = np.copy(state)
    tmp_state = np.zeros_like(state)
    # Create bitstrings for parity check. Key=orbital index. Value=det as int
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
                new_state = construct_ups_state_SA(
                    new_state,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
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
        anni_idxs: Annihilation operator indicies.
        create_idxs: Creation operator indicies.
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


def _propagate_state(
    state: np.ndarray,
    operators: list[FermionicOperator | str],
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure | UccStructure,
    do_folding: bool = True,
) -> np.ndarray:
    """Call propagate_state.

    _propagate_state has another order of the arguments,
    which is needed when using functools.partial.
    """
    return propagate_state(
        operators,
        state,
        idx2det,
        det2idx,
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_active_elec_alpha,
        num_active_elec_beta,
        thetas,
        wf_struct,
        do_folding=do_folding,
    )


def expectation_value(
    bra: np.ndarray,
    operators: list[FermionicOperator | str],
    ket: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure | UccStructure,
    do_folding: bool = True,
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
        wf_struct: Wave function structure object.

    Returns:
        Expectation value.
    """
    # build state vector of operator on ket
    op_ket = propagate_state(
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
        do_folding=do_folding,
    )
    val = bra @ op_ket
    if not isinstance(val, float):
        raise ValueError(f"Calculated expectation value is not a float, got type {type(val)}")
    return val


def expectation_value_SA(
    bra: np.ndarray,
    operators: list[FermionicOperator | str],
    ket: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    wf_struct: UpsStructure,
    do_folding: bool = True,
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
        wf_struct: Wave function structure object.

    Returns:
        Expectation value.
    """
    # build state vector of operator on ket
    op_ket = propagate_state_SA(
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
        do_folding=do_folding,
    )
    val = np.einsum("ij,ij->", bra, op_ket)
    if not isinstance(val, float):
        raise ValueError(f"Calculated expectation value is not a float, got type {type(val)}")
    return val / len(bra)


def construct_ucc_state(
    state: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,  # pylint: disable=unused-argument
    num_active_orbs: int,
    num_virtual_orbs: int,  # pylint: disable=unused-argument
    num_active_elec_alpha: int,  # pylint: disable=unused-argument
    num_active_elec_beta: int,  # pylint: disable=unused-argument
    thetas: Sequence[float],
    ucc_struct: UccStructure,
    dagger: bool = False,
) -> np.ndarray:
    """Construct UCC state by applying UCC unitary to reference state.

    Args:
        state: Reference state vector.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        ucc_struct: UCCStructure object.
        dagger: If true, do dagger unitaries.

    Returns:
        New state vector with unitaries applied.
    """
    # Build up T matrix based on excitations in ucc_struct and given thetas
    T = get_ucc_T(thetas, ucc_struct)
    # mv = functools.partial(
    #    _propagate_state,
    #    operators=[T],
    #    idx2det=idx2det,
    #    det2idx=det2idx,
    #    num_inactive_orbs=num_inactive_orbs,
    #    num_active_orbs=num_active_orbs,
    #    num_virtual_orbs=num_virtual_orbs,
    #    num_active_elec_alpha=num_active_elec_alpha,
    #    num_active_elec_beta=num_active_elec_beta,
    #    thetas=thetas,
    #    wf_struct=ucc_struct,
    #    do_folding=False,
    # )
    # rmv = functools.partial(
    #    _propagate_state,
    #    operators=[-1.0 * T],
    #    idx2det=idx2det,
    #    det2idx=det2idx,
    #    num_inactive_orbs=num_inactive_orbs,
    #    num_active_orbs=num_active_orbs,
    #    num_virtual_orbs=num_virtual_orbs,
    #    num_active_elec_alpha=num_active_elec_alpha,
    #    num_active_elec_beta=num_active_elec_beta,
    #    thetas=thetas,
    #    wf_struct=ucc_struct,
    #    do_folding=False,
    # )

    # linopT = ss.linalg.LinearOperator((len(state), len(state)), matvec=mv, rmatvec=rmv)
    # if dagger:
    #    return ss.linalg.expm_multiply(-linopT, state, traceA=0.0)
    # return ss.linalg.expm_multiply(linopT, state, traceA=0.0)
    # Evil matrix construction
    Tmat = build_operator_matrix(T, idx2det, det2idx, num_active_orbs)
    if dagger:
        return ss.linalg.expm_multiply(-Tmat, state, traceA=0.0)
    return ss.linalg.expm_multiply(Tmat, state, traceA=0.0)


def get_ucc_T(
    thetas: Sequence[float],
    ucc_struct: UccStructure,
) -> FermionicOperator:
    """Construct UCC operator.

    Args:
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        ucc_struct: UCCStructure object.

    Returns:
        UCC operator.
    """
    # Build up T matrix based on excitations in ucc_struct and given thetas
    T = FermionicOperator({}, {})
    for exc_type, exc_indices, theta in zip(
        ucc_struct.excitation_operator_type, ucc_struct.excitation_indicies, thetas
    ):
        if abs(theta) < 10**-14:
            continue
        if exc_type == "sa_single":
            (i, a) = exc_indices
            T += theta * G1_sa(i, a, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = exc_indices
            T += theta * G2_1_sa(i, j, a, b, True)
        elif exc_type == "sa_double_2":
            (i, j, a, b) = exc_indices
            T += theta * G2_2_sa(i, j, a, b, True)
        elif exc_type == "triple":
            (i, j, k, a, b, c) = exc_indices
            T += theta * G3(i, j, k, a, b, c, True)
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = exc_indices
            T += theta * G4(i, j, k, l, a, b, c, d, True)
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = exc_indices
            T += theta * G5(i, j, k, l, m, a, b, c, d, e, True)
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = exc_indices
            T += theta * G6(i, j, k, l, m, n, a, b, c, d, e, f, True)
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return T


def construct_ups_state(
    state: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
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
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_betaa: Number of active beta electrons.
        thetas: Ansatz parameters values.
        ups_struct: Unitary product state structure.
        dagger: If true, do dagger unitaries.

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
            # Create T matrices
            Ta = G1(i * 2, a * 2, True)
            Tb = G1(i * 2 + 1, a * 2 + 1, True)
            # Analytical application on state vector
            tmp = (
                tmp
                + np.sin(A * theta)
                * propagate_state(
                    [Ta],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
                + (1 - np.cos(A * theta))
                * propagate_state(
                    [Ta, Ta],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
            )
            tmp = (
                tmp
                + np.sin(A * theta)
                * propagate_state(
                    [Tb],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
                + (1 - np.cos(A * theta))
                * propagate_state(
                    [Tb, Tb],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
            )
        elif exc_type in ("single", "double"):
            # Create T matrix
            if exc_type == "single":
                (i, a) = exc_indices
                T = G1(i, a, True)
            elif exc_type == "double":
                (i, j, a, b) = exc_indices
                T = G2(i, j, a, b, True)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Analytical application on state vector
            tmp = (
                tmp
                + np.sin(theta)
                * propagate_state(
                    [T],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
                + (1 - np.cos(theta))
                * propagate_state(
                    [T, T],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def construct_ups_state_SA(
    state: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
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
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_betaa: Number of active beta electrons.
        thetas: Ansatz parameters values.
        ups_struct: Unitary product state structure.
        dagger: If true, do dagger unitaries.

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
            # Create T matrices
            Ta = G1(i * 2, a * 2, True)
            Tb = G1(i * 2 + 1, a * 2 + 1, True)
            # Analytical application on state vector
            tmp = (
                tmp
                + np.sin(A * theta)
                * propagate_state_SA(
                    [Ta],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
                + (1 - np.cos(A * theta))
                * propagate_state_SA(
                    [Ta, Ta],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
            )
            tmp = (
                tmp
                + np.sin(A * theta)
                * propagate_state_SA(
                    [Tb],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
                + (1 - np.cos(A * theta))
                * propagate_state_SA(
                    [Tb, Tb],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
            )
        elif exc_type in ("single", "double"):
            # Create T matrix
            if exc_type == "single":
                (i, a) = exc_indices
                T = G1(i, a, True)
            elif exc_type == "double":
                (i, j, a, b) = exc_indices
                T = G2(i, j, a, b, True)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Analytical application on state vector
            tmp = (
                tmp
                + np.sin(theta)
                * propagate_state_SA(
                    [T],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
                + (1 - np.cos(theta))
                * propagate_state_SA(
                    [T, T],
                    tmp,
                    idx2det,
                    det2idx,
                    num_inactive_orbs,
                    num_active_orbs,
                    num_virtual_orbs,
                    num_active_elec_alpha,
                    num_active_elec_beta,
                    thetas,
                    ups_struct,
                    do_folding=False,
                )
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def propagate_unitary(
    state: np.ndarray,
    idx: int,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
) -> np.ndarray:
    """Apply unitary from UPS operator number 'idx' to state.

    Args:
        state: State vector.
        idx: Index of operator in the ups_struct.
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        thetas: Values for ansatz parameters.
        ups_struct: UPS structure object.

    Returns:
        State with unitary applied.
    """
    # Select unitary operation based on idx
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indicies[idx]
    theta = thetas[idx]
    if abs(theta) < 10**-14:
        return np.copy(state)
    if exc_type in ("sa_single",):
        A = 1  # 2**(-1/2)
        (i, a) = exc_indices
        # Create T matrix
        Ta = G1(i * 2, a * 2, True)
        Tb = G1(i * 2 + 1, a * 2 + 1, True)
        # Analytical application on state vector
        tmp = (
            state
            + np.sin(A * theta)
            * propagate_state(
                [Ta],
                state,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
            + (1 - np.cos(A * theta))
            * propagate_state(
                [Ta, Ta],
                state,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
        )
        tmp = (
            tmp
            + np.sin(A * theta)
            * propagate_state(
                [Tb],
                tmp,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
            + (1 - np.cos(A * theta))
            * propagate_state(
                [Tb, Tb],
                tmp,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
        )
    elif exc_type in ("single", "double"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = exc_indices
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T = G2(i, j, a, b, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Analytical application on state vector
        tmp = (
            state
            + np.sin(theta)
            * propagate_state(
                [T],
                state,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
            + (1 - np.cos(theta))
            * propagate_state(
                [T, T],
                state,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def propagate_unitary_SA(
    state: np.ndarray,
    idx: int,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
) -> np.ndarray:
    """Apply unitary from UPS operator number 'idx' to state.

    Args:
        state: State vector.
        idx: Index of operator in the ups_struct.
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        thetas: Values for ansatz parameters.
        ups_struct: UPS structure object.

    Returns:
        State with unitary applied.
    """
    # Select unitary operation based on idx
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indicies[idx]
    theta = thetas[idx]
    if abs(theta) < 10**-14:
        return np.copy(state)
    if exc_type in ("sa_single",):
        A = 1  # 2**(-1/2)
        (i, a) = exc_indices
        # Create T matrix
        Ta = G1(i * 2, a * 2, True)
        Tb = G1(i * 2 + 1, a * 2 + 1, True)
        # Analytical application on state vector
        tmp = (
            state
            + np.sin(A * theta)
            * propagate_state_SA(
                [Ta],
                state,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
            + (1 - np.cos(A * theta))
            * propagate_state_SA(
                [Ta, Ta],
                state,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
        )
        tmp = (
            tmp
            + np.sin(A * theta)
            * propagate_state_SA(
                [Tb],
                tmp,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
            + (1 - np.cos(A * theta))
            * propagate_state_SA(
                [Tb, Tb],
                tmp,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
        )
    elif exc_type in ("single", "double"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = exc_indices
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T = G2(i, j, a, b, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Analytical application on state vector
        tmp = (
            state
            + np.sin(theta)
            * propagate_state_SA(
                [T],
                state,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
            + (1 - np.cos(theta))
            * propagate_state_SA(
                [T, T],
                state,
                idx2det,
                det2idx,
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
                thetas,
                ups_struct,
                do_folding=False,
            )
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def get_grad_action(
    state: np.ndarray,
    idx: int,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs,
    num_active_orbs: int,
    num_virtual_orbs,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
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
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        ups_struct: UPS structure object.

    Returns:
        State with derivative of the idx'th unitary applied.
    """
    # Select unitary operation based on idx
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indicies[idx]
    if exc_type in ("sa_single",):
        # Create T matrix
        A = 1  # 2**(-1/2)
        (i, a) = exc_indices
        Ta = G1(i * 2, a * 2, True)
        Tb = G1(i * 2 + 1, a * 2 + 1, True)
        # Apply missing T factor of derivative
        tmp = propagate_state(
            [A * (Ta + Tb)],
            state,
            idx2det,
            det2idx,
            num_inactive_orbs,
            num_active_orbs,
            num_virtual_orbs,
            num_active_elec_alpha,
            num_active_elec_beta,
            (0.0,),
            ups_struct,
            do_folding=False,
        )
    elif exc_type in ("single", "double"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = exc_indices
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T = G2(i, j, a, b, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Apply missing T factor of derivative
        tmp = propagate_state(
            [T],
            state,
            idx2det,
            det2idx,
            num_inactive_orbs,
            num_active_orbs,
            num_virtual_orbs,
            num_active_elec_alpha,
            num_active_elec_beta,
            (0.0,),
            ups_struct,
            do_folding=False,
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def get_grad_action_SA(
    state: np.ndarray,
    idx: int,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    num_inactive_orbs,
    num_active_orbs: int,
    num_virtual_orbs,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
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
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        ups_struct: UPS structure object.

    Returns:
        State with derivative of the idx'th unitary applied.
    """
    # Select unitary operation based on idx
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indicies[idx]
    if exc_type in ("sa_single",):
        # Create T matrix
        A = 1  # 2**(-1/2)
        (i, a) = exc_indices
        Ta = G1(i * 2, a * 2, True)
        Tb = G1(i * 2 + 1, a * 2 + 1, True)
        # Apply missing T factor of derivative
        tmp = propagate_state_SA(
            [A * (Ta + Tb)],
            state,
            idx2det,
            det2idx,
            num_inactive_orbs,
            num_active_orbs,
            num_virtual_orbs,
            num_active_elec_alpha,
            num_active_elec_beta,
            (0.0,),
            ups_struct,
            do_folding=False,
        )
    elif exc_type in ("single", "double"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = exc_indices
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T = G2(i, j, a, b, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Apply missing T factor of derivative
        tmp = propagate_state_SA(
            [T],
            state,
            idx2det,
            det2idx,
            num_inactive_orbs,
            num_active_orbs,
            num_virtual_orbs,
            num_active_elec_alpha,
            num_active_elec_beta,
            (0.0,),
            ups_struct,
            do_folding=False,
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp
