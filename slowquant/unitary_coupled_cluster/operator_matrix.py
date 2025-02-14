from collections.abc import Sequence

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
) -> tuple[list[int], dict[int, int]]:
    """Get relation between index and determiant.

    Args:
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.

    Returns:
        List to map index to determiant and dictionary to map determiant to index.
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
    return idx2det, det2idx


def build_operator_matrix(
    op: FermionicOperator,
    idx2det: Sequence[int],
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
    op_mat = np.zeros((num_dets, num_dets))  # basis
    # Create bitstrings for parity check. Key=orbital index. Value=det as int
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
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
                nth_bit = (det >> 2 * num_active_orbs - 1 - orb_idx) & 1
                if nth_bit == 0 and fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
                elif nth_bit == 1 and fermi_op.dagger:
                    break
                elif nth_bit == 0 and not fermi_op.dagger:
                    break
                elif nth_bit == 1 and not fermi_op.dagger:
                    det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                    phase_changes += (det & parity_check[orb_idx]).bit_count()
            else:  # nobreak
                val = op.factors[fermi_label] * (-1) ** phase_changes
                if abs(val) > 10**-14:
                    op_mat[det2idx[det], i] += val
    return op_mat


def propagate_state(
    operators: list[FermionicOperator | str],
    state: np.ndarray,
    idx2det: Sequence[int],
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
    It is the first step to a faster matrix multiplication for expectation values.

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
        wf_struct: wave function structure object

    Returns:
        New state.
    """
    if len(operators) == 0:
        return np.copy(state)
    num_dets = len(idx2det)
    new_state = np.copy(state)
    tmp_state = np.zeros(num_dets)
    # Create bitstrings for parity check. Key=orbital index. Value=det as int
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num

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
            # Fold operator to only get active contributions
            if do_folding:
                op_folded = op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
            else:
                op_folded = op
            # loop over all determinants
            for i in range(num_dets):
                if abs(new_state[i]) < 10**-14:
                    continue
                # loop over all strings of annihilation operators in FermionicOperator sum
                for fermi_label in op_folded.factors:
                    det = idx2det[i]
                    phase_changes = 0
                    # evaluate how string of annihilation operator change det
                    # take care of phases using parity_check
                    for fermi_op in op_folded.operators[fermi_label][::-1]:
                        orb_idx = fermi_op.idx
                        nth_bit = (det >> 2 * num_active_orbs - 1 - orb_idx) & 1
                        if nth_bit == 0 and fermi_op.dagger:
                            det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                            phase_changes += (det & parity_check[orb_idx]).bit_count()
                        elif nth_bit == 1 and fermi_op.dagger:
                            break
                        elif nth_bit == 0 and not fermi_op.dagger:
                            break
                        elif nth_bit == 1 and not fermi_op.dagger:
                            det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                            phase_changes += (det & parity_check[orb_idx]).bit_count()
                    else:  # nobreak
                        val = op_folded.factors[fermi_label] * (-1) ** phase_changes
                        if abs(val) > 10**-14:
                            tmp_state[det2idx[det]] += val * new_state[i]  # Update value
            new_state = np.copy(tmp_state)
    return new_state


def propagate_state_SA(
    operators: list[FermionicOperator | str],
    state: np.ndarray,
    idx2det: Sequence[int],
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
    It is the first step to a faster matrix multiplication for expectation values.

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
        wf_struct: wave function structure object

    Returns:
        New state.
    """
    if len(operators) == 0:
        return np.copy(state)
    num_dets = len(idx2det)
    new_state = np.copy(state)
    tmp_state = np.zeros_like(state)
    # Create bitstrings for parity check. Key=orbital index. Value=det as int
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num

    for op in operators[::-1]:
        tmp_state[:, :] = 0.0
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
            # Fold operator to only get active contributions
            if do_folding:
                op_folded = op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
            else:
                op_folded = op
            # loop over all determinants
            for i in range(num_dets):
                if np.max(np.abs(new_state[:, i])) < 10**-14:
                    continue
                # loop over all strings of annihilation operators in FermionicOperator sum
                for fermi_label in op_folded.factors:
                    det = idx2det[i]
                    phase_changes = 0
                    # evaluate how string of annihilation operator change det
                    # take care of phases using parity_check
                    for fermi_op in op_folded.operators[fermi_label][::-1]:
                        orb_idx = fermi_op.idx
                        nth_bit = (det >> 2 * num_active_orbs - 1 - orb_idx) & 1
                        if nth_bit == 0 and fermi_op.dagger:
                            det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                            phase_changes += (det & parity_check[orb_idx]).bit_count()
                        elif nth_bit == 1 and fermi_op.dagger:
                            break
                        elif nth_bit == 0 and not fermi_op.dagger:
                            break
                        elif nth_bit == 1 and not fermi_op.dagger:
                            det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
                            phase_changes += (det & parity_check[orb_idx]).bit_count()
                    else:  # nobreak
                        val = op_folded.factors[fermi_label] * (-1) ** phase_changes
                        if abs(val) > 10**-14:
                            tmp_state[:, det2idx[det]] += val * new_state[:, i]  # Update value
            new_state = np.copy(tmp_state)
    return new_state


def _propagate_state(
    state: np.ndarray,
    operators: list[FermionicOperator | str],
    idx2det: Sequence[int],
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
    idx2det: Sequence[int],
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
        wf_struct: wave function structure object

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
    idx2det: Sequence[int],
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
        wf_struct: wave function structure object

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
    idx2det: Sequence[int],
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


def construct_ups_state(
    state: np.ndarray,
    idx2det: Sequence[int],
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
    idx2det: Sequence[int],
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
    idx2det: Sequence[int],
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
    idx2det: Sequence[int],
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
    idx2det: Sequence[int],
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
    idx2det: Sequence[int],
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
