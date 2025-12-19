import numpy as np

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G1, G1_generalized,
    G2, G2_generalized,
)
from slowquant.unitary_coupled_cluster.util import UpsStructure
from slowquant.unitary_coupled_cluster.operator_state_algebra import bitcount


#@nb.jit(nopython=True)
def generalized_apply_operator(
    state: np.ndarray,
    anni_idxs: np.ndarray,
    create_idxs: np.ndarray,
    num_active_orbs: int,
    parity_check: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    do_unsafe: bool,
    tmp_state: np.ndarray,
    factor: float | complex,
) -> np.ndarray:
    """Apply operator to state for a single state wave function.

    This part is outside of propagate_state for performance reasons,
    i.e., Numba JIT.

    Args:
        state: Original state.
        anni_idxs: Indicies for annihilation operators.
        create_idxs: Indicies for creation operators.
        num_active_orbs: Number of active spatial orbitals.
        parity_check: Array used to check the parity when an operator is applied.
        idx2det: Maps index to determinant.
        det2idx: Maps determinant to index.
        do_unsafe: Do unsafe.
        tmp_state: New state.
        factor: Factor in front of operator.

    Returns:
        New state.
    """
    anni_idxs = anni_idxs[::-1]
    create_idxs = create_idxs[::-1]
    # loop over all determinants in new_state
    for i, det in enumerate(idx2det):
        if abs(state[i]) < 10**-14:
            continue
        phase_changes = 0
        is_killstate = False
        # evaluate how string of annihilation operator change det
        for orb_idx in anni_idxs:
            if (det >> 2 * num_active_orbs - 1 - orb_idx) & 1 == 0:
                # If an annihilation operator works on zero, then we reach kill-state.
                is_killstate = True
                break
            det = det ^ (1 << (2 * num_active_orbs - 1 - orb_idx))
            # take care of phases using parity_check
            phase_changes += bitcount(det & parity_check[orb_idx])
        if is_killstate:
            continue
        for orb_idx in create_idxs:
            if (det >> 2 * num_active_orbs - 1 - orb_idx) & 1 == 1:
                # If creation operator works on one, then we reach kill-state.
                is_killstate = True
                break
            det = det ^ (1 << (2 * num_active_orbs - 1 - orb_idx))
            # take care of phases using parity_check
            phase_changes += bitcount(det & parity_check[orb_idx])
        if is_killstate:
            continue
        if do_unsafe:
            # For some algorithms it is guaranteed that the application of operators will always
            # keep the new determinants within a pre-defined space (in det2idx and idx2det).
            # For these algorithms it is a sign of bug if a keyerror when calling det2idx is found.
            # These algorithms thus does also not need to check for the exsistence of the new determinant
            # in det2idx.
            # For other algorithms this 'safety' is not guaranteed, hence the keyword is called 'do_unsafe'.
            if det not in det2idx:
                continue
        tmp_state[det2idx[det]] += factor * (-1) ** phase_changes * state[i]
    return tmp_state


def generalized_propagate_state(
    operators: list[FermionicOperator | str],
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: list[float | complex] | None = None,
    wf_struct: UpsStructure | None = None,
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
    # Annika has forced this to be of type complex
    tmp_state = np.zeros_like(state,dtype=complex)
    # Create bitstrings for parity check. Contains occupied determinant up to orbital index.
    parity_check = np.zeros(2 * num_active_orbs + 1, dtype=np.int64)
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
                # OBS!!!!!Changed to generalized_construct_ups_state_modified!!!!!!
                new_state = generalized_construct_ups_state_modified(
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
            for fermi_label in op_folded.operators.keys():
                # Separate each annihilation operator string in creation and annihilation indices
                anni_idx = []
                create_idx = []
                for fermi_op in fermi_label:
                    if fermi_op[1]:
                        create_idx.append(fermi_op[0])
                    else:
                        anni_idx.append(fermi_op[0])
                anni_idx = np.array(anni_idx, dtype=np.int64)
                create_idx = np.array(create_idx, dtype=np.int64)
                tmp_state = generalized_apply_operator(
                    new_state,
                    anni_idx,
                    create_idx,
                    num_active_orbs,
                    parity_check,
                    idx2det,
                    det2idx,
                    do_unsafe,
                    tmp_state,
                    op_folded.operators[fermi_label],
                )
            new_state = np.copy(tmp_state)
    return new_state


def generalized_expectation_value(
    bra: np.ndarray,
    operators: list[FermionicOperator | str],
    ket: np.ndarray,
    ci_info: CI_Info,
    thetas: list[float | complex] | None = None,
    wf_struct: UpsStructure | None = None,
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
    op_ket = generalized_propagate_state(
        operators,
        ket,
        ci_info,
        thetas,
        wf_struct,
        do_folding=do_folding,
        do_unsafe=do_unsafe,
    )
    val = bra.conj() @ op_ket

    if val.imag > 1e-10:
        print("Warning: Expectation value is complex!!", val)

    return val.real


def expectation_value_for_gradient(
    bra: np.ndarray,
    operators: list[FermionicOperator | str],
    ket: np.ndarray,
    ci_info: CI_Info,
    thetas: list[float | complex] | None = None,
    wf_struct: UpsStructure | None = None,
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
    op_ket = generalized_propagate_state(
        operators,
        ket,
        ci_info,
        thetas,
        wf_struct,
        do_folding=do_folding,
        do_unsafe=do_unsafe,
    )
    val = bra.conj() @ op_ket

    return val


def generalized_construct_ups_state(
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: list[float | complex],
    ups_struct: UpsStructure,
    dagger: bool = False,
) -> np.ndarray:
    r"""Construct unitary product state by applying UPS unitary to reference state.

    .. math::
        \boldsymbol{U}_N...\boldsymbol{U}_0\left|\nu\right> = \left|\tilde\nu\right>

    #. 10.48550/arXiv.2303.10825, Eq. 15
    #. 10.48550/arXiv.2505.00883, Eq. 45, 47, and, 49 (SA doubles)
    #. 10.48550/arXiv.2505.02984, Eq. 35, D1, and, D2 (SA doubles)

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
            theta = -theta.conjugate()
        if exc_type in ("single", "double"):
            # Create T matrix
            if exc_type == "single":
                (i, a) = np.array(exc_indices) + 2 * offset
                T = G1(i, a, True)
            elif exc_type == "double":
                (i, j, a, b) = np.array(exc_indices) + 2 * offset
                T = G2(i, j, a, b, True)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Analytical application on state vector
            out = (
                out
                + np.sin(theta)
                * generalized_propagate_state(
                    [T],
                    out,
                    ci_info,
                    do_folding=False,
                )
                + (1 - np.cos(theta))
                * generalized_propagate_state(
                    [T, T],
                    out,
                    ci_info,
                    do_folding=False,
                )
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return out


def generalized_construct_ups_state_modified(
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: list[float | complex],
    ups_struct: UpsStructure,
    dagger: bool = False,
) -> np.ndarray:
    r"""Construct unitary product state by applying UPS unitary to reference state.

    .. math::
        \boldsymbol{U}_N...\boldsymbol{U}_0\left|\nu\right> = \left|\tilde\nu\right>

    #. 10.48550/arXiv.2303.10825, Eq. 15
    #. 10.48550/arXiv.2505.00883, Eq. 45, 47, and, 49 (SA doubles)
    #. 10.48550/arXiv.2505.02984, Eq. 35, D1, and, D2 (SA doubles)

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
    n_thetas = int(len(thetas)/2)
    order = 1
    offset = ci_info.space_extension_offset
    if dagger:
        order = -1
    # Loop over all excitation in UPSStructure
    for exc_type, exc_indices, theta_R, theta_I in zip(
        ups_struct.excitation_operator_type[::order], ups_struct.excitation_indices[::order],
        thetas[:n_thetas][::order], thetas[n_thetas:2*n_thetas][::order]
    ):
        if dagger:
            theta_R = -theta_R
            theta_I = -theta_I
        if exc_type in ("single", "double"):
            # Create T matrix Imaginary:
            if abs(theta_I) < 10**-14:
                continue
            else:
                if exc_type == "single":
                    (i, a) = np.array(exc_indices) + 2 * offset
                    T_R = G1_generalized(i, a, True, Real = False)
                elif exc_type == "double":
                    (i, j, a, b) = np.array(exc_indices) + 2 * offset
                    T_R = G2_generalized(i, j, a, b, True, Real = False)
                else:
                    raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Create T matrix Real:
            if abs(theta_R) < 10**-14:
                continue
            else:
                if exc_type == "single":
                    (i, a) = np.array(exc_indices) + 2 * offset
                    T_I = G1_generalized(i, a, True, Real = True)
                elif exc_type == "double":
                    (i, j, a, b) = np.array(exc_indices) + 2 * offset
                    T_I = G2_generalized(i, j, a, b, True, Real = True)
                else:
                    raise ValueError(f"Got unknown excitation type: {exc_type}")
            
            if not dagger:
                if abs(theta_I) < 10**-14:
                    # Analytical application on state vector
                    out = (
                        out
                        + np.sin(theta_I)
                        * generalized_propagate_state(
                            [T_I],
                            out,
                            ci_info,
                            do_folding=False,
                        )
                        + (1 - np.cos(theta_I))
                        * generalized_propagate_state(
                            [T_I, T_I],
                            out,
                            ci_info,
                            do_folding=False,
                        )
                    )
                if abs(theta_R) < 10**-14:
                    # Analytical application on state vector
                    out = (
                        out
                        + np.sin(theta_R)
                        * generalized_propagate_state(
                            [T_R],
                            out,
                            ci_info,
                            do_folding=False,
                        )
                        + (1 - np.cos(theta_R))
                        * generalized_propagate_state(
                            [T_R, T_R],
                            out,
                            ci_info,
                            do_folding=False,
                        )
                    )
            else:
                if abs(theta_R) < 10**-14:
                    # Analytical application on state vector
                    out = (
                        out
                        + np.sin(theta_R)
                        * generalized_propagate_state(
                            [T_R],
                            out,
                            ci_info,
                            do_folding=False,
                        )
                        + (1 - np.cos(theta_R))
                        * generalized_propagate_state(
                            [T_R, T_R],
                            out,
                            ci_info,
                            do_folding=False,
                        )
                    )
                if abs(theta_I) < 10**-14:
                    # Analytical application on state vector
                    out = (
                        out
                        + np.sin(theta_I)
                        * generalized_propagate_state(
                            [T_I],
                            out,
                            ci_info,
                            do_folding=False,
                        )
                        + (1 - np.cos(theta_I))
                        * generalized_propagate_state(
                            [T_I, T_I],
                            out,
                            ci_info,
                            do_folding=False,
                        )
                    )              
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return out



def generalized_propagate_unitary(
    state: np.ndarray,
    idx: int,
    ci_info: CI_Info,
    thetas: list[float | complex],
    ups_struct: UpsStructure,
) -> np.ndarray:
    """Apply unitary from UPS operator number 'idx' to state.

    #. 10.48550/arXiv.2505.00883, Eq. 45, 47, and, 49 (SA doubles)
    #. 10.48550/arXiv.2505.02984, Eq. 35, D1, and, D2 (SA doubles)

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
    if exc_type in ("single", "double"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = np.array(exc_indices) + 2 * offset
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = np.array(exc_indices) + 2 * offset
            T = G2(i, j, a, b, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Analytical application on state vector
        out = (
            state
            + np.sin(theta)
            * generalized_propagate_state(
                [T],
                state,
                ci_info,
                do_folding=False,
            )
            + (1 - np.cos(theta))
            * generalized_propagate_state(
                [T, T],
                state,
                ci_info,
                do_folding=False,
            )
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return out


def generalized_propagate_unitary_modified(
    state: np.ndarray,
    idx: int,
    ci_info: CI_Info,
    thetas: list[float | complex],
    ups_struct: UpsStructure,
) -> np.ndarray:
    """Apply unitary from UPS operator number 'idx' to state.

    #. 10.48550/arXiv.2505.00883, Eq. 45, 47, and, 49 (SA doubles)
    #. 10.48550/arXiv.2505.02984, Eq. 35, D1, and, D2 (SA doubles)

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
    real = True
    if idx >= len(ups_struct.excitation_indices):
        idx = idx - len(ups_struct.excitation_indices)
        real = False

    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indices[idx]
    n_thetas = int(len(thetas)/2)
    theta_R = thetas[idx]
    theta_I = thetas[n_thetas + idx]
    offset = ci_info.space_extension_offset
    
    if exc_type in ("single", "double"):
        if real:
            # Create T matrix Real:
            if abs(theta_R) < 10**-14:
                return np.copy(state)
            else:
                if exc_type == "single":
                    (i, a) = np.array(exc_indices) + 2 * offset
                    T = G1_generalized(i, a, True, Real = True)
                elif exc_type == "double":
                    (i, j, a, b) = np.array(exc_indices) + 2 * offset
                    T = G2_generalized(i, j, a, b, True, Real = True)
                else:
                    raise ValueError(f"Got unknown excitation type: {exc_type}")
                # Analytical application on state vector
                out = (
                    state
                    + np.sin(theta_R)
                    * generalized_propagate_state(
                        [T],
                        state,
                        ci_info,
                        do_folding=False,
                    )
                    + (1 - np.cos(theta_R))
                    * generalized_propagate_state(
                        [T, T],
                        state,
                        ci_info,
                        do_folding=False,
                    )
                )
        else:
            # Create T matrix Imaginary:
            if abs(theta_I) < 10**-14:
                return np.copy(state)
            else:
                if exc_type == "single":
                    (i, a) = np.array(exc_indices) + 2 * offset
                    T = G1_generalized(i, a, True, Real = False)
                elif exc_type == "double":
                    (i, j, a, b) = np.array(exc_indices) + 2 * offset
                    T = G2_generalized(i, j, a, b, True, Real = False)
                else:
                    raise ValueError(f"Got unknown excitation type: {exc_type}")
                # Analytical application on state vector
                out = (
                    state
                    + np.sin(theta_I)
                    * generalized_propagate_state(
                        [T],
                        state,
                        ci_info,
                        do_folding=False,
                    )
                    + (1 - np.cos(theta_I))
                    * generalized_propagate_state(
                        [T, T],
                        state,
                        ci_info,
                        do_folding=False,
                    )
                )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return out


def generalized_get_grad_action(
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
    if exc_type in (
        "single",
        "double",
    ):
        # Create T matrix
        if exc_type == "single":
            (i, a) = np.array(exc_indices) + 2 * offset
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = np.array(exc_indices) + 2 * offset
            T = G2(i, j, a, b, True)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Apply missing T factor of derivative
        tmp = generalized_propagate_state(
            [T],
            state,
            ci_info,
            do_folding=False,
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp



def generalized_get_grad_action_modified(
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
    real = True
    if idx >= len(ups_struct.excitation_indices):
        idx = idx - len(ups_struct.excitation_indices)
        real = False

    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indices[idx]
    offset = ci_info.space_extension_offset
    if exc_type in (
        "single",
        "double",
    ):
        if real:
            # Create T matrix real:
            if exc_type == "single":
                (i, a) = np.array(exc_indices) + 2 * offset
                T = G1_generalized(i, a, True, Real = True)
            elif exc_type == "double":
                (i, j, a, b) = np.array(exc_indices) + 2 * offset
                T = G2_generalized(i, j, a, b, True, Real = True)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Apply missing T factor of derivative
            tmp = generalized_propagate_state(
                [T],
                state,
                ci_info,
                do_folding=False,
            )
        else:
            # Create T matrix imaginary:
            if exc_type == "single":
                (i, a) = np.array(exc_indices) + 2 * offset
                T = G1_generalized(i, a, True, Real = False)
            elif exc_type == "double":
                (i, j, a, b) = np.array(exc_indices) + 2 * offset
                T = G2_generalized(i, j, a, b, True, Real = False)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Apply missing T factor of derivative
            tmp = generalized_propagate_state(
                [T],
                state,
                ci_info,
                do_folding=False,
            )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp

