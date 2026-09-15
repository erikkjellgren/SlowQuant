import math
from collections.abc import Sequence

import numba as nb
import numpy as np
import scipy.sparse as ss

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info, bitcount
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
from slowquant.unitary_coupled_cluster.spin_factorized_algebra import (
    propagate_state_factorized,
    propagate_state_SA_factorized,
)
from slowquant.unitary_coupled_cluster.spin_ordering import alpha_idx, beta_idx
from slowquant.unitary_coupled_cluster.util import UccStructure, UpsStructure


def _init() -> None:
    # This runs the first time file is imported
    print("\nSetting number of Numba threads to 1\n")
    # Defaulting to one thread
    nb.set_num_threads(1)


_init()


@nb.jit(nopython=True)
def apply_operator_serial(
    state: np.ndarray,
    a_string: np.ndarray,
    create_screen: np.ndarray,
    anni_idx: np.ndarray,
    num_active_orbs: int,
    parity_check: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    do_unsafe: bool,
    tmp_state: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Apply operator to state for a single state wave function.

    This part is outside of propagate_state for performance reasons,
    i.e., Numba JIT.

    The algorithm applies an annihilation string to a start determinant to generate a new determinant.
    This is performed in two step.

    Step 1)
    First it is checked if the operator will generate the kill-state.
    This is done by 'bitwise and' with the determinant and annihilation part of the operator,
    and, 'bitwise and' with the determinant and creation part of the operator.
    Note here, that the creation indices that also exist in the annihilation part,
    are assumed to be screened out, this is how number operators are handled.

    Step 2)
    Second, for all determiant that does not end up in kill-state,
    loop through the annihilation string,
    and do a bitflip on the determiant for the given index, and, calculate the phase change.
    No need to check for kill-state in this step, as that is handled by the first step.

    Note on do_unsafe)
    For some algorithms it is guaranteed that the application of operators will always
    keep the new determinants within a pre-defined space (in det2idx and idx2det).
    For these algorithms it is a sign of bug if a keyerror when calling det2idx is found.
    These algorithms thus does also not need to check for the exsistence of the new determinant in det2idx.
    For other algorithms this 'safety' is not guaranteed, hence the keyword is called 'do_unsafe'.

    Args:
        state: Original state.
        a_string: Creation and annihilation operator indices.
        create_screen: Creation operator indices without indices in anni_idx.
        anni_idx: Indices for annihilation operators.
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
    num_spin_orbs_m1 = 2 * num_active_orbs - 1
    anni_mask = 0
    for orb_idx in anni_idx:
        anni_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    create_mask = 0
    for orb_idx in create_screen:
        create_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    for i in range(len(idx2det)):
        det = idx2det[i]
        if (det & anni_mask) != anni_mask:
            continue
        if (det & create_mask) != 0:
            continue
        state_i = state[i]
        if abs(state_i) < 10**-28:
            continue
        phase_changes = 0
        for orb_idx in a_string:
            det = det ^ (1 << (num_spin_orbs_m1 - orb_idx))
            phase_changes += bitcount(det & parity_check[orb_idx])
        if do_unsafe:
            if det not in det2idx:
                continue
        sign = 1.0 - 2.0 * (phase_changes & 1)
        tmp_state[det2idx[det]] += sign * factor * state_i
    return tmp_state


@nb.jit(nopython=True, parallel=True)
def apply_operator_threaded(
    state: np.ndarray,
    a_string: np.ndarray,
    create_idx: np.ndarray,
    anni_screen: np.ndarray,
    num_active_orbs: int,
    parity_check: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    do_unsafe: bool,
    tmp_state: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Apply operator to state for a single state wave function.

    This part is outside of propagate_state for performance reasons,
    i.e., Numba JIT.

    The algorithm removes an annihilation string to a target determinant to generate a new determinant.
    This is performed in two step.

    Step 1)
    First it is checked if the operator will generate the kill-state.
    This is done by 'bitwise and' with the determinant and annihilation part of the operator,
    and, 'bitwise and' with the determinant and creation part of the operator.
    Note here, that the annihilation indices that also exist in the creation part,
    are assumed to be screened out, this is how number operators are handled.

    Step 2)
    Second, for all determiant that does not end up in kill-state,
    loop through the annihilation string,
    and do a bitflip on the determiant for the given index, and, calculate the phase change.
    No need to check for kill-state in this step, as that is handled by the first step.

    Note on do_unsafe)
    For some algorithms it is guaranteed that the application of operators will always
    keep the new determinants within a pre-defined space (in det2idx and idx2det).
    For these algorithms it is a sign of bug if a keyerror when calling det2idx is found.
    These algorithms thus does also not need to check for the exsistence of the new determinant in det2idx.
    For other algorithms this 'safety' is not guaranteed, hence the keyword is called 'do_unsafe'.

    Args:
        state: Original state.
        a_string: Creation and annihilation operator indices.
        create_idx: Creation operator indices.
        anni_screen: Annihilation operator indices without indices in create_idx.
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
    num_spin_orbs_m1 = 2 * num_active_orbs - 1
    create_mask = 0
    for orb_idx in create_idx:
        create_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    anni_mask = 0
    for orb_idx in anni_screen:
        anni_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    for i in nb.prange(len(idx2det)):
        det = idx2det[i]
        if (det & create_mask) != create_mask:
            continue
        if (det & anni_mask) != 0:
            continue
        phase_changes = 0
        for orb_idx in a_string:
            det = det ^ (1 << (num_spin_orbs_m1 - orb_idx))
            phase_changes += bitcount(det & parity_check[orb_idx])
        if do_unsafe:
            if det not in det2idx:
                continue
        sign = 1.0 - 2.0 * (phase_changes & 1)
        tmp_state[i] += sign * factor * state[det2idx[det]]
    return tmp_state


@nb.jit(nopython=True)
def add_operator_matrix(
    op_mat: np.ndarray,
    a_string: np.ndarray,
    create_screen: np.ndarray,
    anni_idx: np.ndarray,
    num_active_orbs: int,
    parity_check: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    do_unsafe: bool,
    factor: float,
) -> np.ndarray:
    """Add matrix representation of annihilation string.

    This part is outside of propagate_state for performance reasons,
    i.e., Numba JIT.

    See 'apply_operator_serial' for algorithmic description.

    Args:
        op_mat: Matrix representation of operator.
        a_string: Creation and annihilation operator indices.
        create_screen: Creation operator indices without indices in anni_idx.
        anni_idx: Indices for annihilation operators.
        num_active_orbs: Number of active spatial orbitals.
        parity_check: Array used to check the parity when an operator is applied.
        idx2det: Maps index to determinant.
        det2idx: Maps determinant to index.
        do_unsafe: Do unsafe.
        factor: Factor in front of operator.

    Returns:
        Operator matrix.
    """
    num_spin_orbs_m1 = 2 * num_active_orbs - 1
    anni_mask = 0
    for orb_idx in anni_idx:
        anni_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    create_mask = 0
    for orb_idx in create_screen:
        create_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    for i in range(len(idx2det)):
        det = idx2det[i]
        if (det & anni_mask) != anni_mask:
            continue
        if (det & create_mask) != 0:
            continue
        phase_changes = 0
        for orb_idx in a_string:
            det = det ^ (1 << (num_spin_orbs_m1 - orb_idx))
            phase_changes += bitcount(det & parity_check[orb_idx])
        if do_unsafe:
            if det not in det2idx:
                continue
        sign = 1.0 - 2.0 * (phase_changes & 1)
        op_mat[det2idx[det], i] += sign * factor
    return op_mat


@nb.jit(nopython=True)
def apply_operator_SA_serial(
    state: np.ndarray,
    a_string: np.ndarray,
    create_screen: np.ndarray,
    anni_idx: np.ndarray,
    num_active_orbs: int,
    parity_check: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    do_unsafe: bool,
    tmp_state: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Apply operator to state for a state-averaged wave function.

    This part is outside of propagate_state for performance reasons,
    i.e., Numba JIT.

    See 'apply_operator_serial' for algorithmic description.

    Args:
        state: Original state.
        a_string: Creation and annihilation operator indices.
        create_screen: Creation operator indices without indices in anni_idx.
        anni_idx: Indices for annihilation operators.
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
    num_spin_orbs_m1 = 2 * num_active_orbs - 1
    anni_mask = 0
    for orb_idx in anni_idx:
        anni_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    create_mask = 0
    for orb_idx in create_screen:
        create_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    for i in range(len(idx2det)):
        det = idx2det[i]
        if (det & anni_mask) != anni_mask:
            continue
        if (det & create_mask) != 0:
            continue
        is_non_zero = False
        for val in state[:, i]:
            if abs(val) > 10**-28:
                is_non_zero = True
                break
        if not is_non_zero:
            continue
        phase_changes = 0
        for orb_idx in a_string:
            det = det ^ (1 << (num_spin_orbs_m1 - orb_idx))
            phase_changes += bitcount(det & parity_check[orb_idx])
        if do_unsafe:
            if det not in det2idx:
                continue
        sign = 1.0 - 2.0 * (phase_changes & 1)
        tmp_state[:, det2idx[det]] += sign * factor * state[:, i]  # Update value
    return tmp_state


@nb.jit(nopython=True, parallel=True)
def apply_operator_SA_threaded(
    state: np.ndarray,
    a_string: np.ndarray,
    create_idx: np.ndarray,
    anni_screen: np.ndarray,
    num_active_orbs: int,
    parity_check: np.ndarray,
    idx2det: np.ndarray,
    det2idx: dict[int, int],
    do_unsafe: bool,
    tmp_state: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Apply operator to state for a state-averaged wave function.

    This part is outside of propagate_state for performance reasons,
    i.e., Numba JIT.

    See 'apply_operator_threaded' for algorithmic description.

    Args:
        state: Original state.
        a_string: Creation and annihilation operator indices.
        create_idx: Creation operator indices.
        anni_screen: Annihilation operator indices without indices in create_idx.
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
    num_spin_orbs_m1 = 2 * num_active_orbs - 1
    create_mask = 0
    for orb_idx in create_idx:
        create_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    anni_mask = 0
    for orb_idx in anni_screen:
        anni_mask |= 1 << (num_spin_orbs_m1 - orb_idx)
    for i in nb.prange(len(idx2det)):
        det = idx2det[i]
        if (det & create_mask) != create_mask:
            continue
        if (det & anni_mask) != 0:
            continue
        phase_changes = 0
        for orb_idx in a_string:
            det = det ^ (1 << (num_spin_orbs_m1 - orb_idx))
            phase_changes += bitcount(det & parity_check[orb_idx])
        if do_unsafe:
            if det not in det2idx:
                continue
        sign = 1.0 - 2.0 * (phase_changes & 1)
        tmp_state[:, i] += sign * factor * state[:, det2idx[det]]  # Update value
    return tmp_state


def embed_spatial_indices(exc_indices: Sequence[int], ci_info: CI_Info) -> list[int]:
    """Embed ansatz spatial orbital indices into the CI space.

    Args:
        exc_indices: Spatial orbital indices, counted from the start of the active space.
        ci_info: Information about the CI space.

    Returns:
        Spatial orbital indices in the CI space.
    """
    return [p + ci_info.space_extension_offset for p in exc_indices]


def embed_spin_indices(exc_indices: Sequence[int], ci_info: CI_Info, num_active_orbs: int) -> list[int]:
    """Embed ansatz spin-orbital indices into the CI space.

    The ansatz is built over num_active_orbs spatial orbitals. When the CI space is extended
    those orbitals sit at an offset inside a larger space, and because the ordering is blocked
    the alpha and beta halves are shifted by different amounts.

    Args:
        exc_indices: Spin-orbital indices in the active space of the ansatz.
        ci_info: Information about the CI space.
        num_active_orbs: Number of active spatial orbitals the ansatz is built over.

    Returns:
        Spin-orbital indices in the CI space.
    """
    if num_active_orbs < 1 and len(exc_indices) > 0:
        # A structure whose builder was never run reports zero, which would silently make every
        # index look like beta.
        raise ValueError("Cannot embed spin-orbital indices without the size of the ansatz space.")
    offset = ci_info.space_extension_offset
    embedded = []
    for idx in exc_indices:
        if idx < num_active_orbs:
            embedded.append(idx + offset)
        else:
            embedded.append(idx - num_active_orbs + offset + ci_info.num_active_orbs)
    return embedded


@nb.jit(nopython=True, cache=True)
def rotate_determinant_pairs(
    states: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    sign: np.ndarray,
    cos_theta: float,
    sin_theta: float,
) -> None:
    r"""Rotate paired determinant amplitudes in place.

    .. math::
        \begin{pmatrix}c_p\\c_q\end{pmatrix} \leftarrow
        \begin{pmatrix}\cos\theta & -\Gamma\sin\theta\\
                       \Gamma\sin\theta & \cos\theta\end{pmatrix}
        \begin{pmatrix}c_p\\c_q\end{pmatrix}

    The pairs are disjoint, so this needs no output vector.

    Args:
        states: States as (number of states, number of determinants), updated in place.
        src: First determinant of each pair.
        dst: Second determinant of each pair.
        sign: Phase :math:`\Gamma` of each pair.
        cos_theta: Cosine of the rotation angle.
        sin_theta: Sine of the rotation angle.
    """
    for pair in range(len(src)):
        p = src[pair]
        q = dst[pair]
        signed_sin = sign[pair] * sin_theta
        for state_idx in range(states.shape[0]):
            amplitude_p = states[state_idx, p]
            amplitude_q = states[state_idx, q]
            states[state_idx, p] = cos_theta * amplitude_p - signed_sin * amplitude_q
            states[state_idx, q] = signed_sin * amplitude_p + cos_theta * amplitude_q


MAX_EXPONENTIAL_BLOCK = 64

# Frequencies and the weight of each power of the generator in the closed form of a
# spin-adapted double, as the branches that used to carry a copy each spelled them. Odd
# powers are weighted by sin(S*theta) and even ones by cos(S*theta)-1. Only reached when
# the generator cannot be blocked, see build_generator_blocks.
SPIN_ADAPTED_DOUBLE_SERIES: dict[str, tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]] = {
    "sa_double_2": (
        (1, math.sqrt(2) / 2),
        (
            (-1, 2 * math.sqrt(2)),
            (1, -4),
            (-2, 2 * math.sqrt(2)),
            (2, -4),
        ),
    ),
    "sa_double_3": (
        (1, math.sqrt(2) / 2),
        (
            (-1, 2 * math.sqrt(2)),
            (1, -4),
            (-2, 2 * math.sqrt(2)),
            (2, -4),
        ),
    ),
    "sa_double_4": (
        (1, math.sqrt(2), math.sqrt(2) / 2, 1 / 2),
        (
            (2 / 3, -math.sqrt(2) / 42, -8 * math.sqrt(2) / 3, 128 / 21),
            (-2 / 3, 1 / 42, 16 / 3, -256 / 21),
            (13 / 3, -math.sqrt(2) / 6, -44 * math.sqrt(2) / 3, 64 / 3),
            (-13 / 3, 1 / 6, 88 / 3, -128 / 3),
            (22 / 3, -math.sqrt(2) / 3, -52 * math.sqrt(2) / 3, 64 / 3),
            (-22 / 3, 1 / 3, 104 / 3, -128 / 3),
            (8 / 3, -4 * math.sqrt(2) / 21, -16 * math.sqrt(2) / 3, 128 / 21),
            (-8 / 3, 4 / 21, 32 / 3, -256 / 21),
        ),
    ),
    "sa_double_5": (
        (math.sqrt(2), math.sqrt(2) / 2, math.sqrt(3) / 3, math.sqrt(3) / 2, math.sqrt(3) / 6),
        (
            (
                math.sqrt(2) / 1150,
                8 * math.sqrt(2) / 5,
                -54 * math.sqrt(3) / 25,
                -16 * math.sqrt(3) / 75,
                432 * math.sqrt(3) / 115,
            ),
            (-1 / 1150, -16 / 5, 162 / 25, 32 / 75, -2592 / 115),
            (
                11 * math.sqrt(2) / 690,
                404 * math.sqrt(2) / 15,
                -171 * math.sqrt(3) / 5,
                -56 * math.sqrt(3) / 15,
                2952 * math.sqrt(3) / 115,
            ),
            (-11 / 690, -808 / 15, 513 / 5, 112 / 15, -17712 / 115),
            (
                133 * math.sqrt(2) / 1725,
                308 * math.sqrt(2) / 3,
                -2718 * math.sqrt(3) / 25,
                -1192 * math.sqrt(3) / 75,
                1368 * math.sqrt(3) / 23,
            ),
            (-133 / 1725, -616 / 3, 8154 / 25, 2384 / 75, -8208 / 23),
            (
                16 * math.sqrt(2) / 115,
                608 * math.sqrt(2) / 5,
                -576 * math.sqrt(3) / 5,
                -112 * math.sqrt(3) / 5,
                6192 * math.sqrt(3) / 115,
            ),
            (-16 / 115, -1216 / 5, 1728 / 5, 224 / 5, -37152 / 115),
            (
                48 * math.sqrt(2) / 575,
                192 * math.sqrt(2) / 5,
                -864 * math.sqrt(3) / 25,
                -192 * math.sqrt(3) / 25,
                1728 * math.sqrt(3) / 115,
            ),
            (-48 / 575, -384 / 5, 2592 / 25, 384 / 25, -10368 / 115),
        ),
    ),
}


@nb.jit(nopython=True, cache=True)
def label_connected_determinants(src: np.ndarray, dst: np.ndarray, num_dets: int) -> np.ndarray:
    """Label each determinant with the connected group of the generator it belongs to.

    Args:
        src: Determinant the generator acts on.
        dst: Determinant it is taken to.
        num_dets: Number of determinants.

    Returns:
        Group label of every determinant.
    """
    parent = np.arange(num_dets)
    for entry in range(len(src)):
        root_a = src[entry]
        root_b = dst[entry]
        while parent[root_a] != root_a:
            parent[root_a] = parent[parent[root_a]]
            root_a = parent[root_a]
        while parent[root_b] != root_b:
            parent[root_b] = parent[parent[root_b]]
            root_b = parent[root_b]
        if root_a != root_b:
            parent[root_a] = root_b
    for det in range(num_dets):
        root = det
        while parent[root] != root:
            root = parent[root]
        parent[det] = root
    return parent


@nb.jit(nopython=True, cache=True)
def rotate_determinant_blocks(
    states: np.ndarray,
    dets: np.ndarray,
    starts: np.ndarray,
    shape: np.ndarray,
    rotations: np.ndarray,
) -> None:
    r"""Apply a small dense rotation to each group of determinants, in place.

    .. math::
        c_{d_r} \leftarrow \sum_s R^{(b)}_{rs} c_{d_s}

    The groups are disjoint, so this needs no output vector.

    Args:
        states: States as (number of states, number of determinants), updated in place.
        dets: Determinants of every group, one group after another.
        starts: Where each group begins in dets, with the end appended.
        shape: Which rotation each group uses.
        rotations: Rotations as (number of distinct groups, size, size).
    """
    buffer = np.empty(rotations.shape[1])
    for group in range(len(starts) - 1):
        start = starts[group]
        size = starts[group + 1] - start
        rotation = rotations[shape[group]]
        for state_idx in range(states.shape[0]):
            state = states[state_idx]
            for row in range(size):
                buffer[row] = state[dets[start + row]]
            for row in range(size):
                total = 0.0
                for col in range(size):
                    total += rotation[row, col] * buffer[col]
                state[dets[start + row]] = total


def build_generator_blocks(op: FermionicOperator, ci_info: CI_Info) -> tuple[np.ndarray, ...] | None:
    r"""Block diagonalize an excitation generator over the determinants it connects.

    An anti-Hermitian generator is real and antisymmetric, so the determinants it connects fall
    into groups it cannot mix, and on each group it is a small antisymmetric matrix. The
    exponential is then block diagonal too,

    .. math::
        \exp\left(\theta\hat{T}\right) = \bigoplus_b \exp\left(\theta T^{(b)}\right)

    with the identity on every determinant the generator annihilates. A single excitation gives
    groups of two and the exponential of each is a Givens rotation; a spin-adapted double gives
    groups of up to eight, of which only a handful are distinct however large the active space
    is, because a group is fixed by the occupation of the few orbitals the generator touches.

    Each fermionic string of the generator is a signed map of determinants on its own, so
    applying it to :math:`v_k = k+1`, which is positive and all different, names the determinant
    each one came from and the sign it picked up. The groups and their matrices follow from
    those maps.

    Args:
        op: Excitation generator, already embedded in the CI space.
        ci_info: Information about the CI space.

    Returns:
        Determinants of every group, where each group starts, which distinct matrix it uses, and
        those matrices. None if the generator leaves the CI space or connects too much of it to
        be worth blocking.
    """
    num_dets = len(ci_info.idx2det)
    ramp = np.arange(1.0, num_dets + 1.0)
    rows, cols, values = [], [], []
    try:
        for string, factor in op.operators.items():
            # Unit weight, so that the sign of the result is the sign the string picked up.
            reached = propagate_state([FermionicOperator({string: 1.0})], ramp, ci_info, do_folding=False)
            taken_to = np.flatnonzero(reached)
            sign = np.sign(reached[taken_to])
            rows.append(taken_to)
            cols.append(np.rint(np.abs(reached[taken_to]) - 1.0).astype(int))
            values.append(factor * sign)
    except KeyError:
        # The generator takes some determinant out of the CI space. The general kernel skips
        # determinants the state is zero on, so it survives that as long as the state stays
        # away from them, and a blocked exponential could not. Leave it to the caller.
        return None
    if not rows:
        return None
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    value = np.concatenate(values)
    if np.any(col < 0) or np.any(col >= num_dets):
        return None

    label = label_connected_determinants(col, row, num_dets)
    # A mask rather than a unique, because the entries run to several times the CI space.
    reached = np.zeros(num_dets, dtype=bool)
    reached[row] = True
    reached[col] = True
    touched = np.flatnonzero(reached)
    if len(touched) == 0:
        return None
    # Determinants of a group sit next to each other once sorted by group, and within a group
    # they keep determinant order, which is the same order in every group of the same shape.
    order = np.argsort(label[touched], kind="stable")
    dets = touched[order]
    group_label = label[dets]
    starts = np.flatnonzero(np.concatenate(([True], group_label[1:] != group_label[:-1])))
    starts = np.concatenate((starts, [len(dets)]))
    sizes = np.diff(starts)
    block_size = int(sizes.max())
    if block_size > MAX_EXPONENTIAL_BLOCK:
        return None

    # Where each determinant sits, and in which group, so the entries can be placed at once.
    group_of = np.empty(num_dets, dtype=int)
    place_of = np.empty(num_dets, dtype=int)
    group_of[dets] = np.repeat(np.arange(len(sizes)), sizes)
    place_of[dets] = np.arange(len(dets)) - np.repeat(starts[:-1], sizes)
    blocks = np.zeros((len(sizes), block_size, block_size))
    np.add.at(blocks, (group_of[col], place_of[row], place_of[col]), value)

    # Only a handful of the groups are distinct, however large the active space is, so they are
    # matched on their contents. Two that differ in the last bit only cost an extra exponential.
    shape = np.empty(len(sizes), dtype=np.int32)
    seen: dict[bytes, int] = {}
    distinct: list[np.ndarray] = []
    for group in range(len(sizes)):
        key = blocks[group].tobytes()
        found = seen.get(key)
        if found is None:
            found = len(distinct)
            seen[key] = found
            distinct.append(blocks[group])
        shape[group] = found
    groups = np.array(distinct)
    if not np.allclose(groups, -np.transpose(groups, (0, 2, 1))):
        # An anti-Hermitian generator has to give antisymmetric groups. If it did not, the maps
        # above did not describe it and the exponential below would be a different operator.
        # Only the distinct ones need checking, and there are never many of those.
        return None
    return dets.astype(np.int32), starts.astype(np.int32), shape, groups


def diagonalize_generator_blocks(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Diagonalize the generator blocks once, so their exponential is cheap at every angle.

    A real antisymmetric :math:`T` makes :math:`i T` Hermitian, so :math:`iT = V\lambda V^\dagger`
    and

    .. math::
        \exp\left(\theta T\right) = V e^{-i\theta\lambda}V^\dagger

    which is real and costs two small matrix products to assemble once the parameter is known.

    Args:
        blocks: Generator blocks as (number of distinct groups, size, size).

    Returns:
        Eigenvectors and eigenvalues of each block.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(1j * blocks)
    return eigenvectors, eigenvalues


def exponentiate_generator_blocks(
    eigenvectors: np.ndarray, eigenvalues: np.ndarray, theta: float
) -> np.ndarray:
    r"""Assemble :math:`\exp(\theta T^{(b)})` for every distinct block.

    Args:
        eigenvectors: Eigenvectors of each block.
        eigenvalues: Eigenvalues of each block.
        theta: Ansatz parameter value.

    Returns:
        Rotation of each distinct block.
    """
    phases = np.exp(-1j * theta * eigenvalues)[:, np.newaxis, :]
    return np.real((eigenvectors * phases) @ np.conjugate(np.transpose(eigenvectors, (0, 2, 1))))


def build_rotation_layout(
    op: FermionicOperator, ci_info: CI_Info
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    r"""Find the determinant pairs that the exponential of a generator rotates.

    An excitation generator :math:`\hat{T} = \hat{T}_{\text{exc}} - \hat{T}_{\text{exc}}^\dagger`
    squares to minus a projector,

    .. math::
        \hat{T}^2 = -\left[\hat{n}_i\left(1-\hat{n}_a\right)
                         + \hat{n}_a\left(1-\hat{n}_i\right)\right] \equiv -\hat{P}

    so that :math:`\hat{T}^3 = -\hat{T}` and the series for the exponential closes,

    .. math::
        \exp\left(\theta\hat{T}\right) = 1 + \sin\theta\,\hat{T}
                                       + \left(1-\cos\theta\right)\hat{T}^2

    Because :math:`\hat{P}` is diagonal, the determinant basis splits into the determinants the
    generator annihilates, which the unitary leaves alone, and pairs
    :math:`\hat{T}\left|p\right> = \Gamma\left|q\right>`,
    :math:`\hat{T}\left|q\right> = -\Gamma\left|p\right>` spanning a two dimensional block. On
    each block the three terms above sum to one Givens rotation by :math:`\Gamma\theta`, which
    can therefore be applied in a single sweep instead of building
    :math:`\hat{T}\left|0\right>` and :math:`\hat{T}^2\left|0\right>` as separate vectors.

    The pairing is read off the generator rather than derived per excitation type: applying it
    to a vector of ones gives :math:`\Gamma` at every reachable determinant, and applying it to
    :math:`v_k = k+1` gives :math:`\Gamma\left(p+1\right)` there, which names the partner. The
    two are then checked against each other, so a generator that is not a signed pairing, such
    as a spin-adapted double, reports None and is left to the caller.

    Args:
        op: Excitation generator, already embedded in the CI space.
        ci_info: Information about the CI space.

    Returns:
        First and second determinant of each pair with its phase, or None if the generator does
        not act as a signed pairing of determinants.
    """
    num_dets = len(ci_info.idx2det)
    ramp = np.arange(1.0, num_dets + 1.0)
    try:
        phases = propagate_state([op], np.ones(num_dets), ci_info, do_folding=False)
        reached = propagate_state([op], ramp, ci_info, do_folding=False)
    except KeyError:
        # The generator takes some determinant out of the CI space. The general kernel skips
        # determinants the state is zero on, so it survives that as long as the state stays
        # away from them, and a rotation built here could not. Leave it to the caller.
        return None
    dst = np.flatnonzero(phases)
    if len(dst) == 0 or not np.allclose(np.abs(phases[dst]), 1.0):
        return None
    src = np.rint(reached[dst] / phases[dst] - 1.0).astype(int)
    if np.any(src < 0) or np.any(src >= num_dets):
        return None
    # Antisymmetry means every pair is reached from both of its ends, so half of them are kept.
    forward = src < dst
    src, dst, sign = src[forward], dst[forward], phases[dst][forward]
    # The pairing has to reproduce the generator exactly, or the exponential below is not the
    # one the rest of the code computes. The ramp has a distinct value on every determinant,
    # so a single comparison covers the whole CI space.
    check = np.zeros(num_dets)
    check[dst] = sign * ramp[src]
    check[src] = -sign * ramp[dst]
    if not np.allclose(check, reached):
        return None
    # Kept narrow because the index arrays are streamed alongside the state.
    return src.astype(np.int32), dst.astype(np.int32), sign.astype(np.int8)


def get_rotation_layout(
    op: FermionicOperator, ci_info: CI_Info, cache_key: tuple[str, tuple[int, ...]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Get the rotation layout of a generator, building it the first time it is asked for.

    The layout depends only on the generator and the CI space, so it is reused across every
    parameter value the optimizer visits.

    Args:
        op: Excitation generator, already embedded in the CI space.
        ci_info: Information about the CI space.
        cache_key: Excitation type and indices naming this generator.

    Returns:
        Rotation layout, or None if the generator does not act as a signed pairing.
    """
    if cache_key not in ci_info.rotation_layouts:
        ci_info.rotation_layouts[cache_key] = build_rotation_layout(op, ci_info)
    return ci_info.rotation_layouts[cache_key]


def get_block_layout(
    op: FermionicOperator, ci_info: CI_Info, cache_key: tuple[str, tuple[int, ...]]
) -> tuple[np.ndarray, ...] | None:
    """Get the blocked form of a generator, building and diagonalizing it the first time.

    The blocks depend only on the generator and the CI space, so they are reused across every
    parameter value the optimizer visits and only their exponential is rebuilt.

    Args:
        op: Excitation generator, already embedded in the CI space.
        ci_info: Information about the CI space.
        cache_key: Excitation type and indices naming this generator.

    Returns:
        Determinants of every group, where each group starts, which distinct block it uses, and
        the eigenvectors and eigenvalues of those blocks. None if the generator cannot be
        blocked.
    """
    if cache_key not in ci_info.block_layouts:
        built = build_generator_blocks(op, ci_info)
        if built is None:
            ci_info.block_layouts[cache_key] = None
        else:
            dets, starts, shape, blocks = built
            eigenvectors, eigenvalues = diagonalize_generator_blocks(blocks)
            ci_info.block_layouts[cache_key] = (dets, starts, shape, eigenvectors, eigenvalues)
    return ci_info.block_layouts[cache_key]


def apply_blocked_exponential(
    states: np.ndarray,
    op: FermionicOperator,
    theta: float,
    ci_info: CI_Info,
    cache_key: tuple[str, tuple[int, ...]],
) -> np.ndarray | None:
    r"""Apply the exponential of a generator through its blocks, if it has them.

    .. math::
        \left|\tilde{\nu}\right> = \exp\left(\theta\hat{T}\right)\left|\nu\right>

    Args:
        states: States as (number of states, number of determinants).
        op: Excitation generator, already embedded in the CI space.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.
        cache_key: Excitation type and indices naming this generator.

    Returns:
        New states, or None if the generator cannot be blocked.
    """
    layout = get_block_layout(op, ci_info, cache_key)
    if layout is None:
        return None
    dets, starts, shape, eigenvectors, eigenvalues = layout
    out = np.copy(states)
    rotate_determinant_blocks(
        out, dets, starts, shape, exponentiate_generator_blocks(eigenvectors, eigenvalues, theta)
    )
    return out


def spin_adapted_double_weight(
    order: int, weights: tuple[float, ...], frequencies: tuple[float, ...], theta: float
) -> float:
    r"""Weight of one power of the generator in the closed form of a spin-adapted double.

    .. math::
        c_m = \sum_f k^{(m)}_f \sin\left(S_f\theta\right) \quad m \text{ odd}, \qquad
        c_m = \sum_f k^{(m)}_f \left(\cos\left(S_f\theta\right)-1\right) \quad m \text{ even}

    Args:
        order: Power of the generator this weight belongs to, counted from one.
        weights: Weight of this power at each frequency.
        frequencies: Frequencies of the generator.
        theta: Ansatz parameter value.

    Returns:
        Weight of this power.
    """
    if order % 2:
        return float(sum(k * np.sin(f * theta) for k, f in zip(weights, frequencies)))
    return float(sum(k * (np.cos(f * theta) - 1) for k, f in zip(weights, frequencies)))


def apply_spin_adapted_double(
    state: np.ndarray,
    op: FermionicOperator,
    exc_type: str,
    theta: float,
    ci_info: CI_Info,
    cache_key: tuple[str, tuple[int, ...]],
) -> np.ndarray:
    r"""Apply the exponential of a spin-adapted double excitation generator to a state.

    .. math::
        \left|\tilde{0}\right> = \exp\left(\theta\hat{T}\right)\left|0\right>

    Unlike a plain excitation these generators carry several frequencies, so their exponential
    is not a rotation of determinant pairs. It is still block diagonal, over groups of at most
    eight determinants, and taking it that way costs one sweep instead of one application of the
    generator per power. When the groups cannot be built the closed form is summed instead,
    which is what the code did everywhere before.

    Args:
        state: State.
        op: Excitation generator, already embedded in the CI space.
        exc_type: Which spin-adapted double this is.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.
        cache_key: Excitation type and indices naming this generator.

    Returns:
        New state.
    """
    blocked = apply_blocked_exponential(state.reshape(1, -1), op, theta, ci_info, cache_key)
    if blocked is not None:
        return blocked.reshape(state.shape)
    frequencies, weights = SPIN_ADAPTED_DOUBLE_SERIES[exc_type]
    out = np.copy(state)
    power = state
    for order, weight in enumerate(weights, start=1):
        power = propagate_state([op], power, ci_info, do_folding=False)
        out += spin_adapted_double_weight(order, weight, frequencies, theta) * power
    return out


def apply_spin_adapted_double_SA(
    states: np.ndarray,
    op: FermionicOperator,
    exc_type: str,
    theta: float,
    ci_info: CI_Info,
    cache_key: tuple[str, tuple[int, ...]],
) -> np.ndarray:
    r"""Apply the exponential of a spin-adapted double to every state of a state average.

    .. math::
        \left|\tilde{\nu}\right> = \exp\left(\theta\hat{T}\right)\left|\nu\right>

    Args:
        states: States as (number of states, number of determinants).
        op: Excitation generator, already embedded in the CI space.
        exc_type: Which spin-adapted double this is.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.
        cache_key: Excitation type and indices naming this generator.

    Returns:
        New states.
    """
    blocked = apply_blocked_exponential(states, op, theta, ci_info, cache_key)
    if blocked is not None:
        return blocked
    frequencies, weights = SPIN_ADAPTED_DOUBLE_SERIES[exc_type]
    out = np.copy(states)
    power = states
    for order, weight in enumerate(weights, start=1):
        power = propagate_state_SA([op], power, ci_info, do_folding=False)
        out += spin_adapted_double_weight(order, weight, frequencies, theta) * power
    return out


def apply_generator_exponential(
    state: np.ndarray,
    op: FermionicOperator,
    theta: float,
    ci_info: CI_Info,
    cache_key: tuple[str, tuple[int, ...]],
) -> np.ndarray:
    r"""Apply the exponential of an excitation generator to a state.

    .. math::
        \left|\tilde{0}\right> = \exp\left(\theta\hat{T}\right)\left|0\right>

    A generator that pairs determinants is applied as a Givens rotation, see
    build_rotation_layout. Anything else, a spin-adapted double in particular, falls back to
    summing the three terms of the closed form.

    Args:
        state: State.
        op: Excitation generator, already embedded in the CI space.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.
        cache_key: Excitation type and indices naming this generator.

    Returns:
        New state.
    """
    layout = get_rotation_layout(op, ci_info, cache_key)
    if layout is not None:
        out = np.copy(state)
        rotate_determinant_pairs(out.reshape(1, -1), *layout, np.cos(theta), np.sin(theta))
        return out
    blocked = apply_blocked_exponential(state.reshape(1, -1), op, theta, ci_info, cache_key)
    if blocked is not None:
        return blocked.reshape(state.shape)
    return (
        state
        + np.sin(theta) * propagate_state([op], state, ci_info, do_folding=False)
        + (1 - np.cos(theta)) * propagate_state([op, op], state, ci_info, do_folding=False)
    )


def apply_generator_exponential_SA(
    states: np.ndarray,
    op: FermionicOperator,
    theta: float,
    ci_info: CI_Info,
    cache_key: tuple[str, tuple[int, ...]],
) -> np.ndarray:
    r"""Apply the exponential of an excitation generator to every state of a state average.

    .. math::
        \left|\tilde{\nu}\right> = \exp\left(\theta\hat{T}\right)\left|\nu\right>

    Args:
        states: States as (number of states, number of determinants).
        op: Excitation generator, already embedded in the CI space.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.
        cache_key: Excitation type and indices naming this generator.

    Returns:
        New states.
    """
    layout = get_rotation_layout(op, ci_info, cache_key)
    if layout is not None:
        out = np.copy(states)
        rotate_determinant_pairs(out, *layout, np.cos(theta), np.sin(theta))
        return out
    blocked = apply_blocked_exponential(states, op, theta, ci_info, cache_key)
    if blocked is not None:
        return blocked
    return (
        states
        + np.sin(theta) * propagate_state_SA([op], states, ci_info, do_folding=False)
        + (1 - np.cos(theta)) * propagate_state_SA([op, op], states, ci_info, do_folding=False)
    )


def build_operator_matrix(op: FermionicOperator, ci_info: CI_Info, do_unsafe: bool = False) -> np.ndarray:
    """Build matrix representation of operator.

    Args:
        op: Fermionic number and spin conserving operator.
        ci_info: Information about the CI space.
        do_unsafe: Ignore elements that are outside the space defined in ci_info. (default: False)
                If not ignored, getting elements outside the space will stop the calculation.

    Returns:
        Matrix representation of operator.
    """
    idx2det = ci_info.idx2det
    det2idx = ci_info.det2idx
    num_active_orbs = ci_info.num_active_orbs
    num_dets = len(idx2det)  # number of spin and particle conserving determinants
    op_mat = np.zeros((num_dets, num_dets), dtype=float)  # basis
    # Create bitstrings for parity check. Contains occupied determinant up to orbital index.
    parity_check = np.zeros(2 * num_active_orbs + 1, dtype=int)
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    # loop over all strings of annihilation operators in FermionicOperator sum
    for fermi_label in op.operators.keys():
        # When screening determinants,
        # no need to consider the annihilation index of an
        # operator that has the same creation index.
        create_screen = np.array([idx for idx in fermi_label[0] if idx not in fermi_label[1]], dtype=np.int64)
        anni_idx = np.array(fermi_label[1], dtype=np.int64)
        a_string = np.array([*fermi_label[1], *fermi_label[0]], dtype=np.int64)
        op_mat = add_operator_matrix(
            op_mat,
            a_string,
            create_screen,
            anni_idx,
            num_active_orbs,
            parity_check,
            idx2det,
            det2idx,
            do_unsafe,
            op.operators[fermi_label],
        )
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
    if len(operators) == 0:
        return np.copy(state)
    idx2det = ci_info.idx2det
    det2idx = ci_info.det2idx
    num_inactive_orbs = ci_info.num_inactive_orbs
    num_active_orbs = ci_info.num_active_orbs
    num_virtual_orbs = ci_info.num_virtual_orbs
    if nb.get_num_threads() == 1:
        is_parallel = False
    else:
        is_parallel = True
    # Every kernel reads the incoming state and writes the outgoing one, so the two are
    # swapped rather than copied. The incoming state belongs to the caller and is only read.
    new_state = state
    tmp_state = np.zeros_like(state, dtype=float)
    tmp_state_is_zero = True
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
            if not tmp_state_is_zero:
                tmp_state[:] = 0.0
            # Fold operator to only get active contributions
            if do_folding:
                op_folded = op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
            else:
                op_folded = op
            # A spin conserving operator factorizes over a CI space that is a product of an
            # alpha and a beta string space, which is much cheaper to apply. Anything else,
            # notably the extended space, falls through to the general kernels below.
            factorized_state = propagate_state_factorized(op_folded, new_state, ci_info, tmp_state)
            if factorized_state is not None:
                new_state, tmp_state = tmp_state, new_state
                tmp_state_is_zero = False
                if tmp_state is state:
                    # The caller's state must never be handed to a kernel to write into.
                    tmp_state = np.zeros_like(state, dtype=float)
                    tmp_state_is_zero = True
                continue
            # loop over all strings of annihilation operators in FermionicOperator sum
            if is_parallel:
                for fermi_label in op_folded.operators.keys():
                    # When screening determinants,
                    # no need to consider the annihilation index of an
                    # operator that has the same creation index.
                    anni_screen = np.array(
                        [idx for idx in fermi_label[1] if idx not in fermi_label[0]], dtype=np.int64
                    )
                    create_idx = np.array(fermi_label[0], dtype=np.int64)
                    a_string = np.array([*fermi_label[0], *fermi_label[1]], dtype=np.int64)
                    tmp_state = apply_operator_threaded(
                        new_state,
                        a_string,
                        create_idx,
                        anni_screen,
                        num_active_orbs,
                        parity_check,
                        idx2det,
                        det2idx,
                        do_unsafe,
                        tmp_state,
                        op_folded.operators[fermi_label],
                    )
            else:
                for fermi_label in op_folded.operators.keys():
                    # When screening determinants,
                    # no need to consider the annihilation index of an
                    # operator that has the same creation index.
                    create_screen = np.array(
                        [idx for idx in fermi_label[0] if idx not in fermi_label[1]], dtype=np.int64
                    )
                    anni_idx = np.array(fermi_label[1], dtype=np.int64)
                    a_string = np.array([*fermi_label[1], *fermi_label[0]], dtype=np.int64)
                    tmp_state = apply_operator_serial(
                        new_state,
                        a_string,
                        create_screen,
                        anni_idx,
                        num_active_orbs,
                        parity_check,
                        idx2det,
                        det2idx,
                        do_unsafe,
                        tmp_state,
                        op_folded.operators[fermi_label],
                    )
            new_state, tmp_state = tmp_state, new_state
            tmp_state_is_zero = False
            if tmp_state is state:
                # The caller's state must never be handed to a kernel to write into.
                tmp_state = np.zeros_like(state, dtype=float)
                tmp_state_is_zero = True
    return new_state


def propagate_state_SA(
    operators: list[FermionicOperator | str],
    state: np.ndarray,
    ci_info: CI_Info,
    thetas: Sequence[float] | None = None,
    wf_struct: UpsStructure | None = None,
    do_folding: bool = True,
    do_unsafe: bool = False,
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
        do_unsafe: Ignore elements that are outside the space defined in ci_info. (default: False)
                If not ignored, getting elements outside the space will stop the calculation.

    Returns:
        New state.
    """
    if len(operators) == 0:
        return np.copy(state)
    idx2det = ci_info.idx2det
    det2idx = ci_info.det2idx
    num_inactive_orbs = ci_info.num_inactive_orbs
    num_active_orbs = ci_info.num_active_orbs
    num_virtual_orbs = ci_info.num_virtual_orbs
    if nb.get_num_threads() == 1:
        is_parallel = False
    else:
        is_parallel = True
    # Every kernel reads the incoming state and writes the outgoing one, so the two are
    # swapped rather than copied. The incoming state belongs to the caller and is only read.
    new_state = state
    tmp_state = np.zeros_like(state, dtype=float)
    tmp_state_is_zero = True
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
            if not tmp_state_is_zero:
                tmp_state[:, :] = 0.0
            # Fold operator to only get active contributions
            if do_folding:
                op_folded = op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
            else:
                op_folded = op
            # A spin conserving operator factorizes over a CI space that is a product of an
            # alpha and a beta string space, which is much cheaper to apply. Anything else,
            # notably the extended space, falls through to the general kernels below.
            factorized_state = propagate_state_SA_factorized(op_folded, new_state, ci_info, tmp_state)
            if factorized_state is not None:
                new_state, tmp_state = tmp_state, new_state
                tmp_state_is_zero = False
                if tmp_state is state:
                    # The caller's state must never be handed to a kernel to write into.
                    tmp_state = np.zeros_like(state, dtype=float)
                    tmp_state_is_zero = True
                continue
            # loop over all strings of annihilation operators in FermionicOperator sum
            if is_parallel:
                for fermi_label in op_folded.operators.keys():
                    # When screening determinants,
                    # no need to consider the annihilation index of an
                    # operator that has the same creation index.
                    anni_screen = np.array(
                        [idx for idx in fermi_label[1] if idx not in fermi_label[0]], dtype=np.int64
                    )
                    create_idx = np.array(fermi_label[0], dtype=np.int64)
                    a_string = np.array([*fermi_label[0], *fermi_label[1]], dtype=np.int64)
                    tmp_state = apply_operator_SA_threaded(
                        new_state,
                        a_string,
                        create_idx,
                        anni_screen,
                        num_active_orbs,
                        parity_check,
                        idx2det,
                        det2idx,
                        do_unsafe,
                        tmp_state,
                        op_folded.operators[fermi_label],
                    )
            else:
                for fermi_label in op_folded.operators.keys():
                    # When screening determinants,
                    # no need to consider the annihilation index of an
                    # operator that has the same creation index.
                    create_screen = np.array(
                        [idx for idx in fermi_label[0] if idx not in fermi_label[1]], dtype=np.int64
                    )
                    anni_idx = np.array(fermi_label[1], dtype=np.int64)
                    a_string = np.array([*fermi_label[1], *fermi_label[0]], dtype=np.int64)
                    tmp_state = apply_operator_SA_serial(
                        new_state,
                        a_string,
                        create_screen,
                        anni_idx,
                        num_active_orbs,
                        parity_check,
                        idx2det,
                        det2idx,
                        do_unsafe,
                        tmp_state,
                        op_folded.operators[fermi_label],
                    )
            new_state, tmp_state = tmp_state, new_state
            tmp_state_is_zero = False
            if tmp_state is state:
                # The caller's state must never be handed to a kernel to write into.
                tmp_state = np.zeros_like(state, dtype=float)
                tmp_state_is_zero = True
    return new_state


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

    val = 0.0
    for a, b in zip(bra, op_ket):
        val += a @ b

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
    T = get_ucc_T(thetas, ucc_struct, ci_info)
    # Evil matrix construction
    Tmat = build_operator_matrix(T, ci_info)
    if dagger:
        return ss.linalg.expm_multiply(-Tmat, state, traceA=0.0)
    return ss.linalg.expm_multiply(Tmat, state, traceA=0.0)


def get_ucc_T(
    thetas: Sequence[float],
    ucc_struct: UccStructure,
    ci_info: CI_Info,
) -> FermionicOperator:
    """Construct UCC operator.

    Args:
        thetas: Active-space parameters.
               Ordered as (S, D, T, ...).
        ucc_struct: UCCStructure object.
        ci_info: Information about the CI space.

    Returns:
        UCC operator.
    """
    # Build up T matrix based on excitations in ucc_struct and given thetas
    T = FermionicOperator({})
    for exc_type, exc_indices, theta in zip(
        ucc_struct.excitation_operator_type, ucc_struct.excitation_indices, thetas
    ):
        if abs(theta) < 10**-28:
            continue
        if exc_type == "sa_single":
            (i, a) = embed_spatial_indices(exc_indices, ci_info)
            T += theta * G1_sa(i, a, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T += theta * G2_sa(i, j, a, b, 1, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_2":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T += theta * G2_sa(i, j, a, b, 2, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_3":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T += theta * G2_sa(i, j, a, b, 3, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_4":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T += theta * G2_sa(i, j, a, b, 4, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_5":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T += theta * G2_sa(i, j, a, b, 5, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "single":
            (i, a) = embed_spin_indices(exc_indices, ci_info, ucc_struct.num_active_orbs)
            T += theta * G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = embed_spin_indices(exc_indices, ci_info, ucc_struct.num_active_orbs)
            T += theta * G2(i, j, a, b, True)
        elif exc_type == "triple":
            (i, j, k, a, b, c) = embed_spin_indices(exc_indices, ci_info, ucc_struct.num_active_orbs)
            T += theta * G3(i, j, k, a, b, c, True)
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = embed_spin_indices(exc_indices, ci_info, ucc_struct.num_active_orbs)
            T += theta * G4(i, j, k, l, a, b, c, d, True)
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = embed_spin_indices(
                exc_indices, ci_info, ucc_struct.num_active_orbs
            )
            T += theta * G5(i, j, k, l, m, a, b, c, d, e, True)
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = embed_spin_indices(
                exc_indices, ci_info, ucc_struct.num_active_orbs
            )
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
    if dagger:
        order = -1
    # Loop over all excitation in UPSStructure
    for exc_type, exc_indices, theta in zip(
        ups_struct.excitation_operator_type[::order], ups_struct.excitation_indices[::order], thetas[::order]
    ):
        if abs(theta) < 10**-28:
            continue
        if dagger:
            theta = -theta
        if exc_type in ("sa_single",):
            A = 1  # 2**(-1/2)
            (i, a) = embed_spatial_indices(exc_indices, ci_info)
            # Create T matrix
            Ta = G1(alpha_idx(i, ci_info.num_active_orbs), alpha_idx(a, ci_info.num_active_orbs), True)
            Tb = G1(beta_idx(i, ci_info.num_active_orbs), beta_idx(a, ci_info.num_active_orbs), True)
            # Analytical application on state vector
            out = apply_generator_exponential(
                out,
                Ta,
                A * theta,
                ci_info,
                ("sa_single_alpha", tuple(exc_indices)),
            )
            out = apply_generator_exponential(
                out,
                Tb,
                A * theta,
                ci_info,
                ("sa_single_beta", tuple(exc_indices)),
            )
        elif exc_type in ("single", "double", "triple", "quadruple", "quintuple", "sextuple", "sa_double_1"):
            # Create T matrix
            if exc_type == "single":
                (i, a) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
                T = G1(i, a, True)
            elif exc_type == "double":
                (i, j, a, b) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
                T = G2(i, j, a, b, True)
            elif exc_type == "triple":
                (i, j, k, a, b, c) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
                T = G3(i, j, k, a, b, c, True)
            elif exc_type == "quadruple":
                (i, j, k, l, a, b, c, d) = embed_spin_indices(
                    exc_indices, ci_info, ups_struct.num_active_orbs
                )
                T = G4(i, j, k, l, a, b, c, d, True)
            elif exc_type == "quintuple":
                (i, j, k, l, m, a, b, c, d, e) = embed_spin_indices(
                    exc_indices, ci_info, ups_struct.num_active_orbs
                )
                T = G5(i, j, k, l, m, a, b, c, d, e, True)
            elif exc_type == "sextuple":
                (i, j, k, l, m, n, a, b, c, d, e, f) = embed_spin_indices(
                    exc_indices, ci_info, ups_struct.num_active_orbs
                )
                T = G6(i, j, k, l, m, n, a, b, c, d, e, f, True)
            elif exc_type == "sa_double_1":
                (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
                T = G2_sa(i, j, a, b, 1, True, num_orbs=ci_info.num_active_orbs)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Analytical application on state vector
            out = apply_generator_exponential(out, T, theta, ci_info, (exc_type, tuple(exc_indices)))
        elif exc_type in ("sa_double_2", "sa_double_3"):
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
            out = apply_spin_adapted_double(out, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
        elif exc_type in ("sa_double_4",):
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
            out = apply_spin_adapted_double(out, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
        elif exc_type in ("sa_double_5",):
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
            out = apply_spin_adapted_double(out, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
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
    if dagger:
        order = -1
    # Loop over all excitation in UPSStructure
    for exc_type, exc_indices, theta in zip(
        ups_struct.excitation_operator_type[::order], ups_struct.excitation_indices[::order], thetas[::order]
    ):
        if abs(theta) < 10**-28:
            continue
        if dagger:
            theta = -theta
        if exc_type in ("sa_single",):
            A = 1  # 2**(-1/2)
            (i, a) = embed_spatial_indices(exc_indices, ci_info)
            # Create T matrices
            Ta = G1(alpha_idx(i, ci_info.num_active_orbs), alpha_idx(a, ci_info.num_active_orbs), True)
            Tb = G1(beta_idx(i, ci_info.num_active_orbs), beta_idx(a, ci_info.num_active_orbs), True)
            # Analytical application on state vector
            out = apply_generator_exponential_SA(
                out,
                Ta,
                A * theta,
                ci_info,
                ("sa_single_alpha", tuple(exc_indices)),
            )
            out = apply_generator_exponential_SA(
                out,
                Tb,
                A * theta,
                ci_info,
                ("sa_single_beta", tuple(exc_indices)),
            )
        elif exc_type in ("single", "double", "triple", "quadruple", "quintuple", "sextuple", "sa_double_1"):
            # Create T matrix
            if exc_type == "single":
                (i, a) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
                T = G1(i, a, True)
            elif exc_type == "double":
                (i, j, a, b) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
                T = G2(i, j, a, b, True)
            elif exc_type == "triple":
                (i, j, k, a, b, c) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
                T = G3(i, j, k, a, b, c, True)
            elif exc_type == "quadruple":
                (i, j, k, l, a, b, c, d) = embed_spin_indices(
                    exc_indices, ci_info, ups_struct.num_active_orbs
                )
                T = G4(i, j, k, l, a, b, c, d, True)
            elif exc_type == "quintuple":
                (i, j, k, l, m, a, b, c, d, e) = embed_spin_indices(
                    exc_indices, ci_info, ups_struct.num_active_orbs
                )
                T = G5(i, j, k, l, m, a, b, c, d, e, True)
            elif exc_type == "sextuple":
                (i, j, k, l, m, n, a, b, c, d, e, f) = embed_spin_indices(
                    exc_indices, ci_info, ups_struct.num_active_orbs
                )
                T = G6(i, j, k, l, m, n, a, b, c, d, e, f, True)
            elif exc_type == "sa_double_1":
                (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
                T = G2_sa(i, j, a, b, 1, True, num_orbs=ci_info.num_active_orbs)
            else:
                raise ValueError(f"Got unknown excitation type: {exc_type}")
            # Analytical application on state vector
            out = apply_generator_exponential_SA(out, T, theta, ci_info, (exc_type, tuple(exc_indices)))
        elif exc_type in ("sa_double_2", "sa_double_3"):
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
            out = apply_spin_adapted_double_SA(
                out, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices))
            )
        elif exc_type in ("sa_double_4",):
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
            out = apply_spin_adapted_double_SA(
                out, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices))
            )
        elif exc_type in ("sa_double_5",):
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
            out = apply_spin_adapted_double_SA(
                out, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices))
            )
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
    if abs(theta) < 10**-28:
        return np.copy(state)
    if exc_type in ("sa_single",):
        A = 1  # 2**(-1/2)
        (i, a) = embed_spatial_indices(exc_indices, ci_info)
        # Create T matrix
        Ta = G1(alpha_idx(i, ci_info.num_active_orbs), alpha_idx(a, ci_info.num_active_orbs), True)
        Tb = G1(beta_idx(i, ci_info.num_active_orbs), beta_idx(a, ci_info.num_active_orbs), True)
        # Analytical application on state vector
        out = apply_generator_exponential(
            state,
            Ta,
            A * theta,
            ci_info,
            ("sa_single_alpha", tuple(exc_indices)),
        )
        out = apply_generator_exponential(out, Tb, A * theta, ci_info, ("sa_single_beta", tuple(exc_indices)))
    elif exc_type in ("single", "double", "triple", "quadruple", "quintuple", "sextuple", "sa_double_1"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G2(i, j, a, b, True)
        elif exc_type == "triple":
            (i, j, k, a, b, c) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G3(i, j, k, a, b, c, True)
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G4(i, j, k, l, a, b, c, d, True)
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = embed_spin_indices(
                exc_indices, ci_info, ups_struct.num_active_orbs
            )
            T = G5(i, j, k, l, m, a, b, c, d, e, True)
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = embed_spin_indices(
                exc_indices, ci_info, ups_struct.num_active_orbs
            )
            T = G6(i, j, k, l, m, n, a, b, c, d, e, f, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 1, True, num_orbs=ci_info.num_active_orbs)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Analytical application on state vector
        out = apply_generator_exponential(state, T, theta, ci_info, (exc_type, tuple(exc_indices)))
    elif exc_type in ("sa_double_2", "sa_double_3"):
        (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
        T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
        out = apply_spin_adapted_double(state, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
    elif exc_type in ("sa_double_4",):
        (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
        T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
        out = apply_spin_adapted_double(state, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
    elif exc_type in ("sa_double_5",):
        (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
        T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
        out = apply_spin_adapted_double(state, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
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
    if abs(theta) < 10**-28:
        return np.copy(state)
    if exc_type in ("sa_single",):
        A = 1  # 2**(-1/2)
        (i, a) = embed_spatial_indices(exc_indices, ci_info)
        # Create T matrix
        Ta = G1(alpha_idx(i, ci_info.num_active_orbs), alpha_idx(a, ci_info.num_active_orbs), True)
        Tb = G1(beta_idx(i, ci_info.num_active_orbs), beta_idx(a, ci_info.num_active_orbs), True)
        # Analytical application on state vector
        out = apply_generator_exponential_SA(
            state,
            Ta,
            A * theta,
            ci_info,
            ("sa_single_alpha", tuple(exc_indices)),
        )
        out = apply_generator_exponential_SA(
            out,
            Tb,
            A * theta,
            ci_info,
            ("sa_single_beta", tuple(exc_indices)),
        )
    elif exc_type in ("single", "double", "triple", "quadruple", "quintuple", "sextuple", "sa_double_1"):
        # Create T matrix
        if exc_type == "single":
            (i, a) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G2(i, j, a, b, True)
        elif exc_type == "triple":
            (i, j, k, a, b, c) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G3(i, j, k, a, b, c, True)
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G4(i, j, k, l, a, b, c, d, True)
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = embed_spin_indices(
                exc_indices, ci_info, ups_struct.num_active_orbs
            )
            T = G5(i, j, k, l, m, a, b, c, d, e, True)
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = embed_spin_indices(
                exc_indices, ci_info, ups_struct.num_active_orbs
            )
            T = G6(i, j, k, l, m, n, a, b, c, d, e, f, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 1, True, num_orbs=ci_info.num_active_orbs)
        else:
            raise ValueError(f"Got unknown excitation type: {exc_type}")
        # Analytical application on state vector
        out = apply_generator_exponential_SA(state, T, theta, ci_info, (exc_type, tuple(exc_indices)))
    elif exc_type in ("sa_double_2", "sa_double_3"):
        (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
        T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
        out = apply_spin_adapted_double_SA(state, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
    elif exc_type in ("sa_double_4",):
        (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
        T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
        out = apply_spin_adapted_double_SA(state, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
    elif exc_type in ("sa_double_5",):
        (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
        T = G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=ci_info.num_active_orbs)
        out = apply_spin_adapted_double_SA(state, T, exc_type, theta, ci_info, (exc_type, tuple(exc_indices)))
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
    if exc_type in ("sa_single",):
        # Create T matrix
        A = 1  # 2**(-1/2)
        (i, a) = embed_spatial_indices(exc_indices, ci_info)
        Ta = G1(alpha_idx(i, ci_info.num_active_orbs), alpha_idx(a, ci_info.num_active_orbs), True)
        Tb = G1(beta_idx(i, ci_info.num_active_orbs), beta_idx(a, ci_info.num_active_orbs), True)
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
        "triple",
        "quadruple",
        "quintuple",
        "sextuple",
        "sa_double_1",
        "sa_double_2",
        "sa_double_3",
        "sa_double_4",
        "sa_double_5",
    ):
        # Create T matrix
        if exc_type == "single":
            (i, a) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G2(i, j, a, b, True)
        elif exc_type == "triple":
            (i, j, k, a, b, c) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G3(i, j, k, a, b, c, True)
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G4(i, j, k, l, a, b, c, d, True)
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = embed_spin_indices(
                exc_indices, ci_info, ups_struct.num_active_orbs
            )
            T = G5(i, j, k, l, m, a, b, c, d, e, True)
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = embed_spin_indices(
                exc_indices, ci_info, ups_struct.num_active_orbs
            )
            T = G6(i, j, k, l, m, n, a, b, c, d, e, f, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 1, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_2":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 2, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_3":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 3, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_4":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 4, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_5":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 5, True, num_orbs=ci_info.num_active_orbs)
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
    if exc_type in ("sa_single",):
        # Create T matrix
        A = 1  # 2**(-1/2)
        (i, a) = embed_spatial_indices(exc_indices, ci_info)
        Ta = G1(alpha_idx(i, ci_info.num_active_orbs), alpha_idx(a, ci_info.num_active_orbs), True)
        Tb = G1(beta_idx(i, ci_info.num_active_orbs), beta_idx(a, ci_info.num_active_orbs), True)
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
        "triple",
        "quadruple",
        "quintuple",
        "sextuple",
        "sa_double_1",
        "sa_double_2",
        "sa_double_3",
        "sa_double_4",
        "sa_double_5",
    ):
        # Create T matrix
        if exc_type == "single":
            (i, a) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G1(i, a, True)
        elif exc_type == "double":
            (i, j, a, b) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G2(i, j, a, b, True)
        elif exc_type == "triple":
            (i, j, k, a, b, c) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G3(i, j, k, a, b, c, True)
        elif exc_type == "quadruple":
            (i, j, k, l, a, b, c, d) = embed_spin_indices(exc_indices, ci_info, ups_struct.num_active_orbs)
            T = G4(i, j, k, l, a, b, c, d, True)
        elif exc_type == "quintuple":
            (i, j, k, l, m, a, b, c, d, e) = embed_spin_indices(
                exc_indices, ci_info, ups_struct.num_active_orbs
            )
            T = G5(i, j, k, l, m, a, b, c, d, e, True)
        elif exc_type == "sextuple":
            (i, j, k, l, m, n, a, b, c, d, e, f) = embed_spin_indices(
                exc_indices, ci_info, ups_struct.num_active_orbs
            )
            T = G6(i, j, k, l, m, n, a, b, c, d, e, f, True)
        elif exc_type == "sa_double_1":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 1, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_2":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 2, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_3":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 3, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_4":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 4, True, num_orbs=ci_info.num_active_orbs)
        elif exc_type == "sa_double_5":
            (i, j, a, b) = embed_spatial_indices(exc_indices, ci_info)
            T = G2_sa(i, j, a, b, 5, True, num_orbs=ci_info.num_active_orbs)
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


def get_determinant_expansion_from_operator_on_HF(
    operator: FermionicOperator,
    num_active_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
) -> tuple[list[float], list[str]]:
    """Get determinant expansion from applying operator to HF state.

    Args:
        operator: Fermionic operator.
        num_active_orbs: Number of active spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.

    Returns:
        Determinant expansion.
    """
    # Blocked ordering, so the alpha string is followed by the beta string.
    hf_det_ = ""
    for p in range(num_active_orbs):
        hf_det_ += "1" if p < num_active_elec_alpha else "0"
    for p in range(num_active_orbs):
        hf_det_ += "1" if p < num_active_elec_beta else "0"
    hf_det = int(hf_det_, 2)

    coeffs = []
    dets = []
    parity_check = {0: 0}
    num = 0
    for i in range(2 * num_active_orbs - 1, -1, -1):
        num += 2**i
        parity_check[2 * num_active_orbs - i] = num
    for anni_string, fac in operator.operators.items():
        det = hf_det
        phase_changes = 0
        is_killstate = False
        # Do annihilation
        for orb_idx in anni_string[1][::-1]:
            nth_bit = (det >> 2 * num_active_orbs - 1 - orb_idx) & 1
            if nth_bit == 0:
                is_killstate = True
                break
            det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
            phase_changes += (det & parity_check[orb_idx]).bit_count()
        # Do creation
        for orb_idx in anni_string[0][::-1]:
            nth_bit = (det >> 2 * num_active_orbs - 1 - orb_idx) & 1
            if nth_bit == 1:
                is_killstate = True
                break
            det = det ^ 2 ** (2 * num_active_orbs - 1 - orb_idx)
            phase_changes += (det & parity_check[orb_idx]).bit_count()
        if not is_killstate:
            val = fac * (-1) ** phase_changes
            coeffs.append(val)
            dets.append(format(det, f"0{2 * num_active_orbs}b"))
    return coeffs, dets
