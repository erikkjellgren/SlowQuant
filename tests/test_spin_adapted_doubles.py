"""Test the blocked exponential of the spin-adapted double excitation generators.

These generators carry several frequencies, so unlike a plain excitation they do not rotate
pairs of determinants. They are still block diagonal over small groups, and the exponential is
taken group by group. The reference is a dense matrix exponential of the generator, built one
matrix element at a time, and the closed form the code used before.
"""

import time

import numpy as np
import pytest
import scipy.linalg

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing, get_indexing_extended
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    SPIN_ADAPTED_DOUBLE_SERIES,
    apply_blocked_exponential,
    apply_spin_adapted_double,
    apply_spin_adapted_double_SA,
    build_generator_blocks,
    build_operator_matrix,
    embed_spatial_indices,
    propagate_state,
    spin_adapted_double_weight,
)
from slowquant.unitary_coupled_cluster.operators import G2_sa

NUM_ORBS = 6
NUM_ELEC = 6
CASES = ("sa_double_2", "sa_double_3", "sa_double_4", "sa_double_5")
THETAS = (0.0, 0.29, -1.4, np.pi / 3, 2.6)


def build_space():
    """Build a small closed-shell CI space.

    Returns:
        CI space information.
    """
    return get_indexing(0, NUM_ORBS, 0, NUM_ELEC // 2, NUM_ELEC // 2)


# The index topology each spin-adapted double is built with. Case two only ever appears with
# one occupied orbital and case three with one virtual, and their closed forms are derived for
# exactly that; handing them four distinct indices gives a generator they do not describe.
TOPOLOGY = {
    "sa_double_2": (0, 0, 2, 3),
    "sa_double_3": (0, 1, 2, 2),
    "sa_double_4": (0, 1, 2, 3),
    "sa_double_5": (0, 1, 2, 3),
}


def generator(exc_type, num_orbs=NUM_ORBS, offset=0):
    """Build the spin-adapted double of one case, with the indices that case is used with.

    Args:
        exc_type: Which spin-adapted double to build.
        num_orbs: Number of spatial orbitals the indices live in.
        offset: Where the active window starts.

    Returns:
        Excitation generator.
    """
    i, j, a, b = (offset + p for p in TOPOLOGY[exc_type])
    return G2_sa(i, j, a, b, int(exc_type[-1]), True, num_orbs=num_orbs)


def series_form(state, op, exc_type, theta, ci_info):
    r"""Apply :math:`\exp(\theta\hat{T})` by summing the closed form.

    This is the path taken when the generator cannot be blocked, and is what the code did
    everywhere before.

    Args:
        state: State.
        op: Excitation generator.
        exc_type: Which spin-adapted double this is.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.

    Returns:
        New state.
    """
    frequencies, weights = SPIN_ADAPTED_DOUBLE_SERIES[exc_type]
    out = np.copy(state)
    power = state
    for order, weight in enumerate(weights, start=1):
        power = propagate_state([op], power, ci_info, do_folding=False)
        out += spin_adapted_double_weight(order, weight, frequencies, theta) * power
    return out


@pytest.mark.parametrize("exc_type", CASES)
@pytest.mark.parametrize("theta", THETAS)
def test_blocked_exponential_matches_the_matrix_exponential(exc_type: str, theta: float) -> None:
    """Test the blocked exponential against a dense exponential of the generator.

    Args:
        exc_type: Which spin-adapted double to test.
        theta: Ansatz parameter value.
    """
    ci_info = build_space()
    op = generator(exc_type)
    state = np.random.default_rng(21).random(len(ci_info.idx2det))
    reference = scipy.linalg.expm(theta * build_operator_matrix(op, ci_info)) @ state
    got = apply_spin_adapted_double(state, op, exc_type, theta, ci_info, (exc_type, (0,)))
    assert np.allclose(got, reference, atol=1e-11), exc_type


@pytest.mark.parametrize("exc_type", CASES)
def test_blocked_exponential_matches_the_closed_form(exc_type: str) -> None:
    """Test the blocked exponential against the closed form it replaces.

    Args:
        exc_type: Which spin-adapted double to test.
    """
    ci_info = build_space()
    op = generator(exc_type)
    state = np.random.default_rng(22).random(len(ci_info.idx2det))
    for theta in THETAS:
        blocked = apply_spin_adapted_double(state, op, exc_type, theta, ci_info, (exc_type, (0,)))
        assert np.allclose(blocked, series_form(state, op, exc_type, theta, ci_info), atol=1e-11), exc_type


@pytest.mark.parametrize("exc_type", CASES)
def test_the_groups_are_small_and_few(exc_type: str) -> None:
    """Test that a generator gives small groups, and few distinct ones, at two active space sizes.

    This is what makes the blocked form worth taking: the exponential is built once per distinct
    group, so the work per parameter value must not grow with the CI space.

    Args:
        exc_type: Which spin-adapted double to test.
    """
    previous = None
    for num_orbs, num_elec in ((6, 6), (8, 8)):
        ci_info = get_indexing(0, num_orbs, 0, num_elec // 2, num_elec // 2)
        built = build_generator_blocks(generator(exc_type, num_orbs), ci_info)
        assert built is not None, exc_type
        _, starts, _, groups = built
        assert np.max(np.diff(starts)) <= 8, exc_type
        # The distinct groups are fixed by the handful of orbitals the generator touches, so
        # their number stays small however large the CI space gets. It is not quite constant:
        # a bigger space can offer an occupation pattern the smaller one had no room for.
        assert len(groups) <= 16, exc_type
        if previous is not None:
            assert abs(len(groups) - previous) <= 2, exc_type
        previous = len(groups)


@pytest.mark.parametrize("exc_type", CASES)
def test_the_unitary_is_orthogonal(exc_type: str) -> None:
    """Test that the blocked exponential preserves the norm and undoes itself.

    Args:
        exc_type: Which spin-adapted double to test.
    """
    ci_info = build_space()
    op = generator(exc_type)
    state = np.random.default_rng(23).random(len(ci_info.idx2det))
    state /= np.linalg.norm(state)
    forward = apply_spin_adapted_double(state, op, exc_type, 0.77, ci_info, (exc_type, (0,)))
    assert abs(np.linalg.norm(forward) - 1.0) < 1e-12, exc_type
    back = apply_spin_adapted_double(forward, op, exc_type, -0.77, ci_info, (exc_type, (0,)))
    assert np.allclose(back, state, atol=1e-11), exc_type


@pytest.mark.parametrize("exc_type", CASES)
def test_the_incoming_state_is_left_alone(exc_type: str) -> None:
    """Test that the blocked exponential does not write into the state it was handed.

    Args:
        exc_type: Which spin-adapted double to test.
    """
    ci_info = build_space()
    op = generator(exc_type)
    state = np.random.default_rng(24).random(len(ci_info.idx2det))
    untouched = np.copy(state)
    apply_spin_adapted_double(state, op, exc_type, 0.4, ci_info, (exc_type, (0,)))
    assert np.array_equal(state, untouched), exc_type


def test_state_averaged_matches_one_state_at_a_time() -> None:
    """Test that every state of a state average is taken through the same exponential."""
    ci_info = build_space()
    states = np.random.default_rng(25).random((3, len(ci_info.idx2det)))
    op = generator("sa_double_4")
    got = apply_spin_adapted_double_SA(states, op, "sa_double_4", 0.51, ci_info, ("sa", (0,)))
    for state_idx in range(len(states)):
        one = apply_spin_adapted_double(
            states[state_idx], op, "sa_double_4", 0.51, ci_info, ("one", (state_idx,))
        )
        assert np.allclose(got[state_idx], one, atol=1e-12)


@pytest.mark.parametrize("exc_type", CASES)
def test_blocking_works_on_a_space_that_is_not_a_spin_product(exc_type: str) -> None:
    """Test the blocked exponential on the extended space of the self-consistent linear response.

    The groups are in determinant indices, so they do not need the CI space to be a product of
    an alpha and a beta string space.

    Args:
        exc_type: Which spin-adapted double to test.
    """
    ci_info = get_indexing_extended(1, NUM_ORBS, 1, NUM_ELEC // 2, NUM_ELEC // 2, 1)
    assert not ci_info.is_spin_product
    i = embed_spatial_indices((0,), ci_info)[0]
    op = generator(exc_type, ci_info.num_active_orbs, offset=i)
    state = np.random.default_rng(26).random(len(ci_info.idx2det))
    assert build_generator_blocks(op, ci_info) is not None, exc_type
    for theta in (0.33, -0.9):
        blocked = apply_spin_adapted_double(state, op, exc_type, theta, ci_info, (exc_type, (1,)))
        assert np.allclose(blocked, series_form(state, op, exc_type, theta, ci_info), atol=1e-11), exc_type


def test_a_generator_that_cannot_be_blocked_reports_nothing() -> None:
    """Test that a generator leaving the CI space gets no groups, so the closed form stays."""
    ci_info = get_indexing_extended(1, NUM_ORBS, 1, NUM_ELEC // 2, NUM_ELEC // 2, 1)
    # Straight out of the inactive orbital, which the extended space only holds singles of.
    op = generator("sa_double_4", ci_info.num_active_orbs, offset=0)
    assert build_generator_blocks(op, ci_info) is None
    assert (
        apply_blocked_exponential(np.zeros((1, len(ci_info.idx2det))), op, 0.2, ci_info, ("outside", (0,)))
        is None
    )


def test_the_blocked_exponential_is_faster_than_the_closed_form() -> None:
    """Test that blocking beats summing the closed form by a wide margin.

    Both paths are timed in the same process on the same generator, so this compares the two
    algorithms rather than the machine. The margin asserted is far below what is measured; the
    point is that the blocked form cannot silently stop being the faster one.
    """
    num_orbs = 10
    ci_info = get_indexing(0, num_orbs, 0, 5, 5)
    op = generator("sa_double_5", num_orbs)
    state = np.random.default_rng(27).random(len(ci_info.idx2det))
    theta = 0.35

    def best(call):
        """Time a call a few times and keep the fastest.

        Args:
            call: Callable taking no arguments.

        Returns:
            Seconds the fastest run took.
        """
        call()
        fastest = np.inf
        for _ in range(5):
            start = time.perf_counter()
            call()
            fastest = min(fastest, time.perf_counter() - start)
        return fastest

    closed = best(lambda: series_form(state, op, "sa_double_5", theta, ci_info))
    blocked = best(
        lambda: apply_spin_adapted_double(state, op, "sa_double_5", theta, ci_info, ("speed", (0,)))
    )
    assert blocked < closed / 3, f"blocked {blocked * 1e6:.0f} us vs closed form {closed * 1e6:.0f} us"
