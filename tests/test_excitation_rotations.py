"""Test the Givens-rotation form of the excitation unitaries.

An excitation generator squares to minus a projector, so the determinant basis splits into
pairs and the exponential is a rotation on each pair. The reference here is deliberately not
the code that does that: it is a dense matrix exponential of the generator, built one matrix
element at a time.
"""

import time

import numpy as np
import pytest
import scipy.linalg

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing, get_indexing_extended
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    apply_generator_exponential,
    apply_generator_exponential_SA,
    build_operator_matrix,
    build_rotation_layout,
    embed_spatial_indices,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import G1, G2, G3, G2_sa
from slowquant.unitary_coupled_cluster.spin_ordering import alpha_idx, beta_idx

NUM_ORBS = 6
NUM_ELEC = 6


def build_space():
    """Build a small closed-shell CI space.

    Returns:
        CI space information.
    """
    return get_indexing(0, NUM_ORBS, 0, NUM_ELEC // 2, NUM_ELEC // 2)


def series_form(state, op, theta, ci_info):
    r"""Apply :math:`\exp(\theta\hat{T})` as the sum of its three terms.

    This is the form the code used before the rotation was introduced, and the form a
    generator that is not a pairing still falls back to.

    Args:
        state: State.
        op: Excitation generator.
        theta: Ansatz parameter value.
        ci_info: Information about the CI space.

    Returns:
        New state.
    """
    return (
        state
        + np.sin(theta) * propagate_state([op], state, ci_info, do_folding=False)
        + (1 - np.cos(theta)) * propagate_state([op, op], state, ci_info, do_folding=False)
    )


def generators(num_orbs):
    """Build one generator of every type the ansatz builders emit.

    Args:
        num_orbs: Number of active spatial orbitals.

    Returns:
        Name and generator pairs.
    """
    a = lambda p: alpha_idx(p, num_orbs)  # noqa: E731
    b = lambda p: beta_idx(p, num_orbs)  # noqa: E731
    return [
        ("single alpha", G1(a(0), a(num_orbs - 1), True)),
        ("single beta", G1(b(1), b(num_orbs - 2), True)),
        ("double same spin", G2(a(0), a(1), a(num_orbs - 2), a(num_orbs - 1), True)),
        ("double mixed spin", G2(a(0), b(1), a(num_orbs - 2), b(num_orbs - 1), True)),
        ("triple", G3(a(0), a(1), b(2), a(num_orbs - 3), a(num_orbs - 2), b(num_orbs - 1), True)),
    ]


@pytest.mark.parametrize("name, op", generators(NUM_ORBS))
@pytest.mark.parametrize("theta", (0.0, 0.37, -1.2, np.pi / 2, 3.0))
def test_rotation_matches_the_matrix_exponential(name: str, op, theta: float) -> None:
    """Test the rotation against a dense exponential of the generator.

    Args:
        name: Name of the generator.
        op: Excitation generator.
        theta: Ansatz parameter value.
    """
    ci_info = build_space()
    state = np.random.default_rng(4).random(len(ci_info.idx2det))
    reference = scipy.linalg.expm(theta * build_operator_matrix(op, ci_info)) @ state
    got = apply_generator_exponential(state, op, theta, ci_info, (name, (0,)))
    assert np.allclose(got, reference, atol=1e-12), name


@pytest.mark.parametrize("name, op", generators(NUM_ORBS))
def test_rotation_matches_the_series_it_replaces(name: str, op) -> None:
    """Test the rotation against the three-term sum it replaces.

    Args:
        name: Name of the generator.
        op: Excitation generator.
    """
    ci_info = build_space()
    state = np.random.default_rng(5).random(len(ci_info.idx2det))
    for theta in (0.11, 0.9, -2.4):
        assert np.allclose(
            apply_generator_exponential(state, op, theta, ci_info, (name, (0,))),
            series_form(state, op, theta, ci_info),
            atol=1e-12,
        ), name


@pytest.mark.parametrize("name, op", generators(NUM_ORBS))
def test_the_unitary_is_orthogonal(name: str, op) -> None:
    """Test that the rotation preserves the norm and undoes itself.

    Args:
        name: Name of the generator.
        op: Excitation generator.
    """
    ci_info = build_space()
    state = np.random.default_rng(6).random(len(ci_info.idx2det))
    state /= np.linalg.norm(state)
    forward = apply_generator_exponential(state, op, 0.83, ci_info, (name, (0,)))
    assert abs(np.linalg.norm(forward) - 1.0) < 1e-12, name
    back = apply_generator_exponential(forward, op, -0.83, ci_info, (name, (0,)))
    assert np.allclose(back, state, atol=1e-12), name


@pytest.mark.parametrize("name, op", generators(NUM_ORBS))
def test_the_incoming_state_is_left_alone(name: str, op) -> None:
    """Test that the rotation does not write into the state it was handed.

    Args:
        name: Name of the generator.
        op: Excitation generator.
    """
    ci_info = build_space()
    state = np.random.default_rng(7).random(len(ci_info.idx2det))
    untouched = np.copy(state)
    apply_generator_exponential(state, op, 0.4, ci_info, (name, (0,)))
    assert np.array_equal(state, untouched), name


def test_state_averaged_rotation_matches_one_state_at_a_time() -> None:
    """Test that every state of a state average is rotated the same way."""
    ci_info = build_space()
    num_dets = len(ci_info.idx2det)
    states = np.random.default_rng(8).random((3, num_dets))
    op = G2(
        alpha_idx(0, NUM_ORBS), beta_idx(1, NUM_ORBS), alpha_idx(4, NUM_ORBS), beta_idx(5, NUM_ORBS), True
    )
    got = apply_generator_exponential_SA(states, op, 0.62, ci_info, ("sa", (0,)))
    for state_idx in range(len(states)):
        one = apply_generator_exponential(states[state_idx], op, 0.62, ci_info, ("one", (state_idx,)))
        assert np.allclose(got[state_idx], one, atol=1e-12)


@pytest.mark.parametrize("case", (1, 2, 3, 4))
def test_spin_adapted_doubles_are_not_a_pairing(case: int) -> None:
    """Test that a spin-adapted double reports no layout and keeps the three-term sum.

    Their generator takes values other than plus or minus one on the determinant basis, so the
    two dimensional block structure the rotation needs is absent.

    Args:
        case: Which of the spin-adapted double generators to build.
    """
    ci_info = build_space()
    op = G2_sa(0, 1, NUM_ORBS - 2, NUM_ORBS - 1, case, True, num_orbs=NUM_ORBS)
    assert build_rotation_layout(op, ci_info) is None


def extended_space():
    """Build the extended space the self-consistent linear response uses.

    Returns:
        CI space information, which is a union of product blocks rather than a product.
    """
    return get_indexing_extended(1, NUM_ORBS, 1, NUM_ELEC // 2, NUM_ELEC // 2, 1)


def test_rotation_works_on_a_space_that_is_not_a_spin_product() -> None:
    """Test the rotation on the extended space, with generators embedded as the ansatz embeds them.

    That space is a union of product blocks rather than a product, so the alpha and beta string
    factorization does not apply to it. The pairing is read off the generator itself, which does
    not depend on the space having that structure.
    """
    ci_info = extended_space()
    assert not ci_info.is_spin_product
    num_orbs = ci_info.num_active_orbs
    state = np.random.default_rng(9).random(len(ci_info.idx2det))
    i, a = embed_spatial_indices((0, NUM_ORBS - 1), ci_info)
    j, b = embed_spatial_indices((1, NUM_ORBS - 2), ci_info)
    cases = (
        ("single alpha", G1(alpha_idx(i, num_orbs), alpha_idx(a, num_orbs), True)),
        ("single beta", G1(beta_idx(i, num_orbs), beta_idx(a, num_orbs), True)),
        (
            "double mixed spin",
            G2(
                alpha_idx(i, num_orbs),
                beta_idx(j, num_orbs),
                alpha_idx(a, num_orbs),
                beta_idx(b, num_orbs),
                True,
            ),
        ),
    )
    for name, op in cases:
        assert build_rotation_layout(op, ci_info) is not None, name
        for theta in (0.25, -1.1):
            assert np.allclose(
                apply_generator_exponential(state, op, theta, ci_info, (name, (1,))),
                series_form(state, op, theta, ci_info),
                atol=1e-12,
            ), name


def test_a_generator_leaving_the_space_falls_back() -> None:
    """Test that a generator which is not closed on the CI space keeps the three-term sum.

    The general kernel skips determinants the state is zero on, so it tolerates such a generator
    for as long as the state stays away from the determinants it cannot represent. A rotation
    cannot, so no layout is built and the old path stays in charge.
    """
    ci_info = extended_space()
    num_orbs = ci_info.num_active_orbs
    # Straight out of the inactive orbital, which the extended space only holds singles of.
    op = G2(
        alpha_idx(0, num_orbs),
        beta_idx(0, num_orbs),
        alpha_idx(num_orbs - 1, num_orbs),
        beta_idx(num_orbs - 1, num_orbs),
        True,
    )
    assert build_rotation_layout(op, ci_info) is None


def test_rotation_is_faster_than_the_series_it_replaces() -> None:
    """Test that the rotation beats the three-term sum by a wide margin.

    Both paths are timed in the same process on the same operator, so this compares the two
    algorithms rather than the machine. The margin asserted is far below what is measured, and
    the point is only that the rotation cannot silently stop being the faster one.
    """
    num_orbs = 10
    ci_info = get_indexing(0, num_orbs, 0, 5, 5)
    op = G1(alpha_idx(0, num_orbs), alpha_idx(num_orbs - 1, num_orbs), True)
    state = np.random.default_rng(10).random(len(ci_info.idx2det))
    theta = 0.3

    def best(call):
        """Time a call a few times and keep the fastest.

        Args:
            call: Callable taking no arguments.

        Returns:
            Seconds the fastest run took.
        """
        call()
        fastest = np.inf
        for _ in range(7):
            start = time.perf_counter()
            call()
            fastest = min(fastest, time.perf_counter() - start)
        return fastest

    series = best(lambda: series_form(state, op, theta, ci_info))
    rotation = best(lambda: apply_generator_exponential(state, op, theta, ci_info, ("speed", (0,))))
    assert rotation < series / 4, f"rotation {rotation * 1e6:.0f} us vs series {series * 1e6:.0f} us"
