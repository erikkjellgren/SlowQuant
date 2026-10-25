"""Test the string-space form of the excitation rotations.

Over a spin product an excitation pairs spin strings rather than determinants, so the rotation
can be held in a form whose size does not grow with the CI space. The reference is a dense
matrix exponential of the generator, and the pairing is also checked against the determinant
form it replaces.
"""

from math import comb

import numpy as np
import pytest
import scipy.linalg

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing, get_indexing_extended
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    apply_generator_exponential,
    apply_generator_exponential_SA,
    build_operator_matrix,
    embed_spatial_indices,
)
from slowquant.unitary_coupled_cluster.operators import G1, G2, G3, G4, G2_sa
from slowquant.unitary_coupled_cluster.spin_factorized_algebra import (
    ROTATION_ALPHA,
    ROTATION_BETA,
    ROTATION_MIXED,
    build_string_rotation_layout,
)
from slowquant.unitary_coupled_cluster.spin_ordering import alpha_idx, beta_idx

NUM_ORBS = 6
THETAS = (0.0, 0.31, -1.2, np.pi / 2, 2.4)


def space(num_orbs=NUM_ORBS, num_alpha=3, num_beta=3):
    """Build a spin-product CI space.

    Args:
        num_orbs: Number of active spatial orbitals.
        num_alpha: Number of alpha electrons.
        num_beta: Number of beta electrons.

    Returns:
        CI space information.
    """
    return get_indexing(0, num_orbs, 0, num_alpha, num_beta)


def generators(num_orbs=NUM_ORBS):
    """One generator of each shape a pairing can take.

    Args:
        num_orbs: Number of active spatial orbitals.

    Returns:
        Name, generator and expected rotation kind.
    """
    a = lambda p: alpha_idx(p, num_orbs)  # noqa: E731
    b = lambda p: beta_idx(p, num_orbs)  # noqa: E731
    return [
        ("single alpha", G1(a(0), a(num_orbs - 1), True), ROTATION_ALPHA),
        ("single beta", G1(b(1), b(num_orbs - 2), True), ROTATION_BETA),
        ("double alpha alpha", G2(a(0), a(1), a(num_orbs - 2), a(num_orbs - 1), True), ROTATION_ALPHA),
        ("double beta beta", G2(b(0), b(1), b(num_orbs - 2), b(num_orbs - 1), True), ROTATION_BETA),
        ("double mixed", G2(a(0), b(1), a(num_orbs - 2), b(num_orbs - 1), True), ROTATION_MIXED),
        (
            "triple two alpha one beta",
            G3(a(0), a(1), b(2), a(num_orbs - 3), a(num_orbs - 2), b(num_orbs - 1), True),
            ROTATION_MIXED,
        ),
        (
            "quadruple two and two",
            G4(
                a(0),
                a(1),
                b(0),
                b(1),
                a(num_orbs - 2),
                a(num_orbs - 1),
                b(num_orbs - 2),
                b(num_orbs - 1),
                True,
            ),
            ROTATION_MIXED,
        ),
    ]


@pytest.mark.parametrize("name, op, kind", generators())
@pytest.mark.parametrize("theta", THETAS)
def test_string_rotation_matches_the_matrix_exponential(name: str, op, kind: int, theta: float) -> None:
    """Test the string rotation against a dense exponential of the generator.

    Args:
        name: Name of the generator.
        op: Excitation generator.
        kind: Which spins the rotation is expected to touch.
        theta: Ansatz parameter value.
    """
    ci_info = space()
    folded = op.get_folded_operator(0, NUM_ORBS, 0)
    state = np.random.default_rng(4).random(ci_info.num_dets)
    reference = scipy.linalg.expm(theta * build_operator_matrix(folded, ci_info)) @ state
    got = apply_generator_exponential(state, folded, theta, ci_info, (name, (0,)))
    assert np.allclose(got, reference, atol=1e-12), name


@pytest.mark.parametrize("name, op, kind", generators())
def test_the_rotation_touches_the_spins_it_should(name: str, op, kind: int) -> None:
    """Test that each generator is recognised as acting on alpha, beta or both.

    Args:
        name: Name of the generator.
        op: Excitation generator.
        kind: Which spins the rotation is expected to touch.
    """
    ci_info = space()
    layout = build_string_rotation_layout(op.get_folded_operator(0, NUM_ORBS, 0), ci_info)
    assert layout is not None, name
    assert layout[0] == kind, name


@pytest.mark.parametrize("name, op, kind", generators())
def test_the_layout_does_not_grow_with_the_ci_space(name: str, op, kind: int) -> None:
    """Test that a rotation is stored per spin string, not per determinant.

    This is the whole point of the string form: doubling the determinant count must not double
    the memory the ansatz holds.

    Args:
        name: Name of the generator.
        op: Excitation generator.
        kind: Which spins the rotation is expected to touch.
    """
    small = space(NUM_ORBS, 3, 3)
    large = space(NUM_ORBS + 2, 4, 4)
    entries = []
    for ci_info, num_orbs in ((small, NUM_ORBS), (large, NUM_ORBS + 2)):
        rebuilt = generators(num_orbs)
        op_here = next(o for label, o, _ in rebuilt if label == name)
        layout = build_string_rotation_layout(op_here.get_folded_operator(0, num_orbs, 0), ci_info)
        assert layout is not None, name
        entries.append(len(layout[1][0]) + len(layout[2][0]))
    determinant_growth = large.num_dets / small.num_dets
    string_growth = (large.num_alpha_strings + large.num_beta_strings) / (
        small.num_alpha_strings + small.num_beta_strings
    )
    entry_growth = entries[1] / entries[0]
    # The claim is that a layout is sized by the spin strings, so it must track their growth and
    # not the determinants', which grow as the square.
    assert entry_growth <= string_growth * 1.2, (
        f"{name}: entries grew {entry_growth:.1f}x against {string_growth:.1f}x for the strings"
    )
    assert entry_growth < determinant_growth / 2, (
        f"{name}: entries grew {entry_growth:.1f}x, close to the determinants' {determinant_growth:.1f}x"
    )


def test_a_spin_adapted_double_has_no_string_pairing() -> None:
    """Test that a generator which is not one excitation and its adjoint reports no layout."""
    ci_info = space()
    op = G2_sa(0, 0, NUM_ORBS - 2, NUM_ORBS - 1, 2, True, num_orbs=NUM_ORBS)
    assert build_string_rotation_layout(op.get_folded_operator(0, NUM_ORBS, 0), ci_info) is None


def test_a_space_that_is_not_a_product_has_no_string_pairing() -> None:
    """Test that the string form is refused off a spin product, leaving the determinant form."""
    ci_info = get_indexing_extended(1, NUM_ORBS, 1, 3, 3, 1)
    assert not ci_info.is_spin_product
    num_orbs = ci_info.num_active_orbs
    i, a = embed_spatial_indices((0, NUM_ORBS - 1), ci_info)
    op = G1(alpha_idx(i, num_orbs), alpha_idx(a, num_orbs), True)
    assert build_string_rotation_layout(op, ci_info) is None
    # And the unitary still works there, through the determinant form.
    state = np.random.default_rng(8).random(ci_info.num_dets)
    reference = scipy.linalg.expm(0.4 * build_operator_matrix(op, ci_info)) @ state
    got = apply_generator_exponential(state, op, 0.4, ci_info, ("extended", (0,)))
    assert np.allclose(got, reference, atol=1e-12)


def test_state_averaged_rotation_matches_one_state_at_a_time() -> None:
    """Test that every state of a state average is rotated the same way."""
    ci_info = space()
    states = np.random.default_rng(9).random((3, ci_info.num_dets))
    op = G2(
        alpha_idx(0, NUM_ORBS), beta_idx(1, NUM_ORBS), alpha_idx(4, NUM_ORBS), beta_idx(5, NUM_ORBS), True
    ).get_folded_operator(0, NUM_ORBS, 0)
    together = apply_generator_exponential_SA(states, op, 0.62, ci_info, ("sa", (0,)))
    for state_idx in range(len(states)):
        alone = apply_generator_exponential(states[state_idx], op, 0.62, ci_info, ("one", (state_idx,)))
        assert np.allclose(together[state_idx], alone, atol=1e-12)


@pytest.mark.parametrize("num_alpha, num_beta", ((3, 3), (4, 2), (2, 4), (4, 3)))
def test_open_shell_spaces(num_alpha: int, num_beta: int) -> None:
    """Test the string rotation where the two spin spaces differ in size.

    A phase error between the spin blocks shows up here and not in a closed-shell space.

    Args:
        num_alpha: Number of alpha electrons.
        num_beta: Number of beta electrons.
    """
    ci_info = space(NUM_ORBS, num_alpha, num_beta)
    assert ci_info.num_dets == comb(NUM_ORBS, num_alpha) * comb(NUM_ORBS, num_beta)
    state = np.random.default_rng(10).random(ci_info.num_dets)
    for name, op, _ in generators():
        folded = op.get_folded_operator(0, NUM_ORBS, 0)
        for theta in (0.37, -0.9):
            reference = scipy.linalg.expm(theta * build_operator_matrix(folded, ci_info)) @ state
            got = apply_generator_exponential(state, folded, theta, ci_info, (name, (num_alpha,)))
            assert np.allclose(got, reference, atol=1e-12), f"{name} at ({num_alpha}a,{num_beta}b)"
