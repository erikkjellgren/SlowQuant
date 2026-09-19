"""Test the per-spin blocked form of the generators that are not pairings.

A spin-adapted double carries several frequencies, so its unitary is not a rotation of
determinant pairs. Over a spin product it still blocks: the strings of each spin fall into
cells the operator connects, and it cannot take a determinant out of the cell pair its two
strings lie in. The reference is a dense matrix exponential of the generator.
"""

import numpy as np
import pytest
import scipy.linalg

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing, get_indexing_extended
from slowquant.unitary_coupled_cluster.operator_state_algebra import build_operator_matrix
from slowquant.unitary_coupled_cluster.operators import G2_sa
from slowquant.unitary_coupled_cluster.spin_factorized_algebra import (
    apply_spin_block_layout,
    build_spin_block_layout,
)

NUM_ORBS = 6
# Case two only appears with one occupied orbital and case three with one virtual.
TOPOLOGY = {1: (0, 1, 2, 3), 2: (0, 0, 2, 3), 3: (0, 1, 2, 2), 4: (0, 1, 2, 3), 5: (0, 1, 2, 3)}
CASES = (1, 2, 3, 4, 5)
THETAS = (0.0, 0.29, -1.1, np.pi / 3, 2.3)


def generator(case, num_orbs=NUM_ORBS):
    """Build the spin-adapted double of one case, folded into the active space.

    Args:
        case: Which spin-adapted double to build.
        num_orbs: Number of active spatial orbitals.

    Returns:
        Excitation generator.
    """
    i, j, a, b = TOPOLOGY[case]
    return G2_sa(i, j, a, b, case, True, num_orbs=num_orbs).get_folded_operator(0, num_orbs, 0)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("theta", THETAS)
def test_matches_the_matrix_exponential(case: int, theta: float) -> None:
    """Test the blocked exponential against a dense exponential of the generator.

    Args:
        case: Which spin-adapted double to test.
        theta: Ansatz parameter value.
    """
    ci_info = get_indexing(0, NUM_ORBS, 0, 3, 3)
    op = generator(case)
    layout = build_spin_block_layout(op, ci_info)
    assert layout is not None, case
    state = np.random.default_rng(2).random(ci_info.num_dets)
    got = np.copy(state)
    apply_spin_block_layout(got.reshape(1, -1), layout, theta, ci_info)
    reference = scipy.linalg.expm(theta * build_operator_matrix(op, ci_info)) @ state
    assert np.allclose(got, reference, atol=1e-12), case


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("num_alpha, num_beta", ((3, 3), (4, 2), (2, 4)))
def test_open_shell_spaces(case: int, num_alpha: int, num_beta: int) -> None:
    """Test the blocked exponential where the two spin spaces differ in size.

    A phase error between the spin blocks shows here and not in a closed-shell space.

    Args:
        case: Which spin-adapted double to test.
        num_alpha: Number of alpha electrons.
        num_beta: Number of beta electrons.
    """
    ci_info = get_indexing(0, NUM_ORBS, 0, num_alpha, num_beta)
    op = generator(case)
    layout = build_spin_block_layout(op, ci_info)
    assert layout is not None, case
    state = np.random.default_rng(3).random(ci_info.num_dets)
    for theta in (0.37, -0.9):
        got = np.copy(state)
        apply_spin_block_layout(got.reshape(1, -1), layout, theta, ci_info)
        reference = scipy.linalg.expm(theta * build_operator_matrix(op, ci_info)) @ state
        assert np.allclose(got, reference, atol=1e-12), f"case {case} at ({num_alpha},{num_beta})"


@pytest.mark.parametrize("case", CASES)
def test_the_unitary_is_orthogonal(case: int) -> None:
    """Test that the blocked exponential preserves the norm and undoes itself.

    Args:
        case: Which spin-adapted double to test.
    """
    ci_info = get_indexing(0, NUM_ORBS, 0, 3, 3)
    layout = build_spin_block_layout(generator(case), ci_info)
    assert layout is not None, case
    state = np.random.default_rng(4).random(ci_info.num_dets)
    state /= np.linalg.norm(state)
    forward = np.copy(state)
    apply_spin_block_layout(forward.reshape(1, -1), layout, 0.83, ci_info)
    assert abs(np.linalg.norm(forward) - 1.0) < 1e-12, case
    apply_spin_block_layout(forward.reshape(1, -1), layout, -0.83, ci_info)
    assert np.allclose(forward, state, atol=1e-12), case


@pytest.mark.parametrize("case", CASES)
def test_the_layout_does_not_grow_with_the_determinant_count(case: int) -> None:
    """Test that the layout is sized by the spin strings, not by the determinants.

    This is the whole point of the per-spin form, and the reason the determinant-indexed blocks
    it replaces could not reach a large active space.

    Args:
        case: Which spin-adapted double to test.
    """
    sizes = []
    for num_orbs, num_elec in ((6, 3), (8, 4)):
        ci_info = get_indexing(0, num_orbs, 0, num_elec, num_elec)
        layout = build_spin_block_layout(generator(case, num_orbs), ci_info)
        assert layout is not None, case
        sizes.append((ci_info.num_dets, sum(item.nbytes for item in layout if hasattr(item, "nbytes"))))
    determinant_growth = sizes[1][0] / sizes[0][0]
    layout_growth = sizes[1][1] / sizes[0][1]
    assert layout_growth < determinant_growth / 2, (
        f"case {case}: layout grew {layout_growth:.1f}x while determinants grew {determinant_growth:.1f}x"
    )


def test_a_space_that_is_not_a_product_has_no_layout() -> None:
    """Test that the per-spin form is refused off a spin product."""
    ci_info = get_indexing_extended(1, NUM_ORBS, 1, 3, 3, 1)
    assert not ci_info.is_spin_product
    assert build_spin_block_layout(generator(4, ci_info.num_active_orbs), ci_info) is None
