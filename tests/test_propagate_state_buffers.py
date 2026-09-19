"""Test that propagating a state leaves the caller's array alone.

propagate_state swaps its two buffers rather than copying them, so the one thing that could
go wrong is handing the caller's array to a kernel to write into. That would corrupt a state
the caller still holds, silently and far from here, so it is checked directly.
"""

import numpy as np
import pytest

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing, get_indexing_extended
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    propagate_state,
    propagate_state_SA,
)
from slowquant.unitary_coupled_cluster.operators import Epq, G2_sa, hamiltonian_0i_0a

NUM_ACTIVE_ORBS = 6
# Closed shell and unequal alpha/beta, since the two take different paths through the algebra.
OCCUPATIONS = ((3, 3), (4, 2))


def random_hamiltonian(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw random integrals with the right permutational symmetry.

    Args:
        seed: Seed of the random number generator.

    Returns:
        One- and two-electron integrals.
    """
    rng = np.random.default_rng(seed)
    h_mo = rng.random((NUM_ACTIVE_ORBS,) * 2)
    h_mo = h_mo + h_mo.T
    g_mo = rng.random((NUM_ACTIVE_ORBS,) * 4)
    g_mo = g_mo + np.transpose(g_mo, (1, 0, 3, 2))
    return h_mo, g_mo + np.transpose(g_mo, (2, 3, 0, 1))


def operator_lists() -> list[list]:
    """Get operator lists covering one, several and no operators.

    Returns:
        Lists of operators to propagate with.
    """
    h_mo, g_mo = random_hamiltonian(0)
    hamiltonian = hamiltonian_0i_0a(h_mo, g_mo, 0, NUM_ACTIVE_ORBS, 0)
    excitation = Epq(0, 2, NUM_ACTIVE_ORBS)
    spin_adapted = G2_sa(0, 1, 2, 3, 3, True, num_orbs=NUM_ACTIVE_ORBS)
    return [[], [hamiltonian], [excitation, excitation], [hamiltonian, excitation, spin_adapted]]


@pytest.mark.parametrize("num_alpha, num_beta", OCCUPATIONS)
def test_propagating_leaves_the_input_state_alone(num_alpha: int, num_beta: int) -> None:
    """Test that the input is unchanged and the result does not share memory with it.

    Args:
        num_alpha: Number of active alpha electrons.
        num_beta: Number of active beta electrons.
    """
    ci_info = get_indexing(0, NUM_ACTIVE_ORBS, 0, num_alpha, num_beta)
    for operators in operator_lists():
        state = np.random.default_rng(1).random(len(ci_info.idx2det))
        untouched = state.copy()
        new_state = propagate_state(operators, state, ci_info)
        assert np.array_equal(state, untouched)
        assert not np.shares_memory(new_state, state)


@pytest.mark.parametrize("num_alpha, num_beta", OCCUPATIONS)
def test_state_averaged_propagating_leaves_the_input_states_alone(num_alpha: int, num_beta: int) -> None:
    """Test the same for the state-averaged path.

    Args:
        num_alpha: Number of active alpha electrons.
        num_beta: Number of active beta electrons.
    """
    ci_info = get_indexing(0, NUM_ACTIVE_ORBS, 0, num_alpha, num_beta)
    for operators in operator_lists():
        states = np.random.default_rng(2).random((3, len(ci_info.idx2det)))
        untouched = states.copy()
        new_states = propagate_state_SA(operators, states, ci_info)
        assert np.array_equal(states, untouched)
        assert not np.shares_memory(new_states, states)


def test_general_kernel_leaves_the_input_state_alone() -> None:
    """Test the same on a CI space that is not a spin product, which cannot be factorized."""
    ci_info = get_indexing_extended(1, 2, 1, 1, 1, 1)
    assert not ci_info.is_spin_product
    state = np.random.default_rng(3).random(len(ci_info.idx2det))
    untouched = state.copy()
    new_state = propagate_state([Epq(0, 1, 4)], state, ci_info, do_unsafe=True)
    assert np.array_equal(state, untouched)
    assert not np.shares_memory(new_state, state)


@pytest.mark.parametrize("num_alpha, num_beta", OCCUPATIONS)
def test_operators_are_applied_one_after_another(num_alpha: int, num_beta: int) -> None:
    """Test that a list of operators gives the same as applying them one at a time.

    Swapping buffers is only correct if each operator reads what the previous one wrote, so
    this pins the chaining that the swap has to preserve.

    Args:
        num_alpha: Number of active alpha electrons.
        num_beta: Number of active beta electrons.
    """
    ci_info = get_indexing(0, NUM_ACTIVE_ORBS, 0, num_alpha, num_beta)
    h_mo, g_mo = random_hamiltonian(4)
    hamiltonian = hamiltonian_0i_0a(h_mo, g_mo, 0, NUM_ACTIVE_ORBS, 0)
    excitation = Epq(0, 2, NUM_ACTIVE_ORBS)
    state = np.random.default_rng(5).random(len(ci_info.idx2det))
    # propagate_state applies its operators right to left, as they act on a ket.
    one_at_a_time = propagate_state([hamiltonian], propagate_state([excitation], state, ci_info), ci_info)
    together = propagate_state([hamiltonian, excitation], state, ci_info)
    assert np.allclose(together, one_at_a_time, atol=1e-12)
