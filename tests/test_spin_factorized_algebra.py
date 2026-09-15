import numpy as np
import pytest

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing, get_indexing_extended
from slowquant.unitary_coupled_cluster.operator_state_algebra import build_operator_matrix
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    G3,
    Epq,
    G1_sa,
    G2_sa,
    hamiltonian_0i_0a,
)
from slowquant.unitary_coupled_cluster.spin_factorized_algebra import (
    apply_mixed_terms,
    apply_sigma3_terms,
    build_sigma3_layout,
    factorize_operator,
    propagate_state_factorized,
    split_spin_string,
    use_sigma3,
)
from slowquant.unitary_coupled_cluster.spin_ordering import alpha_idx, beta_idx

# Closed shell, open shell, and the degenerate cases where a spin space has one string.
CI_SPACES_TESTED = ((4, 2, 2), (5, 3, 2), (6, 3, 3), (6, 4, 2), (5, 2, 0), (4, 4, 2))


def random_hamiltonian(num_active_orbs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Get random one- and two-electron integrals with the right permutational symmetry.

    Args:
        num_active_orbs: Number of active spatial orbitals.
        seed: Seed of the random number generator.

    Returns:
        One- and two-electron integrals.
    """
    rng = np.random.default_rng(seed)
    h_mo = rng.random((num_active_orbs,) * 2)
    h_mo = h_mo + h_mo.T
    g_mo = rng.random((num_active_orbs,) * 4)
    g_mo = g_mo + np.transpose(g_mo, (1, 0, 3, 2))
    g_mo = g_mo + np.transpose(g_mo, (2, 3, 0, 1))
    return h_mo, g_mo


def assert_matches_unfactorized(op, ci_info, seed: int = 0) -> None:
    """Assert the factorized algebra reproduces the general kernel for one operator.

    The reference is built with build_operator_matrix, which goes through the general
    determinant kernel. Going through propagate_state instead would be no test at all, since
    that is the caller the factorized algebra is wired into.

    Args:
        op: Folded fermionic operator.
        ci_info: Information about the CI space.
        seed: Seed of the random number generator.
    """
    state = np.random.default_rng(seed).random(len(ci_info.idx2det))
    reference = build_operator_matrix(op, ci_info) @ state
    tmp_state = np.zeros_like(state)
    factorized = propagate_state_factorized(op, state, ci_info, tmp_state)
    assert factorized is not None
    assert np.allclose(factorized, reference, atol=1e-12)


def test_hamiltonian_matches_unfactorized() -> None:
    """Test that the factorized algebra reproduces the general kernel for a Hamiltonian."""
    for num_active_orbs, num_alpha, num_beta in CI_SPACES_TESTED:
        ci_info = get_indexing(0, num_active_orbs, 0, num_alpha, num_beta)
        h_mo, g_mo = random_hamiltonian(num_active_orbs, seed=num_active_orbs)
        hamiltonian = hamiltonian_0i_0a(h_mo, g_mo, 0, num_active_orbs, 0)
        assert_matches_unfactorized(hamiltonian.get_folded_operator(0, num_active_orbs, 0), ci_info)


def test_excitation_operators_match_unfactorized() -> None:
    """Test that the factorized algebra reproduces the general kernel for the ansatz generators.

    The generators are what construct_ups_state and get_grad_action apply, and unlike the
    Hamiltonian they carry sub-strings of unequal rank on the two spins.
    """
    num_active_orbs = 4
    for num_alpha, num_beta in ((2, 2), (3, 1)):
        ci_info = get_indexing(0, num_active_orbs, 0, num_alpha, num_beta)
        operators = [
            Epq(0, 2, num_active_orbs),
            Epq(1, 1, num_active_orbs),
            G1(alpha_idx(0, num_active_orbs), alpha_idx(2, num_active_orbs), True),
            G1(beta_idx(1, num_active_orbs), beta_idx(3, num_active_orbs), True),
            G2(
                alpha_idx(0, num_active_orbs),
                beta_idx(0, num_active_orbs),
                alpha_idx(2, num_active_orbs),
                beta_idx(2, num_active_orbs),
                True,
            ),
            G2(
                alpha_idx(0, num_active_orbs),
                alpha_idx(1, num_active_orbs),
                alpha_idx(2, num_active_orbs),
                alpha_idx(3, num_active_orbs),
                True,
            ),
            G3(
                alpha_idx(0, num_active_orbs),
                alpha_idx(1, num_active_orbs),
                beta_idx(0, num_active_orbs),
                alpha_idx(2, num_active_orbs),
                alpha_idx(3, num_active_orbs),
                beta_idx(2, num_active_orbs),
                True,
            ),
            G1_sa(0, 2, True, num_orbs=num_active_orbs),
            G2_sa(0, 1, 2, 3, 1, True, num_orbs=num_active_orbs),
            G2_sa(0, 1, 2, 3, 3, True, num_orbs=num_active_orbs),
            Epq(0, 2, num_active_orbs) * Epq(1, 3, num_active_orbs) * Epq(2, 0, num_active_orbs),
        ]
        for op in operators:
            assert_matches_unfactorized(op.get_folded_operator(0, num_active_orbs, 0), ci_info)


def test_repeated_application_matches_unfactorized() -> None:
    """Test that a product of operators still matches, so the cache is not stateful."""
    num_active_orbs = 4
    ci_info = get_indexing(0, num_active_orbs, 0, 2, 2)
    h_mo, g_mo = random_hamiltonian(num_active_orbs, seed=7)
    hamiltonian = hamiltonian_0i_0a(h_mo, g_mo, 0, num_active_orbs, 0).get_folded_operator(
        0, num_active_orbs, 0
    )
    state = np.random.default_rng(3).random(len(ci_info.idx2det))
    matrix = build_operator_matrix(hamiltonian, ci_info)
    for _ in range(3):
        reference = matrix @ state
        tmp_state = np.zeros_like(state)
        state = propagate_state_factorized(hamiltonian, state, ci_info, tmp_state)
        assert np.allclose(state, reference, atol=1e-12)


def test_non_spin_conserving_string_is_rejected() -> None:
    """Test that a string changing the electron count of a spin is not factorized."""
    num_active_orbs = 3
    # One alpha creation and one beta annihilation, a spin flip.
    assert split_spin_string(((0,), (num_active_orbs,)), num_active_orbs) is None
    # A bare annihilation operator.
    assert split_spin_string(((), (0,)), num_active_orbs) is None
    # Spin conserving, so it does factorize.
    assert split_spin_string(((0,), (1,)), num_active_orbs) is not None


def test_extended_space_is_not_factorized() -> None:
    """Test that the extended CI space falls back to the general algebra."""
    ci_info = get_indexing_extended(1, 2, 1, 1, 1, 1)
    assert not ci_info.is_spin_product
    h_mo, g_mo = random_hamiltonian(4, seed=1)
    hamiltonian = hamiltonian_0i_0a(h_mo, g_mo, 0, 4, 0)
    assert factorize_operator(hamiltonian, ci_info) is None


def test_sub_string_split_keeps_descending_order() -> None:
    """Test that splitting a string on spin keeps both sub-strings sorted descending.

    A fermionic string is stored sorted descending, and the excitation map builder reads that
    order as the application order, so the split must not disturb it.
    """
    num_active_orbs = 4
    op_key = ((6, 5, 1, 0), (7, 4, 3, 2))
    split = split_spin_string(op_key, num_active_orbs)
    assert split is not None
    alpha_sub, beta_sub, _ = split
    for sub_string in (alpha_sub, beta_sub):
        for block in sub_string:
            assert list(block) == sorted(block, reverse=True)


@pytest.mark.parametrize("num_active_orbs, num_alpha, num_beta", CI_SPACES_TESTED)
def test_state_norm_is_preserved_by_hermitian_operator(
    num_active_orbs: int, num_alpha: int, num_beta: int
) -> None:
    """Test that the factorized algebra gives a symmetric matrix for a Hermitian operator."""
    ci_info = get_indexing(0, num_active_orbs, 0, num_alpha, num_beta)
    h_mo, g_mo = random_hamiltonian(num_active_orbs, seed=num_active_orbs)
    hamiltonian = hamiltonian_0i_0a(h_mo, g_mo, 0, num_active_orbs, 0).get_folded_operator(
        0, num_active_orbs, 0
    )
    num_dets = len(ci_info.idx2det)
    matrix = np.zeros((num_dets, num_dets))
    for i in range(num_dets):
        unit = np.zeros(num_dets)
        unit[i] = 1.0
        column = propagate_state_factorized(hamiltonian, unit, ci_info, np.zeros(num_dets))
        assert column is not None
        matrix[:, i] = column
    assert np.allclose(matrix, matrix.T, atol=1e-12)


def test_sigma3_matches_term_by_term_mixed_application() -> None:
    """Test the dense contraction against pairing the surviving strings of each mixed term.

    The two are selected by a heuristic, so both have to be exercised deliberately. This also
    covers spaces where a spin sub-string has no surviving excitation at all, which shares an
    arena slice with the next one and so must not be mistaken for it.
    """
    for num_active_orbs, num_alpha, num_beta in (*CI_SPACES_TESTED, (6, 3, 3), (7, 4, 3)):
        ci_info = get_indexing(0, num_active_orbs, 0, num_alpha, num_beta)
        h_mo, g_mo = random_hamiltonian(num_active_orbs, seed=num_active_orbs + num_alpha)
        hamiltonian = hamiltonian_0i_0a(h_mo, g_mo, 0, num_active_orbs, 0).get_folded_operator(
            0, num_active_orbs, 0
        )
        factorized = factorize_operator(hamiltonian, ci_info)
        assert factorized is not None
        if len(factorized.mixed_factor) == 0:
            continue
        state = np.random.default_rng(num_active_orbs).random(len(ci_info.idx2det))
        term_by_term = apply_mixed_terms(
            state,
            np.zeros_like(state),
            ci_info.num_beta_strings,
            factorized.alpha_src,
            factorized.alpha_dst,
            factorized.alpha_sign,
            factorized.beta_src,
            factorized.beta_dst,
            factorized.beta_sign,
            factorized.mixed_alpha_start,
            factorized.mixed_alpha_stop,
            factorized.mixed_beta_start,
            factorized.mixed_beta_stop,
            factorized.mixed_factor,
        )
        layout = build_sigma3_layout(factorized, ci_info.num_alpha_strings)
        contracted = apply_sigma3_terms(
            state,
            np.zeros_like(state),
            ci_info.num_alpha_strings,
            ci_info.num_beta_strings,
            *layout,
        )
        assert np.allclose(contracted, term_by_term, atol=1e-12)


def test_sigma3_is_actually_selected_somewhere() -> None:
    """Test that the dense contraction is reached, so the test above is not vacuous."""
    ci_info = get_indexing(0, 6, 0, 3, 3)
    h_mo, g_mo = random_hamiltonian(6, seed=1)
    hamiltonian = hamiltonian_0i_0a(h_mo, g_mo, 0, 6, 0).get_folded_operator(0, 6, 0)
    factorized = factorize_operator(hamiltonian, ci_info)
    assert factorized is not None
    layout = build_sigma3_layout(factorized, ci_info.num_alpha_strings)
    num_pair_products = int(
        np.sum(
            (factorized.mixed_alpha_stop - factorized.mixed_alpha_start)
            * (factorized.mixed_beta_stop - factorized.mixed_beta_start)
        )
    )
    assert use_sigma3(
        ci_info.num_alpha_strings,
        ci_info.num_beta_strings,
        layout[0].shape[1],
        layout[1].shape[1],
        num_pair_products,
    )
