import itertools
import math

import pytest

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing
from slowquant.unitary_coupled_cluster.operators import a_op
from slowquant.unitary_coupled_cluster.spin_ordering import (
    alpha_idx,
    beta_idx,
    blocked_to_interleaved,
    det_blocked_to_interleaved,
    det_interleaved_to_blocked,
    get_reordering_sign,
    interleaved_to_blocked,
    is_alpha,
    spatial_idx,
    spin_orb_idx,
)

NUM_ORBS_TESTED = (1, 2, 3, 4, 5)


def all_determinants(num_orbs: int):
    """Generate every occupation string of the given length.

    Args:
        num_orbs: Number of spatial orbitals.

    Returns:
        Determinant strings.
    """
    for bits in itertools.product("01", repeat=2 * num_orbs):
        yield "".join(bits)


def test_index_round_trip() -> None:
    """Test that the interleaved and blocked index maps invert each other."""
    for num_orbs in NUM_ORBS_TESTED:
        for i in range(2 * num_orbs):
            assert blocked_to_interleaved(interleaved_to_blocked(i, num_orbs), num_orbs) == i
            assert interleaved_to_blocked(blocked_to_interleaved(i, num_orbs), num_orbs) == i


def test_index_maps_are_bijections() -> None:
    """Test that the index maps permute the full set of spin orbitals."""
    for num_orbs in NUM_ORBS_TESTED:
        expected = set(range(2 * num_orbs))
        assert {interleaved_to_blocked(i, num_orbs) for i in range(2 * num_orbs)} == expected
        assert {blocked_to_interleaved(i, num_orbs) for i in range(2 * num_orbs)} == expected


def test_blocked_index_layout() -> None:
    """Test that alpha occupies the first block and beta the second."""
    for num_orbs in NUM_ORBS_TESTED:
        for p in range(num_orbs):
            assert alpha_idx(p, num_orbs) == p
            assert beta_idx(p, num_orbs) == p + num_orbs
            assert spin_orb_idx(p, "alpha", num_orbs) == alpha_idx(p, num_orbs)
            assert spin_orb_idx(p, "beta", num_orbs) == beta_idx(p, num_orbs)
            # Spatial index and spin are recoverable from the spin-orbital index.
            assert spatial_idx(alpha_idx(p, num_orbs), num_orbs) == p
            assert spatial_idx(beta_idx(p, num_orbs), num_orbs) == p
            assert is_alpha(alpha_idx(p, num_orbs), num_orbs)
            assert not is_alpha(beta_idx(p, num_orbs), num_orbs)


def test_interleaved_index_map_matches_definition() -> None:
    """Test the index map against the interleaved definition, 2p + spin."""
    for num_orbs in NUM_ORBS_TESTED:
        for p in range(num_orbs):
            assert interleaved_to_blocked(2 * p, num_orbs) == alpha_idx(p, num_orbs)
            assert interleaved_to_blocked(2 * p + 1, num_orbs) == beta_idx(p, num_orbs)


def test_determinant_round_trip() -> None:
    """Test that the determinant string converters invert each other."""
    for num_orbs in NUM_ORBS_TESTED:
        for det in all_determinants(num_orbs):
            assert det_blocked_to_interleaved(det_interleaved_to_blocked(det)) == det
            assert det_interleaved_to_blocked(det_blocked_to_interleaved(det)) == det


def test_determinant_conversion_matches_index_map() -> None:
    """Test that the determinant converter moves occupations along the index map."""
    for num_orbs in NUM_ORBS_TESTED:
        for det in all_determinants(num_orbs):
            blocked = det_interleaved_to_blocked(det)
            for i, occ in enumerate(det):
                assert blocked[interleaved_to_blocked(i, num_orbs)] == occ


def test_determinant_integer_splits_into_spin_strings() -> None:
    """Test that a blocked determinant integer splits into an alpha and a beta half.

    This underpins the alpha/beta factorization the operator-state algebra is meant to use.
    """
    for num_orbs in NUM_ORBS_TESTED:
        for det in all_determinants(num_orbs):
            blocked = det_interleaved_to_blocked(det)
            det_int = int(blocked, 2)
            assert det_int >> num_orbs == int(blocked[:num_orbs], 2)
            assert det_int & ((1 << num_orbs) - 1) == int(blocked[num_orbs:], 2)


def reordering_sign_by_inversions(det: str) -> int:
    """Get the reordering sign by explicitly counting permutation inversions.

    Independent brute-force reference for get_reordering_sign.

    Args:
        det: Determinant in interleaved ordering.

    Returns:
        Phase factor from the reordering.
    """
    num_orbs = len(det) // 2
    # Creation operators in the order they appear in the interleaved determinant,
    # labelled by the index they will have in the blocked ordering.
    blocked_order = [interleaved_to_blocked(i, num_orbs) for i, occ in enumerate(det) if occ == "1"]
    inversions = 0
    for i in range(len(blocked_order)):
        for j in range(i + 1, len(blocked_order)):
            if blocked_order[i] > blocked_order[j]:
                inversions += 1
    return (-1) ** inversions


def test_reordering_sign_matches_inversion_count() -> None:
    """Test the reordering sign against an explicit permutation parity."""
    for num_orbs in NUM_ORBS_TESTED:
        for det in all_determinants(num_orbs):
            assert get_reordering_sign(det) == reordering_sign_by_inversions(det)


def reordering_sign_pre_migration(det: str) -> int:
    """Get the reordering sign, frozen copy of the pre-migration qiskit_interface implementation.

    Kept so the relocated implementation stays pinned to the behaviour it replaced.

    Args:
        det: Determinant in interleaved ordering.

    Returns:
        Phase factor from the reordering.
    """
    sign = 1
    alphas = 0
    for i, occ in enumerate(det[::-1]):
        # Doing reverse thus alpha are the uneven
        if i % 2 == 1 and occ == "1":
            alphas += 1
        # Doing the reverse thus beta are the even
        elif i % 2 == 0 and occ == "1":
            if alphas % 2 == 1:
                sign *= -1
    return sign


def test_reordering_sign_matches_pre_migration_implementation() -> None:
    """Test that the relocated reordering sign reproduces the original one exactly."""
    for num_orbs in NUM_ORBS_TESTED:
        for det in all_determinants(num_orbs):
            assert get_reordering_sign(det) == reordering_sign_pre_migration(det)


def test_ci_space_is_a_spin_product() -> None:
    r"""Test that the CI space factorizes into an alpha and a beta string space.

    The determinant index must satisfy :math:`I = I_\alpha N_\beta + I_\beta`, and the
    determinant integer must split into the two spin strings. A future factorized
    operator-state algebra rests on both, so this pins the enumeration order of get_indexing.
    """
    for num_active_orbs in (2, 3, 4):
        for num_active_elec_alpha in range(num_active_orbs + 1):
            for num_active_elec_beta in range(num_active_orbs + 1):
                ci_info = get_indexing(0, num_active_orbs, 0, num_active_elec_alpha, num_active_elec_beta)
                num_alpha = math.comb(num_active_orbs, num_active_elec_alpha)
                num_beta = math.comb(num_active_orbs, num_active_elec_beta)
                assert len(ci_info.idx2det) == num_alpha * num_beta
                assert ci_info.num_alpha_strings == num_alpha
                assert ci_info.num_beta_strings == num_beta
                mask = (1 << num_active_orbs) - 1
                for idx, det in enumerate(ci_info.idx2det):
                    idx_alpha = ci_info.alpha_str2idx[det >> num_active_orbs]
                    idx_beta = ci_info.beta_str2idx[det & mask]
                    assert idx_alpha * ci_info.num_beta_strings + idx_beta == idx


def test_ci_space_determinants_have_correct_occupations() -> None:
    """Test that every determinant in the CI space has the requested alpha and beta occupation."""
    for num_active_orbs in (2, 3, 4):
        for num_active_elec_alpha in range(num_active_orbs + 1):
            for num_active_elec_beta in range(num_active_orbs + 1):
                ci_info = get_indexing(0, num_active_orbs, 0, num_active_elec_alpha, num_active_elec_beta)
                mask = (1 << num_active_orbs) - 1
                for det in ci_info.idx2det:
                    assert (det >> num_active_orbs).bit_count() == num_active_elec_alpha
                    assert (det & mask).bit_count() == num_active_elec_beta


def test_a_op_matches_spin_orb_idx() -> None:
    """Test that a_op's inlined index agrees with the spin_ordering definition.

    a_op writes the blocked index out instead of calling spin_orb_idx, because it runs once per
    term when a Hamiltonian is built. This pins the two together.
    """
    for num_orbs in NUM_ORBS_TESTED:
        for p in range(num_orbs):
            for spin in ("alpha", "beta"):
                for dagger in (True, False):
                    dagger_string, nondagger_string = next(iter(a_op(p, spin, dagger, num_orbs).operators))
                    # A creation operator is stored in the first tuple of the key, an
                    # annihilation operator in the second.
                    (idx,) = dagger_string if dagger else nondagger_string
                    assert idx == spin_orb_idx(p, spin, num_orbs), f"{p} {spin} {num_orbs}"
                    assert len(nondagger_string if dagger else dagger_string) == 0


def test_a_op_rejects_unknown_spin() -> None:
    """Test that a_op still validates the spin argument."""
    with pytest.raises(ValueError, match="alpha"):
        a_op(0, "up", True, 2)
