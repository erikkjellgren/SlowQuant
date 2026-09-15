r"""Test operator folding against explicit full-space matrix elements.

get_folded_operator rewrites a full-space operator as an active-space one, given that the
inactive orbitals are doubly occupied and the virtual orbitals empty. This module checks that
claim directly: it builds the operator matrix over the whole orbital space, restricts it to the
determinants that have the inactive orbitals full and the virtual orbitals empty, and compares
that block to the matrix of the folded operator in the active space.

The point of doing it this way is that the reference needs no reasoning about anticommutation
signs, so it stays a valid check on the folding regardless of the spin-orbital ordering.
"""

import itertools

import numpy as np
import pytest

from slowquant.unitary_coupled_cluster.ci_spaces import det_from_spin_strings, get_indexing
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operator_state_algebra import build_operator_matrix
from slowquant.unitary_coupled_cluster.operators import (
    G3,
    G4,
    Epq,
    epqrs,
    hamiltonian_0i_0a,
    one_elec_op_0i_0a,
)

# (num_inactive_orbs, num_active_orbs, num_virtual_orbs, num_active_elec_alpha, num_active_elec_beta)
SPACES = [
    (0, 2, 0, 1, 1),  # neither inactive nor virtual, folding is a no-op
    (1, 2, 0, 1, 1),  # inactive only
    (0, 2, 1, 1, 1),  # virtual only
    (1, 2, 1, 1, 1),  # both
    (2, 2, 1, 1, 1),  # more inactive than active
    (1, 3, 1, 2, 1),  # unequal alpha and beta occupation
]


def active_det_to_full_det(
    active_det: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
) -> int:
    """Embed an active-space determinant in the full orbital space.

    The inactive orbitals are filled and the virtual orbitals left empty.

    Args:
        active_det: Determinant over the active orbitals.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.

    Returns:
        Determinant over all orbitals.
    """
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    active_mask = (1 << num_active_orbs) - 1
    inactive_string = (1 << num_inactive_orbs) - 1
    # Each spin string is inactive (filled), then active, then virtual (empty).
    alpha = (inactive_string << (num_active_orbs + num_virtual_orbs)) | (
        (active_det >> num_active_orbs) << num_virtual_orbs
    )
    beta = (inactive_string << (num_active_orbs + num_virtual_orbs)) | (
        (active_det & active_mask) << num_virtual_orbs
    )
    return det_from_spin_strings(alpha, beta, num_orbs)


def make_test_operators(
    num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int
) -> dict[str, FermionicOperator]:
    """Build operators that exercise the inactive, active and virtual blocks.

    Args:
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.

    Returns:
        Operators keyed by name.
    """
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    rng = np.random.default_rng(2024)
    h_mo = rng.random((num_orbs, num_orbs))
    h_mo = h_mo + h_mo.T
    g_mo = rng.random((num_orbs, num_orbs, num_orbs, num_orbs))

    operators = {
        "hamiltonian_0i_0a": hamiltonian_0i_0a(
            h_mo, g_mo, num_inactive_orbs, num_active_orbs, num_virtual_orbs
        ),
        "one_elec_op_0i_0a": one_elec_op_0i_0a(h_mo, num_inactive_orbs, num_active_orbs, num_virtual_orbs),
    }

    # Every one- and two-electron excitation over the whole orbital space, so that terms which
    # fold to zero (unpaired inactive operators, any virtual index) are covered as well.
    all_singles = FermionicOperator({})
    for p, q in itertools.product(range(num_orbs), repeat=2):
        all_singles += float(h_mo[p, q]) * Epq(p, q, num_orbs)
    operators["all_singles"] = all_singles

    all_doubles = FermionicOperator({})
    for p, q, r, s in itertools.product(range(num_orbs), repeat=4):
        all_doubles += float(g_mo[p, q, r, s]) * epqrs(p, q, r, s, num_orbs)
    operators["all_doubles"] = all_doubles

    return operators


@pytest.mark.parametrize("space", SPACES)
def test_folded_operator_matches_full_space_block(space: tuple[int, int, int, int, int]) -> None:
    """Test that the folded operator reproduces the full-space matrix block.

    Args:
        space: Orbital space and active electron numbers.
    """
    num_inactive_orbs, num_active_orbs, num_virtual_orbs, num_act_alpha, num_act_beta = space
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs

    # Full orbital space, at the total alpha and beta electron numbers of interest.
    full_ci_info = get_indexing(
        0,
        num_orbs,
        0,
        num_inactive_orbs + num_act_alpha,
        num_inactive_orbs + num_act_beta,
    )
    active_ci_info = get_indexing(
        num_inactive_orbs, num_active_orbs, num_virtual_orbs, num_act_alpha, num_act_beta
    )

    # Rows of the full space that have the inactive orbitals filled and the virtual ones empty,
    # ordered to match the active-space enumeration.
    block_idx = [
        full_ci_info.det2idx[
            active_det_to_full_det(det, num_inactive_orbs, num_active_orbs, num_virtual_orbs)
        ]
        for det in active_ci_info.idx2det
    ]

    for name, op in make_test_operators(num_inactive_orbs, num_active_orbs, num_virtual_orbs).items():
        full_matrix = build_operator_matrix(op, full_ci_info)
        reference = full_matrix[np.ix_(block_idx, block_idx)]
        folded = build_operator_matrix(
            op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs),
            active_ci_info,
        )
        assert np.allclose(folded, reference, atol=1e-10), f"{space} {name}"


def test_folded_high_rank_operators() -> None:
    """Test folding of operator strings longer than four.

    Linear response folds G3 to G6, and nothing else in the suite covers strings of that length,
    so the inactive-orbital bookkeeping in the fold is only exercised here.
    """
    num_inactive_orbs, num_active_orbs, num_virtual_orbs = 1, 4, 0
    num_act_alpha = num_act_beta = 2
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs

    full_ci_info = get_indexing(
        0, num_orbs, 0, num_inactive_orbs + num_act_alpha, num_inactive_orbs + num_act_beta
    )
    active_ci_info = get_indexing(
        num_inactive_orbs, num_active_orbs, num_virtual_orbs, num_act_alpha, num_act_beta
    )
    block_idx = [
        full_ci_info.det2idx[
            active_det_to_full_det(det, num_inactive_orbs, num_active_orbs, num_virtual_orbs)
        ]
        for det in active_ci_info.idx2det
    ]

    # Blocked indices for five spatial orbitals: alpha 0-4, beta 5-9. The inactive orbital is
    # spatial 0, so active alpha is 1-4 and active beta 6-9, occupied up to the second of each.
    operators = {
        "G3 active": G3(1, 2, 6, 3, 4, 8),
        "G3 active anti-hermitian": G3(1, 2, 6, 3, 4, 8, True),
        "G4 active": G4(1, 2, 6, 7, 3, 4, 8, 9),
        "G4 active anti-hermitian": G4(1, 2, 6, 7, 3, 4, 8, 9, True),
        # An unpaired inactive index has to fold away entirely.
        "G3 unpaired inactive": G3(0, 1, 6, 3, 4, 8),
        # Paired inactive operators do contribute, and their sign is the delicate part.
        "inactive number operator times G3": Epq(0, 0, num_orbs) * G3(1, 2, 6, 3, 4, 8),
        "rank 6 spanning inactive": Epq(0, 0, num_orbs) * Epq(1, 3, num_orbs) * Epq(3, 1, num_orbs),
        "rank 6 inactive excitation pair": Epq(0, 1, num_orbs) * Epq(1, 0, num_orbs) * Epq(2, 3, num_orbs),
        "rank 8 spanning inactive": epqrs(0, 0, 1, 3, num_orbs) * epqrs(3, 1, 2, 2, num_orbs),
    }

    seen_ranks: set[int] = set()
    for name, op in operators.items():
        reference = build_operator_matrix(op, full_ci_info)[np.ix_(block_idx, block_idx)]
        folded = build_operator_matrix(
            op.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs),
            active_ci_info,
        )
        assert np.allclose(folded, reference, atol=1e-10), name
        seen_ranks.update(len(key[0]) + len(key[1]) for key in op.operators)
    # The point of this test is the long strings, so make sure they are actually present.
    assert max(seen_ranks) >= 8, seen_ranks
