"""Test that the ansatz builders emit spin-conserving excitations in blocked ordering.

The excitation indices of UpsStructure and UccStructure are spin-orbital indices in the
alpha/beta-blocked convention, so the spin of an index is decided by which half of the register
it sits in. Every excitation the builders emit has to conserve the alpha and the beta electron
number separately. These are cheap structural checks, but they catch a builder that is still
pairing indices the interleaved way, which is otherwise only visible as a wrong energy.
"""

import pytest

from slowquant.unitary_coupled_cluster.util import UccStructure, UpsStructure

NUM_ACTIVE_ORBS = 4
OCC_IDX = [0, 1]
UNOCC_IDX = [2, 3]
# Blocked: alpha orbitals 0-3, beta orbitals 4-7.
OCC_SPIN_IDX = [0, 1, 4, 5]
UNOCC_SPIN_IDX = [2, 3, 6, 7]

UPS_ANSATZE = {
    "tUPS": lambda layout: layout.create_tiled(NUM_ACTIVE_ORBS, {"n_layers": 2, "do_tups": True}),
    "QNP": lambda layout: layout.create_tiled(NUM_ACTIVE_ORBS, {"n_layers": 1, "do_qnp": True}),
    "fUCCSD": lambda layout: layout.create_fUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["S", "D"]},
    ),
    "kSAfUpCCGSD": lambda layout: layout.create_fUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["SAGS", "GpD"]},
    ),
    "fUCC_GS_GD": lambda layout: layout.create_fUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["GS", "GD"]},
    ),
    "fUCC_TQ": lambda layout: layout.create_fUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["T", "Q"]},
    ),
    "fUCC_SAS_SAD": lambda layout: layout.create_fUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["SAS", "SAD"]},
    ),
    "SDSfUCCSD": lambda layout: layout.create_SDSfUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["D"]},
    ),
    "SDSfUCC_GpD": lambda layout: layout.create_SDSfUCC(
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
        {"n_layers": 1, "excitations": ["GpD"]},
    ),
}

# Excitation types whose indices are spatial, and which are spin adapted by construction.
SPATIAL_EXC_TYPES = (
    "sa_single",
    "sa_double_1",
    "sa_double_2",
    "sa_double_3",
    "sa_double_4",
    "sa_double_5",
)


def spin_change(exc_indices: tuple[int, ...], num_active_orbs: int) -> tuple[int, int]:
    """Get the change in alpha and beta electron number of an excitation.

    Args:
        exc_indices: Spin-orbital indices, occupied first and then unoccupied.
        num_active_orbs: Number of active spatial orbitals.

    Returns:
        Change in the number of alpha and of beta electrons.
    """
    half = len(exc_indices) // 2
    occ, unocc = exc_indices[:half], exc_indices[half:]
    num_alpha = sum(1 for a in unocc if a < num_active_orbs) - sum(1 for i in occ if i < num_active_orbs)
    num_beta = sum(1 for a in unocc if a >= num_active_orbs) - sum(1 for i in occ if i >= num_active_orbs)
    return num_alpha, num_beta


@pytest.mark.parametrize("name", sorted(UPS_ANSATZE))
def test_ups_excitations_conserve_spin(name: str) -> None:
    """Test that a UPS ansatz only emits alpha and beta conserving excitations.

    Args:
        name: Key of the ansatz in UPS_ANSATZE.
    """
    layout = UpsStructure()
    UPS_ANSATZE[name](layout)
    assert layout.n_params > 0
    assert layout.num_active_orbs == NUM_ACTIVE_ORBS
    assert len(layout.excitation_indices) == len(layout.excitation_operator_type)
    for exc_type, exc_indices in zip(layout.excitation_operator_type, layout.excitation_indices):
        if exc_type in SPATIAL_EXC_TYPES:
            for p in exc_indices:
                assert 0 <= p < NUM_ACTIVE_ORBS, f"{name} {exc_type} {exc_indices}"
            continue
        for p in exc_indices:
            assert 0 <= p < 2 * NUM_ACTIVE_ORBS, f"{name} {exc_type} {exc_indices}"
        assert spin_change(exc_indices, NUM_ACTIVE_ORBS) == (0, 0), f"{name} {exc_type} {exc_indices}"


@pytest.mark.parametrize(
    "excitations",
    [["S"], ["D"], ["S", "D"], ["SAS", "SAD"], ["T"], ["Q"], ["S", "D", "T", "Q"], ["pD", "GpD"]],
)
def test_ucc_excitations_conserve_spin(excitations: list[str]) -> None:
    """Test that a UCC ansatz only emits alpha and beta conserving excitations.

    Args:
        excitations: Unitary coupled cluster excitation orders.
    """
    layout = UccStructure()
    layout.add_excitations(
        excitations,
        OCC_IDX,
        UNOCC_IDX,
        OCC_SPIN_IDX,
        UNOCC_SPIN_IDX,
        NUM_ACTIVE_ORBS,
    )
    assert layout.n_params > 0
    assert layout.num_active_orbs == NUM_ACTIVE_ORBS
    for exc_type, exc_indices in zip(layout.excitation_operator_type, layout.excitation_indices):
        if exc_type in SPATIAL_EXC_TYPES:
            continue
        assert spin_change(exc_indices, NUM_ACTIVE_ORBS) == (0, 0), f"{excitations} {exc_type} {exc_indices}"
