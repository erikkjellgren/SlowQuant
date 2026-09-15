"""Test the reduced density matrices against their definition.

They are accumulated one alpha string at a time from the CI expansion, which shares its
machinery with the operator algebra, so the reference here is built the slow and obvious way
instead: one expectation value per element, straight from the definition.
"""

import numpy as np
import pytest

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.density_matrix import build_rdm12
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.operators import Epq
from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

WATER = """O 0.0 0.0 0.1035174918; H 0.0 0.7955612117 -0.4640237459;
           H 0.0 -0.7955612117 -0.4640237459"""
NITROGEN = "N 0.0 0.0 0.0; N 0.0 0.0 1.2"
LITHIUM_HYDRIDE = "Li 0.0 0.0 0.0; H 0.0 0.0 1.6"

# Closed shell and, more importantly, unequal alpha and beta occupations.
CASES = (
    (WATER, "STO-3G", (4, 4), "fUCCSD", None),
    (WATER, "STO-3G", ((3, 1), 4), "fUCCSD", None),
    (WATER, "STO-3G", ((4, 2), 5), "tUPS", {"n_layers": 1}),
    (WATER, "STO-3G", ((2, 0), 3), "fUCCSD", None),
    (WATER, "STO-3G", (6, 5), "fUCCSD", None),
    (LITHIUM_HYDRIDE, "STO-3G", (4, 6), "tUPS", {"n_layers": 2}),
    (NITROGEN, "STO-3G", (6, 6), "tUPS", {"n_layers": 1}),
    (NITROGEN, "STO-3G", ((6, 2), 6), "fUCCSD", None),
)


def build_wave_function(molecule, basis, active_space, ansatz, ansatz_options):
    """Build a wave function at a fixed, deterministic parameter vector.

    Args:
        molecule: Molecule specification.
        basis: Basis set name.
        active_space: Active space.
        ansatz: Name of ansatz.
        ansatz_options: Ansatz options.

    Returns:
        Wave function.
    """
    obj = sq.SlowQuant()
    obj.set_molecule(molecule, distance_unit="angstrom")
    obj.set_basis_set(basis)
    obj.init_hartree_fock()
    obj.hartree_fock.run_restricted_hartree_fock()
    wf = WaveFunctionUPS(active_space, obj.hartree_fock.mo_coeff, obj, ansatz, ansatz_options=ansatz_options)
    wf.thetas = list(0.15 * (np.random.default_rng(7).random(len(wf.thetas)) - 0.5))
    if len(wf.kappa):
        wf.kappa = list(0.03 * (np.random.default_rng(8).random(len(wf.kappa)) - 0.5))
    return wf


def reference_rdm12(wf):
    """Build both density matrices one element at a time, straight from the definition.

    Args:
        wf: Wave function.

    Returns:
        One- and two-electron reduced density matrices.
    """
    num_active_orbs = wf.num_active_orbs
    offset = wf.num_inactive_orbs
    rdm1 = np.zeros((num_active_orbs, num_active_orbs))
    rdm2 = np.zeros((num_active_orbs,) * 4)
    for p in range(num_active_orbs):
        for q in range(num_active_orbs):
            rdm1[p, q] = expectation_value(
                wf.ci_coeffs, [Epq(p + offset, q + offset, wf.num_orbs)], wf.ci_coeffs, wf.ci_info
            )
    for p in range(num_active_orbs):
        for q in range(num_active_orbs):
            for r in range(num_active_orbs):
                for s in range(num_active_orbs):
                    value = expectation_value(
                        wf.ci_coeffs,
                        [Epq(p + offset, q + offset, wf.num_orbs) * Epq(r + offset, s + offset, wf.num_orbs)],
                        wf.ci_coeffs,
                        wf.ci_info,
                    )
                    if q == r:
                        value -= rdm1[p, s]
                    rdm2[p, q, r, s] = value
    return rdm1, rdm2


@pytest.mark.parametrize("molecule, basis, active_space, ansatz, ansatz_options", CASES)
def test_density_matrices_match_their_definition(
    molecule: str, basis: str, active_space: tuple, ansatz: str, ansatz_options: dict | None
) -> None:
    """Test both density matrices against one expectation value per element.

    Args:
        molecule: Molecule specification.
        basis: Basis set name.
        active_space: Active space.
        ansatz: Name of ansatz.
        ansatz_options: Ansatz options.
    """
    wf = build_wave_function(molecule, basis, active_space, ansatz, ansatz_options)
    rdm1_reference, rdm2_reference = reference_rdm12(wf)
    assert np.allclose(wf.rdm1, rdm1_reference, atol=1e-12)
    assert np.allclose(wf.rdm2, rdm2_reference, atol=1e-12)


def test_density_matrices_average_over_states() -> None:
    """Test that several states are averaged with equal weight, as expectation_value_SA does."""
    wf = build_wave_function(WATER, "STO-3G", (4, 4), "fUCCSD", None)
    other = np.random.default_rng(11).random(len(wf.ci_coeffs))
    other /= np.linalg.norm(other)
    rdm1_first, rdm2_first = build_rdm12(wf.ci_coeffs, wf.ci_info)
    rdm1_second, rdm2_second = build_rdm12(other, wf.ci_info)
    rdm1_both, rdm2_both = build_rdm12(np.vstack((wf.ci_coeffs, other)), wf.ci_info)
    assert np.allclose(rdm1_both, 0.5 * (rdm1_first + rdm1_second), atol=1e-12)
    assert np.allclose(rdm2_both, 0.5 * (rdm2_first + rdm2_second), atol=1e-12)


SA_CASES = (
    (WATER, "STO-3G", (4, 4), ([[1], [2**-0.5, -(2**-0.5)]], [["11110000"], ["11100100", "11011000"]])),
    (
        WATER,
        "STO-3G",
        (6, 5),
        ([[1], [2**-0.5, -(2**-0.5)]], [["1111110000"], ["1111100100", "1111011000"]]),
    ),
    (
        NITROGEN,
        "STO-3G",
        (6, 6),
        ([[1], [2**-0.5, -(2**-0.5)]], [["111111000000"], ["111101100000", "111110010000"]]),
    ),
)


@pytest.mark.parametrize("molecule, basis, active_space, states", SA_CASES)
def test_state_averaged_density_matrices_match_their_definition(
    molecule: str, basis: str, active_space: tuple, states: tuple
) -> None:
    """Test both density matrices of a state-averaged wave function against their definition.

    Args:
        molecule: Molecule specification.
        basis: Basis set name.
        active_space: Active space.
        states: State-averaging specification.
    """
    obj = sq.SlowQuant()
    obj.set_molecule(molecule, distance_unit="angstrom")
    obj.set_basis_set(basis)
    obj.init_hartree_fock()
    obj.hartree_fock.run_restricted_hartree_fock()
    wf = WaveFunctionSAUPS(
        active_space,
        obj.hartree_fock.mo_coeff,
        obj,
        states,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
    )
    wf.thetas = list(0.15 * (np.random.default_rng(5).random(len(wf.thetas)) - 0.5))
    assert len(wf.ci_coeffs) > 1
    # Averaged over the states with equal weight, which is what expectation_value_SA does.
    rdm1 = np.mean([build_rdm12(state, wf.ci_info)[0] for state in wf.ci_coeffs], axis=0)
    rdm2 = np.mean([build_rdm12(state, wf.ci_info)[1] for state in wf.ci_coeffs], axis=0)
    assert np.allclose(wf.rdm1, rdm1, atol=1e-12)
    assert np.allclose(wf.rdm2, rdm2, atol=1e-12)
