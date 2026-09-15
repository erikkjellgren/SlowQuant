"""Test the Gram form of the reduced density matrices against the element-by-element form.

The Gram form builds one singlet excited state per pair of active orbitals and gets the whole
two-electron density matrix from their inner products. It is selected by a memory heuristic, so
the element-by-element form is reached here by overriding that heuristic.
"""

import numpy as np
import pytest

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.sa_ups_wavefunction as sa_ups_module
import slowquant.unitary_coupled_cluster.ups_wavefunction as ups_module
from slowquant.unitary_coupled_cluster.density_matrix import (
    build_rdm12_as_gram,
    can_build_rdm12_as_gram,
)
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


@pytest.mark.parametrize("molecule, basis, active_space, ansatz, ansatz_options", CASES)
def test_gram_density_matrices_match_element_by_element(
    molecule: str, basis: str, active_space: tuple, ansatz: str, ansatz_options: dict | None
) -> None:
    """Test that both forms of the density matrices agree.

    Args:
        molecule: Molecule specification.
        basis: Basis set name.
        active_space: Active space.
        ansatz: Name of ansatz.
        ansatz_options: Ansatz options.
    """
    original = ups_module.can_build_rdm12_as_gram
    try:
        ups_module.can_build_rdm12_as_gram = lambda *_: False
        reference = build_wave_function(molecule, basis, active_space, ansatz, ansatz_options)
        rdm1_reference = np.array(reference.rdm1)
        rdm2_reference = np.array(reference.rdm2)
    finally:
        ups_module.can_build_rdm12_as_gram = original
    wf = build_wave_function(molecule, basis, active_space, ansatz, ansatz_options)
    assert can_build_rdm12_as_gram(wf.num_active_orbs, len(wf.ci_coeffs))
    assert np.allclose(wf.rdm1, rdm1_reference, atol=1e-12)
    assert np.allclose(wf.rdm2, rdm2_reference, atol=1e-12)


def test_gram_density_matrices_average_over_states() -> None:
    """Test that several states are averaged with equal weight, as expectation_value_SA does."""
    wf = build_wave_function(WATER, "STO-3G", (4, 4), "fUCCSD", None)
    other = np.random.default_rng(11).random(len(wf.ci_coeffs))
    other /= np.linalg.norm(other)
    arguments = (wf.ci_info, wf.num_inactive_orbs, wf.num_active_orbs, wf.num_orbs)
    rdm1_first, rdm2_first = build_rdm12_as_gram(wf.ci_coeffs, *arguments)
    rdm1_second, rdm2_second = build_rdm12_as_gram(other, *arguments)
    rdm1_both, rdm2_both = build_rdm12_as_gram(np.vstack((wf.ci_coeffs, other)), *arguments)
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
def test_gram_density_matrices_match_for_state_averaged(
    molecule: str, basis: str, active_space: tuple, states: tuple
) -> None:
    """Test both forms of the density matrices for a state-averaged wave function.

    The states are averaged with equal weight, which is what expectation_value_SA does, so the
    element-by-element form is the reference here as well.

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

    def build():
        wf = WaveFunctionSAUPS(
            active_space,
            obj.hartree_fock.mo_coeff,
            obj,
            states,
            "tUPS",
            ansatz_options={"n_layers": 1, "skip_last_singles": True},
        )
        wf.thetas = list(0.15 * (np.random.default_rng(5).random(len(wf.thetas)) - 0.5))
        return wf

    original = sa_ups_module.can_build_rdm12_as_gram
    try:
        sa_ups_module.can_build_rdm12_as_gram = lambda *_: False
        reference = build()
        rdm1_reference = np.array(reference.rdm1)
        rdm2_reference = np.array(reference.rdm2)
    finally:
        sa_ups_module.can_build_rdm12_as_gram = original
    wf = build()
    assert len(wf.ci_coeffs) > 1
    assert np.allclose(wf.rdm1, rdm1_reference, atol=1e-12)
    assert np.allclose(wf.rdm2, rdm2_reference, atol=1e-12)
