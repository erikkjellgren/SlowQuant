"""Reference values for open-shell active spaces, taken from before the spin factorization.

These pin the observables that a phase error in the spin factorization would move. The
reference numbers were produced on b9bd343, the commit the factorized algebra was built on, and
independently reproduced on the wf_cleanup branch, which orders spin orbitals in the
interleaved convention instead. Every case has an unequal number of alpha and beta electrons,
including the degenerate ones where a spin string space holds a single string.

The parameters are fixed rather than optimised, so the values are a direct function of the
algebra and carry no dependence on an optimiser trajectory.
"""

import numpy as np
import pytest

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

MOLECULES = {
    "h2o": """O 0.0 0.0 0.1035174918; H 0.0 0.7955612117 -0.4640237459;
              H 0.0 -0.7955612117 -0.4640237459""",
    "h4": "H 0.0 0.0 0.0; H 0.0 0.0 1.0; H 0.0 0.0 2.2; H 0.0 0.0 3.6",
    "n2": "N 0.0 0.0 0.0; N 0.0 0.0 1.2",
}

# name, molecule, basis, active space, ansatz, ansatz options, seed, then the reference
# electronic energy and the squared norms of the one- and two-electron density matrices.
REFERENCES = (
    (
        "h2o_631g_openshell",
        "h2o",
        "6-31G",
        ((3, 1), 4),
        "fUCCSD",
        None,
        702,
        -84.55993259973697,
        5.812433205845293,
        26.81319508003024,
    ),
    (
        "h2o_631g_openshell_0",
        "h2o",
        "6-31G",
        ((5, 1), 6),
        "fUCCSD",
        None,
        800,
        -83.74551743074161,
        7.579331536893926,
        62.665899033925626,
    ),
    (
        "h2o_631g_openshell_1",
        "h2o",
        "6-31G",
        ((4, 2), 6),
        "tUPS",
        {"n_layers": 1},
        801,
        -84.5834021808743,
        10.000000000000007,
        92.00000000000007,
    ),
    (
        "h2o_631g_openshell_2",
        "h2o",
        "6-31G",
        ((5, 3), 6),
        "fUCCSD",
        None,
        802,
        -84.33072129160671,
        13.149527474480099,
        174.63621922690618,
    ),
    (
        "h2o_631g_openshell_3",
        "h2o",
        "6-31G",
        ((6, 2), 6),
        "fUCCSD",
        None,
        803,
        -83.86448898266478,
        11.91729433743735,
        150.67670939899767,
    ),
    (
        "h2o_631g_openshell_4",
        "h2o",
        "6-31G",
        ((2, 0), 5),
        "fUCCSD",
        None,
        804,
        -84.51615089245665,
        1.9312644943919306,
        4.000000000000003,
    ),
    (
        "h2o_empty_beta",
        "h2o",
        "STO-3G",
        ((2, 0), 3),
        "tUPS",
        {"n_layers": 1},
        401,
        -83.5642331549689,
        2.0000000000000004,
        4.000000000000001,
    ),
    (
        "h2o_full_alpha",
        "h2o",
        "STO-3G",
        ((4, 2), 4),
        "fUCCSD",
        None,
        400,
        -83.52879826038179,
        9.99572193554695,
        91.94866322656341,
    ),
    (
        "h2o_openshell_0",
        "h2o",
        "STO-3G",
        ((3, 1), 4),
        "fUCCSD",
        None,
        300,
        -83.50738808076784,
        5.889636081259699,
        27.32292767859598,
    ),
    (
        "h2o_openshell_1",
        "h2o",
        "STO-3G",
        ((3, 1), 4),
        "tUPS",
        {"n_layers": 1},
        301,
        -83.54951233529756,
        5.999998529935129,
        27.9999882969559,
    ),
    (
        "h2o_openshell_2",
        "h2o",
        "STO-3G",
        ((4, 2), 5),
        "fUCCSD",
        None,
        302,
        -83.39306062823648,
        9.538881337504248,
        84.75695160490513,
    ),
    (
        "h2o_openshell_3",
        "h2o",
        "STO-3G",
        ((4, 2), 5),
        "tUPS",
        {"n_layers": 2},
        303,
        -83.57445424634463,
        9.99999912755067,
        91.9999847960191,
    ),
    ("h2o_openshell_4", "h2o", "STO-3G", ((5, 1), 5), "fUCCSD", None, 304, -83.0491118321969, 8.0, 68.0),
    (
        "h2o_openshell_5",
        "h2o",
        "STO-3G",
        ((2, 0), 3),
        "fUCCSD",
        None,
        305,
        -83.5587565669755,
        2.0000000000000004,
        4.000000000000001,
    ),
    ("h2o_openshell_6", "h2o", "STO-3G", ((4, 0), 4), "fUCCSD", None, 306, -83.04960127287077, 4.0, 24.0),
    (
        "h2o_openshell_7",
        "h2o",
        "STO-3G",
        ((3, 3), 5),
        "fUCCSD",
        None,
        307,
        -83.76137662615822,
        10.874701625243661,
        110.20478845062746,
    ),
    (
        "h2o_openshell_8",
        "h2o",
        "STO-3G",
        ((5, 3), 6),
        "fUCCSD",
        None,
        308,
        -83.27824648598194,
        13.085474775623121,
        173.0567510886102,
    ),
    (
        "h2o_openshell_9",
        "h2o",
        "STO-3G",
        ((4, 2), 5),
        "SDSfUCCSD",
        None,
        309,
        -83.45736310271116,
        9.660416232481836,
        86.41952118544144,
    ),
    (
        "h4_631g_openshell",
        "h4",
        "6-31G",
        ((3, 1), 4),
        "fUCCSD",
        None,
        811,
        -3.8829222985379124,
        5.824335943355169,
        26.890464900887064,
    ),
    ("n2_full_alpha_66", "n2", "STO-3G", ((6, 6), 6), "fUCCSD", None, 402, -128.41223990189806, 24.0, 624.0),
    (
        "n2_openshell_62",
        "n2",
        "STO-3G",
        ((6, 2), 6),
        "fUCCSD",
        None,
        320,
        -128.73582976137388,
        11.924627587100154,
        150.79404139360247,
    ),
)


@pytest.mark.parametrize(
    "name, molecule, basis, active_space, ansatz, ansatz_options, seed, energy, rdm1_norm, rdm2_norm",
    REFERENCES,
)
def test_open_shell_reference_values(
    name: str,
    molecule: str,
    basis: str,
    active_space: tuple,
    ansatz: str,
    ansatz_options: dict | None,
    seed: int,
    energy: float,
    rdm1_norm: float,
    rdm2_norm: float,
) -> None:
    """Test that an open-shell wave function reproduces its pre-factorization values.

    Args:
        name: Name of the case.
        molecule: Molecule key.
        basis: Basis set name.
        active_space: Active space as ((num_alpha, num_beta), num_orbs).
        ansatz: Name of ansatz.
        ansatz_options: Ansatz options.
        seed: Seed of the parameter vector.
        energy: Reference electronic energy.
        rdm1_norm: Reference squared norm of the one-electron density matrix.
        rdm2_norm: Reference squared norm of the two-electron density matrix.
    """
    obj = sq.SlowQuant()
    obj.set_molecule(MOLECULES[molecule], distance_unit="angstrom")
    obj.set_basis_set(basis)
    obj.init_hartree_fock()
    obj.hartree_fock.run_restricted_hartree_fock()
    wf = WaveFunctionUPS(active_space, obj.hartree_fock.mo_coeff, obj, ansatz, ansatz_options=ansatz_options)
    rng = np.random.default_rng(seed)
    wf.thetas = list(0.2 * (rng.random(len(wf.thetas)) - 0.5))
    if len(wf.kappa):
        wf.kappa = list(0.05 * (np.random.default_rng(seed + 1).random(len(wf.kappa)) - 0.5))
    assert abs(wf.energy_elec - energy) < 1e-9
    assert abs(float(np.sum(np.asarray(wf.rdm1) ** 2)) - rdm1_norm) < 1e-9
    assert abs(float(np.sum(np.asarray(wf.rdm2) ** 2)) - rdm2_norm) < 1e-9
