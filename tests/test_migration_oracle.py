"""Pin quantities that must survive the spin-blocked migration unchanged.

The reference values in tests/reference_data/migration_oracle.json were generated from the
pre-migration (interleaved) code, see the _meta entry there for the commit. Every quantity is
invariant under a relabelling of spin orbitals, so this module must keep passing at every step
of the migration.

The recompute below deliberately does not go through the wave function classes, so it stays
usable while those are mid-migration. It does use the operator and CI-space API directly, and so
is expected to be updated as that API changes, but the reference JSON must never be regenerated.
"""

import json
import pathlib

import numpy as np
import pytest

import slowquant.SlowQuant as sq
from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing
from slowquant.unitary_coupled_cluster.integral_manager import IntegralManager
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    build_operator_matrix,
    expectation_value,
)
from slowquant.unitary_coupled_cluster.operators import Epq, hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.spin_ordering import det_interleaved_to_blocked

ORACLE_PATH = pathlib.Path(__file__).parent / "reference_data" / "migration_oracle.json"

SYSTEMS = {
    # name: (geometry, basis, cas(num_active_elec, num_active_orbs))
    "h2_sto3g_cas22": ("H 0.0 0.0 0.0; H 0.0 0.0 0.74", "STO-3G", (2, 2)),
    "lih_sto3g_cas22": ("Li 0.0 0.0 0.0; H 1.6717072740 0.0 0.0", "STO-3G", (2, 2)),
    "lih_sto3g_cas44": ("Li 0.0 0.0 0.0; H 1.6717072740 0.0 0.0", "STO-3G", (4, 4)),
    "h2o_sto3g_cas44": (
        "O 0.0 0.0 0.1035174918; H 0.0 0.7955612117 -0.4640237459; H 0.0 -0.7955612117 -0.4640237459",
        "STO-3G",
        (4, 4),
    ),
}


def compute_oracle_values(geometry: str, basis: str, cas: tuple[int, int]) -> dict:
    """Recompute the pinned quantities with the current code.

    Args:
        geometry: Molecular geometry in angstrom.
        basis: Name of atom-centered basis set.
        cas: CAS(num_active_elec, num_active_orbs).

    Returns:
        Computed reference quantities.
    """
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(geometry, distance_unit="angstrom")
    SQobj.set_basis_set(basis)
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    c_mo = SQobj.hartree_fock.mo_coeff

    int_gen = IntegralManager(SQobj)
    num_orbs = len(int_gen.kinetic_energy)
    num_elec = int_gen.num_elec
    num_active_elec, num_active_orbs = cas
    num_inactive_orbs = (num_elec - num_active_elec) // 2
    num_virtual_orbs = num_orbs - num_inactive_orbs - num_active_orbs
    num_active_elec_alpha = num_active_elec // 2
    num_active_elec_beta = num_active_elec // 2

    h_mo = one_electron_integral_transform(c_mo, int_gen.h_ao)
    g_mo = two_electron_integral_transform(c_mo, int_gen.electron_electron_repulsion)

    ci_info = get_indexing(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_active_elec_alpha,
        num_active_elec_beta,
    )

    H = hamiltonian_0i_0a(h_mo, g_mo, num_inactive_orbs, num_active_orbs, num_virtual_orbs)
    H_folded = H.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
    H_mat = build_operator_matrix(H_folded, ci_info)

    num_active_spin_orbs = 2 * num_active_orbs
    # Reference determinants stay human readable, i.e. interleaved, and are converted.
    hf_det = det_interleaved_to_blocked(
        "1" * num_active_elec + "0" * (num_active_spin_orbs - num_active_elec)
    )
    csf_coeffs = np.zeros(len(ci_info.idx2det))
    csf_coeffs[ci_info.det2idx[int(hf_det, 2)]] = 1

    rdm1 = np.zeros((num_active_orbs, num_active_orbs))
    for p in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            rdm1[p - num_inactive_orbs, q - num_inactive_orbs] = expectation_value(
                csf_coeffs, [Epq(p, q, num_orbs)], csf_coeffs, ci_info
            )

    return {
        "num_inactive_orbs": num_inactive_orbs,
        "num_active_orbs": num_active_orbs,
        "num_virtual_orbs": num_virtual_orbs,
        "num_active_elec_alpha": num_active_elec_alpha,
        "num_active_elec_beta": num_active_elec_beta,
        "num_det": len(ci_info.idx2det),
        "ci_eigenvalues": sorted(np.linalg.eigvalsh(H_mat).tolist()),
        "hf_energy_elec": expectation_value(csf_coeffs, [H], csf_coeffs, ci_info),
        "hf_rdm1": rdm1.tolist(),
    }


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_migration_oracle(name: str) -> None:
    """Test that invariants of the pre-migration code are reproduced.

    Args:
        name: Key of the system in SYSTEMS.
    """
    reference = json.loads(ORACLE_PATH.read_text())[name]
    computed = compute_oracle_values(*SYSTEMS[name])

    for key in (
        "num_inactive_orbs",
        "num_active_orbs",
        "num_virtual_orbs",
        "num_active_elec_alpha",
        "num_active_elec_beta",
        "num_det",
    ):
        assert computed[key] == reference[key], f"{name}: {key}"

    # The CI spectrum is basis independent, so it must be reproduced exactly.
    assert np.allclose(computed["ci_eigenvalues"], reference["ci_eigenvalues"], atol=1e-10), (
        f"{name}: ci_eigenvalues"
    )
    assert abs(computed["hf_energy_elec"] - reference["hf_energy_elec"]) < 1e-10, f"{name}: hf_energy_elec"
    assert np.allclose(computed["hf_rdm1"], reference["hf_rdm1"], atol=1e-10), f"{name}: hf_rdm1"
