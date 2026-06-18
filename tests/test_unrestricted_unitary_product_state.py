import numpy as np
from pyscf import gto, scf

from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import (
    UnrestrictedWaveFunctionUPS,
)


def test_ups_oh() -> None:
    """Test unrestricted calculation for OH radical."""
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1",
        basis="STO3G",
        spin=1,
        charge=0,
    )

    mf = scf.UHF(mol)
    mf.kernel()

    WF = UnrestrictedWaveFunctionUPS(
        ((2, 1), 3),
        mf.mo_coeff,
        mol,
        ansatz="fuccsdt",
    )
    WF.run_wf_optimization_1step("SLSQP", True)
    assert abs(WF.energy_elec - -78.59886958623208) < 10**-8


def test_utups_oh_with_oo() -> None:
    """Test unrestricted calculation for OH radical. Using utUPS and orbital optimization"""
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1",
        basis="STO3G",
        spin=1,
        charge=0,
    )

    mf = scf.UHF(mol)
    mf.kernel()

    WF = UnrestrictedWaveFunctionUPS(
        ((2, 1), 3),
        mf.mo_coeff,
        mol,
        ansatz="utups",
        ansatz_options={"n_layers": 2},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("SLSQP", True)
    assert abs(WF.energy_elec - -78.2342622356012) < 10**-8
    assert abs(WF.energy_elec_RDM - -78.2342622356012) < 10**-8


def test_utups_oh_without_oo() -> None:
    """Test unrestricted calculation for OH radical. Using utUPS and no orbital optimization"""
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1",
        basis="STO3G",
        spin=1,
        charge=0,
    )

    mf = scf.UHF(mol)
    mf.kernel()

    WF = UnrestrictedWaveFunctionUPS(
        ((2, 1), 3),
        mf.mo_coeff,
        mol,
        ansatz="utups",
        ansatz_options={"n_layers": 2},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("SLSQP", False)
    assert abs(WF.energy_elec - -78.19082220913847) < 10**-8
    assert abs(WF.energy_elec_RDM - -78.19082220913847) < 10**-8


def test_hfc_oh() -> None:
    """Test calculation of HFC for OH, with utUPS"""
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1",
        basis="STO3G",
        spin=1,
        charge=0,
    )

    mf = scf.UHF(mol)
    mf.kernel()

    WF = UnrestrictedWaveFunctionUPS(
        ((2, 1), 3),
        mf.mo_coeff,
        mol,
        ansatz="utups",
        ansatz_options={"n_layers": 2},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_1step("SLSQP", True)

    def nuclear_g_factor(atom):
        if atom == "H":
            g_k = 2.79284734 * (2 / 1)
        elif atom == "O":
            g_k = -1.89379 * (2 / 5)
        else:
            raise ValueError(f"No nuclear g-value is found for atom: {atom}")
        return g_k

    a_iso = []
    spin = 1
    for atom in mol._atom:
        amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
        mo_basis_a = amp_basis @ WF.c_a_mo
        mo_basis_b = amp_basis @ WF.c_b_mo
        h1mo_a = np.outer(np.conj(mo_basis_a), mo_basis_a)[
            : WF.num_inactive_orbs + WF.num_active_orbs, : WF.num_inactive_orbs + WF.num_active_orbs
        ]
        h1mo_b = np.outer(np.conj(mo_basis_b), mo_basis_b)[
            : WF.num_inactive_orbs + WF.num_active_orbs, : WF.num_inactive_orbs + WF.num_active_orbs
        ]
        rdma = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
        rdmb = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
        rdma[WF.num_inactive_orbs :, WF.num_inactive_orbs :] = WF.rdm1aa
        rdmb[WF.num_inactive_orbs :, WF.num_inactive_orbs :] = WF.rdm1bb
        hfc = np.trace(h1mo_a @ rdma - h1mo_b @ rdmb)

        m = spin * (1 / 2)
        f_k = 400.1186763101158
        g_k = nuclear_g_factor(atom=atom[0])

        a_iso.append(f_k * g_k / m * hfc)

    assert abs(a_iso[0] - -368.2159748785761) < 10**-8
    assert abs(a_iso[1] - 1766.1752889356255) < 10**-8
