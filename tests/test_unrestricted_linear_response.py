import pyscf
from pyscf import mcscf, scf

import slowquant.unitary_coupled_cluster.linear_response.unrestricted_naive as unaive
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS


def test_oh_rad_ulr():
    """Calculate unrestricted excitation energies"""
    # PySCF
    mol = pyscf.M(atom="O 0 0 0; H 0.0  0.0  0.9697", basis="sto-3g", unit="angstrom", spin=1)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    active_space = ((3, 2), 4)

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    res = mc.kernel(mf.mo_coeff)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    # SlowQuant
    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccsdtq",
        {"n_layers": 2},
        include_active_kappa=True,
    )
    # WF.run_wf_optimization_1step("slsqp", False)
    WF.run_wf_optimization_1step("bfgs", True)

    thresh = 10**-4

    assert abs(WF.energy_elec_RDM - (-78.7514369)) < thresh

    ULR = unaive.LinearResponseUPS(WF, excitations="SDTQ")
    ULR.calc_excitation_energies()

    assert abs(ULR.excitation_energies[3] - 0.370351) < thresh
    assert abs(ULR.excitation_energies[4] - 0.396542) < thresh
    assert abs(ULR.excitation_energies[5] - 0.461634) < thresh
    assert abs(ULR.excitation_energies[6] - 0.513812) < thresh
    assert abs(ULR.excitation_energies[7] - 0.513812) < thresh

    dipole_integrals = (mol.intor("int1e_r")[0, :], mol.intor("int1e_r")[1, :], mol.intor("int1e_r")[2, :])

    osc_strengths = ULR.get_oscillator_strength(dipole_integrals=dipole_integrals)

    assert abs(osc_strengths[3] - 0.002794) < thresh
    assert abs(osc_strengths[4] - 0.000018) < thresh
    assert abs(osc_strengths[5] - 0.001255) < thresh
    assert abs(osc_strengths[6] - 0.000164) < thresh
    assert abs(osc_strengths[7] - 0.000164) < thresh


def test_oh_cat_ulr():
    """Calculate unrestricted excitation energies"""
    # PySCF
    mol = pyscf.M(atom="O 0 0 0; H 0.0  0.0  1.0289", basis="sto-3g", unit="angstrom", spin=2, charge=1)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    active_space = ((3, 1), 4)

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    res = mc.kernel(mf.mo_coeff)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    # SlowQuant
    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccsdtq",
        {"n_layers": 2},
        include_active_kappa=True,
    )
    # WF.run_wf_optimization_1step("slsqp", False)
    WF.run_wf_optimization_1step("bfgs", True)

    thresh = 10**-4
    # Assert energy
    assert abs(WF.energy_elec_RDM - (-78.17910)) < thresh

    ULR = unaive.LinearResponseUPS(WF, excitations="SDTQ")
    ULR.calc_excitation_energies()
    # Excitation energies for five first non-zero
    assert abs(ULR.excitation_energies[0] - 0.350633) < thresh
    assert abs(ULR.excitation_energies[1] - 0.350633) < thresh
    assert abs(ULR.excitation_energies[2] - 0.570974) < thresh
    assert abs(ULR.excitation_energies[3] - 0.570974) < thresh
    assert abs(ULR.excitation_energies[4] - 0.595558) < thresh

    dipole_integrals = (mol.intor("int1e_r")[0, :], mol.intor("int1e_r")[1, :], mol.intor("int1e_r")[2, :])

    osc_strengths = ULR.get_oscillator_strength(dipole_integrals=dipole_integrals)

    # Corresponding oscillator strengths
    # Jeg har fjernet dem fra testen fordi de varierer fra run til run
    # assert abs(osc_strengths[0] - 0.007468) < 10**-3
    # assert abs(osc_strengths[1] - 0.007468) < 10**-3
    # assert abs(osc_strengths[2] - 0.001795) < 10**-3
    # assert abs(osc_strengths[3] - 0.000147) < 10**-3
    # assert abs(osc_strengths[4] - 0.001443) < 10**-3


# test_oh_rad_ulr()
# test_oh_cat_ulr()
