import pyscf
from pyscf import mcscf, scf
import numpy as np

from slowquant.unitary_coupled_cluster.linear_response import unrestricted_naive as unaive
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS


def test_H2_STO3g_unrestricted():
    mol = pyscf.M(atom="H 0 0 0; H 0.0  0.0  0.74", basis="sto-3g", unit="angstrom", spin=0)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    active_space = ((1, 1), 2)

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
        "fuccsd",
        {"n_layers": 1},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", True)

    thresh = 10**-8

    assert abs(WF.energy_elec_RDM - (-1.8523881735695829)) < thresh

    # Linear Response
    ULR = unaive.LinearResponseUPS(WF, excitations="SD")
    ULR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    assert abs(ULR.excitation_energies[0] - 0.60651048) < thresh
    assert abs(ULR.excitation_energies[1] - 0.9689314) < thresh
    assert abs(ULR.excitation_energies[2] - 1.62042651) < thresh

def test_H2_631g_unrestricted():
    mol = pyscf.M(atom="H 0 0 0; H 0.0  0.0  0.74", basis="6-31g", unit="angstrom", spin=0)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    active_space = ((1, 1), 2)

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
        "fuccsd",
        {"n_layers": 1},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", True)

    thresh = 10**-8

    assert abs(WF.energy_elec_RDM - (-1.8613387621457351)) < thresh

    # Linear Response
    ULR = unaive.LinearResponseUPS(WF, excitations="SD")
    ULR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    assert abs(ULR.excitation_energies[0] - 0.39432446) < thresh
    assert abs(ULR.excitation_energies[1] - 0.57441343) < thresh
    assert abs(ULR.excitation_energies[2] - 0.84842793) < thresh
    assert abs(ULR.excitation_energies[3] - 1.04317705) < thresh
    assert abs(ULR.excitation_energies[4] - 1.13948175) < thresh
    assert abs(ULR.excitation_energies[5] - 1.33548038) < thresh
    assert abs(ULR.excitation_energies[6] - 1.3659622) < thresh
    assert abs(ULR.excitation_energies[7] - 1.44184491) < thresh
    assert abs(ULR.excitation_energies[8] - 1.8311974) < thresh
    assert abs(ULR.excitation_energies[9] - 1.88481546) < thresh
    assert abs(ULR.excitation_energies[10] - 2.58127848) < thresh

def test_H4_sto3g_unrestricted():
    mol = pyscf.M(atom="H 0 0 0; H 0.0  0.0  0.74; H 0 1.11 0.74; H 0 1.11 0", basis="sto-3g", unit="angstrom", spin=0)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    active_space = ((2, 2), 4)

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
        "fuccsd",
        {"n_layers": 1},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", True)

    thresh = 10**-8

    assert abs(WF.energy_elec_RDM - (-5.226636139885884)) < thresh

    # Linear Response
    ULR = unaive.LinearResponseUPS(WF, excitations="SD")
    ULR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies (first 10)
    assert abs(ULR.excitation_energies[0] - 0.28549575) < thresh
    assert abs(ULR.excitation_energies[1] - 0.47095703) < thresh
    assert abs(ULR.excitation_energies[2] - 0.60275005) < thresh
    assert abs(ULR.excitation_energies[3] - 0.69340688) < thresh
    assert abs(ULR.excitation_energies[4] - 0.91991768) < thresh
    assert abs(ULR.excitation_energies[5] - 0.93709376) < thresh
    assert abs(ULR.excitation_energies[6] - 0.9705003) < thresh
    assert abs(ULR.excitation_energies[7] - 1.12156439) < thresh
    assert abs(ULR.excitation_energies[8] - 1.2348762) < thresh
    assert abs(ULR.excitation_energies[9] - 1.23792261) < thresh


def test_H4_631g_unrestricted():
    mol = pyscf.M(atom="H 0 0 0; H 0.0  0.0  0.74; H 0 1.11 0.74; H 0 1.11 0", basis="6-31g", unit="angstrom", spin=0)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    active_space = ((1, 1), 2)

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
        "fuccsd",
        {"n_layers": 1},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", True)

    thresh = 10**-8

    assert abs(WF.energy_elec_RDM - (-5.248326566402399)) < thresh

    # Linear Response
    ULR = unaive.LinearResponseUPS(WF, excitations="SD")
    ULR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies (first 10)
    assert abs(ULR.excitation_energies[0] - 0) < thresh
    assert abs(ULR.excitation_energies[1] - 0) < thresh
    assert abs(ULR.excitation_energies[2] - 0.123877867) < thresh
    assert abs(ULR.excitation_energies[3] - 0.227193949) < thresh
    assert abs(ULR.excitation_energies[4] - 0.373661173) < thresh
    assert abs(ULR.excitation_energies[5] - 0.459846247) < thresh
    assert abs(ULR.excitation_energies[6] - 0.538933580) < thresh
    assert abs(ULR.excitation_energies[7] - 0.571007818) < thresh
    assert abs(ULR.excitation_energies[8] - 0.573561423) < thresh
    assert abs(ULR.excitation_energies[9] - 0.664073629) < thresh



# test_H2_STO3g_unrestricted()
# test_H2_631g_unrestricted()
# test_H4_sto3g_unrestricted()
# test_H4_631g_unrestricted()
