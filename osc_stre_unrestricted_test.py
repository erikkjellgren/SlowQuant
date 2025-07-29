import pyscf
from pyscf import scf, mcscf

import slowquant.unitary_coupled_cluster.linear_response.unrestricted_naive as unaive
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS

def get_unrestricted_excitation_energy(geometry, basis, active_space, charge=0, spin=0, unit="bohr"):
    """
    Calculate unrestricted excitation energies
    """
    # PySCF

    mol = pyscf.M(atom=geometry, basis=basis, charge=charge, spin=spin, unit=unit)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    mc = mcscf.UCASCI(mf, 4, (2,2))
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
        {"n_layers":1},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_1step("SLSQP", True)

    print("Energy elec", WF.energy_elec_RDM)

    ULR = unaive.LinearResponseUPS(WF, excitations="SD")
    ULR.calc_excitation_energies()
    ULR.get_oscillator_strength()
    print(ULR.excitation_energies)

def test_exc_energy():
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;"""
    basis = 'STO-3G'
    active_space = ((2,1),3)
    charge = 0
    spin=1

    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")

def excita_h2o():
    geometry = """O -0.000000   -0.000000    0.112729
                H -0.000000    0.794937   -0.450915
                H -0.000000   -0.794937   -0.450915"""

    basis = "STO-3G"
    active_space = ((2,2),4)

    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, unit="angstrom")

#test_exc_energy()
excita_h2o()