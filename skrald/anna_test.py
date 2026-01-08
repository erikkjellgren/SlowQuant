import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import naive
def unrestricted(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """Calculate hyperfine coupling constant (fermi-contact term) for a molecule"""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    #X2C
    mf = scf.UHF(mol).sfx2c1e()
    mf.scf()
    mf.kernel()
    h_core=mf.get_hcore()
    # h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # # Slowquant
    # WF = UnrestrictedWaveFunctionUPS(
    #     mol.nelectron,
    #     active_space,
    #     mf.mo_coeff,
    #     h_core,
    #     g_eri,
    #     "fuccsd",
    #     {"n_layers": 2},
    #     include_active_kappa=True,
    # )
    # # print(WF.energy_elec_RDM)
    # WF.run_wf_optimization_1step("bfgs", True)
def restricted(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    #X2C
    mf = scf.HF(mol).sfx2c1e()
    mf.scf()
    mf.kernel()
    h_core=mf.get_hcore()
    # h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    mc = mcscf.CASCI(mf, active_space[1], active_space[0])
    # mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # # Slowquant
    WF =WaveFunctionUPS(
        num_elec=mol.nelectron,
        cas=active_space,
        mo_coeffs=mf.mo_coeff,
        h_ao=h_core,
        g_ao=g_eri,
        ansatz="tups",
        ansatz_options={"n_layers": 2},
        include_active_kappa=True,
    )
    print('Antal elektroner',mol.nelectron)
    WF.run_wf_optimization_1step("bfgs", True)
    LR = naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    #X2C
    mf = scf.HF(mol)
    mf.scf()
    mf.kernel()
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    mc = mcscf.CASCI(mf, active_space[1], active_space[0])
    # mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # # Slowquant
    WF =WaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccsd",
        {"n_layers": 2},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("bfgs", True)
    LR = naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "dyall-v2z"
    active_space_u = ((1, 1), 4)
    active_space = (2, 4)
    charge = 0
    spin = 0
    restricted(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # NR(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )
def h2o():
    geometry = """
    O  0.0   0.0  0.11779
    H  0.0   0.75545  -0.47116;
    H  0.0  -0.75545  -0.47116"""
    basis = "dyall-v2z"
    active_space_u = ((2, 2), 4)
    active_space = (4, 4)
    charge = 0
    spin = 0
    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )
def HI():
    geometry = """H  0.0   0.0  0.0;
        I  0.0  0.0  1.60916 """
    basis = "dyall-v2z"
    active_space = (4, 6)
    charge = 0
    spin = 0
    restricted(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # NR(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
def HBr():
    geometry = """H  0.0   0.0  0.0;
        Br  0.0  0.0  1.41443 """
    basis = "dyall-v2z"
    active_space = (4, 6)
    charge = 0
    spin = 0
    restricted(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # NR(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
###SPIN ELLER RUMLIGE ORBITALER###
# h2()
# h2o()
HI()
# HBr()