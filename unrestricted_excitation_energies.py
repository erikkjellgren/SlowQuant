import pyscf
from pyscf import scf, mcscf
import slowquant.SlowQuant as sq
import io
import numpy as np
import slowquant.unitary_coupled_cluster.linear_response.unrestricted_naive as unaive
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
import slowquant.unitary_coupled_cluster.linear_response.naive as naive

def get_unrestricted_excitation_energy(geometry, basis, active_space, charge=0, spin=0, unit="bohr"):
    """
    Calculate unrestricted excitation energies
    """
    # Info for output file
    print(f'geometry: {geometry}, basis: {basis}, active space: {active_space}, charge: {charge}, spin (2S+1): {spin+1}')
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, charge=charge, spin=spin, unit=unit)
    mol.build()
    mf = scf.UHF(mol)
    # mf = scf.HF(mol) #stable run
    mf.kernel()

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0]) 
    res = mc.kernel(mf.mo_coeff)
    
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    
    # print("hej", mf.mo_coeff)
    # print(mf.mo_coeff)
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
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True)
    # WF.run_wf_optimization_1step("bfgs", True)
    
    # Do ADAPT
    # WF = UnrestrictedWaveFunctionUPS(
    #     mol.nelectron,
    #     active_space,
    #     mf.mo_coeff,
    #     h_core,
    #     g_eri,
    #     "adapt",
    #     {"n_layers":2},
    #     include_active_kappa=True,
    # )

    # WF.do_adapt(["GS", "GD"], orbital_optimization=True)

    print("Electronic energy", WF.energy_elec_RDM)
    # print("Elec energy + nuc repulsion", WF.energy_elec_RDM + mf.energy_nuc())


    # print(WF.ci_coeffs)
    ULR = unaive.LinearResponseUPS(WF, excitations="SD")
    ULR.calc_excitation_energies()
    print(f'excitation energies: {ULR.excitation_energies}')
        
    dipole_integrals = (mol.intor('int1e_r')[0,:],
                        mol.intor('int1e_r')[1,:],
                        mol.intor('int1e_r')[2,:]
                        )
    osc_strengths = ULR.get_oscillator_strength(dipole_integrals=dipole_integrals)
    print(f'oscillator strengths: {osc_strengths}')


def oh_radical(): 
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;"""
    basis = 'sto-3g'
    active_space = ((5,4),6)
    charge = 0
    spin=1

    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")

def OH_cation():
    """
    Test of hfc for OH using unrestricted RDMs
    """
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  1.0289;"""
    basis="STO-3g"
    active_space = ((3,1),4)
    charge = 1
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=2
    
    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit='angstrom')

def NO_radical():
    """
    Test of hfc for OH using unrestricted RDMs
    """
    geometry = """O  0.0   0.0  0.0;
        N  0.0  0.0  1.1508;"""
    basis = "STO-3G"
    active_space = ((1,2),3)
    charge = 0
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=1
    
    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit='angstrom')

def h2(): 
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74;"""
    basis = '6-31g'
    active_space = ((1,1),2)
    charge = 0
    spin=0

    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")

def lih():
    geometry = """ Li 0.0 0.0 0.0;
        H 0.0 0.0 1.5957;"""#1.5949
    basis = 'sto-3g'
    active_space = ((2,2),4)
    charge = 0
    spin = 0
    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")

def h4_rektangle():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74;
        H  0.0  1.11  0.74;
        H  0.0  1.11  0.0;"""
    basis = "6-31g"
    active_space = ((1,1), 2)
    charge = 0
    spin = 0
    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")

def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    # basis = "cc-pvdz"
    basis = "sto-3g"
    #basis = "sto-3g"
    active_space = ((2, 1), 3)
    #active_space = (2, 4)
    charge = 0
    spin = 1
    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")

def beH(): 
    geometry = """Be  0.0   0.0  0.0;
        H  0.0  0.0 1.3426;"""
    basis = 'sto-3g'
    active_space = ((3,2),6)
    charge = 0
    spin=1 #2S

    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")


# OH_cation()
# oh_radical()
# h2()
# h4_rektangle()
# h2_res()
# NO_radical()
# h2_ion()
# lih()
h3()
# beH()