import numpy as np
import pyscf
from pyscf import scf, mcscf, fci
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS



def get_unrestricted_ground_state_energy(geometry, basis, active_space, unit='bohr', charge=0, spin=0):
    """
    Calculate hyperfine coupling constant (fermi-contact term) for a molecule
    """
    #PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    res = mc.kernel(mf.mo_coeff)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    
    fci.direct_uhf.make_rdm12s(mc.ci, mc.ncas, mc.nelecas)
    #mol.hartree_fock.run_unrestricted_hartree_fock()
    print(mf.energy_elec())

    #Slowquant
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

    print(WF.energy_elec, WF.energy_elec_RDM)

basis = '6-311++G**'

def OH_radical():
    """
    Test of hfc for OH using unrestricted RDMs
    """
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;"""
    active_space = ((1,2),3)
    charge = 0
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=1
    
    get_unrestricted_ground_state_energy(geometry=geometry, basis=basis, active_space=active_space, unit='angstrom', charge=charge, spin=spin)

def OH_cation():
    """
    Test of hfc for OH using unrestricted RDMs
    """
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  1.0289;"""
    active_space = ((2,2),4)
    charge = 1
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=2
    
    get_unrestricted_ground_state_energy(geometry=geometry, basis=basis, active_space=active_space, unit='angstrom', charge=charge, spin=spin)

def NO_radical():
    """
    Test of hfc for OH using unrestricted RDMs
    """
    geometry = """O  0.0   0.0  0.0;
        N  0.0  0.0  1.1508;"""
    active_space = ((1,2),3)
    charge = 0
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=1
    
    get_unrestricted_ground_state_energy(geometry=geometry, basis=basis, active_space=active_space, unit='angstrom', charge=charge, spin=spin)


#OH_radical()
NO_radical()
#OH_cation()
