import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group

# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import naive
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, hamiltonian_0i_0a, hamiltonian_1i_1a
from slowquant.unitary_coupled_cluster.operators import a_op_spin

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

    # h_core=mf.get_hcore()
    h_core=mol.intor("int1e_kin")  + mol.intor("int1e_nuc")
    h_1e = mol.intor("int1e_kin")  
    h_nuc=mol.intor("int1e_nuc")

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
    print('Antal elektroner',mol.nelectron)
    WF.run_wf_optimization_1step("bfgs", True)
    
    WF =WaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "ADAPT",
        include_active_kappa=False,
    )
    WF.do_adapt(["GS", "GD"], orbital_optimization=True)
    
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
    # mf = scf.HF(mol)
    mf = scf.GHF(mol)

    mf.scf()
    mf.kernel()
    c=mf.mo_coeff
    e_nuc=mf.energy_nuc()


    #make a random unitary transformation
    u = unitary_group.rvs(c.shape[0])
    print(np.dot(u, u.conj().T))

    C_u = c @ u[0] 

    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    h_1e = mol.intor("int1e_kin")
    h_nuc=mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    mc = mcscf.CASCI(mf, active_space[1], active_space[0])

    # mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # # Slowquant
    

    WF =GeneralizedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "None",
        {"n_layers": 2},
        include_active_kappa=True,
    )
    # WF.run_wf_optimization_1step("bfgs", orbital_optimization=True)
    # print("kappa_real:", WF.kappa_real)
    # print("kappa_imag:", WF.kappa_imag)
    # print("E_opt:", WF._energy_elec)
    # LR = naive.LinearResponse(WF, excitations="SD")
    # LR.calc_excitation_energies()
    # print(LR.excitation_energies)

    #call MO integrals
    g_eri_mo = WF.g_mo
    h_eri_mo=WF.h_mo
   
    num_active_spin_orbs=WF.num_active_spin_orbs
    num_inactive_spin_orbs=WF.num_inactive_spin_orbs
    num_virtual_spin_orbs=WF.num_virtual_spin_orbs

    # print(num_active_spin_orbs)
    # rdm1=WF.rdm1
    # print(rdm1)
    # rdm2=WF.rdm2
    # print(rdm2)
    
    H=generalized_hamiltonian_full_space(h_eri_mo, g_eri_mo, c.shape[0])
    H_test=hamiltonian_0i_0a(h_eri_mo, g_eri_mo,num_inactive_spin_orbs,num_active_spin_orbs)
    test=expectation_value(WF.ci_coeffs, [H], WF.ci_coeffs, WF.ci_info)
    # print(test, test+e_nuc)
    test2=expectation_value(WF.ci_coeffs, [H_test], WF.ci_coeffs, WF.ci_info)
    # print(test2, test2+e_nuc)
    H_1iai=hamiltonian_1i_1a(h_eri_mo, g_eri_mo,num_inactive_spin_orbs,num_active_spin_orbs, num_virtual_spin_orbs)
    test3=expectation_value(WF.ci_coeffs, [H_1iai], WF.ci_coeffs, WF.ci_info)
    
    
    # print('huhuhub',WF.get_orbital_gradient_generalized_test)
    # gradient = np.zeros(len(WF.kappa_spin_idx))
    # for idx, (M,N) in enumerate(WF.kappa_spin_idx):
    #     for P in range(WF.num_inactive_spin_orbs+WF.num_active_spin_orbs):
            
    #         e1 = expectation_value(WF.ci_coeffs, [(a_op_spin(M,True)*a_op_spin(N,False))*H], 
    #                                 WF.ci_coeffs, WF.ci_info)
                        
    #         e1 -= expectation_value(WF.ci_coeffs, [H*(a_op_spin(M,True)*a_op_spin(N,False))], 
    #                                 WF.ci_coeffs, WF.ci_info)
            
    #         gradient[idx]= e1
            
    # print('habab',gradient)
    





def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "631-g"
    active_space_u = ((1, 1), 2)
    active_space = (2, 4)
    charge = 0
    spin = 0

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )

def O2():
    geometry = """O  0.0   0.0  0.0;
        O  0.0  0.0  3"""
    basis = "STO-3G"
    active_space_u = ((1, 1), 4)
    active_space = (2, 4)
    charge = 0
    spin = 0

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    )
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

    print("Restricted HI")
    restricted(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    print("Nonrelativistic HI")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HBr():
    geometry = """H  0.0   0.0  0.0;
        Br  0.0  0.0  1.41443 """
    basis = "dyall-v2z"
    active_space = (4, 6)
    charge = 0
    spin = 0
    print("Restricted HBr")
    restricted(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    
###SPIN ELLER RUMLIGE ORBITALER###

h2()

# O2()

# h2o()

# HI()
# HBr()