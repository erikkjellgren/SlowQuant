import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group

# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive, naive
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value, generalized_propagate_state
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a
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

    print('h core',h_core, '..')
    # mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # # Slowquant

    
    WF =WaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccsd",
        {"n_layers": 0},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, test=True,tol=1e-8)
    LR = naive.LinearResponse(WF, excitations="sd")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
    
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
    
    LR = generalized_naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)

def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    # mf = scf.HF(mol)
    # mf = scf.GHF(mol)
    
    #relativistic X2C
    mf = scf.GHF(mol).sfx2c1e()
    
    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 10000

    mf.scf()
    mf.kernel()
    c=np.array(mf.mo_coeff, dtype=complex)
    print(c)
    # c= np.array([[ 0.2328374 +0.j, -0.2295067 +0.j, -0.09973118+0.j, -0.07138934+0.j,
    #                 0.54049766+0.j,  0.54323164+0.j, -0.79382251+0.j, -0.79382251+0.j],
    #                 [0.19357538+0.j, -0.19080632+0.j, -1.39445627+0.j, -0.99817651+0.j,
    #                 -0.48402791+0.j, -0.48647625+0.j,  0.95419254+0.j,  0.95419254+0.j],
    #                 [ 0.2328374 +0.j, -0.2295067 +0.j,  0.09973118+0.j,  0.07138934+0.j,
    #                 0.54049766+0.j,  0.54323164+0.j,  0.79382251+0.j,  0.79382251+0.j],
    #                 [0.19357538+0.j, -0.19080632+0.j,  1.39445627+0.j,  0.99817651+0.j,
    #                 -0.48402791+0.j, -0.48647625+0.j, -0.95419254+0.j, -0.95419254+0.j],
    #                 [ 0.2295067 +0.j,  0.2328374 +0.j,  0.07138934+0.j, -0.09973118+0.j,
    #                 0.54323164+0.j, -0.54049766+0.j, -0.79382251+0.j,  0.79382251+0.j],
    #                 [ 0.19080632+0.j,  0.19357538+0.j,  0.99817651+0.j, -1.39445627+0.j,
    #                 -0.48647625+0.j,  0.48402791+0.j,  0.95419254+0.j, -0.95419254+0.j],
    #                 [ 0.2295067 +0.j,  0.2328374 +0.j, -0.07138934+0.j,  0.09973118+0.j,
    #                 0.54323164+0.j, -0.54049766+0.j,  0.79382251+0.j, -0.79382251+0.j],
    #                 [ 0.19080632+0.j,  0.19357538+0.j, -0.99817651+0.j,  1.39445627+0.j,
    #                 -0.48647625+0.j,  0.48402791+0.j, -0.95419254+0.j,  0.95419254+0.j]])
    e_nuc=mf.energy_nuc()
    # h_core=mol.intor("int1e_kin")  + mol.intor("int1e_nuc") #non relativistic
    h_1e = mol.intor("int1e_kin")  
    h_nuc=mol.intor("int1e_nuc")



    # #relativistic integrals
    h_core=mf.get_hcore()

    g_eri = mol.intor("int2e")
    mc = mcscf.CASCI(mf, active_space[1], active_space[0])


    #make a random unitary transformation
    u = unitary_group.rvs(c.shape[0]) 
    # print(np.dot(u, u.conj().T))
    C_u = c @ u[0] 
    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    h_1e = mol.intor("int1e_kin")
    h_nuc=mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    mc = mcscf.CASCI(mf, active_space[1], active_space[0])

    # # Slowquant
    
    WF =GeneralizedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        c,
        #C_u,
        h_core,
        g_eri,
        "fUCCSD",
        {"n_layers": 0},
        include_active_kappa=True,
    )
    # WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, test=True,tol=1e-8)
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    # WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=False, tol=1e-5, maxiter = 2000)

    print("E_opt:", WF._energy_elec)
    
    
  
    LR = generalized_naive.LinearResponse(WF, excitations="sd")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)

    #call MO integrals
    g_eri_mo = WF.g_mo
    h_eri_mo=WF.h_mo
    
    num_active_spin_orbs=WF.num_active_spin_orbs
    num_inactive_spin_orbs=WF.num_inactive_spin_orbs
    num_virtual_spin_orbs=WF.num_virtual_spin_orbs
    num_spin_orbs = WF.num_spin_orbs
    
    ci_coeff = WF.ci_coeffs
    ci_info = WF.ci_info
    # print('coeff',ci_coeff)
    # print('info',ci_coeff)
    wf_struct = WF.ups_layout
    
    thetas = np.array([0, 0, 0, 0, -0.11284015184], dtype=float).tolist()
    
    # H = generalized_hamiltonian_0i_0a(h_eri_mo, g_eri_mo,
    #                                 num_inactive_spin_orbs, num_active_spin_orbs)

    H=generalized_hamiltonian_full_space(h_eri_mo, g_eri_mo, num_spin_orbs)
    # thetas = [0, 0, 0, 0, -0.11284015184]

    # # 1) THIS function applies thetas (because of "U")
    # psi = generalized_propagate_state(["U"], ci_coeff, ci_info, thetas=thetas, wf_struct=wf_struct)

    # # 2) Now measure energy
    # tester = generalized_expectation_value(psi, [H], psi, ci_info)

    # H = generalized_hamiltonian_0i_0a(
    #     WF.h_mo, WF.g_mo,
    #     WF.num_inactive_spin_orbs, WF.num_active_spin_orbs
    # )

    #    Build the REFERENCE CI state (the SCF determinant) from ci_info
    #    This finds the determinant with the largest amplitude in WF.ci_coeffs,
    #    which is the reference used by your setup.
    ref_idx = int(np.argmax(np.abs(WF.ci_coeffs)))
    ref = np.zeros_like(WF.ci_coeffs, dtype=np.complex128)
    ref[ref_idx] = 1.0

    thetas = [0, 0, 0, 0, -0.11284015184]

    # 3) Make psi(thetas) and compute energy
    psi = generalized_propagate_state(
        ["U"], ref, WF.ci_info,
        thetas=thetas,
        wf_struct=WF.ups_layout
    )
    E = generalized_expectation_value(psi, [H], psi, WF.ci_info)
    # print(E)

    
    # tester = generalized_expectation_value(
    #         ci_coeff,
    #         [generalized_hamiltonian_0i_0a(h_eri_mo,  g_eri_mo, num_inactive_spin_orbs, num_active_spin_orbs)],
    #         # [generalized_hamiltonian_full_space(h_eri_mo, g_eri_mo, num_spin_orbs)],
    #         ci_coeff,
    #         ci_info, thetas = thetas
    #         )
    
    # print('tester',tester)
    


    # print(num_active_spin_orbs)
    # rdm1=WF.rdm1
    # print(rdm1)
    # rdm2=WF.rdm2
    # print(rdm2)
    
    
    'Test of Hamiltonians'
    H=generalized_hamiltonian_full_space(h_eri_mo, g_eri_mo, c.shape[0])
    H_test=generalized_hamiltonian_0i_0a(h_eri_mo, g_eri_mo,num_inactive_spin_orbs,num_active_spin_orbs)
    test=generalized_expectation_value(WF.ci_coeffs, [H], WF.ci_coeffs, WF.ci_info)
    print(test, test+e_nuc)
    test2=generalized_expectation_value(WF.ci_coeffs, [H_test], WF.ci_coeffs, WF.ci_info)
    print(test2, test2+e_nuc)
    H_1iai=generalized_hamiltonian_1i_1a(h_eri_mo, g_eri_mo,num_inactive_spin_orbs,num_active_spin_orbs, num_virtual_spin_orbs)
    test3=generalized_expectation_value(WF.ci_coeffs, [H_1iai], WF.ci_coeffs, WF.ci_info)
    print(test3, test3+e_nuc)
    
    # # 'Test of gradients'
    # print('Expectation Value',np.round(WF.get_orbital_gradient_generalized_expvalue_real_imag,10))
    # print('From RDMs',np.round(WF.get_orbital_gradient_generalized_real_imag,10))



def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "STO-3G"
    active_space_u = ((1, 1), 4) #spin orbitaler
    # active_space = (2, 4)
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
    basis = "sto-6g"
    active_space_u = ((2, 2), 8)
    # active_space = (2, 4)
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

def HI():
    geometry = """H  0.0   0.0  0.0;
        I  0.0  0.0  1.60916 """
    basis = "STO-3g"
    # active_space = (4, 6)
    active_space = ((2,2), 6)
    charge = 0
    spin = 0

    # print("Restricted HI")
    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
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
    
def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    # basis = "cc-pvdz"
    basis = "sto-3g"
    #basis = "sto-3g"
    active_space = ((1, 2), 6)
    #active_space = (2, 4)
    charge = 0
    spin = 1

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )

def h4_rektangle():
    geometry = """H 0.0 0.0 0.0;
                  H 0.0 0.0 0.74;
                  H 0.0 1.11 0.74;
                  H 0.0 1.11 0.0;"""
    basis = "sto-3g"
    active_space = ((1,1), 4)
    charge = 0
    spin = 0
    NR(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")
  
# h3()
# h2()
h4_rektangle()
