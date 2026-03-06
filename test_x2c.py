import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from scipy.linalg import solve
from pyscf.x2c import sfx2c1e
# from pyscf.x2c.x2c import dip_moment

# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive, naive
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value, generalized_propagate_state
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a, generalized_one_elec_op_0i_0a
from slowquant.unitary_coupled_cluster.operators import a_op_spin
from slowquant.molecularintegrals.integralfunctions import generalized_one_electron_transform

def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    # mf = scf.HF(mol)
    # mf = scf.DHF(mol).x2c1e() #Use GHF!
     
    #relativistic X2C
    # x2c = sfx2c1e.SpinFreeX2CHelper(mol)

    # mf = scf.GHF(mol).sfx2c1e() #spinfree
    # mf = scf.GHF(mol).x2c()
    mf = scf.GHF(mol).x2c
    
    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 50000

    mf.scf()
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)
    print(coeff)
    
    
    
    dip_ao_picture_changed = mf.with_x2c.picture_change(('int1e_r_spinor',
                                           'int1e_sprsp_spinor'))
    # dip_mom = mf.dip_moment(picture_change = True) #with picture change
    # print(dip_mom)

    # print(dip_mom)
    #     dip_ao = mol.intor('int1e_r')
    # dip_ao = mol.intor('int1e_r')
    
    # print(dip_ao)
    
    # dip_ao = dip_ao_picture_changed

 
    "Relativistic integrals"
    h_core=mf.get_hcore()
    g_eri = mol.intor("int2e")
    # # g_eri= mol.intor("int2e_spinor")
    # print('Relativistic her',h_core_rel)
    # print(g_eri)

    # mc = mcscf.CASCI(mf, active_space[1], active_space[0])

    #make a random unitary transformation
    # u = unitary_group.rvs(c.shape[0]) 
    # print(np.dot(u, u.conj().T))
    # C_u = c @ u[0] 
    # mc = mcscf.CASCI(mf, active_space[1], active_space[0])
    
    # # Slowquant
    
    WF =GeneralizedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        coeff,
        #C_u,
        h_core,
        g_eri,
        "fUCCSD",
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    # print(WF.c_mo)

    # WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=False, tol=1e-5, maxiter = 2000)

    print("E_opt:", WF._energy_elec)
    # print("E_opt: (+nuc!)", WF._energy_elec + e_nuc)
    
  
    "Calculate Excitation energies"
    LR = generalized_naive.LinearResponse(WF, excitations="sd")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
    
    
    # "Calculate polarizability"
    prop_grad = LR.get_property_gradient(dip_ao)
    response = solve(LR.hessian, prop_grad)
    alpha = np.einsum('ix,ix->x', prop_grad, response)

    print(f'Polarizabilities:\n \t xx: {alpha[0]:.4f} \t yy: {alpha[1]:.4f} \t zz: {alpha[2]:.4f}')

    "Calculate dipole moments"
    mux = generalized_one_electron_transform(WF.c_mo, dip_ao[0])
    muy = generalized_one_electron_transform(WF.c_mo, dip_ao[1])
    muz = generalized_one_electron_transform(WF.c_mo, dip_ao[2])
    mu_op_x = generalized_one_elec_op_0i_0a(mux, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    mu_op_y = generalized_one_elec_op_0i_0a(muy, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    mu_op_z = generalized_one_elec_op_0i_0a(muz, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    dip_x=generalized_expectation_value(WF.ci_coeffs, [mu_op_x], WF.ci_coeffs, WF.ci_info)
    dip_y=generalized_expectation_value(WF.ci_coeffs, [mu_op_y], WF.ci_coeffs, WF.ci_info)
    dip_z=generalized_expectation_value(WF.ci_coeffs, [mu_op_z], WF.ci_coeffs, WF.ci_info)

    print(f'Electric Dipolemoments:\n \t xx: {dip_x:.4f} \t yy: {dip_y:.4f} \t zz: {dip_z:.4f}')




    #call MO integrals
    g_eri_mo = WF.g_mo
    h_eri_mo=WF.h_mo
    
    num_active_spin_orbs=WF.num_active_spin_orbs
    num_inactive_spin_orbs=WF.num_inactive_spin_orbs
    num_virtual_spin_orbs=WF.num_virtual_spin_orbs
    num_spin_orbs = WF.num_spin_orbs
    
    ci_coeff = WF.ci_coeffs
    mo_coeff = WF.c_mo

    ci_info = WF.ci_info
    # print('coeff',ci_coeff)
    # print('info',ci_coeff)
    wf_struct = WF.ups_layout
    
    # print('heyheyhey',WF.c_mo)
        
    thetas = np.array([0, 0, 0, 0, -0.11284015184], dtype=float).tolist()
    
     
    
    'Test of Hamiltonians'
    # H=generalized_hamiltonian_full_space(h_eri_mo, g_eri_mo, c.shape[0])
    # H_test=generalized_hamiltonian_0i_0a(h_eri_mo, g_eri_mo,num_inactive_spin_orbs,num_active_spin_orbs)
    # test=generalized_expectation_value(WF.ci_coeffs, [H], WF.ci_coeffs, WF.ci_info)
    # print(test, test+e_nuc)
    # test2=generalized_expectation_value(WF.ci_coeffs, [H_test], WF.ci_coeffs, WF.ci_info)
    # print(test2, test2+e_nuc)
    # H_1iai=generalized_hamiltonian_1i_1a(h_eri_mo, g_eri_mo,num_inactive_spin_orbs,num_active_spin_orbs, num_virtual_spin_orbs)
    # test3=generalized_expectation_value(WF.ci_coeffs, [H_1iai], WF.ci_coeffs, WF.ci_info)
    # print(test3, test3+e_nuc)
    
    # # 'Test of gradients'
    # print('Expectation Value',np.round(WF.get_orbital_gradient_generalized_expvalue_real_imag,10))
    # print('From RDMs',np.round(WF.get_orbital_gradient_generalized_real_imag,10))



def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "sto-3g"
    active_space = ((1, 1), 4) #spin orbitaler or spinor basis
    # active_space = (2, 4)
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
    # basis = "sto-3g"
    basis = {'H':'sto-3g','I': 'dyall_dz'}
    # active_space = (4, 6)
    active_space = ((2,2), 6)
    charge = 0
    spin = 0

    # print("Restricted HI")
    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HBr():
    geometry = """H  0.0   0.0  0.0;
        Br  0.0  0.0  1.41443 """
    basis = {'H':'sto-3g','Br': 'dyalldz'}
    # basis = ''
    active_space = ((2,2), 6)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    
def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    # basis = "cc-pvdz"
    basis = "sto-3g"
    #active_space = ((2, 1), 6)
    active_space = ((2,1), 6)
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
    basis = "STO-3g"
    active_space = ((2,2), 8)
    charge = 0
    spin = 0
    NR(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")


def oh_radical(): 
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;"""
    basis = 'sto-3g'
    active_space = ((2,1),6)
    charge = 0
    spin=1
    NR(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")

def BeH(): 
    geometry = """Be  0.0   0.0  0.0;
        H  0.0  0.0  1.3426;"""
    basis = 'sto-3g'
    active_space = ((3,2),12)
    charge = 0
    spin=1
    NR(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")


# h3()
h2()
# h4_rektangle()
# HI()
# HBr()
# oh_radical()
# BeH()


