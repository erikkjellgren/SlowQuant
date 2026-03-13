import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from scipy.linalg import solve
from pyscf.x2c import sfx2c1e
from pyscf import cc

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
    # mf = scf.GHF(mol).x2c

    
    mf = scf.GHF(mol)

    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 50000

    mf.scf()
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)
    # print(np.round(np.array(mf.mo_coeff),3))

    
    
    # dip_ao_picture_changed = mf.with_x2c.picture_change(('int1e_r_spinor',
    #                                        'int1e_sprsp_spinor'))
    # dip_mom = mf.dip_moment() #with picture change DEBYE!
    # print(dip_mom/2.541746)

    # print(dip_mom)
    #     dip_ao = mol.intor('int1e_r')
    dip_ao = mol.intor('int1e_r')
    
    efg_ints =  mol.intor("int1e_ipiprinv") + mol.intor("int1e_ipiprinv").transpose(0,2,1) + 2*mol.intor("int1e_iprinvip")

    # print(dip_ao)
    
    # dip_ao = dip_ao_picture_changed


    # e_nuc=mf.energy_nuc()
    "Non-relativistic integrals"
    h_1e = mol.intor("int1e_kin")  
    h_nuc=mol.intor("int1e_nuc")
    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    # print('Non-relativistic her',h_core)


 
    "Relativistic integrals"
    # h_core=mf.get_hcore()
    # g_eri = mol.intor("int2e")
    # # g_eri= mol.intor("int2e_spinor")
    # print('Relativistic her',h_core_rel)
    # print(g_eri)

    # mc = mcscf.CASCI(mf, active_space[1], active_space[0])

    #make a random unitary transformation
    # u = unitary_group.rvs(c.shape[0]) 
    # print(np.dot(u, u.conj().T))
    # C_u = c @ u[0] 
    # mc = mcscf.CASCI(mf, active_space[1], active_space[0])
    


#     #OH-
#     a_coeff = np.array([[ 9.94283934e-01, -2.47147351e-01,  8.84345292e-02, -1.04931221e-16,
#    4.98504451e-17,  8.87761432e-02],
#  [ 2.46612046e-02,  9.23518351e-01, -4.60553400e-01,  5.63817654e-16,
#   -3.29178647e-16, -6.02389631e-01],
#  [ 6.41847722e-20,  3.70605100e-18, -6.64972350e-16, -9.54228556e-02,
#    9.95436828e-01,  3.48320571e-18],
#  [-2.40347744e-19,  3.37265339e-17,  1.12383261e-15,  9.95436828e-01,
#    9.54228556e-02,  3.50915198e-18],
#  [ 3.78067107e-03,  1.10921733e-01,  6.99509652e-01, -8.07764775e-16,
#    2.20170396e-16, -8.60048674e-01],
#  [-6.73367151e-03,  1.66893530e-01,  4.98356847e-01, -6.64122021e-16,
#    5.92613678e-16,  1.14798745e+00]])

#     b_coeff = np.array([[ 9.94838460e-01, -2.35098607e-01,  1.06423085e-01,  1.11493778e-18,
#     1.41895386e-17,  9.53070591e-02],
#     [ 2.23637389e-02, 8.64871528e-01, -5.39424304e-01, -3.10852643e-18,
#     -9.11874175e-17, -6.23695073e-01],
#     [ 3.34318062e-19, 3.05002044e-18,  1.35789227e-16,  9.95436828e-01,
#     -9.54228556e-02,  1.35923643e-17],
#     [-1.84716405e-19, -4.10677643e-17, -1.36337584e-16,  9.54228556e-02,
#     9.95436828e-01, -7.13968485e-17],
#     [ 3.50286689e-03,  1.15481521e-01,  6.60661375e-01, -1.24566491e-16,
#     -6.15518707e-17, -8.89659314e-01],
#     [-6.15012305e-03, 2.49307848e-01,  5.26233905e-01, -8.52475019e-18,
#     1.61881459e-16,  1.12027638e+00]]
#     )

#     coeff = np.zeros((2*len(a_coeff),2*len(a_coeff)))
#     coeff[:len(a_coeff), :len(a_coeff)] = a_coeff
#     coeff[len(a_coeff):,len(a_coeff):] = b_coeff

    #H3 STO-3G ((2,1),6)
#     a_coeff = np.array([[ 3.74625869e-01, -5.98253789e-01,  9.96507453e-01],
#  [ 3.74627020e-01, -5.98267504e-01, -9.96498786e-01],
#  [ 4.75551085e-01,  1.12478016e+00, -7.46562486e-06]], dtype=complex)
    
#     b_coeff = np.array([[ 4.86934286e-01,  9.96504863e-01, -5.11030627e-01],
#  [ 4.86929093e-01, -9.96501377e-01, -5.11042372e-01],
#  [ 2.40466742e-01, -7.68214821e-06,  1.19726981e+00]], dtype=complex)

#     coeff = np.zeros((2*len(a_coeff),2*len(a_coeff)))
#     coeff[:len(a_coeff), :len(a_coeff)] = a_coeff
#     coeff[len(a_coeff):,len(a_coeff):] = b_coeff

    # coeff[0::2, :len(a_coeff)] = a_coeff.T   # even rows: columns of a in left block
    # coeff[1::2, len(a_coeff):] = b_coeff.T   # odd rows: columns of b in right block

    # print(coeff)

    # coeff[:len(a_coeff), 0::2] = a_coeff
    # coeff[len(a_coeff):, 1::2] = b_coeff

    # n = len(a_coeff)
    # coeff = np.zeros((2*n, 2*n), dtype=complex)
    # coeff[:n, 0::2] = a_coeff   # alpha AOs, even columns = alpha MOs
    # coeff[n:, 1::2] = b_coeff   # beta AOs, odd columns = beta MOs

    # # Slowquant
    
    WF =GeneralizedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        coeff,
        #C_u,
        h_core,
        g_eri,
        "fUCCSDT",
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    # print(WF.c_mo)

    # WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=False, tol=1e-5, maxiter = 2000)

    print("E_opt:", WF._energy_elec)
    # print("E_opt: (+nuc!)", WF._energy_elec + e_nuc)
    
    print(WF.ci_coeffs)
  
    "Calculate Excitation energies"
    LR = generalized_naive.LinearResponse(WF, excitations="sd")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
    
    "Calculate polarizability"
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


    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuclear_dipole = np.einsum('i,ij->j', charges, coords)


    print('Nuclear dipole moments',nuclear_dipole)

    print(f'Total Dipolemoments:\n \t xx: {-dip_x+nuclear_dipole[0]:.4f} \t yy: {-dip_y+nuclear_dipole[1]:.4f} \t zz: {-dip_z+nuclear_dipole[2]:.4f}')



    "Calculate electric field gradients"
    efgx = generalized_one_electron_transform(WF.c_mo, efg_ints[0])
    efgy = generalized_one_electron_transform(WF.c_mo, efg_ints[1])
    efgz = generalized_one_electron_transform(WF.c_mo, efg_ints[2])
    efg_op_x = generalized_one_elec_op_0i_0a(efgx, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    efg_op_y = generalized_one_elec_op_0i_0a(efgy, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    efg_op_z = generalized_one_elec_op_0i_0a(efgz, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    efg_x=generalized_expectation_value(WF.ci_coeffs, [efg_op_x], WF.ci_coeffs, WF.ci_info)
    efg_y=generalized_expectation_value(WF.ci_coeffs, [efg_op_y], WF.ci_coeffs, WF.ci_info)
    efg_z=generalized_expectation_value(WF.ci_coeffs, [efg_op_z], WF.ci_coeffs, WF.ci_info)


    print(f'Electric Field Gradients:\n \t xx: {efg_x:.4f} \t yy: {efg_y:.4f} \t zz: {efg_z:.4f}')








        
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    nao = mol.nao

    for A in range(mol.natm):
        # --- Electronic contribution ---
        with mol.with_rinv_origin(mol.atom_coord(A)): #Sets the RA point
            f2 = (mol.intor("int1e_ipiprinv") 
                + mol.intor("int1e_ipiprinv").transpose(0, 2, 1) 
                + 2 * mol.intor("int1e_iprinvip"))  # (9, nao, nao)
        
        f2 = f2.reshape(3, 3, nao, nao)  # (3, 3, nao, nao)

        # compute expectation values for all 3x3 components
        efg_elec = np.zeros((3, 3))
        for alpha in range(3):
            for beta in range(3):
                mo = generalized_one_electron_transform(WF.c_mo, f2[alpha, beta])
                op = generalized_one_elec_op_0i_0a(mo, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs)
                efg_elec[alpha, beta] = generalized_expectation_value(
                    WF.ci_coeffs, [op], WF.ci_coeffs, WF.ci_info).real

        # make traceless by subtracting 1/3 * Tr from diagonal
        trace = np.trace(efg_elec) / 3
        for alpha in range(3):
            efg_elec[alpha, alpha] -= trace

        efg_elec *= -1  # electrons have charge -1

        # --- Nuclear contribution (already traceless) ---
        efg_nuc = np.zeros((3, 3))
        for B in range(mol.natm):
            if B == A:
                continue
            R_AB = coords[A] - coords[B]
            r = np.linalg.norm(R_AB)
            efg_nuc += charges[B] * (3 * np.outer(R_AB, R_AB) - np.eye(3) * r**2) / r**5

        # --- Total ---
        efg_total = efg_elec + efg_nuc

        print(f"EFG at atom {A} ({mol.atom_symbol(A)}):")
        print(f"  xx={efg_total[0,0]:.4f}  xy={efg_total[0,1]:.4f}  xz={efg_total[0,2]:.4f}")
        print(f"  yy={efg_total[1,1]:.4f}  yz={efg_total[1,2]:.4f}")
        print(f"  zz={efg_total[2,2]:.4f}")
        print(f"  Trace: {np.trace(efg_total):.2e}")
        print(f"  Symmetric: {np.allclose(efg_total, efg_total.T)}")











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
    basis = "631-g"
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
    active_space = ((1,1), 4)
    charge = 0
    spin = 0
    NR(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")


def oh_radical(): 
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;"""
    basis = 'sto-3g'
    active_space = ((5,4),12)
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

def LiH():
    geometry = """H  0.0   0.0  0.0;
                Li  0.8  0.0   0.0;"""
    basis = 'sto-3g'
    active_space = ((2,2),12)
    charge = 0
    spin=0
    NR(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom")


h3()
# h2()
# h4_rektangle()
# HI()
# HBr()
# oh_radical()
# BeH()
# h2o()
# LiH()
