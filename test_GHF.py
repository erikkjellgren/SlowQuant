import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from scipy.linalg import solve
from pyscf.x2c import sfx2c1e
from pyscf import cc
import scipy.linalg

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

    # mf = scf.GHF(mol).sfx2c1e() #spinfree
    # mf = scf.GHF(mol).x2c()
    # mf = scf.GHF(mol).x2c

    
    # mf = scf.GHF(mol).x2c()
    mf = scf.GHF(mol)

    # mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF

    mf.max_cycle = 50000

    # mf.scf()
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)
    # print(np.round(np.array(mf.mo_coeff),3))

    e_nuc=mf.energy_nuc()
    # "Non-relativistic integrals"
    # h_1e = mol.intor("int1e_kin")  
    # h_nuc=mol.intor("int1e_nuc")
    # h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    # g_eri = mol.intor("int2e")
    # print('Non-relativistic her',h_core)


 
    # "Relativistic integrals"
    # h_core=mf.get_hcore()
    # g_eri = mol.intor("int2e")

    # mc = mcscf.CASCI(mf, active_space[1], active_space[0])

    #make a random unitary transformation
    # u = unitary_group.rvs(c.shape[0]) 
    # print(np.dot(u, u.conj().T))
    # C_u = c @ u[0] 
    # mc = mcscf.CASCI(mf, active_space[1], active_space[0])
    
    # # Slowquant
    
    WF =GeneralizedWaveFunctionUPS(
        # mol.nelectron,
        active_space,
        coeff,
        #C_u,
        mol,
        "fUCCSDTQ",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    # print(WF.c_mo)

    # WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=False, tol=1e-5, maxiter = 2000)

    print("E_opt:", WF._energy_elec)
    # print("E_opt: (+nuc!)", WF._energy_elec + e_nuc)
    
    # print(WF.ci_coeffs)


    
    dip_ao = build_x2c_pc_operator(mf, mol, "int1e_r", 'int1e_sprsp', c, x2c=False, picture_change=False)


    # print("C shape:", WF.c_mo.shape)
    # print("dip_ao shape:", dip_ao.shape)
    # print("dip_ao[0] shape:", dip_ao[0].shape)

    "Calculate Excitation energies"
    LR = generalized_naive.LinearResponse(WF, excitations="sd")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)

    
    # "Calculate polarizability"
    prop_grad = LR.get_property_gradient(dip_ao) #Computes property gradient V
    response = solve(LR.hessian, prop_grad) # solve (E-h_bar omega S)X=V (the solution/responsevector) with omega =0 response = solve(LR.hessian- omega LR.metric, prop_grad) for non-static?
    alpha = np.einsum('ix,ix->x', prop_grad, response)

    print(f'Polarizabilities:\n \t xx: {alpha[0]:.4f} \t yy: {alpha[1]:.4f} \t zz: {alpha[2]:.4f}')

    "Calculate dipole moments"
    mux = generalized_one_electron_transform(WF.c_mo, dip_ao[0], x2c=False)
    muy = generalized_one_electron_transform(WF.c_mo, dip_ao[1], x2c=False)
    muz = generalized_one_electron_transform(WF.c_mo, dip_ao[2], x2c=False)
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


    # print('Nuclear dipole moments',nuclear_dipole)

    print(f'Total Dipolemoments:\n \t xx: {-dip_x+nuclear_dipole[0]:.4f} \t yy: {-dip_y+nuclear_dipole[1]:.4f} \t zz: {-dip_z+nuclear_dipole[2]:.4f}')



    # "Electric field gradients"
    # coords = mol.atom_coords()
    # charges = mol.atom_charges()

    # for A in range(mol.natm):
    #     int_pc = build_x2c_pc_operator_efg(mf, mol, A, c, x2c=False, picture_change=False)  # (3, 3, 2*nao_c, 2*nao_c)

    #     efg_elec = np.zeros((3, 3)) #create the EFG matrix
    #     for alpha in range(3):
    #         for beta in range(3):
    #             mo = generalized_one_electron_transform(WF.c_mo, int_pc[alpha, beta], x2c=False)
    #             op = generalized_one_elec_op_0i_0a(mo, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs)
    #             efg_elec[alpha, beta] = generalized_expectation_value(
    #                 WF.ci_coeffs, [op], WF.ci_coeffs, WF.ci_info
    #             ) #har fjernet .real
    #             # print('GFG real?', np.isreal(efg_elec[alpha, beta]))

    #     # Make traceless
    #     trace = np.trace(efg_elec) / 3
    #     for alpha in range(3):
    #         efg_elec[alpha, alpha] -= trace

    #     efg_elec *= -1  # electrons charge -1

    #     #Nuclear part
    #     efg_nuc = np.zeros((3, 3))
    #     for B in range(mol.natm):
    #         if B == A:
    #             continue
    #         R_AB = coords[B] - coords[A] #A os expansion point
    #         r = np.linalg.norm(R_AB)
    #         # efg_nuc += charges[B] * (3 * np.outer(R_AB, R_AB) - np.eye(3) * r**2) / r**5
    #         for alpha in range(3):
    #             for beta in range(3):
    #                 efg_nuc[alpha, beta] += charges[B] * (3 * R_AB[alpha] * R_AB[beta]/r**5 - (alpha == beta) / r**3) 


    #     #Total EFG
    #     efg_total = efg_elec + efg_nuc

    #     print(f"EFG at atom {A} ({mol.atom_symbol(A)}):")
    #     print(f"  xx={efg_total[0,0]:.4f}  xy={efg_total[0,1]:.4f}  xz={efg_total[0,2]:.4f}")
    #     print(f"  yy={efg_total[1,1]:.4f}  yz={efg_total[1,2]:.4f}")
    #     print(f"  zz={efg_total[2,2]:.4f}")
    #     print(f"  Trace: {np.trace(efg_total):.2e}")
    #     print(f"  Symmetric: {np.allclose(efg_total, efg_total.T)}")


    



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




"Calculate Properties"
def block_diagonal_matrix(mat):
    # print(scipy.linalg.block_diag(mat, mat))
    return scipy.linalg.block_diag(mat, mat)


def _sigma_dot(prp4: np.ndarray) -> np.ndarray:
    "Mapping the 4 Pauli coefficients to correct positions of the spin-orbital block matrix"
    w, x, y, z = prp4
    return np.block([
        [w + z,        x - 1j * y],
        [x + 1j * y,   w - z     ]
    ])
def build_x2c_pc_operator(mf, mol, int_LL, int_SS, c, x2c=False, picture_change=False): 
    with mol.with_common_orig((0, 0, 0)):
        if x2c==False:
            return mol.intor_symmetric(int_LL)    
        else:
            if picture_change:
                xmol = mf.with_x2c.get_xmol()[0]
                nao = xmol.nao
                r = xmol.intor_symmetric(int_LL)                          # (3, nao_x, nao_x)
                r_so = np.array([block_diagonal_matrix(x) for x in r])   # (3, 2*nao_x, 2*nao_x)
                c1 = 0.5 / c
                sprsp = xmol.intor_symmetric(int_SS).reshape(3, 4, nao, nao)
                sprsp_so = np.array([_sigma_dot(x * c1**2) for x in sprsp])
                return mf.with_x2c.picture_change((r_so, sprsp_so))       # (3, 2*nao_c, 2*nao_c) 
            else:
                nao_c = mol.nao
                r = mol.intor_symmetric(int_LL)                           # (3, nao_c, nao_c)
                r_so = np.array([block_diagonal_matrix(x) for x in r])   # (3, 2*nao_c, 2*nao_c) 
            return r_so




def build_x2c_pc_operator_efg(mf, mol, atom_idx, c, x2c=False, picture_change=False):
    with mol.with_rinv_origin(mol.atom_coord(atom_idx)):
        xmol = mf.with_x2c.get_xmol()[0]
        nao_x = xmol.nao
        nao_c = mol.nao
        c1 = 0.5 / c

        if picture_change:
            efg_ao = (
                xmol.intor("int1e_ipiprinv")
                + xmol.intor("int1e_ipiprinv").transpose(0, 2, 1)
                + 2 * xmol.intor("int1e_iprinvip")
            )  # (9, nao_x, nao_x)
            f2_LL_spinor = np.array([block_diagonal_matrix(x) for x in efg_ao])  # (9, 2*nao_x, 2*nao_x)
            f2_SS = xmol.intor("int1e_ipsprinvspip").reshape(9, 4, nao_x, nao_x)
            f2_SS_spinor = np.array([_sigma_dot(x * c1**2) for x in f2_SS])
            ao_efg = mf.with_x2c.picture_change((f2_LL_spinor, f2_SS_spinor))  # (9, 2*nao_c, 2*nao_c)
            nao_out = nao_c
        else:
            efg_ao = (
                mol.intor("int1e_ipiprinv")                       
                + mol.intor("int1e_ipiprinv").transpose(0, 2, 1)
                + 2 * mol.intor("int1e_iprinvip")
            )  # (9, nao_c, nao_c)
            ao_efg = np.array([block_diagonal_matrix(x) for x in efg_ao])  # (9, 2*nao_c, 2*nao_c)
            nao_out = nao_c  

        ao_efg = 0.5 * (ao_efg + ao_efg.conj().transpose(0, 2, 1))
        ao_efg = ao_efg.reshape(3, 3, 2 * nao_out, 2 * nao_out)
        ao_efg = 0.5 * (ao_efg + ao_efg.transpose(1, 0, 2, 3))

    return ao_efg




def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "STO-3G"
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
    # basis = {'H':'sto-3g','Br': 'dyalldz'}
    basis = {'H':'sto-3g','Br': 'x2c-SVPall.nw'}
    # basis = ''
    active_space = ((2,2), 6)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
from pyscf.gto.basis import load
import pyscf.gto as gto
def HCl():
    geometry = """H  0.0   0.0  0.0;
        Cl  0.0  0.0  1.27455 """
    # basis = {'H':'sto-3g','Cl': 'x2c-SVPall.nw'}
    basis = {'H': gto.uncontract(load('x2c-SVPall.nw', 'H')),
                'Cl': gto.uncontract(load('x2c-SVPall.nw', 'Cl'))}
    active_space = ((2,2), 6) #spin orbitaler or spinor basis
    # active_space = (2, 4)
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
    basis = "631-g"
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



# h3()
# h2()
# h4_rektangle()
# HI()
# HBr()
# oh_radical()
# BeH()
# h2o()
# LiH()
HCl()
