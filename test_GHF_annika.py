import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm


# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a
from slowquant.unitary_coupled_cluster.generalized_density_matrix import get_orbital_gradient_generalized_real_imag, get_orbital_gradient_expvalue_real_imag

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)




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

    
    # WF =WaveFunctionUPS(
    #     mol.nelectron,
    #     active_space,
    #     mf.mo_coeff,
    #     h_core,
    #     g_eri,
    #     "fuccsd",
    #     {"n_layers": 2},
    #     include_active_kappa=True,
    # )
    # print('Antal elektroner',mol.nelectron)
    # WF.run_wf_optimization_1step("bfgs", True)
    
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


def my_ao2mo_1e(nmo,nao,int_1e_nuc,int_1e_kin,C):
    one_e_int_kin = np.zeros(shape=(nmo,nmo))
    one_e_int_nuc = np.zeros(shape=(nmo,nmo))

    for a in range(nmo):
        for b in range(nmo):
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0
            for mu in range(nao):
                for nu in range(nao):
                    term1 += C[mu,a]*C[nu,b]*int_1e_nuc[mu,nu]
                    term2 += C[nao+mu,a]*C[nao+nu,b]*int_1e_nuc[mu,nu]
                    term3 += C[mu,a]*C[nu,b]*int_1e_kin[mu,nu]
                    term4 += C[nao+mu,a]*C[nao+nu,b]*int_1e_kin[mu,nu]
            one_e_int_nuc[a,b] = term1 + term2 
            one_e_int_kin[a,b] = term3 + term4
    return np.add(one_e_int_kin, one_e_int_nuc)

def ao2mo_2e_new(nmo,nao,C,int_2e_inp):
    two_e_int = np.zeros(shape=(nmo,nmo,nmo,nmo))

    int_2e = np.array(int_2e_inp)

    temp1_1 = np.zeros(shape=(nmo,nao,nao,nao))
    temp2_1 = np.zeros(shape=(nmo,nmo,nao,nao))
    temp3_1 = np.zeros(shape=(nmo,nmo,nmo,nao))
    temp1_2 = np.zeros(shape=(nmo,nao,nao,nao))
    temp2_2 = np.zeros(shape=(nmo,nmo,nao,nao))
    temp3_2 = np.zeros(shape=(nmo,nmo,nmo,nao))
    temp1_3 = np.zeros(shape=(nmo,nao,nao,nao))
    temp2_3 = np.zeros(shape=(nmo,nmo,nao,nao))
    temp3_3 = np.zeros(shape=(nmo,nmo,nmo,nao))
    temp1_4 = np.zeros(shape=(nmo,nao,nao,nao))
    temp2_4 = np.zeros(shape=(nmo,nmo,nao,nao))
    temp3_4 = np.zeros(shape=(nmo,nmo,nmo,nao))

    for a in range(nmo):
        for mu in range(nao):  
            temp1_1[a,:,:,:] += C[mu,a]*int_2e[mu,:,:,:]
            temp1_2[a,:,:,:] += C[mu,a]*int_2e[mu,:,:,:]
            temp1_3[a,:,:,:] += C[nao+mu,a]*int_2e[mu,:,:,:]
            temp1_4[a,:,:,:] += C[nao+mu,a]*int_2e[mu,:,:,:]
        for b in range(nmo):
            for nu in range(nao):  
                temp2_1[a,b,:,:] += C[nu,b]*temp1_1[a,nu,:,:]
                temp2_2[a,b,:,:] += C[nu,b]*temp1_2[a,nu,:,:]
                temp2_3[a,b,:,:] += C[nao+nu,b]*temp1_3[a,nu,:,:]
                temp2_4[a,b,:,:] += C[nao+nu,b]*temp1_4[a,nu,:,:]
            for c in range(nmo):
                for la in range(nao):  
                    temp3_1[a,b,c,:] += C[la,c]*temp2_1[a,b,la,:]
                    temp3_2[a,b,c,:] += C[nao+la,c]*temp2_2[a,b,la,:]
                    temp3_3[a,b,c,:] += C[la,c]*temp2_3[a,b,la,:]
                    temp3_4[a,b,c,:] += C[nao+la,c]*temp2_4[a,b,la,:]
                for d in range(nmo):
                    for si in range(nao):  
                        two_e_int[a,b,c,d] += C[si,d]*temp3_1[a,b,c,si]
                        two_e_int[a,b,c,d] += C[nao+si,d]*temp3_2[a,b,c,si] 
                        two_e_int[a,b,c,d] += C[si,d]*temp3_3[a,b,c,si] 
                        two_e_int[a,b,c,d] += C[nao+si,d]*temp3_4[a,b,c,si] 

    return two_e_int

def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    #X2C
    uhf = scf.UHF(mol)
    uhf.chkfile = "/home/annika4ee/SlowQuant/uhf_guess.chk"
    uhf.scf()
    uhf.kernel()

    nmo = uhf.mo_coeff[0].shape[1]

    # small random anti-Hermitian
    epsilon = 0.6  # controls "step size"
    X = np.random.randn(nmo, nmo) + 1j*np.random.randn(nmo, nmo)
    A = epsilon * (X - X.conj().T)/2  # make anti-Hermitian
    # unitary
    U_small = expm(A)
    c_guess = (uhf.mo_coeff[0] @ U_small, uhf.mo_coeff[1] @ U_small)

    # load original chkfile content (optional)
    data = chkfile.load("/home/annika4ee/SlowQuant/uhf_guess.chk",'scf')

    # overwrite the MO coefficients
    data['mo_coeff'] = c_guess

    # dump everything back into a new chkfile
    chkfile.dump('/home/annika4ee/SlowQuant/uhf_guess.chk','scf', data)


    mf = scf.GHF(mol)
    mf.chkfile = '/home/annika4ee/SlowQuant/uhf_guess.chk'

    # Change initial guess:
    #mf.init_guess = "chkfile"

    mf.scf()
    mf.kernel()
    c=np.array(mf.mo_coeff,dtype=complex)

    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    h_1e = mol.intor("int1e_kin")
    h_nuc=mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    mc = mcscf.CASCI(mf, active_space[1], active_space[0])

    # mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # # Slowquant

     # small random anti-Hermitian
    eps = 0.3  # controls "step size"
    X_anti = np.random.randn(c.shape[0],c.shape[0]) + 1j*np.random.randn(c.shape[0],c.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    U_step = expm(A_mat)

    c_u = c @ U_step



    WF = GeneralizedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        c,
        h_core,
        g_eri,
        "fuccsd",
        {"n_layers": 2},
        include_active_kappa=True,
    )

    print(mf.energy_elec()[0])

    print("Nr. of spin orbitals:", WF.num_spin_orbs)
    print("Nr. of kappas:", len(WF.kappa_spin_idx))
    print("Nr. of spin orbitals:", WF.num_spin_orbs)
    print("Nr. of inactive spin orbitals:", WF.num_inactive_spin_orbs)
    print("Nr. of active spin orbitals:", WF.num_active_spin_orbs)
    print("Nr. of virtual spin orbitals:", WF.num_virtual_spin_orbs)



    #print("Nr. of occ active spind idx shifted orbitals:", WF.active_occ_spin_idx_shifted)
    #print("Nr. of unocc active spind idx shifted orbitals:",WF.active_unocc_spin_idx_shifted)

    H=generalized_hamiltonian_full_space(WF.h_mo, WF.g_mo,WF.num_spin_orbs)
    H2=generalized_hamiltonian_0i_0a(WF.h_mo, WF.g_mo, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs)
    H3=generalized_hamiltonian_1i_1a(WF.h_mo, WF.g_mo, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs, WF.num_virtual_spin_orbs)

    test_energy=expectation_value(WF.ci_coeffs, [H], WF.ci_coeffs, WF.ci_info)
    test_energy2=expectation_value(WF.ci_coeffs, [H2], WF.ci_coeffs, WF.ci_info)
    test_energy3=expectation_value(WF.ci_coeffs, [H3], WF.ci_coeffs, WF.ci_info)

    print(test_energy)
    print(test_energy2)
    print(test_energy3)


    my_gradient_before = get_orbital_gradient_generalized_real_imag(WF.h_mo,
        WF.g_mo,
        WF.kappa_spin_idx,
        WF.num_inactive_spin_orbs, 
        WF.num_active_spin_orbs,
        WF.rdm1,
        WF.rdm2)

    print("my gradient_before:",np.round(my_gradient_before,10))


    total_gradient_before = get_orbital_gradient_expvalue_real_imag(
        WF.ci_coeffs,
        WF.ci_info,
        WF.h_mo,
        WF.g_mo,
        WF.num_spin_orbs,
        WF.kappa_spin_idx)
            
    print('total gradient_before',np.round(total_gradient_before,10))


    #WF.run_wf_optimization_1step("BFGS",orbital_optimization=True,test=True)


    '''my_gradient_after = get_orbital_gradient_generalized_real_imag(WF.h_mo,
        WF.g_mo,
        WF.kappa_spin_idx,
        WF.num_inactive_spin_orbs, 
        WF.num_active_spin_orbs,
        WF.rdm1,
        WF.rdm2)

    print("my gradient after:",np.round(my_gradient_after,3))

    total_gradient_after = get_orbital_gradient_expvalue_real_imag(
        WF.ci_coeffs,
        WF.ci_info,
        WF.h_mo,
        WF.g_mo,
        WF.num_spin_orbs,
        WF.kappa_spin_idx)
            
    print('total gradient_after',np.round(total_gradient_after,3))'''


    '''WF.do_adapt(
        operator_pool = ["gs","gd"],
        orbital_optimization = True,
    )'''


    '''LR = naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)'''


def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    active_space = ((1, 1), 4)
    #active_space = (2, 4)
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

def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    basis = "cc-pvdz"
    #basis = "631-g"
    #basis = "sto-3g"
    active_space = ((1, 2), 4)
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

def LiH():
    geometry = """H  0.0   0.0  0.0;
        Li  0.0  0.0  1"""
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    active_space = ((1, 1), 8)
    #active_space = (2, 4)
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

def h2o():
    geometry = """
    O  0.0   0.0  0.11779 
    H  0.0   0.75545  -0.47116;
    H  0.0  -0.75545  -0.47116"""
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    basis = "sto-3g"
    #active_space = ((5, 5), 14)
    active_space = ((1,1),4)
    #active_space = (2, 4)
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

h2o()


# h2o()

# HI()
# HBr()