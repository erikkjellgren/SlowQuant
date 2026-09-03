import numpy as np

# Compatibility patch for old code expecting np.complex
if not hasattr(np, "complex"):
    np.complex = complex

if not hasattr(np, "float"):
    np.float = float

if not hasattr(np, "int"):
    np.int = int


import pyscf
from pyscf import mcscf, scf, gto, x2c, lib
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
from pyscf.scf.dhf import _visscher_ssss_correction
from collections import defaultdict
import h5py
from DIRACparser_functions import read_dirac_file
import basis_set_exchange as bse


import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyscf')

from pyscf.prop.ssc.dhf import SSC
from pyscf.prop.ssc.dhf import sa01sa01_integral
from pyscf.prop.nmr import dhf as nmr_dhf

from pyscf.scf import dhf, hf
from pyscf.prop.ssc.rhf import SSC as SSC_rhf



# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive_DHF
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value_energy
from slowquant.unitary_coupled_cluster.generalized_operators import DHF_hamiltonian_full_space, DHF_hamiltonian_0i_0a, DHF_hamiltonian_1i_1a, DHF_hamiltonian_full_space
from slowquant.unitary_coupled_cluster.generalized_density_matrix_DHF import ( get_orbital_gradient_generalized_real_imag,
get_orbital_gradient_expvalue_real_imag, get_nonsplit_gradient_expvalue, 
get_gradient_finite_diff, get_electronic_energy_generalized, RDM2, get_orbital_response_hessian_block,
)

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)

from slowquant.molecularintegrals.integralfunctions import DHF_one_electron_transform, DHF_two_electron_transform

c = lib.param.LIGHT_SPEED

def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, cart = False)


    # rhf = scf.RHF(mol)


    # rhf.conv_tol = 1e-8        # Energy convergence (Hartree)
    # rhf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    # rhf.max_cycle = 500
    # rhf.kernel()

    # sscobj_rhf = SSC_rhf(rhf)
    # sscobj_rhf.cphf = True
    # sscobj_rhf.conv_tol = 1e-9
    # sscobj_rhf.verbose = 5
    # sscobj_rhf.with_fcsd = True
    # sscobj_rhf.kernel()


    #mol = build_ukb_mol(mol_init)


    #uhf = scf.UHF(mol)
    #uhf.kernel()

    #mf_n = dhf.DHF(mol)        # bypass scf.DHF factory — always gives DHF, never RDHF
    #mf = scf.newton(mf_n)

    #print(type(mf))   # should be _SecondOrderDHF


    hf.remove_overlap_zero_eigenvalue = False

    mf = scf.dhf.DHF(mol)


    mf.conv_tol = 1e-10        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 500
    # mf.with_ssss


    '''#DIRAC_dict = read_dirac_file("OPERATORS.h5")

    #hcore_dirac = build_hcore_from_dirac(DIRAC_dict)

    #mf.get_hcore = lambda *args: hcore_dirac

    #mf.get_init_guess = lambda *args: initial_guess_from_hcore(hcore_dirac, mol.nelectron)'''

    mf.kernel()


    sscobj = SSC(mf)
    sscobj.cphf = True
    sscobj.conv_tol = 1e-9
    sscobj.mb = "RMB"
    sscobj.verbose = 5
    sscobj.with_fcsd = True
    #jj = sscobj.kernel()


    nmr = nmr_dhf.NMR(mf)
    nmr.cphf = True
    nmr.mb = 'RMB'      # or 'RKB'
    nmr.gauge_orig = None #[0,0,0]  # GIAO vs. # [0,0,0]

    shielding = nmr.kernel()

    sigma_iso = np.trace(shielding, axis1=1, axis2=2) / 3
    print(sigma_iso)


    '''# print("PySCF e11 (raw, before Hz conversion):", jj)
    # print("PySCF Tr(e11)/3:", np.trace(jj[0])/3)


    # Getting integrals for response:
    # Paramagnetic: (natm, 3, n4c, n4c)
    h1 = make_h1_ao(mol)

    #print(np.linalg.norm(h1))

    # Diamagnetic: (natm, natm, 3, 3, n4c, n4c)
    h2 = make_h2_ao(mol)
    #h2 = np.zeros_like(h1)'''


    C_MO=np.array(mf.mo_coeff,dtype=complex)

    #h_core = mf.get_hcore()
    #g_eri = np.array([mol.intor("int2e_spinor"), mol.intor('int2e_spsp1spsp2_spinor')*(0.0625/c**4),
    #                  mol.intor('int2e_spsp2_spinor')*(0.25/c**2), mol.intor('int2e_spsp1_spinor')*(0.25/c**2)],dtype=np.complex128)

    #print(g_eri.shape)
    #dip_int = mol.intor("int1e_r")

    #size = C_MO.shape[0] // 2


    # small random anti-Hermitian
    #eps = 0.001  # controls "step size"
    #X_anti = np.random.randn(C_MO.shape[0],C_MO.shape[0]) + 1j*np.random.randn(C_MO.shape[0],C_MO.shape[0])
    #A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    #U_step = expm(A_mat)

    #C_U = C_MO @ U_step

    S_ovlp = mf.get_ovlp()

    def theta(C):
        """
        Time reversal for PySCF 4c ordering:
        [large 2c spinor block, small 2c spinor block]
        """

        nao, nmo = C.shape

        if nao % 2 != 0:
            raise ValueError("Expected even number of 4c AO functions")

        n2 = nao // 2

        out = np.zeros_like(C)

        # large component block
        L = C[:n2].conj()
        out[:n2:2, :] = -L[1::2, :]
        out[1:n2:2, :] =  L[0::2, :]

        # small component block
        S = C[n2:].conj()
        out[n2::2, :] = -S[1::2, :]
        out[n2+1::2, :] = S[0::2, :]

        return out

    
    Cbar = theta(C_MO)

    M = C_MO.conj().T @ S_ovlp @ Cbar

    K_pairs = []
    M_values = []

    for p in range(C_MO.shape[1]):
        q = int(np.argmax(abs(M[p,:])))
        if (p,q) not in K_pairs and (q,p) not in K_pairs: 
            K_pairs.append((p,q))
            M_values.append(np.max(abs(M[p,:])))

    #print("Kramers pairs")
    #print(K_pairs)
    #print(M_values)


    # WF2 = GeneralizedWaveFunctionUPS(
    #     active_space,
    #     C_MO,
    #     mol, 
    #     K_pairs,
    #     False,
    #     "fUCCSD",
    #     {"n_layers": 0, "is_spin_conserving" : False},
    #     include_active_kappa=True,
    # )

    # np.random.seed(20)
    # if len(WF2.thetas) > 0:
    #     real = np.random.uniform(-0.05,0.05,len(WF2.thetas_real))
    #     #imag = np.zeros_like(WF.thetas_imag)
    #     imag = np.random.uniform(-0.05,0.05,len(WF2.thetas_real))
    #     WF2.set_thetas(real, imag)


    # data = np.load("LiH((1,1),4).npz") 
    # data = np.load("HF((5,5),12).npz") 
    # data = np.load("LiH((2,2),6).npz") 
    # data = np.load("HF((1,1),4).npz")
    # data = np.load("H2-6-31g-J((1,1),4).npz")
    data = np.load("HF((2,2),6).npz")
    # data = np.load("H2-dyallv2z((1,1),6).npz")

    WF2 = GeneralizedWaveFunctionUPS(
        active_space,
        data["c_mo"],
        #C_MO,
        mol,
        K_pairs,
        False,
        "fUCCSD",
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )
    WF2.set_thetas(data["thetas_real"], data["thetas_imag"])

    #print("Kramers rotations:")
    #print(WF2.kappa_spin_idx)
    #print(WF2.kappa_spin_idx_ep)


    '''# WF = GeneralizedWaveFunctionUPS(
    #     active_space,
    #     #C_MO,
    #     C_U,
    #     mol,
    #     K_pairs,
    #     False,
    #     "fUCCSD",
    #     {"n_layers": 0, "is_spin_conserving" : False},
    #     include_active_kappa=True,
    # )'''

    print("DHF", mf.energy_elec()[0])

    # h_mo = DHF_one_electron_transform(C_MO, h_core)
    # g_mo = DHF_two_electron_transform(C_MO, g_eri)

    print("Nr. of kappas:", len(WF2.kappa_spin_idx))
    print("Nr. of spin orbitals:", WF2.num_spin_orbs)
    print("Nr. of inactive spin orbitals:", WF2.num_inactive_spin_orbs)
    print("Nr. of active spin orbitals:", WF2.num_active_spin_orbs)
    print("Nr. of virtual spin orbitals:", WF2.num_virtual_spin_orbs)
    print("Nr. of positronic spin orbitals:", WF2.num_spin_orbs_NES)
    print("Inactive spin orbitals idx:", WF2.inactive_spin_idx)
    print("Active spin orbitals idx:", WF2.active_spin_idx)
    print("Virtual spin orbitals idx:", WF2.virtual_spin_idx)
    print("Positronic spin orbitals idx:", WF2.positronic_spin_idx)
    print("Active occupied:", WF2.active_occ_spin_idx)
    print("Active unoccupied:", WF2.active_unocc_spin_idx)
    #print("qs: ", WF2.kappa_no_activeactive_spin_idx_resp)

    '''H = DHF_hamiltonian_full_space(h_mo[size:,size:], g_mo[size:,size:,size:,size:], WF.num_spin_orbs_NES)

    #H = generalized_hamiltonian_full_space(h_mo, g_mo, WF.num_spin_orbs)


    #print(WF.rdm1)

    # E_tester = get_electronic_energy_generalized(
    #             h_mo,
    #             g_mo,
    #             WF.num_spin_orbs_NES,
    #             WF.num_inactive_spin_orbs,
    #             WF.num_active_spin_orbs,
    #             WF.rdm1,
    #             WF.rdm2,
    #         )
    
    # print(E_tester)
    #print(_visscher_ssss_correction(mf,mf.make_rdm1()))
    #print(E_tester + _visscher_ssss_correction(mf,mf.make_rdm1()))


    # E2 = generalized_expectation_value_energy(
    #             WF.ci_coeffs,
    #             # [generalized_hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_spin_orbs, self.num_active_spin_orbs)],
    #             [H],
    #             WF.ci_coeffs,
    #             WF.ci_info,
    #         )

    # print(E2)


    # print("noactive_active", WF.kappa_no_activeactive_spin_idx)
    # print("noactive_active resp", WF.kappa_no_activeactive_spin_idx_resp)
    # print("Kappas ep:", WF.kappa_spin_idx_ep)
    # print("Kappas ee:", WF.kappa_spin_idx)



    #print(WF.ci_info.num_inactive_orbs)
    #print(WF.ci_info.num_active_orbs)  
    #print(WF.ci_info.num_virtual_orbs)
    #print(WF.ci_info.num_positronic_orbs)
    #print(WF.ci_info.idx2det)

    #print(RDM2(12, 13, 12, 13, 12, 8, 4, WF.rdm1, WF.rdm2))

    #print(RDM2(12, 13, 13, 12, 12, 8, 4, WF.rdm1, WF.rdm2))

    # err = 0
    # for p in range(C_MO.shape[1]):
    #     for q in range(C_MO.shape[1]):
    #         for r in range(C_MO.shape[1]):
    #             for s in range(C_MO.shape[1]):
    #                 err = max(err, abs(
    #                     RDM2(p,q,r,s,WF.num_spin_orbs_NES, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs, WF.rdm1, WF.rdm2) 
    #                   + RDM2(r,q,p,s,WF.num_spin_orbs_NES, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs, WF.rdm1, WF.rdm2)
    #                 ))
    # print("err", err)

    #WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 10000)

    # print("rdm2 PySCF")
    # with np.printoptions(precision=4):
    #             print(np.round(WF2.rdm2, 4))

    # print("rdm2 Mine")
    # with np.printoptions(precision=4):
    #         print(np.round(WF.rdm2, 4))

    # D_pyscf = C_MO @ C_MO.conj().T

    # D_mine = WF.c_mo @ WF.c_mo.conj().T

    # print("D PySCF")
    # with np.printoptions(precision=4):
    #             print(np.round(D_pyscf, 4))

    # print("D Mine")
    # with np.printoptions(precision=4):
    #         print(np.round(D_mine, 4))


    # gradient_ee = get_orbital_gradient_generalized_real_imag(
    #             WF.h_mo,
    #             WF.g_mo,
    #             WF.kappa_spin_idx,
    #             WF.num_spin_orbs_NES,
    #             WF.num_inactive_spin_orbs,
    #             WF.num_active_spin_orbs,
    #             WF.rdm1,
    #             WF.rdm2,
    #         )

    # gradient_ep = - get_orbital_gradient_generalized_real_imag(
    #             WF.h_mo_ep,
    #             WF.g_mo_ep,
    #             WF.kappa_spin_idx_ep,
    #             WF.num_spin_orbs_NES,
    #             WF.num_inactive_spin_orbs,
    #             WF.num_active_spin_orbs,
    #             WF.rdm1,
    #             WF.rdm2,
    #         )
    
    # print("max gradient ee", np.max(np.abs(gradient_ee)))
    # print("max gradient ep", np.max(np.abs(gradient_ep)))

    #print(WF._calc_gradient_optimization_DHF(WF.kappa_real + WF.kappa_imag, theta_optimization=False, kappa_ee_optimization=True,kappa_ep_optimization=True))

    #kappas = np.concatenate([WF.kappa_real, WF.kappa_real_ep, WF.kappa_imag, WF.kappa_imag_ep])

    # gradient_test = get_orbital_gradient_generalized_real_imag(
    #             WF.h_mo,
    #             WF.g_mo,
    #             WF.kappa_spin_idx,
    #             WF.num_spin_orbs_NES,
    #             WF.num_inactive_spin_orbs,
    #             WF.num_active_spin_orbs,
    #             WF.rdm1,
    #             WF.rdm2,
    #         )
    #print(np.round(gradient_test,5))

    # gradient_test_ep = get_orbital_gradient_generalized_real_imag(
    #         WF.h_mo,
    #         WF.g_mo,
    #         WF.kappa_spin_idx_ep,
    #         WF.num_spin_orbs_NES,
    #         WF.num_inactive_spin_orbs,
    #         WF.num_active_spin_orbs,
    #         WF.rdm1,
    #         WF.rdm2,
    #     )
    #print(np.round(gradient_test_ep,5))


    # gradient_test_ep = get_orbital_gradient_generalized_real_imag(
    #     WF.h_mo,
    #     WF.g_mo,
    #     WF.kappa_no_activeactive_spin_idx_resp,
    #     WF.num_spin_orbs_NES,
    #     WF.num_inactive_spin_orbs,
    #     WF.num_active_spin_orbs,
    #     WF.rdm1,
    #     WF.rdm2,
    # )
    #print(np.round(gradient_test_ep,5))

    # hess = get_orbital_response_hessian_block(
    #     WF.h_mo,
    #     WF.g_mo,
    #     WF.kappa_no_activeactive_spin_idx_dagger,
    #     WF.kappa_no_activeactive_spin_idx,
    #     WF.num_spin_orbs_NES, 
    #     WF.num_inactive_spin_orbs,
    #     WF.num_active_spin_orbs,
    #     WF.rdm1,
    #     WF.rdm2,
    #     )
    
    # print(f"Hermiticity check of the Hessian: max|E2 - E2†| = "
    #         f"{np.max(np.abs(hess - hess.conj().T)):.2e}")  

    # E_tester_post = get_electronic_energy_generalized(
    #             WF.h_mo[size:,size:],
    #             WF.g_mo[size:,size:,size:,size:],
    #             WF.num_spin_orbs_NES,
    #             WF.num_inactive_spin_orbs,
    #             WF.num_active_spin_orbs,
    #             WF.rdm1,
    #             WF.rdm2,
    #         )
    
    # print(E_tester_post)

    # print(WF.kappa_no_activeactive_spin_idx_resp)


    U = WF2.c_mo.conj().T @ S_ovlp @ WF.c_mo

    # with np.printoptions(precision=4):
    #     print(np.round(U, 4))

    # with np.printoptions(precision=4):
    #     print(np.round(U.conj().T @ U, 4))

    # with np.printoptions(precision=4):
    #     print(np.round(WF2.c_mo, 4))
    #     print(np.round(WF.c_mo, 4)) 

    def time_reversal_matrix(n_spatial):
        """
        Time reversal operator in the Dirac representation.

        Basis ordering assumed:
            (large alpha, large beta, small alpha, small beta)
            for each spatial basis function

        Returns the matrix part of T, excluding complex conjugation.
        """
        sigma_y = np.array(
            [[0, -1j],
            [1j, 0]],
            dtype=complex
        )

        block = np.zeros((4, 4), dtype=complex)
        block[:2, :2] = sigma_y
        block[2:, 2:] = sigma_y

        # T = -i sigma_y K
        return -1j * np.kron(np.eye(n_spatial), block)


    def check_kramers_symmetry(C, S, occ_idx):
        """
        Check Kramers symmetry of a DHF determinant.

        Parameters
        ----------
        C : ndarray
            4-component MO coefficient matrix
            shape (n4c, nmo)

        S : ndarray
            4-component overlap matrix
            shape (n4c, n4c)

        occ_idx : list/int array
            occupied orbital indices

        Returns
        -------
        deviation : float
            ||P - TPT^-1||
        """

        n4c = C.shape[0]
        n_spatial = n4c // 4

        Tmat = time_reversal_matrix(n_spatial)

        # occupied projector
        C_occ = C[:, occ_idx]

        P = C_occ @ C_occ.conj().T @ S

        # Time reversed density:
        # T P T^-1 = Tmat P* Tmat^\dagger
        P_TR = Tmat @ P.conj() @ Tmat.conj().T

        deviation = np.linalg.norm(P - P_TR)

        print("Kramers breaking measure =", deviation)
        print("relative =", deviation / np.linalg.norm(P))

        return deviation

    occ_idx = [WF.num_spin_orbs_NES,
          WF.num_spin_orbs_NES + 1]

    check_kramers_symmetry(WF.c_mo, S_ovlp, occ_idx)

    def time_reverse(spinor):
        """
        Time reversal for a 4-component Dirac spinor basis.

        Assumes basis ordering:
        [L alpha, L beta, S alpha, S beta]
        for each spatial AO.

        spinor length = 4 * nao
        """

        nao = len(spinor) // 4

        T4 = np.array([
            [0, -1j, 0, 0],
            [1j,  0, 0, 0],
            [0,  0, 0, -1j],
            [0,  0, 1j,  0],
        ], dtype=complex)

        T = np.kron(np.eye(nao), T4)

        return T @ spinor.conj()


    def check_kramers_pair(alpha, beta, S=None):

        beta_from_alpha = time_reverse(alpha)

        if S is None:
            overlap = np.vdot(beta, beta_from_alpha)
            norm = np.linalg.norm(beta)*np.linalg.norm(beta_from_alpha)

        else:
            overlap = beta.conj().T @ S @ beta_from_alpha
            norm = np.sqrt(
                np.real(beta.conj().T @ S @ beta)
                *
                np.real(beta_from_alpha.conj().T @ S @ beta_from_alpha)
            )

        overlap /= norm

        print("Kramers overlap =", overlap)
        print("Deviation =", np.sqrt(max(0,1-abs(overlap)**2)))

        return overlap

    alpha_occ = WF.c_mo[:, 4]
    beta_occ  = WF.c_mo[:, 5]

    # check_kramers_pair(alpha_occ, beta_occ, S_ovlp)'''

    #WF.run_wf_optimization_2step_DHF(optimizer_name = "l-bfgs-b", orbital_optimization = True, tol = 1e-10, maxiter = 1000)


    #Optimization:
    #WF2.run_wf_optimization_2step_DHF(optimizer_name = "l-bfgs-b", orbital_optimization = True, tol = 1e-12, maxiter = 1000)

    # np.savez(
    #     "H2-6-311g_pp_ss-J((1,1),6)",
    #     #"H2-6-31g-J((1,1),4)",
    #     c_mo=WF2.c_mo,
    #     thetas_real=WF2.thetas_real,
    #     thetas_imag=WF2.thetas_imag
    #     )


    LR2 = generalized_naive_DHF.LinearResponse(WF2, excitations="SD", screen = True)
    LR2.calc_excitation_energies()

    print("PySCF:", sigma_iso)

    LR2.get_shieldings_4comp_iso(RMB_GIAO = True, output = True)


    #LR = generalized_naive_DHF.LinearResponse(WF, excitations="S")

    #LR.calc_excitation_energies()
    #print("Excitation energies:",  LR.excitation_energies[LR.excitation_energies < 1e4])

    #shieldings = LR.get_shieldings_4comp_iso(hm, hBm, hB, gB, sB)
    #print("Shieldings after:")
    #print(shieldings)



    
    #print(np.round(LR.get_transition_dipole(dip_int).real,5))
    #print(LR.get_oscillator_strengths(dip_int))

    '''# h1_shield = make_h1_ao_shield(mol)
    # h2_shield = make_h2_ao_shield(mol)
    # hb_shield = make_h_B_ao(mol)
    # g_ssss, g_lsss = make_h_B_2e_ao(mol)

    # shieldings = LR.get_shieldings_4comp_iso(h1_shield, h2_shield, hb_shield, g_ssss, g_lsss)
    # print("Shieldings:")
    # print(shieldings)


    # SSCC = LR.get_SSCC_4comp_iso(h1, h2)
    # for I in range(SSCC.shape[0]):
    #     for J in range(I+1, SSCC.shape[1]):
    #         print(f"K({mol.atom_symbol(I)}{I} - {mol.atom_symbol(J)}{J}) = {SSCC[I,J]:.5f} Hz")




    # kappa = WF.kappa_no_activeactive_spin_idx
    # kappa_dag = WF.kappa_no_activeactive_spin_idx_dagger

    # n = len(kappa)

    # D = np.zeros((n,n), dtype=complex)  # dagger side
    # E = np.zeros((n,n), dtype=complex)  # normal side

    # # D[new dagger, old dagger]
    # for i, (T, Uidx) in enumerate(kappa_dag):
    #     for j, (M, N) in enumerate(kappa_dag):
    #         D[i,j] = np.conj(U[M,T]) * U[N,Uidx]

    # # E[new normal, old normal]
    # for i, (T, Uidx) in enumerate(kappa):
    #     for j, (M, N) in enumerate(kappa):
    #         E[i,j] = np.conj(U[M,T]) * U[N,Uidx]


    # A_transformed = D @ LR2.A @ E.T
    # B_transformed = D @ LR2.B @ D.T

    # print(np.max(np.abs(A_transformed - LR.A)))
    # print(np.max(np.abs(B_transformed - LR.B)))





    # nNES = WF.num_spin_orbs_NES
    # nocc = WF.num_inactive_spin_orbs + WF.num_active_spin_orbs

    # print("occ→virt", np.linalg.norm(U[nNES:nNES+nocc, nNES+nocc:]))
    # print("virt→occ", np.linalg.norm(U[nNES+nocc:, nNES:nNES+nocc]))
    # print("NES→occ ", np.linalg.norm(U[:nNES, nNES:nNES+nocc]))
    # print("occ→NES ", np.linalg.norm(U[nNES:nNES+nocc, :nNES]))


    # U = WF2.c_mo.conj().T @ S_ovlp @ WF.c_mo
    # print("U unitary check:", np.abs(U.conj().T @ U - np.eye(U.shape[0])).max())

    # h_transformed = U.conj().T @ WF2.h_mo @ U
    # print("h_mo consistency:", np.abs(h_transformed - WF.h_mo).max())

    # h_test = DHF_one_electron_transform(WF.c_mo,h_core)

    # print("h_mo consistency:", np.abs(h_test - WF.h_mo).max())

    # C_test = WF2.c_mo @ U
    # print(np.linalg.norm(C_test - WF.c_mo))

    # print(np.trace(WF2.h_mo[6:8,6:8]), np.trace(WF.h_mo[6:8,6:8]))

    # D_mine = WF.c_mo[:, 4:6] @ WF.c_mo[:, 4:6].conj().T
    # D_pyscf = WF2.c_mo[:, 4:6] @ WF2.c_mo[:, 4:6].conj().T
    # print(np.abs(D_mine - D_pyscf).max())

    # print(np.abs(WF.h_mo - WF.h_mo.conj().T).max())                          # h Hermiticity
    # print(np.abs(WF.g_mo - WF.g_mo.transpose(2,3,0,1)).max())               # g exchange symmetry
    # print(np.abs(WF.g_mo - WF.g_mo.transpose(1,0,3,2).conj()).max())'''





def split_general_contraction(basis_dict):
    """Convert generally-contracted shells into segmented (nctr=1) shells."""
    new_basis = {}
    for elem, shells in basis_dict.items():
        new_shells = []
        for shell in shells:
            l = shell[0]
            rows = shell[1:]                     # [exp, c1, c2, ..., cN] per primitive
            ncontr = len(rows[0]) - 1
            for col in range(ncontr):
                new_shells.append([l] + [[row[0], row[col+1]] for row in rows])
        new_basis[elem] = new_shells
    return new_basis

def H2():
    geometry = """
                H   1.0215   -0.0252    0.5645
                H   1.1785   -0.5748    1.0355
                """  #0.74
    #basis = "cc-pvdz"
    #J_631g = bse.get_basis('6-31g-J', elements=['H'], fmt='nwchem')
    # raw = {atom: gto.parse(bse.get_basis('6-31G-J', elements=[Z], fmt='nwchem', header=False))
    #    for atom, Z in [('H', 1)]}   # do this per element you use
    # fixed_basis = split_general_contraction(raw)
    # basis = fixed_basis
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    #dyall_cv2z = bse.get_basis('dyall-cv2z', elements=['H'], fmt='nwchem')
    J_6_31g = bse.get_basis('6-31g-J', elements=['H'], fmt='nwchem')
    J_6_311g_pp_ss = bse.get_basis('6-311++g**-J', elements=['H'], fmt='nwchem')
    # with open('dyall2zp_H.nwchem', 'w') as f:
    #     f.write(dyall_v2z)
    #     f.close()
    basis = dyall_v2z
    #basis = dyall_cv2z
    #basis = "sto-3g"
    #basis = "sto-6g"
    #basis = "631-g"
    #basis = "6-311-g"
    #basis = J_6_31g
    #basis = J_6_311g_pp_ss
    #active_space = ((1, 1), 8)
    #active_space = ((1, 1), 6)
    active_space = ((1,1), 2)
    #active_space = ((1,1),4)
    #active_space = (2, 4)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def N2():
    geometry = geometry = """
    N   0.000000   0.000000  -1.097700
    N   0.000000   0.000000   1.097700
                            """  

    #basis = "631-g"
    #dyall_v2z = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    # with open('dyall2zp_H.nwchem', 'w') as f:
    #     f.write(dyall_v2z)
    #     f.close()
    #basis = dyall_v2z
    basis = "sto-3g"
    #basis = "sto-6g"
    active_space = ((7, 7), 14)
    #active_space = ((1, 1), 2)
    #active_space = (2, 4)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def O2():
    geometry = """O  0.0   0.0  0.0;
                  O  0.0  0.0  1.00"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    # with open('dyall2zp_H.nwchem', 'w') as f:
    #     f.write(dyall_v2z)
    #     f.close()
    #basis = dyall_v2z
    basis = "sto-3g"
    #basis = "sto-6g"
    active_space = ((2, 2), 6)
    #active_space = (2, 4)
    charge = 0
    spin = 2
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def H3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    #basis = "def-2-svp"
    active_space = ((2, 1), 3)
    #active_space = (2, 4)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def LiH():
    geometry = """H  0.0   0.0  0.0;
        Li  0.0  0.0  1.3"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
    #basis = "sto-6g"
    active_space = ((1, 1), 4)
    #active_space = ((2,2), 6)
    #active_space = ((2,2), 12)
    #active_space = ((2,2),4)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def BeH2():
    geometry = """Be   0.000000   0.000000   0.000000
                  H    0.000000   0.000000   1.326000
                  H    0.000000   0.000000  -1.326000"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
    #basis = "sto-6g"
    active_space = ((3, 3), 6)
    #active_space = ((1,1), 4)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HF():
    geometry = """H  0.0   0.0  0.0;
                  F  0.0  0.0  0.9168"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H', 'F'], fmt='nwchem')
    #J_631g = bse.get_basis('6-31g-J', elements=['H', 'F'], fmt='nwchem')
    basis = "sto-3g"
    #basis = dyall_v2z
    #basis= J_631g
    #basis = "aug-cc-pvtz-J"
    active_space = ((2, 2), 6)
    #active_space = ((5, 5), 10)
    #active_space = ((1,1), 4)
    #active_space = ((5,5), 12)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def H2O():
    geometry = """
    O  0.0   0.0  0.11779 
    H  0.000001   0.75545  -0.47116;
    H  0.0  -0.75545  -0.47116"""
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
    #basis = "sto-6g"
    #active_space = ((5, 5), 10)
    #active_space = ((5, 5), 14)
    #active_space = ((3,3),8)
    #active_space = ((2, 2), 6)
    active_space = ((1,1),4)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HI():
    geometry = """H  0.0   0.0  0.0;
        I  0.0  0.0  1.60916 """
    #basis = "dyall-v2z"
    basis = "sto-3g"
    active_space = ((27,27), 54)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HBr():
    geometry = """H  0.0   0.0  0.0;
        Br  0.0  0.0  1.41443 """
    #basis = "dyall-v2z"
    basis = "sto-3g"
    active_space = ((18,18), 36)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HCl():
    geometry = """H  0.0   0.0  0.0;
                  Cl  0.0  0.0  1.41443 """  # 1.41443
    #basis = "dyall-v2z"
    basis = "sto-3g"
    #active_space = ((2,2), 6)
    active_space = ((9,9), 18)
    #active_space = ((),)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    
def CuH():
    geometry = """Cu  0.000000   0.000000   0.000000
                  H   0.000000   0.000000   1.463 """  
    #basis = "dyall-v2z"
    basis = "sto-3g"
    #active_space = ((2,2), 6)
    active_space = ((1,1), 4)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def AgH():
    geometry = """Ag  0.000000   0.000000   0.000000
                  H   0.000000   0.000000   1.622 """  
    #basis = "dyall-v2z"
    basis = "sto-3g"
    #active_space = ((2,2), 6)
    active_space = ((2,2), 6)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def N3():
    geometry = """N
                  N 1 1.4823
                  N 1 1.4823 2 49.2 """  
    basis = "6-31g"
    active_space = ((5,4), 18)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )



###RUN SCRIPT###

HF()
