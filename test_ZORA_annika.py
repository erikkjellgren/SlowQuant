import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c, lib, gto
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
import basis_set_exchange as bse
from zora_build_new import read_zora_so
from zora_build_new_sf import read_zora_sf
import struct
from pyscf.dft import mura_knowles, gen_grid, gauss_chebyshev


# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value_energy
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a
from slowquant.unitary_coupled_cluster.generalized_density_matrix import get_orbital_gradient_generalized_real_imag, get_orbital_gradient_expvalue_real_imag, get_nonsplit_gradient_expvalue, get_gradient_finite_diff, get_electronic_energy_generalized

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)

from slowquant.molecularintegrals.integralfunctions import DHF_one_electron_transform, DHF_two_electron_transform



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
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, cart = False)
    mol.build()
    #X2C
    uhf = scf.UHF(mol)
    uhf.chkfile = "/home/annika4ee/SlowQuant/uhf_guess.chk"
    #uhf.scf()
    uhf.kernel()

    nmo = uhf.mo_coeff[0].shape[1]

    # small random anti-Hermitian
    epsilon = 0  # controls "step size"
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
    #chkfile.dump('/home/annika4ee/SlowQuant/uhf_guess.chk','scf', data)


    
    # RKS:
    h_core_pyscf_1dim = mol.intor('int1e_kin') + mol.intor('int1e_nuc')

    H_zora_so, H_zora_sf = read_zora_so("HI_SO.zora_so")

    h_core_sf_1dim = H_zora_sf[:mol.nao,:mol.nao]

    
    mf_nr = scf.RKS(mol)
    mf_nr.xc = 'PBE0'
    mf_nr.conv_tol = 1e-8
    mf_nr.conv_tol = 1e-8        # Energy convergence (Hartree)
    mf_nr.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf_nr.max_cycle = 1000
    mf_nr.grids.atom_grid = {
    'H':  (99, 434),
    'I':  (99, 434),
    }
    #mf_nr.grids.prune = None
    mf_nr.grids.radi_method = mura_knowles 
    #mf_nr.grids.radii_adjust = None
    mf_nr.grids.becke_scheme = gen_grid.stratmann 
    mf_nr.grids.build()

    mf_nr.kernel()




    mf_nr = scf.RKS(mol)
    mf_nr.xc = 'PBE0'
    mf_nr.conv_tol = 1e-8        # Energy convergence (Hartree)
    mf_nr.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf_nr.max_cycle = 1000
    mf_nr.grids.atom_grid = {
    'H':  (99, 434),
    'I':  (99, 434),
    }
    #mf_nr.grids.prune = None
    mf_nr.grids.radi_method = mura_knowles 
    #mf_nr.grids.radii_adjust = None
    mf_nr.grids.becke_scheme = gen_grid.stratmann 
    mf_nr.grids.build()

    mf_nr.get_hcore = lambda *args: h_core_pyscf_1dim + h_core_sf_1dim

    mf_nr.kernel()



    chkfile.dump('/home/annika4ee/SlowQuant/uhf_guess.chk','dft', mf_nr.mo_coeff)





    mf = scf.GKS(mol)
    mf.xc = "pbe0"
    #mf.chkfile = '/home/annika4ee/SlowQuant/uhf_guess.chk'

    # Change initial guess:
    #mf.init_guess = "chkfile"
    mf.conv_tol = 1e-8        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 1000
    # mf.grids.atom_grid = {
    #     'H':  (70, 194),
    #     'Li': (70, 194),
    # }

    mf.grids.atom_grid = {
    'H':  (99, 434),
    'I':  (99, 434),
    }

    # Key: control small density cutoff
    #mf.grids.eps = 1e-14          # default 1e-14; larger -> lower accuracy

    # Key: control XC functional interpolation
    #mf._numint._cint.libcint_precision = 1e-14   # controls integral precision
    mf.grids.prune = True
    mf.grids.radi_method = mura_knowles 
    mf.grids.radii_adjust = None
    mf.grids.becke_scheme = gen_grid.stratmann 
    mf.grids.build()

    
    # T = mol.intor("int1e_kin")
    # T_2c = np.kron(np.eye(2), T)

    # V = mol.intor("int1e_nuc")
    # V_2c = np.kron(np.eye(2), V)

    # px = -1j * mol.intor('int1e_ipovlp', comp=3)[0]
    # py = -1j * mol.intor('int1e_ipovlp', comp=3)[1]
    # pz = -1j * mol.intor('int1e_ipovlp', comp=3)[2]

    # PxVPx = px.conj().T @ V @ px
    # PyVPy = py.conj().T @ V @ py
    # PzVPz = pz.conj().T @ V @ pz

    # PxVPy = px.conj().T @ V @ py
    # PxVPz = px.conj().T @ V @ pz

    # PyVPx = py.conj().T @ V @ px
    # PyVPz = py.conj().T @ V @ pz

    # PzVPx = pz.conj().T @ V @ px
    # PzVPy = pz.conj().T @ V @ py

    # c = lib.param.LIGHT_SPEED

    # sigma_x = np.array([[0, 1],
    #                 [1, 0]], dtype=complex)

    # sigma_y = np.array([[0, -1j],
    #                 [1j,  0]], dtype=complex)

    # sigma_z = np.array([[1,  0],
    #                 [0, -1]], dtype=complex)
    
    # scalar = PxVPx + PyVPy + PzVPz

    # scalar_2c = np.kron(np.eye(2), scalar)

    # Wx = PyVPz - PzVPy
    # Wy = PzVPx - PxVPz
    # Wz = PxVPy - PyVPx

    # soc_2c = (np.kron(sigma_x, 1j * Wx) + np.kron(sigma_y, 1j * Wy) + np.kron(sigma_z, 1j * Wz))


    # W_0 = 0.25*(1/c**2)*(scalar_2c+soc_2c) # Remember to put back the soc_2c!!

    # #W = np.linalg.inv(np.linalg.inv(W_0) - np.linalg.inv(T_2c))
 
    # #hcore = T_2c + W + V_2c





    h_core_pyscf = np.kron(np.eye(2), h_core_pyscf_1dim)

    h_core_tot = (h_core_pyscf + H_zora_so)

    mf.get_hcore = lambda *args: h_core_tot

    mf.kernel()




    # # Build overlap matrix and compute AO norms
    # S = mol.intor('int1e_ovlp')        # AO overlap (spatial)
    # norms = np.sqrt(np.diag(S))        # 1D array of length nAO
    # # ==============================
    # # 3. Scale H_ZORA_corr to PySCF AO norms
    # # ==============================
    # # Repeat norms for 2c spinors (alpha, beta)
    # norms_2c = np.tile(norms, 2)  # length = 2*nAO
    # # Vectorized scaling
    # H_ZORA_scaled = H_zora / (norms_2c[:, None] * norms_2c[None, :])


    # One-electron energy: T + Vne



    



    # Get 2c density matrix
    dm0 = mf.make_rdm1()
    dms = [dm0]

    # One-electron energy
    hcore = mf.get_hcore()
    e_one = (dm0 @ hcore).trace().real

    # Exchange-Correlation energy — use get_veff and extract exc directly
    veff = mf.get_veff(mf.mol, dm0)
    exc = veff.exc  # XC energy stored on the veff object

    # Coulomb energy
    ecoul = veff.ecoul

    # Nuclear repulsion
    e_nuc = mf.mol.energy_nuc()

    # Total energy
    e_tot = e_one + ecoul + exc + e_nuc

    print("=== 2c GKS Energy Breakdown ===")
    print(f"One electron energy      = {e_one.real:20.12f} Eh")
    print(f"Coulomb energy           = {ecoul:20.12f} Eh")
    print(f"Exchange-Corr. energy    = {exc:20.12f} Eh")
    print(f"Nuclear repulsion energy = {e_nuc:20.12f} Eh")
    print(f"Total DFT energy         = {e_tot.real:20.12f} Eh")
    print("================================\n")





    # # Standard overlap from PySCF
    # S = mol.intor('int1e_ovlp')
    # S_4x4 = np.kron(np.eye(2), S)

    # # Scalar ZORA scaling (block diagonal, alpha and beta)
    # S_scale = np.zeros((2*nbf, 2*nbf))
    # S_scale[:nbf, :nbf] = zora_scale_sf[0]  # alpha block
    # S_scale[nbf:, nbf:] = zora_scale_sf[1]  # beta block

    # # SO ZORA scaling from ga_dens_so source:
    # # z: D_im[bb] - D_im[aa]  -> diagonal blocks, imaginary
    # # x: -D_im[ab] - D_im[ba] -> off-diagonal blocks, imaginary  
    # # y: D_re[ab] - D_re[ba]  -> off-diagonal blocks, real
    # # soxyz = ['z','y','x'] -> scale_so(1)=z, scale_so(2)=y, scale_so(3)=x

    # S_scale_so = np.zeros((2*nbf, 2*nbf), dtype=complex)

    # # z component: diagonal blocks, imaginary
    # S_scale_so[:nbf, :nbf] -= 1j * zora_scale_so[0]  # -i*z in alpha block
    # S_scale_so[nbf:, nbf:] += 1j * zora_scale_so[0]  # +i*z in beta block

    # # y component: off-diagonal blocks, real
    # S_scale_so[:nbf, nbf:] += zora_scale_so[1]   # +y in alpha-beta block
    # S_scale_so[nbf:, :nbf] -= zora_scale_so[1]   # -y in beta-alpha block

    # # x component: off-diagonal blocks, imaginary
    # S_scale_so[:nbf, nbf:] -= 1j * zora_scale_so[2]  # -i*x in alpha-beta block
    # S_scale_so[nbf:, :nbf] -= 1j * zora_scale_so[2]  # -i*x in beta-alpha block

    # # Full modified overlap
    # S_zora = S_4x4.astype(complex) + S_scale.astype(complex) + S_scale_so

    # # Set modified overlap in PySCF
    # #mf.spin_square = lambda *args: (0.0, 1.0)  # skip spin_square check
    # #mf.get_ovlp = lambda *args: S_zora

    # def compute_zora_scaling_correction(mf, nbf, zora_scale_sf, zora_scale_so):
    #     """
    #     Compute scaled ZORA energy correction from van Lenthe, Baerends, Snijders
    #     J. Chem. Phys. 101, 9783 (1994) and NWChem dft_zora_scale_so source.

    #     E_scaled = E_ZORA + Σ_i [ -ε_i * s_i / (1 + s_i) ]

    #     where s_i = <ψ_i|scale_sf(a)|ψ_ia> + <ψ_i|scale_sf(b)|ψ_ib>
    #             + <ψ_i|scale_so(z)|ψ_iz> + <ψ_i|scale_so(y)|ψ_iy>
    #             + <ψ_i|scale_so(x)|ψ_ix>

    #     following exactly NWChem's dft_zora_scale_so with soxyz=['z','y','x']
    #     """
    #     mo_coeff  = mf.mo_coeff
    #     mo_energy = mf.mo_energy
    #     mo_occ    = mf.mo_occ

    #     ener_scal = 0.0
    #     for iorb in range(len(mo_occ)):
    #         if mo_occ[iorb] > 0:
    #             psi   = mo_coeff[:, iorb]
    #             psi_a = psi[:nbf]
    #             psi_b = psi[nbf:]

    #             # Density matrix blocks D = psi psi† (one orbital)
    #             D_re_aa = np.outer(psi_a.real, psi_a.real) + np.outer(psi_a.imag, psi_a.imag)
    #             D_re_bb = np.outer(psi_b.real, psi_b.real) + np.outer(psi_b.imag, psi_b.imag)
    #             D_re_ab = np.outer(psi_a.real, psi_b.real) + np.outer(psi_a.imag, psi_b.imag)
    #             D_re_ba = np.outer(psi_b.real, psi_a.real) + np.outer(psi_b.imag, psi_a.imag)
    #             D_im_aa = np.outer(psi_a.imag, psi_a.real) - np.outer(psi_a.real, psi_a.imag)
    #             D_im_bb = np.outer(psi_b.imag, psi_b.real) - np.outer(psi_b.real, psi_b.imag)
    #             D_im_ab = np.outer(psi_a.imag, psi_b.real) - np.outer(psi_a.real, psi_b.imag)
    #             D_im_ba = np.outer(psi_b.imag, psi_a.real) - np.outer(psi_b.real, psi_a.imag)

    #             # ga_dens_sf: scalar density contributions
    #             ener_sf = (np.sum(D_re_aa * zora_scale_sf[0]) +
    #                     np.sum(D_re_bb * zora_scale_sf[1]))

    #             # ga_dens_so: SO density contributions
    #             # soxyz = ['z','y','x'] -> scale_so(1)=z, scale_so(2)=y, scale_so(3)=x
    #             dens_so_z =  D_im_bb - D_im_aa          # z component
    #             dens_so_y =  D_re_ab - D_re_ba          # y component
    #             dens_so_x = -D_im_ab - D_im_ba          # x component

    #             ener_so = (np.sum(dens_so_z * zora_scale_so[0]) +
    #                     np.sum(dens_so_y * zora_scale_so[1]) +
    #                     np.sum(dens_so_x * zora_scale_so[2]))

    #             # s_i = total scaling expectation value
    #             ener_tot   = ener_sf + ener_so

    #             # NWChem formula: ener_scal -= (ε_i / (1 + s_i)) * s_i
    #             zora_denom = 1.0 + ener_tot
    #             eval_scal  = mo_energy[iorb] / zora_denom
    #             ener_scal -= eval_scal * ener_tot * mo_occ[iorb]

    #     return ener_scal




    

    

    # print("h_core_pyscf shape:", h_core_pyscf.shape)
    # print("H_zora shape:", H_zora.shape)
    # print("h_core_pyscf[0,0]:", h_core_pyscf[0,0])
    # print("H_zora[0,0]:", H_zora[0,0])
    # print("h_core_tot[0,0]:", h_core_tot[0,0])
    # print("H_zora max diagonal:", np.max(np.abs(np.diag(H_zora).real)))
    # print("H_zora is Hermitian:", np.max(np.abs(H_zora - H_zora.conj().T)))

 



    c_MO=np.array(mf.mo_coeff,dtype=complex)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h_1e = mol.intor("int1e_kin")
    h_nuc = mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    dip_int = mol.intor("int1e_r")
    #mc = mcscf.CASCI(mf, active_space[1], active_space[0])

    # mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # # Slowquant

     # small random anti-Hermitian
    eps = 0.5  # controls "step size"
    X_anti = np.random.randn(c_MO.shape[0],c_MO.shape[0]) + 1j*np.random.randn(c_MO.shape[0],c_MO.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    U_step = expm(A_mat)

    c_u = c_MO @ U_step

    print("MAX", np.max(c_MO.imag))
    #print(mf.mo_coeff)



    WF = GeneralizedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        #c_u,
        c_MO,
        #h_core,
        h_core_pyscf,
        #h_core_tot,
        g_eri,
        "fuccsd",
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )


    #rhf = scf.RHF(mol)

    #rhf.kernel()

    print(mf.energy_elec()[0])
    print(mf.energy_nuc())
    #print(mf.energy_elec()[0]+0.715104390540)
    print(mol.nao)

    # After mf.kernel():
    #ener_scal = compute_zora_scaling_correction(mf, nbf, zora_scale_sf, zora_scale_so)
    #E_scaled = mf.e_tot + ener_scal



    #print(test_energy)

    '''E_tester = get_electronic_energy_generalized(
                WF.h_mo,
                WF.g_mo,
                WF.num_inactive_spin_orbs,
                WF.num_active_spin_orbs,
                WF.rdm1,
                WF.rdm2,
            )
    
    print(E_tester)'''


    print("Nr. of kappas:", len(WF.kappa_spin_idx))
    print("Nr. of spin orbitals:", WF.num_spin_orbs)
    print("Nr. of inactive spin orbitals:", WF.num_inactive_spin_orbs)
    print("Nr. of active spin orbitals:", WF.num_active_spin_orbs)
    print("Nr. of virtual spin orbitals:", WF.num_virtual_spin_orbs)


    '''H=generalized_hamiltonian_full_space(WF.h_mo, WF.g_mo, WF.num_spin_orbs)
    H2=generalized_hamiltonian_0i_0a(WF.h_mo, WF.g_mo, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs)
    H3=generalized_hamiltonian_1i_1a(WF.h_mo, WF.g_mo, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs, WF.num_virtual_spin_orbs)

    test_energy=expectation_value(WF.ci_coeffs, [H], WF.ci_coeffs, WF.ci_info)
    test_energy2=expectation_value(WF.ci_coeffs, [H2], WF.ci_coeffs, WF.ci_info)
    test_energy3=expectation_value(WF.ci_coeffs, [H3], WF.ci_coeffs, WF.ci_info)

    print(test_energy)
    print(test_energy2)
    print(test_energy3)'''


    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 10000)


    '''E_tester = get_electronic_energy_generalized(
            WF.h_mo,
            WF.g_mo,
            WF.num_inactive_spin_orbs,
            WF.num_active_spin_orbs,
            WF.rdm1,
            WF.rdm2,
        )
    
    print(E_tester)'''



    LR = generalized_naive.LinearResponse(WF, excitations="SD")  #, excitations="SD")
    LR.calc_excitation_energies()
    print("Excitation energies")
    print(LR.excitation_energies)
    print("Transition dipole moments")
    print(np.round(LR.get_transition_dipole(dip_int).real,5))
    print("Oscillator strengths")
    print(np.round(LR.get_oscillator_strengths(dip_int),5))



def h2():
    geometry = """H  0.0   0.0  -0.37;
                  H  0.0  0.0  0.37"""
    #basis = "cc-pvtz"
    basis = "631-g"
    #basis = "sto-3g"
    #basis = "sto-6g"
    dyall2zp_H = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    with open('dyall2zp_H.nwchem', 'w') as f:
        f.write(dyall2zp_H)
        f.close()
    #basis = {'H': gto.basis.load('dyall2zp_H.nwchem', 'H')}
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

def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-6g"
    #basis = ""
    active_space = ((2, 1), 12)
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
    #basis = "631-g"
    basis = "sto-3g"
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
    all_bases = bse.get_all_basis_names()
    dyall = [b for b in all_bases if 'dyall' in b.lower()]
    #print(dyall)
    dyall2zp_H = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    dyall2zp_O = bse.get_basis('dyall-v2z', elements=['O'], fmt='nwchem')
    with open('dyall2zp_H.nwchem', 'w') as f:
        f.write(dyall2zp_H)
        f.close()
    with open('dyall2zp_O.nwchem', 'w') as f:
        f.write(dyall2zp_O)
        f.close()
    basis = {'H': gto.basis.load('dyall2zp_H.nwchem', 'H'),'O': gto.basis.load('dyall2zp_O.nwchem', 'O')}
    #basis = "dyallv2z"
    #basis = "cc-pvdz"
    #basis = "631-g"
    #basis = "sto-3g"
    #basis = "sto-6g"
    #active_space = ((5, 5), 14)
    active_space = ((2,2),6)
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
        I  0.0  0.0  3.04 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    dyall = bse.get_basis('dyall-acv3z', elements=['I'], fmt='nwchem')
    with open('dyall.nwchem', 'w') as f:
        f.write(dyall)
        f.close()
    basis = {
    'H':  "def2-svp",
    #'I': bse.get_basis('jorge-dzp-zora', elements=['I'], fmt='nwchem'),
    #'I':  "def2-tzvppd",
    #'I': "tzp-zora",
    'I':  gto.basis.load('dyall_I.nw', 'I'),
    }
    active_space = (4, 6)
    charge = 0
    spin = 0

    #print("Restricted HI")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    print("Nonrelativistic HI")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="au"
    )

def HCl():
    geometry = """H  0.0   0.0  0.0;
        Cl  0.0  0.0  1.1275 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    basis = "sto-3g"
    active_space = ((9,9), 20)
    charge = 0
    spin = 0
    #print("Restricted HBr")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    #print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom",
    )

def HBr():
    geometry = """H  0.0   0.0  0.0;
        Br  0.0  0.0  2.672 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    #basis = "sto-6g"
    basis ="def2-svp"
    #basis = "sto-3g"
    #basis = "cc-pvtz"
    # basis = {
    # 'H':  gto.basis.load('basis_hbr.nw', 'H'),
    # 'Br': gto.basis.load('basis_hbr.nw', 'Br'),
    # }
    # basis = {
    # 'H':  gto.basis.load('basis_hbr.nw', 'H'),
    # 'Br': gto.basis.load('basis_hbr.nw', 'Br'),
    # }
    # basis = {
    # 'H':  gto.basis.load('nwchem_def2svp.nw', 'H'),
    # 'Br': gto.basis.load('nwchem_def2svp.nw', 'Br'),
    # }
    
    #active_space = ((18,18), 38)
    active_space = ((1,1), 4)
    charge = 0
    spin = 0
    #print("Restricted HBr")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    #print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="au",
    )

def HF():
    geometry = """H  0.0   0.0  0.0;
        F  0.0  0.0  1.7328 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    #basis = "sto-6g"
    #basis ="def2-svp"
    #basis = "cc-pvdz"
    basis = "6-31g"
    #active_space = ((18,18), 38)
    active_space = ((1,1), 4)
    charge = 0
    spin = 0
    #print("Restricted HBr")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    #print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="au",
    )
    
def LiH():
    geometry = """H  0.0   0.0  0.0;
                  Li  0.0  0.0  3.015 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    basis = "sto-3g"
    #basis ="def2-svp"
    #basis = "cc-pvdz"
    #basis = "6-31g"
    #active_space = ((18,18), 38)
    active_space = ((1,1), 4)
    charge = 0
    spin = 0
    #print("Restricted HBr")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    #print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="au",
    )

def OH():
    geometry = """O  0.0   0.0  0.0;
                  H  0.0  0.0  1.833 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    basis = "sto-3g"
    #basis ="def2-svp"
    #basis = "cc-pvdz"
    #basis = "6-31g"
    #active_space = ((18,18), 38)
    active_space = ((1,1), 4)
    charge = -1
    spin = 2
    #print("Restricted HBr")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    #print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="au",
    )


###SPIN ELLER RUMLIGE ORBITALER###

h2()


# h2o()
# HI()
# HBr()