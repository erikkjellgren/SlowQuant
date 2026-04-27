import numpy as np
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



# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive_DHF
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value_energy
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a, DHF_hamiltonian_full_space
from slowquant.unitary_coupled_cluster.generalized_density_matrix_DHF import ( get_orbital_gradient_generalized_real_imag,
get_orbital_gradient_expvalue_real_imag, get_nonsplit_gradient_expvalue, 
get_gradient_finite_diff, get_electronic_energy_generalized, RDM2, get_orbital_response_hessian_block,
)

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)

from slowquant.molecularintegrals.integralfunctions import DHF_one_electron_transform, DHF_two_electron_transform

def unpack_triangular_old(vec):
    """
    Unpack lower-triangular packed Hermitian matrix.
    n*(n+1)/2 = vec.size
    """
    n = int((np.sqrt(1 + 8*vec.size) - 1) // 2)
    assert n*(n+1)//2 == vec.size, \
        f"Size {vec.size} is not a valid triangular number"
    mat = np.zeros((n, n), dtype=complex)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = vec[idx]
            mat[j, i] = vec[idx].conj()
            idx += 1
    return mat

def unpack_triangular(vec):
    """
    Convert packed lower-triangular Hermitian matrix
    into full (n x n) matrix.
    """
    n = int((np.sqrt(1 + 8 * vec.size) - 1) // 2)

    if n * (n + 1) // 2 != vec.size:
        raise ValueError(f"Invalid packed size: {vec.size}")

    mat = np.zeros((n, n), dtype=complex)

    idx = 0
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = vec[idx]
            mat[j, i] = vec[idx].conj()
            idx += 1

    return mat

def build_ukb_mol(molecule):
    """
    Build UKB mol matching DIRAC's basis construction.
    
    Key insight: each primitive in the large component basis
    independently generates small component functions via sigma.p.
    Even if two shells share exponents (SP shells in STO-3G),
    they must each contribute their own small component functions.
    We force this by making every primitive its own shell.
    """

    ukb_basis = {}

    for atom_symbol in set([a[0] for a in molecule._atom]):
        large_basis = molecule._basis[atom_symbol]

        #print(f"\n{atom_symbol} large basis shells:")
        small_shells = []

        for shell in large_basis:
            l = shell[0]
            exponents = [row[0] for row in shell[1:]]
            #print(f"  l={l}, exponents={exponents}")

            for zeta in exponents:
                # l+1 always
                small_shells.append([l + 1, [zeta, 1.0]])
                #print(f"    -> small l={l+1}, zeta={zeta:.6f}")
                # l-1 only if l > 0
                if l > 0:
                    small_shells.append([l - 1, [zeta, 1.0]])
                    #print(f"    -> small l={l-1}, zeta={zeta:.6f}")

        # NO deduplication at all -- every primitive independently
        # generates its own small component functions, matching DIRAC
        ukb_basis[atom_symbol] = large_basis + small_shells

        n_small = sum(2*s[0]+1 for s in small_shells)
        #print(f"  -> {len(small_shells)} small shells, "
        #      f"{n_small} small component functions")

    mol_ukb = gto.M(
        atom=molecule.atom,
        basis=ukb_basis,
        charge=molecule.charge,
        spin=molecule.spin,
        unit=molecule.unit,
        cart = molecule.cart,
    )

    n_large = molecule.nao_nr()
    n_ukb   = mol_ukb.nao_nr()
    n_small = n_ukb - n_large

    print(f"\nLarge component AOs : {n_large}")
    print(f"Small component AOs : {n_small}")
    print(f"UKB total AOs       : {n_ukb}")
    if n_ukb % 4 != 0:
        print(f"Warning: {n_ukb} not divisible by 4")

    return mol_ukb

def inspect_dirac_dict(contents):
    print("\nAll ao_matrices keys and shapes:")
    for key, val in contents["ao_matrices"].items():
        n = val.size
        # Check if valid triangular number
        candidate_n = int((np.sqrt(1 + 8*n) - 1) // 2)
        is_triangular = (candidate_n*(candidate_n+1)//2 == n)
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
              f"size={n}, triangular={is_triangular} "
              f"({'→ '+str(candidate_n)+'x'+str(candidate_n) if is_triangular else ''}) ")

def build_hcore_from_dirac(contents, c=137.035999084):
    """
    Build full 4-component Dirac hcore from DIRAC OPERATORS.h5.
    Returns 64x64 hcore including UKB large + small components.
    """
    ao = contents["ao_matrices"]

    # unpack triangular matrices
    S  = unpack_triangular(ao["OVERLAP TFFT"])
    V  = unpack_triangular(ao["MOLFIELDTFFT"])
    Tx = unpack_triangular(ao["XDIPVEL FTTF"])
    Ty = unpack_triangular(ao["YDIPVEL FTTF"])
    Tz = unpack_triangular(ao["ZDIPVEL FTTF"])

    # kinetic term: T = -c/2 * (XDIPVEL + YDIPVEL + ZDIPVEL)
    T = (-c / 2.0) * (Tx + Ty + Tz)

    # rest mass term: +c^2 for large component, -c^2 for small component
    n = S.shape[0]
    nL = n // 2
    T_rest = np.zeros_like(S)
    T_rest[:nL, :nL] =  c**2 * S[:nL, :nL]
    T_rest[nL:, nL:] = -c**2 * S[nL:, nL:]

    # full 4c hcore
    hcore = T + V + T_rest

    # sanity check
    print("hcore shape:", hcore.shape)
    print("Hermitian:", np.allclose(hcore, hcore.conj().T, atol=1e-8))

    return hcore

def initial_guess_from_hcore(hcore, nocc):
    # Diagonalize hcore
    e, C = np.linalg.eigh(hcore)
    # Occupied orbitals -> first nocc positive-energy eigenvectors
    C_occ = C[:, :nocc]
    # Build density matrix: D = C_occ @ C_occ.H
    D = C_occ @ C_occ.conj().T
    return D

def make_h2_ao(mol):
    """
    Diamagnetic property Hessian integrals for all atom pairs.

    Returns h2 of shape (natm, natm, 3, 3, n4c, n4c), complex
    h2[I, J, alpha, beta, :, :] is the (alpha,beta) component of Z_I Z_J
    Note: h2[I,J] == h2[J,I] (symmetric in atom indices)
    """
    n2c = mol.nao_2c()
    n4c = n2c * 2
    natm = mol.natm

    h2 = np.zeros((natm, natm, 3, 3, n4c, n4c), dtype=np.complex128)

    for I in range(natm):
        for J in range(I, natm):
            orig1 = mol.atom_coord(I)
            orig2 = mol.atom_coord(J)
            a01a01 = sa01sa01_integral(mol, orig1, orig2)  # (3, 3, n2c, n2c)

            block = np.zeros((3, 3, n4c, n4c), dtype=np.complex128)
            block[:, :, n2c:, :n2c] =  0.5 * a01a01
            block[:, :, :n2c, n2c:] =  0.5 * a01a01.conj().transpose(0, 1, 3, 2)

            h2[I, J] = block
            h2[J, I] = block  # symmetric

    return h2  # (natm, natm, 3, 3, n4c, n4c)

def make_h1_ao_old(mol):
    """
    Paramagnetic property gradient integrals for all atoms.

    Returns h1 of shape (natm, 3, n4c, n4c), complex
    h1[I, alpha, :, :] is the alpha-component of Z_I
    """
    n2c = mol.nao_2c()
    n4c = n2c * 2
    natm = mol.natm

    h1 = np.zeros((natm, 3, n4c, n4c), dtype=np.complex128)

    for I in range(natm):
        mol.set_rinv_origin(mol.atom_coord(I))
        a01int = mol.intor('int1e_sa01sp_spinor', comp=3)  # (3, n2c, n2c)
        for k in range(3):
            h1[I, k, :n2c, n2c:] =  0.5 * a01int[k]
            h1[I, k, n2c:, :n2c] =  0.5 * a01int[k].conj().T

    return h1  # (natm, 3, n4c, n4c)


def make_h1_ao_other(mol):
    """
    Construct relativistic SSC operator:
        κ_M = α × (r_M / r_M^3)

    Returns:
        h1: (natm, 3, n4c, n4c) complex array
    """
    n2c = mol.nao_2c()
    n4c = 2 * n2c
    natm = mol.natm

    h1 = np.zeros((natm, 3, n4c, n4c), dtype=np.complex128)

    # --- Pauli matrices ---
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    sigmas = [sigma_x, sigma_y, sigma_z]

    # --- Build α matrices in 4c block form ---
    # α = [[0, σ],
    #      [σ, 0]]
    def alpha_matrix(sigma):
        return np.block([
            [np.zeros((n2c, n2c)), sigma_tensor(sigma, n2c)],
            [sigma_tensor(sigma, n2c), np.zeros((n2c, n2c))]
        ])

    # Expand σ to AO dimension
    def sigma_tensor(sigma, n2c):
        # Each AO has spinor structure → kron with identity over spatial AOs
        nao = n2c // 2
        return np.kron(np.eye(nao), sigma)

    alpha = [alpha_matrix(s) for s in sigmas]  # αx, αy, αz

    for I in range(natm):
        mol.set_rinv_origin(mol.atom_coord(I))

        # r_i / r^3 integrals (spinor)
        # shape: (3, n2c, n2c)
        rinv = mol.intor('int1e_rinv_spinor', comp=3)

        # Build 4c version of r/r^3 operator
        r4 = np.zeros((3, n4c, n4c), dtype=complex)
        for a in range(3):
            r4[a, :n2c, :n2c] = rinv[a]
            r4[a, n2c:, n2c:] = rinv[a]

        # --- Cross product κ = α × r/r^3 ---
        kappa = [None]*3

        # κx = α_y r_z - α_z r_y
        kappa[0] = alpha[1] @ r4[2] - alpha[2] @ r4[1]

        # κy = α_z r_x - α_x r_z
        kappa[1] = alpha[2] @ r4[0] - alpha[0] @ r4[2]

        # κz = α_x r_y - α_y r_x
        kappa[2] = alpha[0] @ r4[1] - alpha[1] @ r4[0]

        for a in range(3):
            h1[I, a] = kappa[a]

    return h1


def make_h1_ao_other_old(mol):
    """
    Construct relativistic SSC perturbation operator h1 in AO basis.

    h1[I, a] corresponds to the operator κ_{I,a} entering:
        << κ_I ; κ_J >>

    Returns:
        h1: (natm, 3, n4c, n4c) complex array
    """
    n2c = mol.nao_2c()
    n4c = 2 * n2c
    natm = mol.natm

    h1 = np.zeros((natm, 3, n4c, n4c), dtype=np.complex128)

    for I in range(natm):
        # Set origin at nucleus I
        mol.set_rinv_origin(mol.atom_coord(I))

        # This is the KEY integral
        # shape: (3, n2c, n2c)
        a01 = mol.intor('int1e_sa01sp_spinor', comp=3)

        # Assemble into 4-component matrix
        for a in range(3):
            # Upper-right block (large → small)
            h1[I, a, :n2c, n2c:] = a01[a]

            # Lower-left block (small → large)
            h1[I, a, n2c:, :n2c] = a01[a].conj().T

            # Other blocks remain zero

    return h1


def make_h1_ao_diff_int(mol):
    """
    Construct the relativistic (alpha x r_M / r_M^3) perturbation operator
    in the 4-component AO basis under Restricted Kinetic Balance (RKB).

    The operator alpha x r_M / r_M^3 is purely off-diagonal in the
    large/small block structure (since alpha is off-diagonal), so only
    the LS and SL blocks are non-zero. The full relativistic content,
    including what becomes diamagnetic in the NR limit, is recovered by
    summing the response over the complete electron + positron spectrum.

    h1[I, a] corresponds to the a-th Cartesian component of the operator
    for nucleus I, entering e.g.:
        << kappa_I ; kappa_J >>

    Returns:
        h1: (natm, 3, n4c, n4c) complex array
    """
    n2c = mol.nao_2c()
    n4c = 2 * n2c
    natm = mol.natm

    h1 = np.zeros((natm, 3, n4c, n4c), dtype=np.complex128)

    for I in range(natm):
        # Set origin at nucleus I
        mol.set_rinv_origin(mol.atom_coord(I))

        # (sigma x r_M / r_M^3)(sigma.p) in spinor basis
        # shape: (3, n2c, n2c)
        a01 = mol.intor('int1e_cg_sa10sp_spinor', comp=3)

        for a in range(3):
            # Upper-right block: LS (large -> small)
            h1[I, a, :n2c, n2c:] = a01[a]

            # Lower-left block: SL (small -> large)
            h1[I, a, n2c:, :n2c] = a01[a].conj().T

    return h1

def make_h1_ao(mol):
    """
    Construct the 4-component relativistic matrix elements of

        (alpha x r_M) / r_M^3

    in the AO spinor basis under Restricted Kinetic Balance (RKB).

    Uses int1e_sa01sp_spinor = (nabla-rinv cross sigma | sigma dot p),
    which gives sigma x (r_M / r_M^3) with RKB encoded in the sp suffix.
    Since nabla(1/r_M) = -r_M/r_M^3, the sign is:

        nabla-rinv cross sigma = -r_M/r_M^3 x sigma = sigma x r_M/r_M^3

    The operator is purely off-diagonal (LS/SL only) since alpha is
    off-diagonal in the large/small block structure.

    Returns:
        h1: (natm, 3, n4c, n4c) complex array
    """
    #c = lib.param.LIGHT_SPEED
    
    n2c = mol.nao_2c()
    n4c = 2 * n2c
    natm = mol.natm

    h1 = np.zeros((natm, 3, n4c, n4c), dtype=np.complex128)

    for I in range(natm):
        mol.set_rinv_origin(mol.atom_coord(I))

        # (nabla-rinv cross sigma | sigma dot p)
        # = (sigma x r_M/r_M^3 | sigma dot p)  [RKB included via sp]
        # shape: (3, n2c, n2c)
        a01 = mol.intor('int1e_sa01sp_spinor', comp=3)  *.5#*(0.25/c**2)  

        for a in range(3):
                h1[I, a, :n2c, n2c:] = a01[a]      # LS block
                h1[I, a, n2c:, :n2c] = a01[a].conj().T  # SL block

    return h1


def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, cart = False)

    #mol = build_ukb_mol(mol_init)


    #uhf = scf.UHF(mol)
    #uhf.kernel()



    mf = scf.DHF(mol)
    mf.conv_tol = 1e-8        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 500
    #mf.with_ssss


    #DIRAC_dict = read_dirac_file("OPERATORS.h5")

    #hcore_dirac = build_hcore_from_dirac(DIRAC_dict)

    #mf.get_hcore = lambda *args: hcore_dirac

    #mf.get_init_guess = lambda *args: initial_guess_from_hcore(hcore_dirac, mol.nelectron)


    mf.kernel()


    sscobj = SSC(mf)
    sscobj.cphf = True
    sscobj.mb = "RMB"
    sscobj.verbose = 5
    jj = sscobj.kernel()



    # print("PySCF e11 (raw, before Hz conversion):", jj)
    # print("PySCF Tr(e11)/3:", np.trace(jj[0])/3)



    # Getting integrals for response:
    # Paramagnetic: (natm, 3, n4c, n4c)
    h1 = make_h1_ao(mol)

    #print(np.linalg.norm(h1))

    # Diamagnetic: (natm, natm, 3, 3, n4c, n4c)
    h2 = make_h2_ao(mol)






    C_MO=np.array(mf.mo_coeff,dtype=complex)

    c = lib.param.LIGHT_SPEED

    h_core = mf.get_hcore()
    g_eri = np.array([mol.intor("int2e_spinor"), mol.intor('int2e_spsp1spsp2_spinor')*(0.0625/c**4),
                      mol.intor('int2e_spsp2_spinor')*(0.25/c**2), mol.intor('int2e_spsp1_spinor')*(0.25/c**2)],dtype=np.complex128)

    #dip_int = mol.intor("int1e_r")

    size = int(C_MO.shape[1]/2)


    # small random anti-Hermitian
    eps = 0.0001  # controls "step size"
    X_anti = np.random.randn(C_MO.shape[0],C_MO.shape[0]) + 1j*np.random.randn(C_MO.shape[0],C_MO.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    U_step = expm(A_mat)

    C_U = C_MO @ U_step


    WF = GeneralizedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        C_MO,
        #C_U,
        h_core,
        g_eri,
        "fuccsd",
        {"n_layers":1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )


    print("DHF", mf.energy_elec()[0])

    h_mo = DHF_one_electron_transform(C_MO, h_core)

    g_mo = DHF_two_electron_transform(C_MO, g_eri)

    H = DHF_hamiltonian_full_space(h_mo[size:,size:], g_mo[size:,size:,size:,size:], WF.num_spin_orbs_NES)

    #H = generalized_hamiltonian_full_space(h_mo, g_mo, WF.num_spin_orbs)


    #print(WF.rdm1)

    E_tester = get_electronic_energy_generalized(
                h_mo,
                g_mo,
                WF.num_spin_orbs_NES,
                WF.num_inactive_spin_orbs,
                WF.num_active_spin_orbs,
                WF.rdm1,
                WF.rdm2,
            )
    
    print(E_tester)
    #print(_visscher_ssss_correction(mf,mf.make_rdm1()))
    #print(E_tester + _visscher_ssss_correction(mf,mf.make_rdm1()))


    E2 = generalized_expectation_value_energy(
                WF.ci_coeffs,
                # [generalized_hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_spin_orbs, self.num_active_spin_orbs)],
                [H],
                WF.ci_coeffs,
                WF.ci_info,
            )

    print(E2)



    print("Nr. of kappas:", len(WF.kappa_spin_idx))


    print("Nr. of spin orbitals:", WF.num_spin_orbs)
    print("Nr. of inactive spin orbitals:", WF.num_inactive_spin_orbs)
    print("Nr. of active spin orbitals:", WF.num_active_spin_orbs)
    print("Nr. of virtual spin orbitals:", WF.num_virtual_spin_orbs)
    print("Nr. of positronic spin orbitals:", WF.num_spin_orbs_NES)
    print("Inactive spin orbitals idx:", WF.inactive_spin_idx)
    print("Active spin orbitals idx:", WF.active_spin_idx)
    print("Virtual spin orbitals idx:", WF.virtual_spin_idx)
    print("Positronic spin orbitals idx:", WF.positronic_spin_idx)
    print("Active occupied:",WF.active_occ_spin_idx)
    print("Active unoccupied:",WF.active_unocc_spin_idx)
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

    WF.run_wf_optimization_2step_DHF(optimizer_name = "l-bfgs-b", orbital_optimization = True,tol = 1e-10, maxiter = 1000)

    #kappas = np.concatenate([WF.kappa_real, WF.kappa_real_ep, WF.kappa_imag, WF.kappa_imag_ep])

    gradient_test = get_orbital_gradient_generalized_real_imag(
                WF.h_mo,
                WF.g_mo,
                WF.kappa_spin_idx,
                WF.num_spin_orbs_NES,
                WF.num_inactive_spin_orbs,
                WF.num_active_spin_orbs,
                WF.rdm1,
                WF.rdm2,
            )
    #print(np.round(gradient_test,5))

    gradient_test_ep = get_orbital_gradient_generalized_real_imag(
            WF.h_mo,
            WF.g_mo,
            WF.kappa_spin_idx_ep,
            WF.num_spin_orbs_NES,
            WF.num_inactive_spin_orbs,
            WF.num_active_spin_orbs,
            WF.rdm1,
            WF.rdm2,
        )
    #print(np.round(gradient_test_ep,5))


    gradient_test_ep = get_orbital_gradient_generalized_real_imag(
        WF.h_mo,
        WF.g_mo,
        WF.kappa_no_activeactive_spin_idx_resp,
        WF.num_spin_orbs_NES,
        WF.num_inactive_spin_orbs,
        WF.num_active_spin_orbs,
        WF.rdm1,
        WF.rdm2,
    )
    #print(np.round(gradient_test_ep,5))

    hess = get_orbital_response_hessian_block(
        WF.h_mo,
        WF.g_mo,
        WF.kappa_no_activeactive_spin_idx_dagger,
        WF.kappa_no_activeactive_spin_idx,
        WF.num_spin_orbs_NES, 
        WF.num_inactive_spin_orbs,
        WF.num_active_spin_orbs,
        WF.rdm1,
        WF.rdm2,
        )
    
    print(f"Hermiticity check of the Hessian: max|E2 - E2†| = "
            f"{np.max(np.abs(hess - hess.conj().T)):.2e}")  

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

    LR = generalized_naive_DHF.LinearResponse(WF, excitations="S")
    LR.calc_excitation_energies()
    print("Excitation energies:", LR.excitation_energies)
    #print(np.round(LR.get_transition_dipole(dip_int).real,5))
    #print(LR.get_oscillator_strengths(dip_int))
    SSCC = LR.get_SSCC_4comp_iso(h1, h2)
    for I in range(SSCC.shape[0]):
        for J in range(I+1, SSCC.shape[1]):
            print(f"K({mol.atom_symbol(I)}{I} - {mol.atom_symbol(J)}{J}) = {SSCC[I,J]:.4f} Hz")



def H2():
    geometry = """H  0.0   0.0  0.0;
                  H  0.0  0.0  0.74"""
    #basis = "cc-pvdz"
    basis = "631-g"
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    # with open('dyall2zp_H.nwchem', 'w') as f:
    #     f.write(dyall_v2z)
    #     f.close()
    #basis = dyall_v2z
    #basis = "sto-3g"
    #basis = "sto-6g"
    active_space = ((1, 1),2)
    #active_space = (2, 4)
    charge = 0
    spin = 0
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
    #basis = ""
    active_space = ((2, 1),6)
    #active_space = (2, 4)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def LiH():
    geometry = """H  0.0   0.0  0.0;
        Li  0.0  0.0  1"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
    active_space = ((2, 2), 8)
    #active_space = (2, 4)
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
    basis = "sto-3g"
    #active_space = ((1, 1), 4)
    active_space = ((2, 2), 6)
    #active_space = (2, 4)
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
    #active_space = ((5, 5), 14)
    #active_space = ((3,3),8)
    active_space = ((2, 2), 6)
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
    active_space = (4, 6)
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
    active_space = ((3,3), 8)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HCl():
    geometry = """H  0.0   0.0  0.0;
                  Cl  0.0  0.0  1.41443 """
    #basis = "dyall-v2z"
    basis = "sto-3g"
    #active_space = ((2,2), 6)
    active_space = ((9,9), 18)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    
###RUN SCRIPT###

H2O()
