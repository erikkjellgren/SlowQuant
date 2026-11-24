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
from pyscf.data import nist
from pyscf.tools import dump_mat


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
    print("Active space:", {active_space})
    print("Basis:", basis)
    print("Spin:", spin)
    print("Charge:", charge)
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, cart = False, nucmod='gaussian')

    Etol = 1e-10
    Gtol = 1e-8
    Maxiter_DHF = 1000

    hf.remove_overlap_zero_eigenvalue = False
    mf = scf.dhf.DHF(mol)
    mf.conv_tol = Etol        # Energy convergence (Hartree)
    mf.conv_tol_grad = Gtol   # Optional: gradient convergence
    mf.max_cycle = Maxiter_DHF

    print("PySCF settings:")
    print("Energy tolerance:", Etol)
    print("Gradient tolerance:", Gtol)
    print("Max iterations:", Maxiter_DHF)

    mf.kernel()

    # Coupling constants PySCF:
    sscobj = SSC(mf)
    sscobj.cphf = True
    sscobj.conv_tol = 1e-9
    sscobj.mb = "RKB"
    sscobj.verbose = 5
    sscobj.with_fcsd = True
    jj = sscobj.kernel()

    # Build the K matrix the same way PySCF does
    mol = sscobj.mol
    nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
    unit = nist.HARTREE2J / nist.PLANCK * nuc_magneton**2
    iso_K = unit * np.einsum('kii->k', jj) / 3

    ktensor = np.zeros((mol.natm, mol.natm))
    for k, (i, j) in enumerate(sscobj.nuc_pair):
        ktensor[i, j] = ktensor[j, i] = iso_K[k]

    # Shieldings PySCF:
    nmr = nmr_dhf.NMR(mf)
    nmr.cphf = True
    nmr.mb = 'RMB'      # or 'RKB'
    nmr.gauge_orig = None  # GIAO vs. # [0,0,0]

    shielding = nmr.kernel()

    sigma_iso = np.trace(shielding, axis1=1, axis2=2) / 3
    print(sigma_iso)

    C_MO=np.array(mf.mo_coeff,dtype=complex)

    print("Max imag c_mo:",np.max(np.abs(C_MO.imag)))

   
    # Kramers restricted operators?
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

    
    tol = 1e-14
    max_iter = 10000
    act_kappa = True
    spin_conserving = False
    orb_opt = True
    optimizer = "l-bfgs-b"
    nl = 1


    #data = np.load("H2_dyall_v2z_new_116.npz")

    # WF object:
    WF = GeneralizedWaveFunctionUPS(
        active_space,
        #data["c_mo"],
        C_MO,
        mol, 
        K_pairs,
        False,
        "fUCCSD",
        {"n_layers": nl, "is_spin_conserving" : spin_conserving},
        include_active_kappa=act_kappa,
    )


    #WF.set_thetas(data["thetas_real"], data["thetas_imag"])

    Tr_seed = 20
    Ti_seed = 40
    
    if len(WF.thetas) > 0:
        np.random.seed(Tr_seed)
        real = np.random.uniform(-0.05,0.05,len(WF.thetas_real))

        np.random.seed(Ti_seed)
        imag = np.random.uniform(-0.05,0.05,len(WF.thetas_real))

        #imag = np.zeros_like(WF.thetas_imag)

        WF.set_thetas(real, imag)


    print("DHF", mf.energy_elec()[0])
    print("Shielding method:", "RMB GIAO")
    print("Tolerance:", tol)
    print("Max iterations:", max_iter)
    print("Spin conservation:", spin_conserving)
    print("Active kappa", act_kappa)
    print("Optimizer:", optimizer)
    print("Orbital optimization:", orb_opt)
    print("Theta real seed:", Tr_seed)
    print("Theta imag seed:", Ti_seed)
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
    print("Active occupied:", WF.active_occ_spin_idx)
    print("Active unoccupied:", WF.active_unocc_spin_idx)
    #print("qs: ", WF2.kappa_no_activeactive_spin_idx_resp)


    # Optimization:
    WF.run_wf_optimization_2step_DHF(optimizer_name = optimizer, orbital_optimization = orb_opt, tol = tol, maxiter = max_iter)

    print("Max imag c_mo:", np.max(np.abs(WF.c_mo.imag)))
    print("Max imag theta:", np.max(np.abs(WF.thetas_imag)))

    #  WF:
    np.savez(
        "H2_dyall_UCCCSD_test",
        c_mo=WF.c_mo,
        thetas_real=WF.thetas_real,
        thetas_imag=WF.thetas_imag
        )

    print("Electronic energy:", WF.energy_elec)

    LR = generalized_naive_DHF.LinearResponse(WF, excitations="SD", screen = True)
    LR.calc_excitation_energies()

    print("PySCF shieldings:", sigma_iso)

    LR.get_shieldings_4comp_iso(RMB_GIAO = True)


    label = ['%2d %-2s' % (ia, mol.atom_symbol(ia)) for ia in range(mol.natm)]

    print("PySCF spin-spin coupling constants:")
    dump_mat.dump_tri(sscobj.stdout, ktensor, label)

    LR.get_SSCC_4comp_iso()




def H2():
    geometry = """
                H   1.0215   -0.0252    0.5645
                H   1.1785   -0.5748    1.0355
                """  #0.74
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    basis = dyall_v2z
    active_space = ((1,1),6)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )


def H3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    basis = dyall_v2z
    active_space = ((2, 1), 6)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def LiH():
    geometry = """H  0.0   0.0  0.0;
        Li  0.0  0.0  1.3"""
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H', 'Li'], fmt='nwchem')
    basis = dyall_v2z
    active_space = ((1, 1), 4)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def BeH2():
    geometry = """Be   0.000000   0.000000   0.000000
                  H    0.000000   0.000000   1.326000
                  H    0.000000   0.000000  -1.326000"""
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H', 'Be'], fmt='nwchem')
    basis = dyall_v2z
    active_space = ((2, 2), 6)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HF():
    geometry = """H  0.0   0.0  0.0;
                  F  0.0  0.0  0.9175"""
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H', 'F'], fmt='nwchem')
    basis = dyall_v2z
    active_space = ((2, 2), 6)
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
    dyall_v2z = bse.get_basis('dyall-v2z', elements=['H', 'O'], fmt='nwchem')
    basis = dyall_v2z
    active_space = ((2,2),6)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

###RUN SCRIPT###
H2()
