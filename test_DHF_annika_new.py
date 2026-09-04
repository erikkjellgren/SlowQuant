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

    hf.remove_overlap_zero_eigenvalue = False
    mf = scf.dhf.DHF(mol)
    mf.conv_tol = 1e-10        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 500

    mf.kernel()

    # Coupling constants PySCF:
    sscobj = SSC(mf)
    sscobj.cphf = True
    sscobj.conv_tol = 1e-9
    sscobj.mb = "RMB"
    sscobj.verbose = 5
    sscobj.with_fcsd = True
    #jj = sscobj.kernel()

    # Shieldings PySCF:
    nmr = nmr_dhf.NMR(mf)
    nmr.cphf = True
    nmr.mb = 'RMB'      # or 'RKB'
    nmr.gauge_orig = None #[0,0,0]  # GIAO vs. # [0,0,0]

    shielding = nmr.kernel()

    sigma_iso = np.trace(shielding, axis1=1, axis2=2) / 3
    print(sigma_iso)

    C_MO=np.array(mf.mo_coeff,dtype=complex)

   
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


    # WF object:
    WF = GeneralizedWaveFunctionUPS(
        active_space,
        C_MO,
        mol, 
        K_pairs,
        False,
        "fUCCSD",
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    np.random.seed(20)
    if len(WF.thetas) > 0:
        real = np.random.uniform(-0.05,0.05,len(WF.thetas_real))
        #imag = np.zeros_like(WF.thetas_imag)
        imag = np.random.uniform(-0.05,0.05,len(WF.thetas_real))
        WF.set_thetas(real, imag)


    # data = np.load("LiH((1,1),4).npz") 
    # data = np.load("HF((5,5),12).npz") 
    # data = np.load("LiH((2,2),6).npz") 
    # data = np.load("HF((1,1),4).npz")
    # data = np.load("H2-6-31g-J((1,1),4).npz")
    # data = np.load("HF((2,2),6).npz")
    # data = np.load("H2-dyallv2z((1,1),6).npz")

    # WF = GeneralizedWaveFunctionUPS(
    #     active_space,
    #     data["c_mo"],
    #     #C_MO,
    #     mol,
    #     K_pairs,
    #     False,
    #     "fUCCSD",
    #     {"n_layers": 1, "is_spin_conserving" : False},
    #     include_active_kappa=True,
    # )
    # WF.set_thetas(data["thetas_real"], data["thetas_imag"])


    print("DHF", mf.energy_elec()[0])
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


    #Optimization:
    WF.run_wf_optimization_2step_DHF(optimizer_name = "l-bfgs-b", orbital_optimization = True, tol = 1e-10, maxiter = 1000)

    # Save WF:
    np.savez(
        "HF-6-31g_pp",
        c_mo=WF.c_mo,
        thetas_real=WF.thetas_real,
        thetas_imag=WF.thetas_imag
        )

    LR = generalized_naive_DHF.LinearResponse(WF, excitations="SD", screen = True)
    LR.calc_excitation_energies()

    print("PySCF:", sigma_iso)

    LR.get_shieldings_4comp_iso(RMB_GIAO = True, output = True)



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
    #basis = "sto-3g"
    #basis = dyall_v2z
    #basis= J_631g
    #basis = "aug-cc-pvtz-J"
    L_atom = bse.get_basis('6-311g**', elements=['F'], fmt='nwchem')
    basis={
        'H': '6-311g**',
        'F': L_atom,
    }
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
    #basis = "sto-3g"
    L_atom = bse.get_basis('6-311g**', elements=['I'], fmt='nwchem')
    basis={
        'H': '6-311g**',
        'F': L_atom,
    }
    active_space = ((2, 2), 6)
    #active_space = ((27,27), 54)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HBr():
    geometry = """H  0.0   0.0  0.0;
        Br  0.0  0.0  1.41443 """
    #basis = "dyall-v2z"
    #basis = "sto-3g"
    L_atom = bse.get_basis('6-311g**', elements=['Br'], fmt='nwchem')
    basis={
        'H': '6-311g**',
        'F': L_atom,
    }
    active_space = ((2, 2), 6)
    #active_space = ((18,18), 36)
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HCl():
    geometry = """H  0.0   0.0  0.0;
                  Cl  0.0  0.0  1.41443 """  # 1.41443
    #basis = "dyall-v2z"
    #basis = "sto-3g"
    L_atom = bse.get_basis('6-311g**', elements=['Cl'], fmt='nwchem')
    basis={
        'H': '6-311g**',
        'F': L_atom,
    }
    active_space = ((2, 2), 6)
    #active_space = ((9,9), 18)
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
