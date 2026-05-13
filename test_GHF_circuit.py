import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm


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

# qWF imports:
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper


# qWF global settings:
sampler = Sampler()
mapper = JordanWignerMapper()



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
    epsilon = 0.0  # controls "step size"
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
    #mf.chkfile = '/home/annika4ee/SlowQuant/uhf_guess.chk'

    # Change initial guess:
    #mf.init_guess = "chkfile"
    mf.conv_tol = 1e-8        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 1000

    mf.scf()
    mf.kernel()
    c=np.array(mf.mo_coeff,dtype=complex)



    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h_1e = mol.intor("int1e_kin")
    h_nuc = mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    dip_int = mol.intor("int1e_r")
    #mc = mcscf.CASCI(mf, active_space[1], active_space[0])

    # mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # # Slowquant

    # small random anti-Hermitian
    eps = 0.005  # controls "step size"
    X_anti = np.random.randn(c.shape[0],c.shape[0]) + 1j*np.random.randn(c.shape[0],c.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    U_step = expm(A_mat)

    c_u = c @ U_step


    print(np.round(c.real,3))


    WF = GeneralizedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        c,
        h_core,
        g_eri,
        "fuccsd",
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )



def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
    #basis = "sto-6g"
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
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
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
    
# Run simulation:

h2()

