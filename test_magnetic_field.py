import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c, lib, gto
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
import basis_set_exchange as bse
import struct
from pathlib import Path
import os


# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value_energy
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a, get_HcoreB, get_HcoreB_z
from slowquant.unitary_coupled_cluster.generalized_density_matrix import get_orbital_gradient_generalized_real_imag, get_orbital_gradient_expvalue_real_imag, get_nonsplit_gradient_expvalue, get_gradient_finite_diff, get_electronic_energy_generalized

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)

from slowquant.molecularintegrals.integralfunctions import DHF_one_electron_transform, DHF_two_electron_transform


def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, cart = False)
    mol.build()

    tol_GHF = 1e-10
    tol_GHF_g = 1e-8
    GHF_max_cycle = 5000

    mf = scf.GHF(mol)
    mf.conv_tol = tol_GHF        # Energy convergence (Hartree)
    mf.conv_tol_grad = tol_GHF_g    # Optional: gradient convergence
    mf.max_cycle = GHF_max_cycle

    # Modyfing the Hcore
    # B in T
    B = np.array([.2, .5, .3]) ; B = (B * (2.3505 * 1e5) / np.linalg.norm(B).tolist()).tolist() ; orig = (0, 0, 0)

    hcore = mf.get_hcore().astype(complex)
    hcoreB_corr = get_HcoreB(mf, mol, B = B, orig = orig)
    mf.get_hcore = lambda *args: hcore + hcoreB_corr

    # Runnign PySCF
    mf.kernel()

    # MO coefficients:
    c_MO=np.array(mf.mo_coeff,dtype=complex)
    print("MAX imag component in C_MO directly from pyscf", np.max(c_MO.imag))

    print("Nr. of orbital energies:", len(mf.mo_energy))
    print("Orbital energies:")
    for i in range(len(mf.mo_energy)):
        print("Orbital", i+1, "energy:", np.round(mf.mo_energy[i],4))

    mf_nofield = scf.GHF(mol)
    mf_nofield.conv_tol = tol_GHF        # Energy convergence (Hartree)
    mf_nofield.conv_tol_grad = tol_GHF_g   # Optional: gradient convergence
    mf_nofield.max_cycle = GHF_max_cycle
    mf_nofield.kernel()
    c_MO_nofield=np.array(mf_nofield.mo_coeff,dtype=complex)
    print("MAX imag component in C_MO directly from pyscf without field", np.max(c_MO_nofield.imag))

    ghf_seed = 42

    # Small step
    np.random.seed(ghf_seed)
    eps = 0.07
    X_anti = np.random.randn(c_MO.shape[0],c_MO.shape[0]) + 1j*np.random.randn(c_MO.shape[0],c_MO.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    step = expm(A_mat)
    c_u = c_MO @ step

    pyscf_GHF = GeneralizedWaveFunctionUPS(
        active_space,
        c_MO,
        mol,
        "fuccsd",
        ansatz_options = {"n_layers": 0, "is_spin_conserving" : False},
        include_active_kappa=True,
        B = B,
        orig = orig,
    )

    pyscf_GHF.spin_analysis()

    GHF = GeneralizedWaveFunctionUPS(
        active_space,
        c_u,
        mol,
        "fuccsd",
        ansatz_options = {"n_layers": 0, "is_spin_conserving" : False},
        include_active_kappa=True,
        B = B,
        orig = orig,
    )

    optimizer = "l-bfgs-b"
    orb_opt = True
    tolerance = 1e-10
    maxiter = 10000

    GHF.run_wf_optimization_1step(optimizer, orbital_optimization=orb_opt, tol=tolerance, maxiter = maxiter)

    GHF.spin_analysis()

    WF = GeneralizedWaveFunctionUPS(
        active_space,
        GHF.c_mo,
        mol,
        "fuccsd",
        ansatz_options = {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
        B = B,
        orig = orig,
    )

    print("PySCF electronic energy", mf.energy_elec()[0])
    print("SQ GHF electronic energy", GHF.energy_elec)
    print("Epsilon value:", eps)
    print("Nr. of kappas:", len(WF.kappa_spin_idx))
    print("Nr. of spin orbitals:", WF.num_spin_orbs)
    print("Nr. of inactive spin orbitals:", WF.num_inactive_spin_orbs)
    print("Nr. of active spin orbitals:", WF.num_active_spin_orbs)
    print("Nr. of virtual spin orbitals:", WF.num_virtual_spin_orbs)

    bounds = [-0.05,0.05]
    rd_seed = 70
    rd_seed2 = 50
    np.random.seed(rd_seed)
    new_thetas_real = np.random.uniform(bounds[0], bounds[1], len(WF.thetas_real)).tolist()
    np.random.seed(rd_seed2)
    new_thetas_imag = np.random.uniform(bounds[0], bounds[1], len(WF.thetas_real)).tolist()
    #new_thetas_imag = np.zeros_like(WF.thetas_imag)
    WF.set_thetas(new_thetas_real, new_thetas_imag)


    # Printing settings:
    print("Tolerance GHF:", tol_GHF)
    print("Tolerance for the gradient GHF:", tol_GHF_g)
    print("Random seed for GHF")
    print("Randsom seed for thetas real component:", rd_seed)
    print("Randsom seed for thetas imag component:", rd_seed2)
    print("Bounds for thetas:", bounds)
    print("GHF max cycles:", GHF_max_cycle)
    print("Optimizer:", optimizer)
    print("Orbital optimization:", orb_opt)
    print("Tolerance UCCSD energy:", tolerance)
    print("Maxiter UCCSD:", maxiter)
    print("Magnetic field in T:", B)
    print("Origin of the magnetic field,", orig)

    WF.run_wf_optimization_1step(optimizer, orbital_optimization=orb_opt, tol=tolerance, maxiter = maxiter)

    print("\nElectronic GHF energy from PySCF:", mf.energy_elec()[0])
    print("\nFinal electronic UCCSD energy:", WF.energy_elec)
    print("Max imag component of thetas", np.max(WF.thetas_imag))
    WF.spin_analysis()

    # Saving the data:
    directory = os.getcwd()
    name = "data_mfield_H3_def2svp_2_1_6_imag"
    j,k = 0,0
    while j < 100:
        if j < 10:
            if os.path.exists("%s/%s_UCCSD_0%s.npz" % (directory, name, j)):
                k = j + 1
        else:
            if os.path.exists("%s/%s_UCCSD_%s.npz" % (directory, name, j)):
                k = j +1
        j += 1

    if k < 10:
        k = f"0{k}"

    data_file = Path("%s_UCCSD_%s.npz" % (name, k))

    print("\nName of the UCCSD data file:", data_file)

    np.savez(
        data_file,
        c_mo=WF.c_mo,
        thetas_real=WF.thetas_real,
        thetas_imag=WF.thetas_imag
        )




def h2():
    geometry = """H  0.0   0.0  0;
                  H  0.0   0.0  0.74"""
    #basis = "cc-pvtz"
    basis = "631-g"
    #basis = "sto-3g"
    #basis = "sto-6g"
    dyall2zp_H = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    with open('dyall2zp_H.nwchem', 'w') as f:
        f.write(dyall2zp_H)
        f.close()
    #basis = {'H': gto.basis.load('dyall2zp_H.nwchem', 'H')}
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
    #basis = "631-g"
    #basis = "sto-3g"
    basis = "def-2-svp"
    #basis = ""
    active_space = ((2, 1), 6)
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
                  Li  0.0  0.0  1.595"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
    active_space = ((1, 1), 6)
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

def HF():
    geometry = """H  0.0   0.0  0.0;
                  F  0.0  0.0  0.917"""
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    #active_space = ((2, 2), 12)
    #active_space = ((2, 2), 10)
    active_space = ((3,3), 8)
    #active_space = ((2, 2), 6)
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
    #basis = {'H': gto.basis.load('dyall2zp_H.nwchem', 'H'),'O': gto.basis.load('dyall2zp_O.nwchem', 'O')}
    #basis = "dyallv2z"
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    #basis = "sto-6g"
    #active_space = ((5, 5), 14)
    active_space = ((2,2),8)
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
    #basis = "dyall-v2z"
    basis = "cc-pvdz"
    active_space = (4, 6)
    charge = 0
    spin = 0

    #print("Restricted HI")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    print("Nonrelativistic HI")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HCl():
    geometry = """H  0.0   0.0  0.0;
        Cl  0.0  0.0  1.1275 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    basis = "sto-3g"
    active_space = ((3,3), 8)
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
        Br  0.0  0.0  1.41443 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    basis = "sto-3g"
    active_space = ((18,18), 38)
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

def BeH():
    geometry = """Be 0.0 0.0 0.0;
                  H 0.0 0.0 1.3426"""
    basis="sto-3g"
    active_space = ((1,2),6)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom",
    )
  

###SPIN ELLER RUMLIGE ORBITALER###

#h2o()
#LiH()
h3()
#h2o()
# HI()
# HBr()