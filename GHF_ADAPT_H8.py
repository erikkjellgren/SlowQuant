import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
import matplotlib
import basis_set_exchange as bse
from pyscf.gto import basis as bs
from pathlib import Path
import os

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
#from qiskit_aer.primitives import SamplerV2, QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, InterleavedQubitMapper
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp



def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("Active space:", {active_space})
    print("Basis", {basis})
    print("Geometry", {geometry})
    print("Spin", {spin})
    print("Charge",{charge})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin) 
    mol.build()

    # GHF
    tole = 1e-10
    tolg = 1e-8

    print("GHF in PySCF:")
    print("Convergence tolerance energy:", tole)
    print("Convergence tolerance gradient:", tolg)

    mf = scf.GHF(mol)
    mf.conv_tol = tole        # Energy convergence (Hartree)
    mf.conv_tol_grad = tolg   # Optional: gradient convergence
    mf.max_cycle = 100000

    mf.kernel()

    print("GHF electronic energy:", mf.energy_elec()[0])
    print("GHF nuclear energy:", mf.energy_nuc())

    c_mo = np.array(mf.mo_coeff,dtype=complex)

    method = "adapt"
    spin_consv = False
    active_k = True
    orb_opt = True
    optimizer = "l-bfgs-b"
    rd_seed = 42
#    bounds = [-0.05,0.05]
    tolerance = 1e-10
    #nl = 1
    max_iter = 100000
    eps = 0.07  # controls "step size"

    print("WF optimization:")
    print("Method:", method)
    print("Is spin conserving:", spin_consv)
    print("Include active kappa:", active_k)
    print("Orbital optimization:", orb_opt)
    print("Optimizer:",optimizer)
    print("Random seed GHF:", rd_seed)
#    print("Bounds for initiation of thetas:", bounds)
    print("Convergence tolerance:", tolerance)
#    print("Number of layers:", nl)
    print("Max iterations:",max_iter)
    print("Epsilon", eps)
    print("Real GHF start")

    np.random.seed(rd_seed)

    # Small real step
    #X = np.random.randn(c_mo.shape[0], c_mo.shape[0])
    #A_mat = eps * (X - X.T) / 2   # real antisymmetric


    # Small step
    X_anti = np.random.randn(c_mo.shape[0],c_mo.shape[0]) + 1j*np.random.randn(c_mo.shape[0],c_mo.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    step = expm(A_mat)
    c_u = c_mo @ step


    #data = np.load("data_H8_complex_ADAPT_00.npz")
    #c_mo_loaded, thetas_real, thetas_imag = data["c_mo"], data["thetas_real"], data["thetas_imag"]
    #print("Max imag component H8:", np.max(np.abs(thetas_imag)))


    # GHF
    print("\n\nStarting GHF reoptimization:")

    GHF = GeneralizedWaveFunctionUPS(
        active_space,
        c_u,
        mol,
        method,
        ansatz_options = {"n_layers": 0, "is_spin_conserving" : spin_consv},
        include_active_kappa=active_k,
    )

    GHF.run_wf_optimization_1step(optimizer, orbital_optimization=orb_opt, tol = tolerance, maxiter = max_iter)

    print("\nGHF electronic energy after reoptimization:", GHF.energy_elec)
    GHF.spin_analysis()

    # Saving the data:
    directory = os.getcwd()
    name = "data_H8"

    j,k = 0,0
    while j < 100:
        if j < 10:
            if os.path.exists("%s/%s_GHF_0%s.npz" % (directory, name, j)):
                k = j + 1
        else:
            if os.path.exists("%s/%s_GHF_%s.npz" % (directory, name, j)):
                k = j +1
        j += 1

    if k < 10:
        k = f"0{k}"

    data_file_GHF = Path("%s_GHF_%s.npz" % (name, k))

    print("\nName of the GHF data file:", data_file_GHF)

    # np.savez(
    #     data_file_GHF,
    #     c_mo=GHF.c_mo
    #     )


    # UCCSD
    print("\nStarting ADAPT WF optimization:")

    WF = GeneralizedWaveFunctionUPS(
        active_space,
        GHF.c_mo,
        mol,
        method,
        ansatz_options = {"is_spin_conserving" : spin_consv},
        include_active_kappa=active_k,
    )

    WF.do_adapt(["s","d"], maxiter = max_iter, grad_threshold = 1e-6 , tol = tolerance, maxiter_1step = max_iter, orbital_optimization=orb_opt)

    print("\nFinal electronic GHF-ADAPT energy:", WF.energy_elec)
    WF.spin_analysis()


    # Saving the data:
    j,k = 0,0
    while j < 100:
        if j < 10:
            if os.path.exists("%s/%s_ADAPT_0%s.npz" % (directory, name, j)):
                k = j + 1
        else:
            if os.path.exists("%s/%s_ADAPT_%s.npz" % (directory, name, j)):
                k = j +1
        j += 1

    if k < 10:
        k = f"0{k}"

    data_file = Path("%s_ADAPT_%s.npz" % (name, k))

    print("\nName of the ADAPT data file:", data_file)

    # np.savez(
    #     data_file,
    #     c_mo=WF.c_mo,
    #     thetas_real=WF.thetas_real,
    #     thetas_imag=WF.thetas_imag
    #     )



def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    #basis = "sto-3g"
    basis = "def-2-svp"
    active_space = ((2, 1), 6)
    charge = 0
    spin = 1

    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )


def h8():
    geometry =   """H   0.590434   0.590434   0.590434
                    H   0.590434  -0.590434  -0.590434
                    H  -0.590434   0.590434  -0.590434
                    H  -0.590434  -0.590434   0.590434
                    H  -0.984056  -0.984056  -0.984056
                    H  -0.984056   0.984056   0.984056
                    H   0.984056  -0.984056   0.984056
                    H   0.984056   0.984056  -0.984056"""
    basis = "6-31g"
    active_space = ((3, 3), 16) # maks 8-9 i 20 siger Pernille, gtUPS og tiled-M0 for 
    charge = 2
    spin = 0

    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )


def HF():
    geometry = """H  0.0   0.0  0.0;
                  F  0.0  0.0  0.917"""
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    #active_space = ((2, 2), 12)
    #active_space = ((2, 2), 10)
    #active_space = ((3,3), 8)
    active_space = ((2, 2), 6)
    charge = 0
    spin = 0

    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

h3()
