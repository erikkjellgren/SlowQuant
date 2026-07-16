import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
import matplotlib
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

    print("GHF")
    print("Convergence tolerance energy:", tole)
    print("Convergence tolerance energy:", tolg)

    mf = scf.GHF(mol)
    mf.conv_tol = tole        # Energy convergence (Hartree)
    mf.conv_tol_grad = tolg   # Optional: gradient convergence
    mf.max_cycle = 1000

    mf.kernel()

    c_mo = np.array(mf.mo_coeff,dtype=complex)

    method = "fUCCSD"
    spin_consv = False
    active_k = True
    orb_opt = True
    optimizer = "l-bfgs-b"
    rd_seed = 42
    bounds = [-0.5,0.5]
    tolerance = 1e-10
    nl = 1
    max_iter=10000

    directory = os.getcwd() + "/"
    name = "data_H7_new"

    j = 0
    while j < 30:
        if os.path.exists("%s/%s_%s.npz" % (directory,name,j)):
            k = j
        j+=1

    data_file = Path("%s/%s_%s.npz" % (directory,name,j))

    print("WF optimization:")
    print("Method:", method)
    print("Is spin conserving:", spin_consv)
    print("Include Active kappa:", active_k)
    print("Orbital optimization:", orb_opt)
    print("Optimizer:",optimizer)
    print("Random seed:", rd_seed)
    print("Bounds for initiation of thetas:", bounds)
    print("Convergence tolerance:", tolerance)
    print("Number of layers:", nl)
    print("Max iterstions:",max_iter)
    print("Name of data file:",data_file)


    WF = GeneralizedWaveFunctionUPS(
        active_space,
        c_mo,
        mol,
        method,
        ansatz_options = {"n_layers": nl, "is_spin_conserving" : spin_consv},
        include_active_kappa=active_k,
    )

    np.random.seed(rd_seed)
    new_thetas_real = np.random.uniform(bounds[0], bounds[1], len(WF.thetas_real)).tolist()
    new_thetas_imag = np.zeros_like(WF.thetas_imag)

    WF.set_thetas(new_thetas_real, new_thetas_imag)

    WF.run_wf_optimization_2step(optimizer, orbital_optimization=orb_opt, tol = tolerance, maxiter = max_iter)

    print("Final electronic energy:", WF.energy_elec)

    np.savez(
        "data_H7_new_EXTRA",
        c_mo=WF.c_mo,
        thetas_real=WF.thetas_real,
        thetas_imag=WF.thetas_imag
        )

    np.savez(
        data_file,
        c_mo=WF.c_mo,
        thetas_real=WF.thetas_real,
        thetas_imag=WF.thetas_imag
        )




def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    basis = "def-2-svp"
    active_space = ((2, 1), 6)
    charge = 0
    spin = 1

    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def h5():
    geometry = """  H   0.850651   0.000000   0.000000
                    H   0.262866   0.809017   0.000000
                    H  -0.688191   0.500000   0.000000
                    H  -0.688191  -0.500000   0.000000
                    H   0.262866  -0.809017   0.000000  """
    basis = "def-2-svp"
    active_space = ((3, 2), 10)
    charge = 0
    spin = 1

    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def h7():
    geometry = """  H   1.152382   0.000000   0.000000
                    H   0.718499   0.900969   0.000000
                    H  -0.256328   1.123490   0.000000
                    H  -1.038362   0.500000   0.000000
                    H  -1.038362  -0.500000   0.000000
                    H  -0.256328  -1.123490   0.000000
                    H   0.718499  -0.900969   0.000000  """
    basis = "6-31g"
    active_space = ((4, 3), 14)
    charge = 0
    spin = 1

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

def Cu3():
    geometry = """Cu   0.000000   0.000000   0.000000;
                  Cu   0.000000   0.000000   2.260000;
                  Cu   0.000000   1.883000   1.250000"""
    basis = "def-2-svp"
    active_space = ((2, 1), 6)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

h7()
