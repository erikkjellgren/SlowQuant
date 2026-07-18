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
from qiskit_aer.primitives import Sampler, SamplerV2
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, InterleavedQubitMapper
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer.noise import NoiseModel



def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    print("Basis", {basis})
    print("Geometry", {geometry})
    print("Spin", {spin})
    print("Charge",{charge})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.GHF(mol)
    mf.conv_tol = 1e-10        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 1000

    mf.kernel()

    c_mo = np.array(mf.mo_coeff,dtype=complex)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    mapper = JordanWignerMapper()
    backend = None
    sampler = SamplerV2()

    method = "fUCCSD"
    spin_consv = False
    active_k = True
    orb_opt = True
    optimizer = "l-bfgs-b"
    rd_seed = 42
    bounds = [-0.5,0.5]
    tolerance = 1e-10
    nl = 1
    shots = 10000
    M0 = True
    post_select = False
    max_iter = 10000

    print("WF optimization:")
    print("Method:", method)
    print("Backend:",backend)
    print("Shots:", shots)
    print("M0:", M0)
    print("Post selection:", post_select)
    print("Is spin conserving:", spin_consv)
    print("Include Active kappa:", active_k)
    print("Orbital optimization:", orb_opt)
    print("Optimizer:",optimizer)
    print("Random seed:", rd_seed)
    print("Bounds for initiation of thetas:", bounds)
    print("Convergence tolerance:", tolerance)
    print("Number of layers:", nl)
    print("Max iterations:", max_iter)

    directory = os.getcwd() + "/"
    name = "data_H3_shot_def2svp"

    j,k = 0,0
    while j < 30:
        if os.path.exists("%s/%s_%s.npz" % (directory,name,j)):
            k = j
        j+=1

    if k < 10:
        k = f"0{k}"

    data_file = Path("%s_%s.npz" % (name,k))
    

    QI1 = QuantumInterface(
        sampler,
        method, # Ansatz
        mapper,
        ansatz_options = {"n_layers": nl, "is_spin_conserving" : spin_consv},
        shots = shots,
        do_M_ansatz0=M0, # default is false
        do_postselection=post_select,
    )

    qWF1 = GeneralizedWaveFunctionCircuit(
        mol.nelectron,
        active_space,
        c_mo,
        h_core,
        g_eri,
        QI1,
        include_active_kappa = active_k,
    )

    QI1.get_info()

    np.random.seed(rd_seed)
    new_thetas_real = np.random.uniform(bounds[0], bounds[1], len(qWF1.thetas_real)).tolist()
    new_thetas_imag = np.zeros_like(qWF1.thetas_imag)

    qWF1.set_thetas_initial(new_thetas_real, new_thetas_imag)

    qWF1.run_wf_optimization_2step(optimizer, orbital_optimization=orb_opt, tol=tolerance, maxiter = max_iter)

    qWF1.energy_elec()

    np.savez(
        data_file,
        c_mo=qWF1.c_mo,
        thetas_real=qWF1.thetas_real,
        thetas_imag=qWF1.thetas_imag
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

def h2():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  0.000000   0.000000       0.740000;"""
    basis = "sto-3g"
    active_space = ((1, 1), 4)
    charge = 0
    spin = 0

    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

h3()