import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
import matplotlib


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


# Connect to IBM cloud
service = QiskitRuntimeService(channel="ibm_quantum_platform", token="?", instance="Random stuff") # Alternative backend: "Random stuff-eu"
# Find least busy backend
backend = service.least_busy(operational=True, simulator=False)

print("We will use the quantum device: ", backend)


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

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    mapper = JordanWignerMapper()
    sampler = SamplerV2(mode=backend)

    method = "fUCCSD"
    spin_consv = False
    active_k = True
    orb_opt = True
    optimizer = "l-bfgs-b"
    rd_seed = 42
    bounds = [-0.5,0.5]
    tolerance = 1e-10
    nl = 1
    shots = 50000
    M0 = True
    post_select = False
    max_iter = 10000

    print("Started from classical thetas from 'data_H3_6-31g.npz' corresponding to GHF_paper_annika_H3_01.out")

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
    #print("Max iterations:", max_iter)
    

    QI = QuantumInterface(
        sampler,
        method, # Ansatz
        mapper,
        ansatz_options = {"n_layers": nl, "is_spin_conserving" : spin_consv},
        shots = shots,
        do_M_ansatz0=M0, # default is false
    )

    data = np.load("data_H3_6-31g.npz")


    qWF = GeneralizedWaveFunctionCircuit(
        mol.nelectron,
        active_space,
        data["c_mo"],
        h_core,
        g_eri,
        QI,
        include_active_kappa = active_k,
    )

    QI.get_info()

    qWF.set_thetas_initial(data["theta_real"], data["theta_imag"])

    print("Final energy from hardware calculation:", qWF.energy_elec)



def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    basis = "6-31g"
    active_space = ((2, 1), 6)
    charge = 0
    spin = 1

    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

h3()