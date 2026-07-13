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
from qiskit_aer.primitives import SamplerV2, QiskitRuntimeService
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, InterleavedQubitMapper
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp


# Connect to IBM cloud
service = QiskitRuntimeService(channel="ibm_quantum_platform", token="insert-your-token", instance="insert-yourinstance")
# Find least busy backend
backend = service.least_busy(operational=True, simulator=False)

print("We will use the quantum device: ", backend)


def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.GHF(mol)
    mf.conv_tol = 1e-10        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-10   # Optional: gradient convergence
    mf.max_cycle = 1000

    mf.scf()
    mf.kernel()

    c_mo = np.array(mf.mo_coeff,dtype=complex)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    mapper = JordanWignerMapper()
    sampler = SamplerV2(mode=backend)

    QI = QuantumInterface(
        sampler,
        "fUCCSD", # Ansatz
        mapper,
        ansatz_options = {"n_layers": 1, "is_spin_conserving" : False},
        ISA=False, # default is false
        shot= 50000,
        do_M_ansatz0=True, # default is false
        do_postselection=True, # default is false
    )

    qWF = GeneralizedWaveFunctionCircuit(
        mol.nelectron,
        active_space,
        c_mo,
        h_core,
        g_eri,
        QI,
        include_active_kappa = True,
    )

    QI.get_info()

    np.random.seed(42)
    new_thetas_real = np.random.uniform(-0.05, 0.05, len(qWF.thetas)).tolist()
    new_thetas_imag = np.zeros_like(qWF.thetas)

    qWF.set_thetas_initial(new_thetas_real, new_thetas_imag)

    qWF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10)



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