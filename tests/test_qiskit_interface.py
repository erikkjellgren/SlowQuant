import numpy as np
import pyscf
from numpy.testing import assert_allclose
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.noise.errors import (
    ReadoutError,
    amplitude_damping_error,
    phase_damping_error,
)
from qiskit_aer.primitives import Sampler as SamplerAer
from qiskit_aer.primitives import SamplerV2 as SamplerV2Aer
from qiskit_ibm_runtime import SamplerV2 as SamplerV2IBM
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

import slowquant.qiskit_interface.linear_response.allprojected as q_allprojected
import slowquant.qiskit_interface.linear_response.naive as q_naive
import slowquant.qiskit_interface.linear_response.projected as q_projected
import slowquant.SlowQuant as sq
from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.sa_circuit_wavefunction import WaveFunctionSACircuit
from slowquant.unitary_coupled_cluster.linear_response import (
    allprojected,
    naive,
)
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["u1", "u2", "u3"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
noise_model.add_all_qubit_quantum_error(amplitude_damping_error(0.02), ["u1", "u2", "u3"], warnings=False)
noise_model.add_all_qubit_quantum_error(phase_damping_error(0.03), ["u1", "u2", "u3"], warnings=False)
noise_model.add_all_qubit_readout_error(ReadoutError([[0.95, 0.05], [0.1, 0.9]]))


def test_LiH_naive() -> None:
    """Test LiH ooVQE with rotosolve + naive LR with sampler from QiskitAer."""
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # SlowQuant
    WF = WaveFunctionUCC(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )

    # Optimize WF
    WF.run_wf_optimization_1step("SLSQP", True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        WF.c_mo,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_wf_optimization_2step("rotosolve", True)

    # LR with SQ
    LR = naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()

    # LR with QSQ
    qLR = q_naive.quantumLR(qWF, "SD")

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    assert np.allclose(excitation_energies, LR.excitation_energies, atol=10**-4)

    solution = [
        0.12945857,
        0.17873002,
        0.17873002,
        0.60460103,
        0.64662807,
        0.7405599,
        0.7405599,
        1.00273234,
        2.07482701,
        2.13719975,
        2.13719975,
        2.45509343,
        2.95423121,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_projected() -> None:
    """Test LiH ooVQE with rotosolve + projected LR sampler from QiskitAer."""
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Conventional UCC wave function
    WF = WaveFunctionUCC(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    # CircuitWF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    # Pass converged UCC orbitals to circuit wave function but still do optimization (just a speed-up)
    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        WF.c_mo,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", True)

    # LR with QSQ
    qLR = q_projected.quantumLR(qWF, "SD")

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    solution = [
        0.1294585,
        0.17872992,
        0.17872992,
        0.60460117,
        0.64662822,
        0.74056037,
        0.74056037,
        1.00273275,
        2.07482698,
        2.13719974,
        2.13719974,
        2.45509396,
        2.95423188,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_allprojected() -> None:
    """Test LiH ooVQE with rotosolve + allprojected LR with sampler from QiskitAer."""
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Conventional UCC wave function
    WF = WaveFunctionUCC(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )

    # Optimize WF
    WF.run_wf_optimization_1step("SLSQP", True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    # Pass converged UCC orbitals to circuit wave function but still do optimization (just a speed-up)
    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        WF.c_mo,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_wf_optimization_2step("rotosolve", True)

    # LR with SQ
    LR = allprojected.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()

    # LR with QSQ
    qLR = q_allprojected.quantumLR(qWF, "SD")

    qLR.run()
    excitation_energies = qLR.get_excitation_energies()

    assert np.allclose(excitation_energies, LR.excitation_energies, atol=10**-4)

    solution = [
        0.12961665,
        0.18079167,
        0.18079167,
        0.60483162,
        0.6469434,
        0.74930517,
        0.74930517,
        1.00301143,
        2.07493044,
        2.13725045,
        2.13725045,
        2.45535443,
        2.95513784,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_naive_sampler_ISA() -> None:
    """Test LiH ooVQE with rotosolve + naive LR with sampler from QiskitAer."""
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # SlowQuant
    WF = WaveFunctionUCC(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )

    # Optimize WF
    WF.run_wf_optimization_1step("SLSQP", True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper, ISA=True)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        WF.c_mo,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_wf_optimization_2step("rotosolve", True)

    # LR with QSQ
    qLR = q_naive.quantumLR(qWF, "SD")

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    solution = [
        0.12945828,
        0.17872932,
        0.17872932,
        0.60460038,
        0.64662872,
        0.74055989,
        0.74055989,
        1.00273248,
        2.07482743,
        2.13719967,
        2.13719967,
        2.45509327,
        2.95422909,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_oscillator_strength() -> None:
    """Test oscillator strength for various LR parametrizations."""
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    x, y, z = mol.intor("int1e_r", comp=3)

    # SlowQuant
    WF = WaveFunctionUCC(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )

    # Optimize WF
    WF.run_wf_optimization_1step("SLSQP", True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        WF.c_mo,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_wf_optimization_2step("rotosolve", True)

    # naive LR with QSQ
    qLR_naive = q_naive.quantumLR(qWF, "SD")
    qLR_naive.run(do_rdm=True)
    qLR_naive.get_excitation_energies()
    qLR_naive.get_normed_excitation_vectors()
    osc_strengths = qLR_naive.get_oscillator_strength([x, y, z])

    solution = [
        0.04994476,
        0.24117344,
        0.24117344,
        0.15814894,
        0.16656511,
        0.01038248,
        0.01038248,
        0.00625838,
        0.06238359,
        0.12886307,
        0.12886307,
        0.04601737,
        0.00390778,
    ]

    assert np.allclose(osc_strengths, solution, atol=10**-5)

    # proj LR with QSQ
    qLR_proj = q_projected.quantumLR(qWF, "SD")
    qLR_proj.run(do_rdm=True)
    qLR_proj.get_excitation_energies()
    qLR_proj.get_normed_excitation_vectors()
    osc_strengths = qLR_proj.get_oscillator_strength([x, y, z])

    solution = [
        0.04994581,
        0.24117141,
        0.24117141,
        0.15813749,
        0.16659558,
        0.01038159,
        0.01038159,
        0.00626614,
        0.062484,
        0.12886242,
        0.12886242,
        0.0460578,
        0.00391062,
    ]

    assert np.allclose(osc_strengths, solution, atol=10**-5)

    # allproj LR with QSQ
    qLR_allproj = q_allprojected.quantumLR(qWF, "SD")
    qLR_allproj.run()
    qLR_allproj.get_excitation_energies()
    qLR_allproj.get_normed_excitation_vectors()
    osc_strengths = qLR_allproj.get_oscillator_strength([x, y, z])

    solution = [
        0.05010188,
        0.25086563,
        0.25086563,
        0.16218005,
        0.16126645,
        0.01835985,
        0.01835985,
        0.00673626,
        0.06319923,
        0.13384523,
        0.13384523,
        0.04670278,
        0.00384235,
    ]

    assert np.allclose(osc_strengths, solution, atol=10**-5)


def test_gradient_optimizer_H2() -> None:
    """Test that an optimizer using gradients works."""
    # Define molecule
    atom = "H .0 .0 .0; H .735 .0 .0"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))
    QI = QuantumInterface(sampler, "fUCCD", mapper)

    WF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    WF.run_wf_optimization_2step("SLSQP", False)
    assert abs(WF.energy_elec - -1.8572750819575072) < 10**-6


def test_sampler_changes() -> None:
    """Test primitive changes."""
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    mapper = ParityMapper(num_particles=(1, 1))

    # Ideal Sampler
    sampler = SamplerAer()
    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    assert QI.max_shots_per_run == 100000
    assert QI.shots is None

    # Change to shot noise simulator reset shots
    sampler = SamplerAer(transpile_options={"optimization_level": 0})
    qWF.change_primitive(sampler)
    QI.shots = 10000

    assert QI.max_shots_per_run == 100000
    assert QI.shots == 10000

    # Change shots in QI
    QI.shots = 200000

    assert QI.max_shots_per_run == 100000
    assert QI.shots == 200000
    assert QI._circuit_multipl == 2

    # Change limit
    QI.max_shots_per_run = 50000

    assert QI.max_shots_per_run == 50000
    assert QI.shots == 200000
    assert QI._circuit_multipl == 4

    QI.shots = None

    sampler = SamplerV2Aer()
    qWF.change_primitive(sampler)

    assert QI.shots == 10000
    assert QI._transpiled is True
    assert QI.ISA is True


def test_shots() -> None:
    """Test if shots work.

    This just runs a simulation with some shots checking that nothing is broken with qiskit aer.
    No values are compared.
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    sampler = SamplerAer(transpile_options={"optimization_level": 0})
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper, shots=10)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    _ = qWF.energy_elec


def test_fUCC_h2o() -> None:
    """Test fUCC for a (4,4) active space."""
    atom = "O .0 .0 0.1035174918; H .0 0.7955612117 -0.4640237459; H .0 -0.7955612117 -0.46402374590;"
    basis = "sto-3g"
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    sampler = SamplerAer()
    mapper = JordanWignerMapper()
    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    WF = WaveFunctionCircuit(
        mol.nelectron,
        (4, 4),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    WF.run_wf_optimization_2step("RotoSolve", False)
    assert abs(WF.energy_elec - -83.96650295692562) < 10**-6


def test_samplerV2() -> None:
    """Test SamplerV2.

    This just runs a simulation with some shots, checking that nothing is broken.
    No values are compared.
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    sampler = SamplerV2Aer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper, shots=10)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    _ = qWF.energy_elec


def test_samplerV2_ibm() -> None:
    """Test SamplerV2 IBM.

    This just runs a simulation with some shots, checking that nothing is broken.
    No values are compared.
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    aer = AerSimulator()
    sampler = SamplerV2IBM(mode=aer)
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper, shots=10)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    _ = qWF.energy_elec


def test_custom() -> None:
    """Test custom Ansatz."""
    # Define molecule
    atom = "H .0 .0 .0; H .0 .0 1.0"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper, shots=None)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", True, is_silent_subiterations=True)
    energy = qWF._calc_energy_elec()

    qc = qWF.QI.circuit.copy()
    qc_param = qWF.QI.parameters
    qc_H = hamiltonian_0i_0a(qWF.h_mo, qWF.g_mo, qWF.num_inactive_orbs, qWF.num_active_orbs)
    qc_H = qc_H.get_folded_operator(qWF.num_inactive_orbs, qWF.num_active_orbs, qWF.num_virtual_orbs)

    # Define the Sampler
    sampler = SamplerAer()

    # Initialize QI with custom qc object and mapper (JW, see above)
    QI = QuantumInterface(sampler, qc, mapper, shots=None)

    # Construct circuit
    QI.construct_circuit(2, (1, 1))

    # Define parameters
    QI.parameters = qc_param

    assert abs(QI.quantum_expectation_value(qc_H) - energy) < 10**-8


def test_H2_sampler_layout() -> None:
    """Test composing of circuits when complicated layout is applied."""
    # Define molecule
    atom = "H .0 .0 .0; H .0 .0 1.0"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = JordanWignerMapper()

    QI = QuantumInterface(sampler, "fUCCSD", mapper, ISA=True)

    qWF = WaveFunctionCircuit(
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_wf_optimization_2step("rotosolve", True)

    QI.update_pass_manager({"backend": FakeTorino()})

    QI._reset_cliques()

    assert np.allclose(qWF._calc_energy_elec(), -1.6303275411526188, atol=10**-6)


def test_mitigation_nocm() -> None:
    """Test mitigations without coupling map."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0           0.0  0.0;
        H  0.735           0.0  0.0;
        """,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    mapper = JordanWignerMapper()
    QI = QuantumInterface(
        sampler,
        "tUPS",
        mapper,
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        ISA=False,
        do_M_mitigation=False,
        do_M_ansatz0=False,
        do_postselection=False,
    )

    qWF = WaveFunctionCircuit(
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_mo,
        h_core,
        g_eri,
        QI,
    )
    qWF.thetas = WF.thetas

    assert abs(qWF._calc_energy_elec() + 9.418341779479798) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0]

    QI.update_mitigation_flags(do_postselection=True)
    assert abs(qWF._calc_energy_elec() + 9.6024842933555) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8]

    QI.update_mitigation_flags(do_postselection=False, do_M_ansatz0=True)
    assert abs(qWF._calc_energy_elec() + 9.683868077270974) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3]

    QI.update_mitigation_flags(do_postselection=True)
    assert abs(qWF._calc_energy_elec() + 9.703076881083204) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3, 11]

    QI.update_mitigation_flags(do_M_ansatz0_plus=True)
    assert abs(qWF._calc_energy_elec() + 9.703076881083204) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3, 11, 15]


def test_mitigation() -> None:
    """Test mitigations."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0           0.0  0.0;
        H  0.735           0.0  0.0;
        """,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    mapper = JordanWignerMapper()
    QI = QuantumInterface(
        sampler,
        "tUPS",
        mapper,
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        ISA=True,
        do_M_mitigation=False,
        do_M_ansatz0=False,
        do_postselection=False,
        pass_manager_options={"backend": FakeTorino(), "seed_transpiler": 1234},
    )

    qWF = WaveFunctionCircuit(
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_mo,
        h_core,
        g_eri,
        QI,
    )
    qWF.thetas = WF.thetas

    assert abs(qWF._calc_energy_elec() + 9.233709939942546) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0]

    QI.update_mitigation_flags(do_postselection=True)
    assert abs(qWF._calc_energy_elec() + 9.530439599929217) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8]

    QI.update_mitigation_flags(do_postselection=False, do_M_ansatz0=True)
    assert abs(qWF._calc_energy_elec() + 9.661825573106206) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3]

    QI.update_mitigation_flags(do_postselection=True)
    assert abs(qWF._calc_energy_elec() + 9.711973925840587) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3, 11]

    QI.update_mitigation_flags(do_M_ansatz0_plus=True)
    assert abs(qWF._calc_energy_elec() + 9.711973925840587) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3, 11, 15]


def test_state_average_layout() -> None:
    """Test RDM1 calculation with SA in the presence of complicated layout."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0           0.0  0.0;
        H  0.735           0.0  0.0;
        """,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")
    SQobj.init_hartree_fock()

    SQobj.hartree_fock.run_restricted_hartree_fock()
    c_mo = SQobj.hartree_fock.mo_coeff
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    WF = WaveFunctionSAUPS(
        SQobj.molecule.number_electrons,
        (2, 4),
        c_mo,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["11000000"],
                ["10010000", "01100000"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 1},
    )
    WF.run_wf_optimization_1step("SLSQP")

    sampler = SamplerAer()
    mapper = JordanWignerMapper()
    QI = QuantumInterface(
        sampler,
        "tUPS",
        mapper,
        ansatz_options={"n_layers": 1},
        ISA=True,
        pass_manager_options={"backend": FakeTorino(), "seed_transpiler": 1234},
    )

    QWF = WaveFunctionSACircuit(
        SQobj.molecule.number_electrons,
        (2, 4),
        c_mo,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["11000000"],
                ["10010000", "01100000"],
            ],
        ),
        QI,
    )
    QWF.thetas = WF.thetas

    assert_allclose(QWF.rdm1, WF.rdm1, atol=10**-6)


def test_state_average_M() -> None:
    """Test Energy calculation with SA in the presence of complicated layout and M_Ansatz0."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0           0.0  0.0;
        H  0.735           0.0  0.0;
        """,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()

    SQobj.hartree_fock.run_restricted_hartree_fock()
    c_mo = SQobj.hartree_fock.mo_coeff
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    WF = WaveFunctionSAUPS(
        SQobj.molecule.number_electrons,
        (2, 2),
        c_mo,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["1100"],
                ["1001", "0110"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 1},
    )
    WF.run_wf_optimization_1step("SLSQP")

    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    mapper = JordanWignerMapper()
    QI = QuantumInterface(
        sampler,
        "tUPS",
        mapper,
        ansatz_options={"n_layers": 1},
        ISA=True,
        do_M_mitigation=True,
        do_M_ansatz0=True,
        do_postselection=True,
        pass_manager_options={"backend": FakeTorino(), "seed_transpiler": 1234},
    )

    QWF = WaveFunctionSACircuit(
        SQobj.molecule.number_electrons,
        (2, 2),
        c_mo,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["1100"],
                ["1001", "0110"],
            ],
        ),
        QI,
    )
    QWF.thetas = WF.thetas

    assert abs(QWF._calc_energy_elec() + 1.4240939758312483) < 10**-6  # type: ignore


def test_state_average_Mplus() -> None:
    """Test Energy calculation with SA in the presence of complicated layout and with M_Ansatz0+."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0           0.0  0.0;
        H  0.735           0.0  0.0;
        """,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()

    SQobj.hartree_fock.run_restricted_hartree_fock()
    c_mo = SQobj.hartree_fock.mo_coeff
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    WF = WaveFunctionSAUPS(
        SQobj.molecule.number_electrons,
        (2, 2),
        c_mo,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["1100"],
                ["1001", "0110"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 1},
    )
    WF.run_wf_optimization_1step("SLSQP")

    sampler = SamplerAer()
    mapper = JordanWignerMapper()
    QI = QuantumInterface(sampler, "tUPS", mapper, ansatz_options={"n_layers": 1})

    QWF = WaveFunctionSACircuit(
        SQobj.molecule.number_electrons,
        (2, 2),
        c_mo,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["1100"],
                ["1001", "0110"],
            ],
        ),
        QI,
    )
    QWF.thetas = WF.thetas

    assert abs(WF._sa_energy - QWF._calc_energy_elec()) < 10**-6  # type: ignore

    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    QWF.change_primitive(sampler)

    QI.shots = None
    QI.ISA = True
    QI.update_pass_manager({"backend": FakeTorino(), "seed_transpiler": 1234})

    # No EM
    QI._reset_cliques()
    assert abs(QWF._calc_energy_elec() + 9.436258997213987) < 10**-6  # type: ignore  # CSFs option 1

    # EM with M_Ansatz0
    QI.update_mitigation_flags(do_M_mitigation=True, do_M_ansatz0=True)

    assert abs(QWF._calc_energy_elec() + 9.596224644030176) < 10**-6  # type: ignore  # CSFs option 4

    # EM with M_Ansatz0+
    QI.update_mitigation_flags(do_M_ansatz0_plus=True)
    assert abs(QWF._calc_energy_elec() + 9.608563673328995) < 10**-6  # type: ignore  # CSFs option 1

    # EM with M_Ansatz0 and postselection
    QI.update_mitigation_flags(do_postselection=True, do_M_ansatz0_plus=False)
    assert abs(QWF._calc_energy_elec() + 9.638637411200133) < 10**-6  # type: ignore  # CSFs option 4


def test_no_saving() -> None:
    """Test Energy calculation with SA for the no saving options."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0           0.0  0.0;
        H  0.735           0.0  0.0;
        """,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()

    SQobj.hartree_fock.run_restricted_hartree_fock()
    c_mo = SQobj.hartree_fock.mo_coeff
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    WF = WaveFunctionSAUPS(
        SQobj.molecule.number_electrons,
        (2, 2),
        c_mo,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["1100"],
                ["1001", "0110"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 1},
    )
    WF.run_wf_optimization_1step("SLSQP")

    sampler = SamplerAer()
    mapper = JordanWignerMapper()
    QI = QuantumInterface(sampler, "tUPS", mapper, ansatz_options={"n_layers": 1})

    QWF = WaveFunctionSACircuit(
        SQobj.molecule.number_electrons,
        (2, 2),
        c_mo,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["1100"],
                ["1001", "0110"],
            ],
        ),
        QI,
    )
    QWF.thetas = WF.thetas

    QI._save_paulis = False
    assert abs(WF._sa_energy - QWF._calc_energy_elec()) < 10**-6  # type: ignore

    QI._do_cliques = False
    assert abs(WF._sa_energy - QWF._calc_energy_elec()) < 10**-6  # type: ignore

    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    QWF.change_primitive(sampler)

    QI.shots = None
    QI.ISA = True
    QI.update_pass_manager({"backend": FakeTorino(), "seed_transpiler": 1234})

    QI.update_mitigation_flags(do_postselection=False, do_M_ansatz0=True)
    assert abs(QWF._calc_energy_elec() + 9.596224644030176) < 10**-6  # type: ignore


def test_variance_nocm() -> None:
    """Test variance calculation and noise model for singel reference simualtion."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0           0.0  0.0;
        H  0.735           0.0  0.0;
        """,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    mapper = JordanWignerMapper()
    QI = QuantumInterface(
        sampler,
        "tUPS",
        mapper,
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        ISA=False,
        do_M_mitigation=False,
        do_M_ansatz0=False,
        do_postselection=False,
    )

    qWF = WaveFunctionCircuit(
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_mo,
        h_core,
        g_eri,
        QI,
    )
    qWF.thetas = WF.thetas

    assert abs(qWF._calc_energy_elec() + 9.418341779479798) < 10**-6  # type: ignore
    assert abs(QI.quantum_variance(qWF._get_hamiltonian()) - 0.10216972771269525) < 10**-6  # type: ignore

    QI.update_mitigation_flags(do_postselection=True)
    assert abs(qWF._calc_energy_elec() + 9.6024842933555) < 10**-6  # type: ignore
    assert abs(QI.quantum_variance(qWF._get_hamiltonian()) - 0.05290092431551458) < 10**-6  # type: ignore


def test_variance() -> None:
    """Test variance calculation with noise model and full backend transpilation."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0           0.0  0.0;
        H  0.735           0.0  0.0;
        """,
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    mapper = JordanWignerMapper()
    QI = QuantumInterface(
        sampler,
        "tUPS",
        mapper,
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        ISA=True,
        do_M_mitigation=False,
        do_M_ansatz0=False,
        do_postselection=False,
        pass_manager_options={"backend": FakeTorino(), "seed_transpiler": 1234},
    )

    qWF = WaveFunctionCircuit(
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_mo,
        h_core,
        g_eri,
        QI,
    )
    qWF.thetas = WF.thetas

    assert abs(qWF._calc_energy_elec() + 9.233709939942546) < 10**-6  # type: ignore
    assert abs(QI.quantum_variance(qWF._get_hamiltonian()) - 0.1366806110474569) < 10**-6  # type: ignore

    QI.update_mitigation_flags(do_postselection=True)
    assert abs(qWF._calc_energy_elec() + 9.530439599929217) < 10**-6  # type: ignore
    assert abs(QI.quantum_variance(qWF._get_hamiltonian()) - 0.08153932076245432) < 10**-6  # type: ignore
