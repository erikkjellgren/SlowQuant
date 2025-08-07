# pylint: disable=too-many-lines
# pylint: disable=protected-access
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

import slowquant.qiskit_interface.linear_response.allprojected as q_allprojected  # pylint: disable=consider-using-from-import
import slowquant.qiskit_interface.linear_response.naive as q_naive  # pylint: disable=consider-using-from-import
import slowquant.qiskit_interface.linear_response.projected as q_projected  # pylint: disable=consider-using-from-import
import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.allprojected as allprojected  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.naive as naive  # pylint: disable=consider-using-from-import
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.sa_wavefunction import WaveFunctionSA
from slowquant.qiskit_interface.wavefunction import WaveFunction
from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["u1", "u2", "u3"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
noise_model.add_all_qubit_quantum_error(amplitude_damping_error(0.02), ["u1", "u2", "u3"], warnings=False)
noise_model.add_all_qubit_quantum_error(phase_damping_error(0.03), ["u1", "u2", "u3"], warnings=False)
noise_model.add_all_qubit_readout_error(ReadoutError([[0.95, 0.05], [0.1, 0.9]]))


def test_LiH_naive_samplerQiskit() -> None:
    """
    Test LiH ooVQE with rotosolve + naive LR with sampler from Qiskit
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # SlowQuant
    WF = WaveFunctionUCC(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )

    # Optimize WF
    WF.run_ucc(True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # LR with SQ
    LR = naive.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    # LR with QSQ
    qLR = q_naive.quantumLR(qWF)

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    assert np.allclose(excitation_energies, LR.excitation_energies, atol=10**-4)

    solution = [
        0.12947075,
        0.17874853,
        0.17874853,
        0.60462373,
        0.64663037,
        0.74060052,
        0.74060052,
        1.00275465,
        2.0748271,
        2.13720201,
        2.13720201,
        2.45509667,
        2.95432578,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_naive() -> None:
    """
    Test LiH ooVQE with rotosolve + naive LR with sampler from QiskitAer
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # SlowQuant
    WF = WaveFunctionUCC(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )

    # Optimize WF
    WF.run_ucc(True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # LR with SQ
    LR = naive.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    # LR with QSQ
    qLR = q_naive.quantumLR(qWF)

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    assert np.allclose(excitation_energies, LR.excitation_energies, atol=10**-4)

    solution = [
        0.12947075,
        0.17874853,
        0.17874853,
        0.60462373,
        0.64663037,
        0.74060052,
        0.74060052,
        1.00275465,
        2.0748271,
        2.13720201,
        2.13720201,
        2.45509667,
        2.95432578,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_projected() -> None:
    """
    Test LiH ooVQE with rotosolve + projected LR sampler from QiskitAer
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # LR with QSQ
    qLR = q_projected.quantumLR(qWF)

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    solution = [
        0.12947075,
        0.17874853,
        0.17874853,
        0.60462373,
        0.64663037,
        0.74060052,
        0.74060052,
        1.00275465,
        2.0748271,
        2.13720201,
        2.13720201,
        2.45509667,
        2.95432578,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_dumb_projected() -> None:
    """
    Test LiH ooVQE with rotosolve + projected LR with sampler from QiskitAer
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # LR with QSQ
    qLR = q_projected.quantumLR(qWF)

    qLR._run_no_saving(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    solution = [
        0.12947075,
        0.17874853,
        0.17874853,
        0.60462373,
        0.64663037,
        0.74060052,
        0.74060052,
        1.00275465,
        2.0748271,
        2.13720201,
        2.13720201,
        2.45509667,
        2.95432578,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_allprojected() -> None:
    """
    Test LiH ooVQE with rotosolve + allprojected LR with sampler from QiskitAer
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # SlowQuant
    WF = WaveFunctionUCC(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )

    # Optimize WF
    WF.run_ucc(True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # LR with SQ
    LR = allprojected.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    # LR with QSQ
    qLR = q_allprojected.quantumLR(qWF)

    qLR.run()
    excitation_energies = qLR.get_excitation_energies()

    assert np.allclose(excitation_energies, LR.excitation_energies, atol=10**-4)

    print(excitation_energies)

    solution = [
        0.12961625,
        0.18079147,
        0.18079147,
        0.60483322,
        0.6469466,
        0.74931037,
        0.74931037,
        1.00301551,
        2.07493174,
        2.13725269,
        2.13725269,
        2.45535992,
        2.95516418,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_dumb_allprojected() -> None:
    """
    Test LiH ooVQE with rotosolve + dumb allprojected LR with sampler from QiskitAer
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # LR with QSQ
    qLR = q_allprojected.quantumLR(qWF)

    qLR.run()
    excitation_energies = qLR.get_excitation_energies()

    print(excitation_energies)

    solution = [
        0.12961625,
        0.18079147,
        0.18079147,
        0.60483322,
        0.6469466,
        0.74931037,
        0.74931037,
        1.00301551,
        2.07493174,
        2.13725269,
        2.13725269,
        2.45535992,
        2.95516418,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_naive_sampler_ISA() -> None:
    """
    Test LiH ooVQE with rotosolve + naive LR with sampler from QiskitAer
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper, ISA=True)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # LR with QSQ
    qLR = q_naive.quantumLR(qWF)

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    solution = [
        0.12947075,
        0.17874853,
        0.17874853,
        0.60462373,
        0.64663037,
        0.74060052,
        0.74060052,
        1.00275465,
        2.0748271,
        2.13720201,
        2.13720201,
        2.45509667,
        2.95432578,
    ]

    assert np.allclose(excitation_energies, solution, atol=10**-6)


def test_LiH_oscillator_strength() -> None:
    """
    Test oscillator strength for various LR parametrizations
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    x, y, z = mol.intor("int1e_r", comp=3)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # naive LR with QSQ
    qLR_naive = q_naive.quantumLR(qWF)
    qLR_naive.run(do_rdm=True)
    qLR_naive.get_excitation_energies()
    qLR_naive.get_normed_excitation_vectors()
    osc_strengths = qLR_naive.get_oscillator_strength([x, y, z])

    solution = [
        0.04993035,
        0.24117267,
        0.24117267,
        0.15818932,
        0.16642583,
        0.01036042,
        0.01036042,
        0.00625735,
        0.06238003,
        0.12886178,
        0.12886178,
        0.04602256,
        0.00390723,
    ]

    assert np.allclose(osc_strengths, solution, atol=10**-6)

    # proj LR with QSQ
    qLR_proj = q_projected.quantumLR(qWF)
    qLR_proj.run(do_rdm=True)
    qLR_proj.get_excitation_energies()
    qLR_proj.get_normed_excitation_vectors()
    osc_strengths = qLR_proj.get_oscillator_strength([x, y, z])

    solution = [
        0.04993178,
        0.24117267,
        0.24117267,
        0.15817858,
        0.16644551,
        0.01036042,
        0.01036042,
        0.00626061,
        0.06238002,
        0.12886178,
        0.12886178,
        0.04602259,
        0.00390724,
    ]

    assert np.allclose(osc_strengths, solution, atol=10**-6)

    # allproj LR with QSQ
    qLR_allproj = q_allprojected.quantumLR(qWF)
    qLR_allproj.run()
    qLR_allproj.get_excitation_energies()
    qLR_allproj.get_normed_excitation_vectors()
    osc_strengths = qLR_allproj.get_oscillator_strength([x, y, z])

    solution = [
        0.05008157,
        0.25084325,
        0.25084325,
        0.16221272,
        0.16126769,
        0.01835635,
        0.01835635,
        0.0067395,
        0.06319573,
        0.13384356,
        0.13384356,
        0.04670223,
        0.00384224,
    ]

    assert np.allclose(osc_strengths, solution, atol=10**-6)


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

    WF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    WF.run_vqe_2step("SLSQP", False)
    assert abs(WF.energy_elec - -1.8572750819575072) < 10**-6


def test_sampler_changes() -> None:
    """
    Test primitive changes
    """
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

    qWF = WaveFunction(
        mol.nao * 2,
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
    """
    Test if shots work.
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

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    print(qWF.energy_elec)


def test_fUCC_h2o() -> None:
    """Test fUCC for a (4,4) active space."""
    atom = "O .0 .0 0.1035174918; H .0 0.7955612117 -0.4640237459; H .0 -0.7955612117 -0.46402374590;"
    basis = "sto-3g"
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    sampler = SamplerAer()
    mapper = JordanWignerMapper()
    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    WF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (4, 4),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    WF.run_vqe_2step("RotoSolve", False)
    assert abs(WF.energy_elec - -83.96650295692562) < 10**-6


def test_samplerV2() -> None:
    """
    Test SamplerV2
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

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    print(qWF.energy_elec)


def test_samplerV2_ibm() -> None:
    """
    Test SamplerV2 IBM
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

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    print(qWF.energy_elec)


def test_custom() -> None:
    """
    Test custom Ansatz.
    """
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

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )
    qWF.run_vqe_2step("rotosolve", True, is_silent_subiterations=True)
    energy = qWF._calc_energy_elec()

    qc = qWF.QI.ansatz_circuit.copy()
    qc_param = qWF.QI.parameters
    qc_H = qWF._get_hamiltonian()

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
    """
    Test composing of circuits when complicated layout is applied.
    """
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

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    QI.update_pass_manager({"backend": FakeTorino()})

    QI._reset_cliques()

    assert np.allclose(qWF._calc_energy_elec(), -1.6303275411526188, atol=10**-6)


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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    WF.run_ups(True)

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

    qWF = WaveFunction(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_trans,
        h_core,
        g_eri,
        QI,
    )
    qWF.ansatz_parameters = WF.thetas

    assert abs(qWF._calc_energy_elec() + 9.258549202172054) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0]

    QI.update_mitigation_flags(do_postselection=True)
    assert abs(qWF._calc_energy_elec() + 9.531143400515425) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8]

    QI.update_mitigation_flags(do_postselection=False, do_M_ansatz0=True)
    assert abs(qWF._calc_energy_elec() + 9.665136097395242) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3]

    QI.update_mitigation_flags(do_postselection=True)
    assert abs(qWF._calc_energy_elec() + 9.711819110417231) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3, 11]

    QI.update_mitigation_flags(do_M_ansatz0_plus=True)
    assert abs(qWF._calc_energy_elec() + 9.711819110417231) < 10**-6  # type: ignore
    assert list(QI.saver[12].cliques[0].distr.data.keys()) == [0, 8, 3, 11, 15]


def test_state_average_layout() -> None:
    """
    Test RDM1 calculation with SA in the presence of complicated layout.
    """
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
        SQobj.molecule.number_bf * 2,
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
    WF.run_saups(False)

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

    QWF = WaveFunctionSA(
        SQobj.molecule.number_bf * 2,
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
    QWF.ansatz_parameters = WF.thetas

    assert_allclose(QWF.rdm1, WF.rdm1, atol=10**-6)


def test_state_average_M() -> None:
    """
    Test Energy calculation with SA in the presence of complicated layout and M_Ansatz0
    """
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
        SQobj.molecule.number_bf * 2,
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
    WF.run_saups(False)

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

    QWF = WaveFunctionSA(
        SQobj.molecule.number_bf * 2,
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
    QWF.ansatz_parameters = WF.thetas

    assert abs(QWF._calc_energy_elec() + 1.417013381867924) < 10**-6  # type: ignore


def test_state_average_Mplus() -> None:
    """
    Test Energy calculation with SA in the presence of complicated layout and with M_Ansatz0+
    """
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
        SQobj.molecule.number_bf * 2,
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
    WF.run_saups(False)

    sampler = SamplerAer()
    mapper = JordanWignerMapper()
    QI = QuantumInterface(sampler, "tUPS", mapper, ansatz_options={"n_layers": 1})

    QWF = WaveFunctionSA(
        SQobj.molecule.number_bf * 2,
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
    QWF.ansatz_parameters = WF.thetas

    assert abs(WF._sa_energy - QWF._calc_energy_elec()) < 10**-6  # type: ignore

    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    QWF.change_primitive(sampler)

    QI.shots = None
    QI.ISA = True
    QI.update_pass_manager({"backend": FakeTorino(), "seed_transpiler": 1234})

    # No EM
    QI._reset_cliques()
    assert abs(QWF._calc_energy_elec() + 9.437068894682653) < 10**-6  # type: ignore  # CSFs option 1

    # EM with M_Ansatz0
    QI.update_mitigation_flags(do_M_mitigation=True, do_M_ansatz0=True)

    assert abs(QWF._calc_energy_elec() + 9.594604931255208) < 10**-6  # type: ignore  # CSFs option 4

    # EM with M_Ansatz0+
    QI.update_mitigation_flags(do_M_ansatz0_plus=True)
    assert abs(QWF._calc_energy_elec() + 9.607824917072104) < 10**-6  # type: ignore  # CSFs option 1

    # EM with M_Ansatz0 and postselection
    QI.update_mitigation_flags(do_postselection=True, do_M_ansatz0_plus=False)
    assert abs(QWF._calc_energy_elec() + 9.635783384095342) < 10**-6  # type: ignore  # CSFs option 4


def test_no_saving() -> None:
    """
    Test Energy calculation with SA for the no saving options
    """
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
        SQobj.molecule.number_bf * 2,
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
    WF.run_saups(False)

    sampler = SamplerAer()
    mapper = JordanWignerMapper()
    QI = QuantumInterface(sampler, "tUPS", mapper, ansatz_options={"n_layers": 1})

    QWF = WaveFunctionSA(
        SQobj.molecule.number_bf * 2,
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
    QWF.ansatz_parameters = WF.thetas

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
    assert abs(QWF._calc_energy_elec() + 9.594604931255208) < 10**-6  # type: ignore


def test_variance() -> None:
    """Test variance calculation."""
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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    WF.run_ups(True)

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

    qWF = WaveFunction(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_trans,
        h_core,
        g_eri,
        QI,
    )
    qWF.ansatz_parameters = WF.thetas

    qWF._calc_energy_elec()
    assert abs(QI.quantum_variance(qWF._get_hamiltonian()) - 0.13590797869982368) < 10**-6  # type: ignore

    QI.update_mitigation_flags(do_postselection=True)
    qWF._calc_energy_elec()
    assert abs(QI.quantum_variance(qWF._get_hamiltonian()) - 0.08330602766877596) < 10**-6  # type: ignore
