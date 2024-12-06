# pylint: disable=too-many-lines
# pylint: disable=protected-access
import numpy as np
import pyscf
from numpy.testing import assert_allclose
from qiskit.primitives import Estimator, Sampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
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


def test_LiH_naive_estimator() -> None:
    """
    Test LiH ooVQE with rotosolve + naive LR with estimator from Qiskit
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
    estimator = Estimator()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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
    estimator = Sampler()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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

    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))
    QI = QuantumInterface(estimator, "fUCCD", mapper)

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

    # Ideal Estimator
    estimator = Estimator()
    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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

    # Ideal Sampler
    sampler = Sampler()
    qWF.change_primitive(sampler)

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

    estimator = Estimator()
    mapper = JordanWignerMapper()
    QI = QuantumInterface(estimator, "fUCCSD", mapper)

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

    noise_model = NoiseModel.from_backend(FakeTorino())  # That might change
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

    # This might fail if the noise model of FakeTorinto changes.
    # But if it fails check if the Noise mode is the issue. Do NOT ignore this failing!
    assert abs(QWF._calc_energy_elec() + 1.3755439434495917) < 10**-6  # type: ignore


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

    assert abs(WF._sa_energy - QWF._calc_energy_elec()) < 10**-6

    noise_model = NoiseModel.from_backend(FakeTorino())
    sampler = SamplerAer(backend_options={"noise_model": noise_model})
    QWF.change_primitive(sampler)

    QI.shots = None
    QI.ISA = True
    QI.do_postselection = False

    QI.update_pass_manager({"backend": FakeTorino(), "seed_transpiler": 1234})

    QI.do_M_mitigation = False
    QI.do_M_ansatz0 = False

    QI._reset_cliques()

    assert abs(QWF._calc_energy_elec() + 9.60851106217584) < 10**-6  # type: ignore  # CSFs option 1

    QI.do_M_mitigation = True
    QI.do_M_ansatz0 = True
    QI.redo_M_mitigation()

    QI._reset_cliques()

    assert abs(QWF._calc_energy_elec() + 9.633426170009107) < 10**-6  # type: ignore  # CSFs option 4

    QI._reset_cliques()
    assert (
        abs(QWF._calc_energy_elec(M_per_superpos=True) + 9.635276750167002) < 10**-6  # type: ignore
    )  # CSFs option 1

    QI.do_postselection = True
    QI._reset_cliques()

    assert abs(QWF._calc_energy_elec() + 9.636464216617595) < 10**-6  # type: ignore  # CSFs option 4
