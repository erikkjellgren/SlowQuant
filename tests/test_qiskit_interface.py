import numpy as np
import pyscf
from qiskit.primitives import Estimator, Sampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as SamplerAer
from qiskit_aer.primitives import SamplerV2 as SamplerV2Aer
from qiskit_ibm_runtime import SamplerV2 as SamplerV2IBM
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

import slowquant.qiskit_interface.linear_response.allprojected as q_allprojected  # pylint: disable=consider-using-from-import
import slowquant.qiskit_interface.linear_response.naive as q_naive  # pylint: disable=consider-using-from-import
import slowquant.qiskit_interface.linear_response.projected as q_projected  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.allprojected as allprojected  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.naive as naive  # pylint: disable=consider-using-from-import
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.wavefunction import WaveFunction
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
    )

    # Optimize WF
    WF.run_ucc("SD", True)

    # Optimize WF with QSQ
    estimator = Estimator()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "tUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        WF.c_trans,
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

    solution = [0.12945826, 0.17872954, 0.17872954, 0.60460094, 0.64662905,
       0.74056115, 0.74056115, 1.00273293, 2.07482709, 2.13719974,
       2.13719974, 2.45509448, 2.95423197]

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
    )

    # Optimize WF
    WF.run_ucc("SD", True)

    # Optimize WF with QSQ
    estimator = Sampler()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "tUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        WF.c_trans,
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

    solution = [0.12945846, 0.17872984, 0.17872984, 0.60460091, 0.64662826,
       0.74055997, 0.74055997, 1.00273236, 2.074827  , 2.13719972,
       2.13719972, 2.45509362, 2.95423092]

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
    )

    # Optimize WF
    WF.run_ucc("SD", True)

    # Optimize WF with QSQ
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "tUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        WF.c_trans,
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

    solution = [0.12945857, 0.17873002, 0.17873002, 0.60460103, 0.64662807,
       0.7405599 , 0.7405599 , 1.00273234, 2.07482701, 2.13719975,
       2.13719975, 2.45509343, 2.95423121]

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

    # SlowQuant
    WF = WaveFunctionUCC(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
    )

    # Optimize WF
    WF.run_ucc("SD", True)

    # Optimize WF with QSQ
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "tUCCSD", mapper)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        WF.c_trans,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    qWF.run_vqe_2step("rotosolve", True)

    # LR with QSQ
    qLR = q_projected.quantumLR(qWF)

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    solution = [0.1294585 , 0.17872992, 0.17872992, 0.60460117, 0.64662822,
       0.74056037, 0.74056037, 1.00273275, 2.07482698, 2.13719974,
       2.13719974, 2.45509396, 2.95423188]

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

    QI = QuantumInterface(estimator, "tUCCSD", mapper)

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

    qLR._run_no_saving(do_rdm=True)  # pylint: disable=protected-access
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
    )

    # Optimize WF
    WF.run_ucc("SD", True)

    # Optimize WF with QSQ
    estimator = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "tUCCSD", mapper)

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

    QI = QuantumInterface(estimator, "tUCCSD", mapper)

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

    QI = QuantumInterface(sampler, "tUCCSD", mapper, ISA=True)

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

    QI = QuantumInterface(estimator, "tUCCSD", mapper)

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
    QI = QuantumInterface(estimator, "tUCCD", mapper)

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
    QI = QuantumInterface(estimator, "tUCCSD", mapper)

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
    assert QI._circuit_multipl == 2  # pylint: disable=protected-access

    # Change limit
    QI.max_shots_per_run = 50000

    assert QI.max_shots_per_run == 50000
    assert QI.shots == 200000
    assert QI._circuit_multipl == 4  # pylint: disable=protected-access

    QI.shots = None

    sampler = SamplerV2Aer()
    qWF.change_primitive(sampler)

    assert QI.shots == 10000
    assert QI._transpiled is True  # pylint: disable=protected-access
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

    QI = QuantumInterface(sampler, "tUCCSD", mapper, shots=10)

    qWF = WaveFunction(
        mol.nao * 2,
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

    QI = QuantumInterface(sampler, "tUCCSD", mapper, shots=10)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    _ = qWF.energy_elec


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

    QI = QuantumInterface(sampler, "tUCCSD", mapper, shots=10)

    qWF = WaveFunction(
        mol.nao * 2,
        mol.nelectron,
        (2, 2),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )

    _ = qWF.energy_elec


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

    QI = QuantumInterface(sampler, "tUCCSD", mapper, shots=None)

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
    energy = qWF._calc_energy_elec()  # pylint: disable=protected-access

    qc = qWF.QI.circuit.copy()
    qc_param = qWF.QI.parameters
    qc_H = qWF._get_hamiltonian()  # pylint: disable=protected-access

    # Define the Sampler
    sampler = SamplerAer()

    # Initialize QI with custom qc object and mapper (JW, see above)
    QI = QuantumInterface(sampler, qc, mapper, shots=None)

    # Construct circuit
    QI.construct_circuit(2, (1, 1))

    # Define parameters
    QI.parameters = qc_param

    assert abs(QI.quantum_expectation_value(qc_H) - energy) < 10**-8
