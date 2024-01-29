# Import PySCF, SlowQuant, DMDM
import sys

import numpy as np
import pyscf
from qiskit.primitives import Estimator, Sampler
from qiskit_nature.second_q.mappers import ParityMapper

import slowquant.qiskit_interface.linear_response.allprojected as q_allprojected

# from slowquant.qiskit_interface.linear_response import quantumLR
import slowquant.qiskit_interface.linear_response.naive as q_naive
import slowquant.qiskit_interface.linear_response.projected as q_projected
import slowquant.unitary_coupled_cluster.linear_response.allprojected as allprojected
import slowquant.unitary_coupled_cluster.linear_response.naive as naive
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.wavefunction import WaveFunction
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_LiH_naive_estimator() -> None:
    """
    Test LiH ooVQE with rotosolve + naive LR with estimator
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

    QI = QuantumInterface(estimator, "UCCSD", mapper)

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

    qLR.run()
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


def test_LiH_projected_estimator() -> None:
    """
    Test LiH ooVQE with rotosolve + projected LR with estimator
    """
    # Define molecule
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto-3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # Optimize WF with QSQ
    estimator = Estimator()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(estimator, "ErikSD", mapper)

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

    qLR.run()
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


def test_LiH_allprojected_estimator() -> None:
    """
    Test LiH ooVQE with rotosolve + allprojected LR with estimator
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

    QI = QuantumInterface(estimator, "ErikSD", mapper)

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


def test_LiH_naive_sampler() -> None:
    """
    Test LiH ooVQE with rotosolve + naive LR with sampler
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
    sampler = Sampler()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "UCCSD", mapper)

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

    qLR.run()
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
