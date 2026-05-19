import numpy as np
import pyscf

from qiskit_aer.primitives import Sampler as SamplerAer
from qiskit_nature.second_q.mappers import ParityMapper

import slowquant.qiskit_interface.linear_response.allprojected as q_allprojected
import slowquant.qiskit_interface.linear_response.naive as q_naive
import slowquant.qiskit_interface.linear_response.projected as q_projected
from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.linear_response import (
    allprojected,
    naive,
    projected,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


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
        (2, 2),
        rhf.mo_coeff,
        mol,
        "SD",
    )

    # Optimize WF
    WF.run_wf_optimization_1step("BFGS", True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunctionCircuit(
        (2, 2),
        WF.c_mo,
        mol,
        QI,
    )

    qWF.run_wf_optimization_2step("rotosolve", True)

    # LR with SQ
    LR = naive.LinearResponse(WF, excitations="SD", triplet=True)
    LR.calc_excitation_energies()

    # LR with QSQ
    qLR = q_naive.quantumLR(qWF, "SD", triplet=True)

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    assert np.allclose(excitation_energies, LR.excitation_energies, atol=10**-5)


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
        (2, 2),
        rhf.mo_coeff,
        mol,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # CircuitWF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    # Pass converged UCC orbitals to circuit wave function but still do optimization (just a speed-up)
    qWF = WaveFunctionCircuit(
        (2, 2),
        WF.c_mo,
        mol,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", True)

    # LR with SQ
    LR = projected.LinearResponse(WF, excitations="SD", triplet=True)
    LR.calc_excitation_energies()

    # LR with QSQ
    qLR = q_projected.quantumLR(qWF, "SD", triplet=True)

    qLR.run(do_rdm=True)
    excitation_energies = qLR.get_excitation_energies()

    assert np.allclose(excitation_energies, LR.excitation_energies, atol=10**-5)


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
        (2, 2),
        rhf.mo_coeff,
        mol,
        "SD",
    )

    # Optimize WF
    WF.run_wf_optimization_1step("BFGS", True)

    # Optimize WF with QSQ
    sampler = SamplerAer()
    mapper = ParityMapper(num_particles=(1, 1))

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    # Pass converged UCC orbitals to circuit wave function but still do optimization (just a speed-up)
    qWF = WaveFunctionCircuit(
        (2, 2),
        WF.c_mo,
        mol,
        QI,
    )

    qWF.run_wf_optimization_2step("rotosolve", True)

    # LR with SQ
    LR = allprojected.LinearResponse(WF, excitations="SD", triplet=True)
    LR.calc_excitation_energies()

    # LR with QSQ
    qLR = q_allprojected.quantumLR(qWF, "SD", triplet=True)

    qLR.run()
    excitation_energies = qLR.get_excitation_energies()

    assert np.allclose(excitation_energies, LR.excitation_energies, atol=10**-5)

test_LiH_allprojected()
test_LiH_naive()
test_LiH_projected()