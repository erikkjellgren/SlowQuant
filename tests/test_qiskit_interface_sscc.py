import numpy as np
import pyscf

from qiskit_aer.primitives import Sampler as SamplerAer
from qiskit_nature.second_q.mappers import ParityMapper

from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC

from slowquant.unitary_coupled_cluster.properties import properties
from slowquant.qiskit_interface.properties import properties as q_properties

def test_H2_sto3g_naive():
    """
    Test of spin-spin coupling constants for naive LR with H2(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.0;
            H  1.39  0.0  0.0;"""
    basis = 'STO-3G'
    active_space = (2,2)

    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit='bohr')
    rhf = mol.RHF().run()
    mo_coeff = rhf.mo_coeff

    # SlowQuant
    WF = WaveFunctionUCC(
        active_space,
        mo_coeff,
        mol,
        "SD",
    )
    WF.run_wf_optimization_1step('SLSQP', False)

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

    # SSCC with SQ
    prop = properties(WF)
    dso, pso, fc, sd = prop.get_spin_spin_coupling_constant()
    total_sq = dso + pso + fc + sd

    # SSCC with QSQ
    q_prop = q_properties(qWF)
    dso, pso, fc, sd = q_prop.get_spin_spin_coupling_constant()
    total_qsq = dso + pso + fc + sd

    assert np.allclose(total_sq, total_qsq, atol=10**-4)


def test_LiH_sto3g_naive():
    """
    Test of spin-spin coupling constants for naive LR with LiH(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.0;
            Li  1.5  0.0  0.0;"""
    basis = "STO-3G"
    active_space = (2,2)

    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit='bohr')
    rhf = mol.RHF().run()
    mo_coeff = rhf.mo_coeff

    # SlowQuant
    WF = WaveFunctionUCC(
        active_space,
        mo_coeff,
        mol,
        "SD",
    )
    WF.run_wf_optimization_1step('SLSQP', True)

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

    # SSCC with SQ
    prop = properties(WF)
    dso, pso, fc, sd = prop.get_spin_spin_coupling_constant()
    total_sq = dso + pso + fc + sd

    # SSCC with QSQ
    q_prop = q_properties(qWF)
    dso, pso, fc, sd = q_prop.get_spin_spin_coupling_constant()
    total_qsq = dso + pso + fc + sd
    
    assert np.allclose(total_sq, total_qsq, atol=10**-4)
