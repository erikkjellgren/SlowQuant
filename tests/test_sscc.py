import numpy as np
import pyscf

from qiskit_aer.primitives import Sampler as SamplerAer
from qiskit_nature.second_q.mappers import JordanWignerMapper

from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.properties import properties

def test_sscc_H2_sto3g():
    """
    Test of spin-spin coupling constants with H2(2,2)/STO-3G for naive, projected, selfconsistent and statetransfer LR
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
    mapper = JordanWignerMapper()

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunctionCircuit(
        active_space,
        WF.c_mo,
        mol,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", False)

    print("\nNaive")
    # with SQ
    prop = properties(WF, lr_formulation="naive")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_naive = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="naive")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_qnaive = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    print("\nProjected")
    # with SQ
    prop = properties(WF, lr_formulation="projected")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_proj = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="projected")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_qproj = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    print("\nSelfconsistent")
    # with SQ
    prop = properties(WF, lr_formulation="selfconsistent")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_sc = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    print("\nStatetransfer")
    # with SQ
    prop = properties(WF, lr_formulation="statetransfer")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_st = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    j = np.array([j_naive, j_qnaive, j_proj, j_qproj, j_sc, j_st])

    thresh = 10**-3

    # Check coupling constant - reference dalton mcscf
    assert np.all(abs(381.5641 - j[:,0]) < thresh)


def test_sscc_LiH_sto3g():
    """
    Test of spin-spin coupling constants with LiH(2,2)/STO-3G for naive, projected, selfconsistent and statetransfer LR
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
    mapper = JordanWignerMapper()

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunctionCircuit(
        active_space,
        WF.c_mo,
        mol,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", True)

    print("\nNaive")
    # with SQ
    prop = properties(WF, lr_formulation="naive")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_naive = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="naive")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_qnaive = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    print("\nProjected")
    # with SQ
    prop = properties(WF, lr_formulation="projected")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_proj = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="projected")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_qproj = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    print("\nSelfconsistent")
    # with SQ
    prop = properties(WF, lr_formulation="selfconsistent")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_sc = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    print("\nStatetransfer")
    # with SQ
    prop = properties(WF, lr_formulation="statetransfer")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_st = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    j = np.array([j_naive, j_qnaive, j_proj, j_qproj, j_sc, j_st])
    
    thresh = 10**-2

    # Check coupling constant - reference dalton mcscf
    assert np.all(abs(-68.2687 - j[:,0]) < thresh)

def test_sscc_LiH_sto3g_projected_q():
    """
    Test of spin-spin coupling constants for LiH(2,2)/STO-3G with allprojected and projected_statetransfer LR
    """
    geometry = """H  0.0   0.0  0.7;
            Li  0.0  0.0  -0.7;"""
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
    mapper = JordanWignerMapper()

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunctionCircuit(
        active_space,
        WF.c_mo,
        mol,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", True)

    print("\nAllprojected")
    # with SQ
    prop = properties(WF, lr_formulation="allprojected")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_proj = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="allprojected")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_qproj = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    assert np.allclose(j_proj, j_qproj)

    print("\nProjected_statetransfer")
    # with SQ
    prop = properties(WF, lr_formulation="projected_statetransfer")
    dso, pso, fc, sd = prop.get_spin_spin_coupling_tensor()
    j_st = np.trace(dso + pso + fc + sd, axis1=1, axis2=2) / 3

    assert np.allclose(j_proj,j_st)

test_sscc_H2_sto3g()
test_sscc_LiH_sto3g()
test_sscc_LiH_sto3g_projected_q()