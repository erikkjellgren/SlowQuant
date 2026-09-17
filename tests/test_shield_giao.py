import numpy as np
import pyscf

from qiskit_aer.primitives import Sampler as SamplerAer
from qiskit_nature.second_q.mappers import JordanWignerMapper

from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.properties import properties

def test_shield_H2_sto3g():
    """
    Test of NMR shielding constants for with H2(2,2)/STO-3G with naive, project, statetransfer and selfconsistent LR
    """
    geometry = """H  0.0   0.0  0.7;
            H  0.0  0.0  -0.7;"""
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
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_naive = np.trace(dia + para, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="naive")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_qnaive = np.trace(dia + para, axis1=1, axis2=2) / 3

    print("\nProjected")
    # with SQ
    prop = properties(WF, lr_formulation="projected")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_proj = np.trace(dia + para, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="projected")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_qproj = np.trace(dia + para, axis1=1, axis2=2) / 3

    print("\nStatetransfer")
    # with SQ
    prop = properties(WF, lr_formulation="statetransfer")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_st = np.trace(dia + para, axis1=1, axis2=2) / 3

    print("\nSelfconsistent")
    # with SQ
    prop = properties(WF, lr_formulation="selfconsistent")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_sc = np.trace(dia + para, axis1=1, axis2=2) / 3

    shield = np.array([shield_naive, shield_proj, shield_st, shield_sc, shield_qnaive, shield_qproj])

    thresh = 10**-4

    # Check shielding constant - reference dalton mcscf
    assert np.all(abs(shield[:,0] - 27.5399) < thresh)
    assert np.all(abs(shield[:,1] - 27.5399) < thresh)

def test_shield_LiH_sto3g():
    """
    Test of NMR shielding constants for LiH(2,2)/STO-3G with naive, project, statetransfer and selfconsistent LR
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

    print("\nNaive")
    # with SQ
    prop = properties(WF, lr_formulation="naive")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_naive = np.trace(dia + para, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="naive")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_qnaive = np.trace(dia + para, axis1=1, axis2=2) / 3

    print("\nProjected")
    # with SQ
    prop = properties(WF, lr_formulation="projected")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_proj = np.trace(dia + para, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="projected")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_qproj = np.trace(dia + para, axis1=1, axis2=2) / 3

    print("\nStatetransfer")
    prop = properties(WF, lr_formulation="statetransfer")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_st = np.trace(dia + para, axis1=1, axis2=2) / 3

    print("\nSelfconsistent")
    prop = properties(WF, lr_formulation="selfconsistent")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_sc = np.trace(dia + para, axis1=1, axis2=2) / 3

    shield = np.array([shield_naive, shield_proj, shield_st, shield_sc, shield_qnaive, shield_qproj])

    thresh = 10**-3

    # Check shielding constant - reference dalton mcscf
    assert np.all(abs(shield[:,0] - 12.4078) < thresh)
    assert np.all(abs(shield[:,1] - 69.0138) < thresh)

def test_shield_LiH_sto3g_allprojected():
    """
    Test of NMR shielding constants for LiH(2,2)/STO-3G with allprojected LR
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

    # with SQ
    prop = properties(WF, lr_formulation="allprojected")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield = np.trace(dia + para, axis1=1, axis2=2) / 3

    # with QSQ
    prop = properties(qWF, lr_formulation="allprojected")
    dia, para = prop.get_nuclear_shielding_tensor_giao()
    shield_q = np.trace(dia + para, axis1=1, axis2=2) / 3

    # Check shielding constant - reference dalton mcscf
    assert np.allclose(shield, shield_q)
