import pyscf
import numpy as np

from qiskit_aer.primitives import Sampler as SamplerAer
from qiskit_nature.second_q.mappers import JordanWignerMapper

from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.properties import Properties


def test_H2_sto3g_naive_q():
    """
    Test of polarisability for naive, projected, selfconsistent and statetransfer LR with H2(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;"""
    # PySCF
    mol = pyscf.M(atom=geometry, basis='sto-3g', unit='angstrom')
    rhf = mol.RHF().run()
    mo_coeff = rhf.mo_coeff

    # SlowQuant
    WF = WaveFunctionUCC(
        (2,2),
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
        (2,2),
        WF.c_mo,
        mol,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", False)

    print("\nNaive")
    prop = Properties(WF, "naive")
    alpha_naive = prop.get_polarisability()

    # qLR
    prop = Properties(qWF, "naive")
    alpha_qnaive = prop.get_polarisability()

    print("\nProjected")
    prop = Properties(WF, "proj")
    alpha_proj = prop.get_polarisability()

    # qLR
    prop = Properties(qWF, "proj")
    alpha_qproj = prop.get_polarisability()

    print("\nSelfconsistent")
    prop = Properties(WF, "sc")
    alpha_sc = prop.get_polarisability()

    print("\nStatetransfer")
    prop = Properties(WF, "st")
    alpha_st = prop.get_polarisability()

    alpha = np.array([alpha_naive, alpha_qnaive, alpha_proj, alpha_qproj, alpha_sc, alpha_st])

    thresh = 10**-4

    # Check excitation energies - reference dalton mcscf
    assert np.all(abs(alpha[:,0,0] - 2.775271948863) < thresh)
    assert np.all(abs(alpha[:,1,1] - 0.0) < thresh)
    assert np.all(abs(alpha[:,2,2] - 0.0) < thresh)


def test_LiH_sto3g_naive_q():
    """
    Test of polarisability for naive, projected, selfconsistent and statetransfer LR with LiH(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.0;
            Li  0.8  0.0  0.0;"""
        # PySCF
    mol = pyscf.M(atom=geometry, basis='sto-3g', unit='angstrom')
    rhf = mol.RHF().run()
    mo_coeff = rhf.mo_coeff

    # SlowQuant
    WF = WaveFunctionUCC(
        (2,2),
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
        (2,2),
        WF.c_mo,
        mol,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", False)

    print("\nNaive")
    prop = Properties(WF, "naive")
    alpha_naive = prop.get_polarisability()

    # qLR
    prop = Properties(qWF, "naive")
    alpha_qnaive = prop.get_polarisability()

    print("\nProjected")
    prop = Properties(WF, "proj")
    alpha_proj = prop.get_polarisability()

    # qLR
    prop = Properties(qWF, "proj")
    alpha_qproj = prop.get_polarisability()

    print("\nSelfconsistent")
    prop = Properties(WF, "sc")
    alpha_sc = prop.get_polarisability()

    print("\nStatetransfer")
    prop = Properties(WF, "st")
    alpha_st = prop.get_polarisability()

    alpha = np.array([alpha_naive, alpha_qnaive, alpha_proj, alpha_qproj, alpha_sc, alpha_st])

    thresh = 10**-2

    # Check excitation energies - reference dalton mcscf
    assert np.all(abs(alpha[:,0,0] - 0.5238650005008) < thresh)
    assert np.all(abs(alpha[:,1,1] - 20.01552907544) < thresh)
    assert np.all(abs(alpha[:,2,2] - 20.01552907544) < thresh)


def test_LiH_sto3g_proj_q():
    """
    Test of polarisability for allprojected and projectedstatetransfer LR with LiH(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.0;
            Li  0.8  0.0  0.0;"""
        # PySCF
    mol = pyscf.M(atom=geometry, basis='sto-3g', unit='angstrom')
    rhf = mol.RHF().run()
    mo_coeff = rhf.mo_coeff

    # SlowQuant
    WF = WaveFunctionUCC(
        (2,2),
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
        (2,2),
        WF.c_mo,
        mol,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", False)

    print("\nAllprojected")
    prop = Properties(WF, "allproj")
    alpha_allproj = prop.get_polarisability()

    # qLR
    prop = Properties(qWF, "allproj")
    alpha_qallproj = prop.get_polarisability()

    print("\nProjected-statetransfer")
    prop = Properties(WF, "projst")
    alpha_projst = prop.get_polarisability()

    thresh = 10**-4

    assert np.allclose(alpha_allproj, alpha_qallproj, atol=thresh)
    assert np.allclose(alpha_allproj, alpha_projst, atol=thresh)
