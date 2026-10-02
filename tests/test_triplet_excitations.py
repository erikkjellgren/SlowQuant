import numpy as np
from qiskit_aer.primitives import Sampler as SamplerAer
from qiskit_nature.second_q.mappers import JordanWignerMapper

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.properties import Properties

def test_H2_sto3g_triplet():
    """
    Test of triplet excitation energies for naive, projected, selfconsistent and statetransfer LR with H2(2,2)/STO-3G
    """
    geometry = """H  0.0  0.0  0.0;
            H  0.74  0.0  0.0;"""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(geometry, distance_unit='angstrom')
    SQobj.set_basis_set('sto-3g')

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # Wavefunction
    WF = WaveFunctionUCC(
        (2,2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
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
        SQobj,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", False)

    # Linear Response
    print("\nNaive")
    prop = Properties(WF, "naive")
    excita_naive, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    # with qLR
    prop = Properties(qWF, "naive")
    excita_qnaive, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    print("\nProjected")
    prop = Properties(WF, "proj")
    excita_proj, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)    

    # with qLR
    prop = Properties(qWF, "proj")
    excita_qproj, _ = prop.get_excitation_energies(triplet=True, osc_strs=True) 

    print("\nSelfconsistent")
    prop = Properties(WF, "sc")
    excita_sc, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    print("\nStatetransfer")
    prop = Properties(WF, "st")
    excita_st, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    excita = np.array([excita_naive, excita_qnaive, excita_proj, excita_qproj, excita_sc, excita_st])

    thresh = 10**-4

    # Check excitation energies
    assert np.all(abs(excita[:,0] - 0.606510) < thresh)


def test_LiH_sto3g_triplet():
    """
    Test of triplet excitation energies for naive, projected, selfconsistent and statetransfer LR with LiH(2,2)/STO-3G
    """
    # Slowquant Object with parameters and setup
    geometry = """Li  0.0  0.0  0.0;
            H  0.8  0.0  0.0;"""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(geometry, distance_unit='angstrom')
    SQobj.set_basis_set('sto-3g')

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # Wavefunction
    WF = WaveFunctionUCC(
        (2,2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
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
        SQobj,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", False)

    # Linear Response
    print("\nNaive")
    prop = Properties(WF, "naive")
    excita_naive, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    # with qLR
    prop = Properties(qWF, "naive")
    excita_qnaive, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    print("\nProjected")
    prop = Properties(WF, "proj")
    excita_proj, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)   

    # with qLR
    prop = Properties(qWF, "proj")
    excita_qproj, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)  

    print("\nSelfconsistent")
    prop = Properties(WF, "sc")
    excita_sc, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    print("\nStatetransfer")
    prop = Properties(WF, "st")
    excita_st, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    excita = np.array([excita_naive, excita_qnaive, excita_proj, excita_qproj, excita_sc, excita_st])

    thresh = 10**-4

    # Check excitation energies
    assert np.all(abs(excita[:,0] - 0.102103) < thresh)
    assert np.all(abs(excita[:,1] - 0.138627) < thresh)
    assert np.all(abs(excita[:,2] - 0.138627) < thresh)
    assert np.all(abs(excita[:,3] - 0.459503) < thresh)
    assert np.all(abs(excita[:,4] - 0.676406) < thresh)
    assert np.all(abs(excita[:,5] - 0.676406) < thresh)
    assert np.all(abs(excita[:,6] - 0.786396) < thresh)
    assert np.all(abs(excita[:,7] - 2.120330) < thresh)
    assert np.all(abs(excita[:,8] - 2.195168) < thresh)
    assert np.all(abs(excita[:,9] - 2.195168) < thresh)
    assert np.all(abs(excita[:,10] - 2.597856) < thresh)
    assert np.all(abs(excita[:,11] - 3.060383) < thresh)


def test_LiH_sto3g_triplet_proj_q():
    """
    Test of triplet excitation energies for allprojected and projected-statetransfer LR with LiH(2,2)/STO-3G
    """
    # Slowquant Object with parameters and setup
    geometry = """Li  0.0  0.0  0.0;
            H  0.8  0.0  0.0;"""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(geometry, distance_unit='angstrom')
    SQobj.set_basis_set('sto-3g')

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # Wavefunction
    WF = WaveFunctionUCC(
        (2,2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
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
        SQobj,
        QI,
    )
    qWF.run_wf_optimization_2step("rotosolve", False)

    # Linear Response
    print("\nAllprojected")
    prop = Properties(WF, "allproj")
    excita_allproj, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    # with qLR
    prop = Properties(qWF, "allproj")
    excita_qallproj, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    print("\nProjected-statetransfer")
    prop = Properties(WF, "projst")
    excita_projst, _ = prop.get_excitation_energies(triplet=True, osc_strs=True)

    excita = np.array([excita_allproj, excita_qallproj, excita_projst])

    thresh = 10**-4

    # Check excitation energies
    assert np.all(abs(excita[:,0] - 0.102369) < thresh)
    assert np.all(abs(excita[:,1] - 0.142590) < thresh)
    assert np.all(abs(excita[:,2] - 0.142590) < thresh)
    assert np.all(abs(excita[:,3] - 0.459633) < thresh)
    assert np.all(abs(excita[:,4] - 0.713103) < thresh)
    assert np.all(abs(excita[:,5] - 0.713103) < thresh)
    assert np.all(abs(excita[:,6] - 0.789325) < thresh)
    assert np.all(abs(excita[:,7] - 2.120450) < thresh)
    assert np.all(abs(excita[:,8] - 2.195230) < thresh)
    assert np.all(abs(excita[:,9] - 2.195230) < thresh)
    assert np.all(abs(excita[:,10] - 2.598082) < thresh)
    assert np.all(abs(excita[:,11] - 3.063790) < thresh)
