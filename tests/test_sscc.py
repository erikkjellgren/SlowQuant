import numpy as np
import pyscf

from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.properties import properties

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

    prop = properties(WF)
    dso, pso, fc, sd = prop.get_spin_spin_coupling_constant()
    total = dso + pso + fc + sd

    thresh = 10**-3

    # Check coupling constant - reference dalton mcscf
    assert abs(381.5641 - np.trace(total[0,:,:])/3) < thresh


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

    prop = properties(WF)
    dso, pso, fc, sd = prop.get_spin_spin_coupling_constant()
    total = dso + pso + fc + sd
    
    thresh = 10**-2

    # Check coupling constant - reference dalton mcscf
    assert abs(-68.2687 - np.trace(total[0,:,:])/3) < thresh
