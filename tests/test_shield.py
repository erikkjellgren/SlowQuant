import numpy as np
import pyscf

from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.linear_response import naive
from slowquant.unitary_coupled_cluster.properties import properties


def test_H2_sto3g_naive():
    """
    Test of NMR shielding constants for naive LR with H2(2,2)/STO-3G
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

    prop = properties(WF, property_options={"excitations": "SD", "lr_formulation": naive})
    dia, para = prop.get_nuclear_shielding_tensor()

    thresh = 10**-4

    # Check shielding constant - reference dalton mcscf
    assert abs(32.9334 - np.trace(dia[0,:,:] + para[0,:,:]) / 3) < thresh
    assert abs(32.9334 - np.trace(dia[1,:,:] + para[1,:,:]) / 3) < thresh


def test_LiH_sto3g_naive():
    """
    Test of NMR shielding constants for naive LR with LiH(2,2)/STO-3G
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

    prop = properties(WF)
    dia, para = prop.get_nuclear_shielding_tensor()

    thresh = 10**-3

    # Check shielding constant - reference dalton mcscf
    assert abs(38.7983 - np.trace(dia[0,:,:] + para[0,:,:]) / 3) < thresh
    assert abs(72.9730 - np.trace(dia[1,:,:] + para[1,:,:]) / 3) < thresh
