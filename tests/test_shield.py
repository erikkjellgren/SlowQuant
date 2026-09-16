import numpy as np
import pyscf

from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.linear_response import naive, projected, statetransfer, selfconsistent
from slowquant.unitary_coupled_cluster.properties import properties


def test_H2_sto3g():
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

    print("Naive")
    prop_naive = properties(WF, property_options={"excitations": "SD", "lr_formulation": naive})
    dia_naive, para_naive = prop_naive.get_nuclear_shielding_tensor()
    shield_naive = np.trace(dia_naive + para_naive, axis1=1, axis2=2) / 3

    print("Projected")
    prop_proj = properties(WF, property_options={"excitations": "SD", "lr_formulation": projected})
    dia_proj, para_proj = prop_proj.get_nuclear_shielding_tensor()
    shield_proj = np.trace(dia_proj + para_proj, axis1=1, axis2=2) / 3

    print("Statetransfer")
    prop_st = properties(WF, property_options={"excitations": "SD", "lr_formulation": statetransfer})
    dia_st, para_st = prop_st.get_nuclear_shielding_tensor()
    shield_st = np.trace(dia_st + para_st, axis1=1, axis2=2) / 3

    print("Selfconsistent")
    prop_sc = properties(WF, property_options={"excitations": "SD", "lr_formulation": selfconsistent})
    dia_sc, para_sc = prop_sc.get_nuclear_shielding_tensor()
    shield_sc = np.trace(dia_sc + para_sc, axis1=1, axis2=2) / 3

    shield = np.array([shield_naive, shield_proj, shield_st, shield_sc])

    thresh = 10**-4

    # Check shielding constant - reference dalton mcscf
    assert np.all(abs(shield[:,0] - 32.9334) < thresh)
    assert np.all(abs(shield[:,1] - 32.9334) < thresh)


def test_LiH_sto3g():
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

    print("Naive")
    prop_naive = properties(WF, property_options={"excitations": "SD", "lr_formulation": naive})
    dia_naive, para_naive = prop_naive.get_nuclear_shielding_tensor()
    shield_naive = np.trace(dia_naive + para_naive, axis1=1, axis2=2) / 3

    print("Projected")
    prop_proj = properties(WF, property_options={"excitations": "SD", "lr_formulation": projected})
    dia_proj, para_proj = prop_proj.get_nuclear_shielding_tensor()
    shield_proj = np.trace(dia_proj + para_proj, axis1=1, axis2=2) / 3

    print("Statetransfer")
    prop_st = properties(WF, property_options={"excitations": "SD", "lr_formulation": statetransfer})
    dia_st, para_st = prop_st.get_nuclear_shielding_tensor()
    shield_st = np.trace(dia_st + para_st, axis1=1, axis2=2) / 3

    print("Selfconsistent")
    prop_sc = properties(WF, property_options={"excitations": "SD", "lr_formulation": selfconsistent})
    dia_sc, para_sc = prop_sc.get_nuclear_shielding_tensor()
    shield_sc = np.trace(dia_sc + para_sc, axis1=1, axis2=2) / 3

    shield = np.array([shield_naive, shield_proj, shield_st, shield_sc])

    thresh = 10**-3

    # Check shielding constant - reference dalton mcscf
    assert np.all(abs(shield[:,0] - 38.7983) < thresh)
    assert np.all(abs(shield[:,1] - 72.9730) < thresh)
