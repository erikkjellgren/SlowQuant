import numpy as np
from pyscf.data import nist
import pyscf

import slowquant.unitary_coupled_cluster.linear_response.naive as naive  # pylint: disable=consider-using-from-import
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.molecularintegrals.integralfunctions import one_electron_integral_transform
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.operators import one_elec_op_0i_0a


def get_shield(geometry, basis, active_space, charge=0, unit='bohr'):
    """
    Calculate the NMR shielding constant
    """
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, charge=charge, unit=unit)
    rhf = mol.RHF().run()
    mo_coeff = rhf.mo_coeff

    # SlowQuant
    WF = WaveFunctionUCC(
        active_space,
        mo_coeff,
        mol,
        "SD",
    )

    # Optimize WF
    if active_space[1] == mol.nao:
        WF.run_wf_optimization_1step('SLSQP', False)
    else:
        WF.run_wf_optimization_1step('SLSQP', True)
    print("Energy elec", WF.energy_elec)


    # full space 1-RDM
    RDM1 = np.zeros((WF.num_inactive_orbs + WF.num_active_orbs + WF.num_virtual_orbs, WF.num_inactive_orbs + WF.num_active_orbs + WF.num_virtual_orbs))
    RDM1[:WF.num_inactive_orbs,:WF.num_inactive_orbs] += np.eye(WF.num_inactive_orbs) * 2
    RDM1[WF.num_inactive_orbs:WF.num_inactive_orbs + WF.num_active_orbs,WF.num_inactive_orbs:WF.num_inactive_orbs + WF.num_active_orbs] += WF.rdm1

    # Diamagnetic term
    atoms = WF.int_gen.atom_coordinates
    dia = np.zeros((len(atoms), 3, 3))

    for i in range(len(atoms)):
        dia_i = []
        origin = atoms[i,:]
        dia_ao = WF.int_gen.diamagnetic_shielding(common_orig=origin, rinv_orig=origin)

        for comp in dia_ao:
            dia_mo = one_electron_integral_transform(WF.c_mo, comp)
            dia_op = one_elec_op_0i_0a(dia_mo, WF.num_inactive_orbs, WF.num_active_orbs)
            dia_i.append(expectation_value(WF.ci_coeffs, [dia_op], WF.ci_coeffs, WF.ci_info))
        
        dia_i = np.array(dia_i).reshape((3,3))
        dia[i,:,:] = dia_i - dia_i.trace() * np.eye(3)
    

    # Paramagnetic term
    LR = naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    para = LR.get_paramagnetic_shielding()

    # Converting units
    sigma_dia   = np.einsum('xii->x', dia  * nist.ALPHA**2 * 1e6) / 3
    sigma_para  = np.einsum('xii->x', para * nist.ALPHA**2 * 1e6) / 3
    sigma_total = sigma_dia + sigma_para

    print('Shielding (in ppm):')
    for i in range(mol.natm):
        print(f'{i}: \tTotal={sigma_total[i]:.4f} \tDia={sigma_dia[i]:.4f} \tPara={sigma_para[i]:.4f}')

    return sigma_total

def test_H2_sto3g_naive():
    """
    Test of NMR shielding constants for naive LR with H2(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.7;
            H  0.0  0.0  -0.7;"""
    basis = 'STO-3G'
    active_space = (2,2)

    sigma = get_shield(geometry=geometry, basis=basis, active_space=active_space, unit='bohr')
    
    thresh = 10**-4

    # Check shielding constant - reference dalton mcscf
    assert abs(32.9334 - sigma[0]) < thresh
    assert abs(32.9334 - sigma[1]) < thresh


def test_LiH_sto3g_naive():
    """
    Test of NMR shielding constants for naive LR with LiH(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.7;
            Li  0.0  0.0  -0.7;"""
    basis = "STO-3G"
    active_space = (2,2)

    sigma = get_shield(geometry=geometry, basis=basis, active_space=active_space, unit='bohr')
    
    thresh = 10**-3

    # Check shielding constant - reference dalton mcscf
    assert abs(38.7983 - sigma[0]) < thresh
    assert abs(72.9730 - sigma[1]) < thresh
