import numpy as np
from pyscf.data import nist
import pyscf
from scipy.linalg import solve

import slowquant.unitary_coupled_cluster.linear_response.naive as naive  # pylint: disable=consider-using-from-import
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


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
    dia = np.zeros((mol.natm, 3, 3))

    for i in range(mol.natm):
        print(type(mol.atom_coord(i)))
        mol.set_common_orig(mol.atom_coord(i))
        mol.set_rinv_origin(mol.atom_coord(i))

        dia_ao = mol.intor('int1e_cg_a11part', comp=9)
        print(dia_ao.shape)
        mo_integrals = np.einsum('uj,xuv,vi->xij', WF.c_mo, dia_ao, WF.c_mo)
        e11 = np.einsum('xij,ij->x', mo_integrals, RDM1).reshape(3,3)
        dia[i,:,:] = e11 - e11.trace() * np.eye(3)


    # Singlet Linear Response
    LR = naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()

    #Paramagnetic term
    para = np.zeros_like(dia)

    for i in range(mol.natm):
        mol.set_common_orig(mol.atom_coord(i))
        mol.set_rinv_origin(mol.atom_coord(i))

        # PSO
        ao = mol.intor('int1e_prinvxp', 3)
        property_gradient = LR.get_property_gradient(ao)
        pso = solve(LR.hessian, property_gradient)
        # Anguar Momentum
        ao = mol.intor('int1e_cg_irxp', 3) / 2
        am = LR.get_property_gradient(ao)
        # Paramagnetic term
        para[i,:,:] -= np.einsum('ix,iy->xy', pso, am)


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

test_H2_sto3g_naive()