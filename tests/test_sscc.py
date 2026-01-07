import numpy as np
from pyscf.data.gyro import get_nuc_g_factor
from pyscf.data import nist
import pyscf
from scipy.linalg import solve

import slowquant.unitary_coupled_cluster.linear_response.naive as naive  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.naive_triplet as naive_t  # pylint: disable=consider-using-from-import
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def dso_integral(mol, orig1, orig2):
    '''Integral of vec{r}vec{r}/(|r-orig1|^3 |r-orig2|^3)
    Ref. JCP, 73, 5718'''
    NUMINT_GRIDS = 30
    from pyscf import gto
    t, w = np.polynomial.legendre.leggauss(NUMINT_GRIDS)
    a = (1+t)/(1-t) * .8
    w *= 2/(1-t)**2 * .8
    fakemol = gto.Mole()
    fakemol._atm = np.asarray([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
    fakemol._bas = np.asarray([[0, 1, NUMINT_GRIDS, 1, 0, 3, 3+NUMINT_GRIDS, 0]],
                                 dtype=np.int32)
    p_cart2sph_factor = 0.488602511902919921
    fakemol._env = np.hstack((orig2, a**2, a**2*w*4/np.pi**.5/p_cart2sph_factor))
    fakemol._built = True

    pmol = mol + fakemol
    pmol.set_rinv_origin(orig1)
    # <nabla i, j | k>  k is a fictitious basis for numerical integraion
    mat1 = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, pmol.nbas))
    # <i, j | nabla k>
    mat  = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(mol.nbas, pmol.nbas, 0, mol.nbas, 0, mol.nbas))
    mat += mat1.transpose(0,3,1,2) + mat1.transpose(0,3,2,1)
    return mat

def _atom_gyro_list(mol):
    gyro = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in mol.nucprop:
            prop = mol.nucprop[symb]
            mass = prop.get('mass', None)
            gyro.append(get_nuc_g_factor(symb, mass))
        else:
            # Get default isotope
            gyro.append(get_nuc_g_factor(symb))
    return np.array(gyro)

def convert_unit(e11, mol, nuc_pair):
    # unit conversions
    e11 = e11*nist.ALPHA**4
    nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2Hz = nist.HARTREE2J / nist.PLANCK
    unit = au2Hz * nuc_magneton ** 2
    iso_ssc = unit * np.einsum('kii->k', e11) / 3
    natm = mol.natm
    ktensor = np.zeros((natm,natm))
    for k, (i, j) in enumerate(nuc_pair):
        ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
    gyro = _atom_gyro_list(mol)
    jtensor = np.einsum('ij,i,j->ij', ktensor, gyro, gyro)
    return jtensor

def get_sscc(geometry, basis, active_space, charge=0, unit='bohr'):
    """
    Calculate the spin-spin coupling constant for a system
    """
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, charge=charge, unit=unit)
    rhf = mol.RHF().run()
    mo_coeff = rhf.mo_coeff

    nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]

    # SlowQuant
    WF = WaveFunctionUCC(
        mol.nelectron,
        active_space,
        mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "SD",
    )

    # Optimize WF
    WF.run_wf_optimization_1step('SLSQP', True)
    print("Energy elec", WF.energy_elec)

    # DSO term
    dso = np.zeros((len(nuc_pair), 3, 3))
    RDM1 = np.zeros((WF.num_inactive_orbs + WF.num_active_orbs + WF.num_virtual_orbs, WF.num_inactive_orbs + WF.num_active_orbs + WF.num_virtual_orbs))
    RDM1[:WF.num_inactive_orbs,:WF.num_inactive_orbs] += np.eye(WF.num_inactive_orbs) * 2
    RDM1[WF.num_inactive_orbs:WF.num_inactive_orbs + WF.num_active_orbs,WF.num_inactive_orbs:WF.num_inactive_orbs + WF.num_active_orbs] += (
    WF.rdm1
    )
    for k, (i,j) in enumerate(nuc_pair):
        dso_ao = dso_integral(mol, mol.atom_coord(i), mol.atom_coord(j)).reshape(9, mol.nao, mol.nao)
        mo_integrals = np.einsum('uj,xuv,vi->xij', WF.c_mo, dso_ao, WF.c_mo)
        a11 = -np.trace(mo_integrals @ RDM1, axis1=1, axis2=2).reshape(3,3)
        dso[k,:,:] += a11 - a11.trace() * np.eye(3)

    # Singlet Linear Response
    LR = naive.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    # PSO
    h1 = []
    d1 = []
    for i in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(i))
        h1ao = mol.intor('int1e_prinvxp', 3)
        property_gradient = LR.get_property_gradient(h1ao)
        h1.append(property_gradient)
        d1.append(solve(LR.hessian, property_gradient))

    pso = np.zeros_like(dso)
    for k, (i,j) in enumerate(nuc_pair):
        pso[k,:,:] -= np.einsum('ix,iy->xy', d1[i], h1[j])

    # Triplet Linear Response
    LR = naive_t.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    # FC
    h1 = []
    d1 = []
    for atom in mol._atom:
        amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
        h1ao = np.array([np.outer(np.conj(amp_basis), amp_basis)]) * nist.G_ELECTRON * 2/3 * np.pi
        property_gradient = LR.get_property_gradient(h1ao)
        h1.append(property_gradient)
        d1.append(solve(LR.hessian, property_gradient))

    fc = np.zeros_like(dso)
    for k, (i,j) in enumerate(nuc_pair):
        fc[k,:,:] -= np.ones((3,3)) * np.einsum('ix,iy->xy',d1[i], h1[j])

    # SD+FC
    h1 = []
    d1 = []
    for i in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(i))
        a01p = mol.intor('int1e_sa01sp', 12).reshape(3, 4, mol.nao, mol.nao) * nist.G_ELECTRON / 4
        h1ao = -(a01p[:,:3] + a01p[:,:3].transpose(0,1,3,2))
        property_gradient = LR.get_property_gradient(h1ao)
        h1.append(property_gradient)
        d1.append(solve(LR.hessian, property_gradient))

    fcsd = np.zeros_like(dso)
    for k, (i,j) in enumerate(nuc_pair):
        fcsd[k,:,:] -= np.einsum('ixw,iyw->xy', d1[i], h1[j])

    j_tensor_dso  = convert_unit(dso, mol, nuc_pair)
    j_tensor_pso  = convert_unit(pso, mol, nuc_pair)
    j_tensor_fcsd = convert_unit(fcsd, mol, nuc_pair)
    j_tensor_fc   = convert_unit(fc, mol, nuc_pair)
    j_tensor_total = j_tensor_dso + j_tensor_pso + j_tensor_fcsd

    print('SSCC (in Hz):')
    for (i,j) in nuc_pair:
        print(f'{i} {j}: \tDSO={j_tensor_dso[i,j]:.4f} \tPSO={j_tensor_pso[i,j]:.4f} \tSD={j_tensor_fcsd[i,j]-j_tensor_fc[i,j]:.4f} \tFC={j_tensor_fc[i,j]:.4f} \tTotal={j_tensor_total[i,j]:.4f}')

    return j_tensor_total

def test_H2_sto3g_naive():
    """
    Test of sscc for naive LR
    """
    geometry = """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;"""
    basis = 'STO-3G'
    active_space = (2,2)

    sscc = get_sscc(geometry=geometry, basis=basis, active_space=active_space, unit='angstrom')
    
    thresh = 10**-2

    # Check coupling constant - reference dalton mcscf
    assert abs(383.685042 - sscc[0,1]) < thresh


def test_LiH_sto3g_naive():
    """
    Test of sscc for naive LR
    """
    geometry = """H  0.0   0.0  0.0;
            Li  0.8  0.0  0.0;"""
    basis = "STO-3G"
    active_space = (2,2)

    sscc = get_sscc(geometry=geometry, basis=basis, active_space=active_space, unit='angstrom')
    
    thresh = 10**-2

    # Check coupling constant - reference dalton mcscf
    assert abs(-63.380071 - sscc[0,1]) < thresh

test_H2_sto3g_naive()
test_LiH_sto3g_naive()
