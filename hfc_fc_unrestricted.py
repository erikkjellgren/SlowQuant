import numpy as np
import pyscf
from pyscf import scf, mcscf, fci
from pyscf.data import nist
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS


def get_hcf_fc_unrestricted(geometry, basis, active_space, unit='bohr', charge=0, spin=0):
    """
    Calculate hyperfine coupling constant (fermi-contact term) for a molecule
    """
    print("active space:", {active_space})
    #PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])

    res = mc.kernel(mf.mo_coeff)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    
    #Slowquant
    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccsdt",
        {"n_layers":2},
        include_active_kappa=True,
    )

    # WF.run_wf_optimization_1step("bfgs", True)

    # FC
    """ a_{iso}^K = \frac{f_k}{2\pi M} \bigg\{\bigg [[A^K_{\alpha}]_I - [A^K_{\beta}]_I\bigg] + \bigg[[A^K_{\alpha}]_A \Gamma^{[1]}_{\alpha} - [A^K_{\beta}]_A \Gamma^{[1]}_{\beta}\bigg] \bigg\}"""
    for atom in mol._atom:
        print(atom[0])
        amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
        mo_basis_a = amp_basis@WF.c_a_mo
        mo_basis_b = amp_basis@WF.c_b_mo
        h1mo_a = np.outer(np.conj(mo_basis_a), mo_basis_a)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs]
        h1mo_b = np.outer(np.conj(mo_basis_b), mo_basis_b)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs] 
        rdma = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
        rdmb = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
        rdma[WF.num_inactive_orbs: , WF.num_inactive_orbs:] = WF.rdm1aa
        rdmb[WF.num_inactive_orbs: , WF.num_inactive_orbs:] = WF.rdm1bb
        hfc = np.trace(h1mo_a@rdma  - h1mo_b@rdmb)
        g_k = 0
        m = 0
        if atom[0] == "H":
            g_k = 5.58569468
        if atom[0] == "O":
            g_k = -0.757516
        if atom[0] == "N":
            g_k = 0.40376100
        if np.absolute(WF.num_active_elec_alpha - WF.num_active_elec_beta) == 1:
            m = 0.5
        if np.absolute(WF.num_active_elec_alpha - WF.num_active_elec_beta) == 3:
            m = 1
        
        print("HFC without factor:", hfc)
        print("HFC:", 400.12*g_k/m*hfc, "MHz")
    



def OH_rad_hfc():
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;"""
    basis = '6311++gss-j'
    active_space = ((2,1),3)
    charge = 0
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=1
    
    get_hcf_fc_unrestricted(geometry=geometry, basis=basis, active_space=active_space, unit='angstrom', charge=charge, spin=spin)

def OH_cat_hfc():
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  1.0289;"""
    basis="6311++gss-j"
    active_space = ((3,1),4)
    charge = 1
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=2
    
    get_hcf_fc_unrestricted(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit='angstrom')

def NO_rad_hfc():
    geometry = """O  0.0   0.0  0.0;
        N  0.0  0.0  1.1508;"""
    basis = "STO-3G"
    active_space = ((2,1),3)
    charge = 0
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=1
    
    get_hcf_fc_unrestricted(geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit='angstrom')


OH_rad_hfc()
OH_cat_hfc()
NO_rad_hfc()

