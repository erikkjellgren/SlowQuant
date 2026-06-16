import numpy as np
import pyscf
from pyscf import scf, mcscf
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.qiskit_interface.unrestricted_circuit_wavefunction import UnrestrictedWaveFunctionCircuit
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from slowquant.qiskit_interface.interface import QuantumInterface

def hfc_qiskit(geometry, basis, active_space, unit='bohr', charge=0, spin=0):
    r""" Calculate fermi-contact term of the hyperfine coupling constant
        Wavefunction optimized classical, and then turn into a circuit
        """
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    
    #Slowquant
    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccsd",
        {"n_layers":1},
        include_active_kappa=True,
    )
    print(WF.energy_elec_RDM)
    WF.run_wf_optimization_1step("bfgs", True)

    #Mapper
    mapper = JordanWignerMapper()
    #Sampler
    primitive = Sampler(run_options={"shots": None})
    QI = QuantumInterface(primitive, "fUCCSD", mapper, ansatz_options={"n_layers":1})
    qWF = UnrestrictedWaveFunctionCircuit(
        mol.nelectron,
        active_space,
        (WF.c_a_mo, WF.c_b_mo),
        h_core,
        g_eri,
        QI,
    )
    qWF.thetas = WF.thetas


    # FC
    r""" a_{iso}^K = \frac{f_k}{2\pi M} \bigg\{\bigg [[A^K_{\alpha}]_I - [A^K_{\beta}]_I\bigg] + \bigg[[A^K_{\alpha}]_A \Gamma^{[1]}_{\alpha} - [A^K_{\beta}]_A \Gamma^{[1]}_{\beta}\bigg] \bigg\}"""
    for atom in mol._atom:
        print(atom[0])
        amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
        mo_basis_a = amp_basis@WF.c_a_mo
        mo_basis_b = amp_basis@WF.c_b_mo
        h1mo_a = np.outer(np.conj(mo_basis_a), mo_basis_a)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs]
        h1mo_b = np.outer(np.conj(mo_basis_b), mo_basis_b)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs] 
        # print("a_alpha", h1mo_a)
        # print("a_beta", h1mo_b)
        rdma = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
        rdmb = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
        rdma[WF.num_inactive_orbs: , WF.num_inactive_orbs:] = WF.rdm1aa
        rdmb[WF.num_inactive_orbs: , WF.num_inactive_orbs:] = WF.rdm1bb
        # print("alpha", rdma, "beta", rdmb)
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
        if np.absolute(WF.num_active_elec_alpha - WF.num_active_elec_beta) == 2:
            m = 1
        
        print("HFC without factor:", hfc)
        print("HFC:", 400.12*g_k/m*hfc, "MHz")


def OH_rad_hfc():
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;"""
    basis = '6-311++gss-j'
    active_space = ((2,1),3)
    charge = 0
    #the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin=1
    
    hfc_qiskit(geometry=geometry, basis=basis, active_space=active_space, unit='angstrom', charge=charge, spin=spin)

OH_rad_hfc()