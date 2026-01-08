import numpy as np
import pyscf
from pyscf import scf, mcscf
from pyscf.data import nist
import sys
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS


# Read geometry file
def my_read_xyz_file(inp):
    inp = inp[0]
    try:
        with open(inp, "r") as file:
            input_names = ["basis", "spin", "charge", "active_space"]
            input = ["", int(0), int(0), ()]
            geometry_list = []
            geometry = ""
            active_space_list = []
            lines = file.readlines()
            l = iter(lines)
            for line in l:
                new_line = line.strip().split()
                if len(new_line) >=5:
                    for a in range(len(input_names)):
                        if type(input[a]) == str:
                            input[a] = new_line[a+1]
                        elif type(input[a]) == int:
                            input[a] = int(new_line[a+1])
                        elif type(input[a]) == tuple:
                            for char in new_line[a+1]:
                                try:
                                    active_space_list.append(int(char))
                                except:
                                    continue
                elif len(new_line) == 4:
                    for a in range(len(new_line)):
                        geometry_list.append(new_line[a])
                    geometry_list.append(";")
                else:
                    continue
            active_space = ((active_space_list[0], active_space_list[1]), active_space_list[2])
            input[-1] = active_space
            geometry = " ".join(geometry_list)
    except FileNotFoundError: 
        print(f"file: {inp} not found")
    
    return input, geometry

# Define constants
mu0 = 1.256637 * 10 ** (-6)  # Vacuum permeability with units [N * A^-2]

me = 9.10938215 * 10 ** (-31)  # Mass of electron with units [kg]

ec = 1.602176487 * 10 ** (-19)  # Elementary charge with units [C]

muN = 5.05078324 * 10 ** (-27)  # Nuclear magneton with units [J * T^-1]

muP = 1.672621637 * 10 ** (-27)  # Proton mass [kg}

ge = 2.002319304386  # g-factor of electron [unitless]

a_0 = 5.29177210544 * 10 ** (-11) # Bohr radius [m]

def calculate_constant():
    r"""\frac{f_k}{2 \pi} = \frac{g_e e \mu_0 \mu_N}{12 m_e a_0^3 \pi} = 400.12 MHz"""
    constant = (ge * ec * mu0 * muN) / (12 * me * (a_0 ** 3) * np.pi)
    return constant*10**(-6)

# Define nuclear g-factors
def nuclear_g_factor(atom):
    if atom == "H":
        g_k = 2.79284734 * (2/1)
    elif atom == "B":
        g_k = 2.6886489 * (2/3)
    elif atom == "C":
        g_k = 0.7024118 * (2/1)
    elif atom == "N":
        g_k = 0.403761
    elif atom == "O":
        g_k = -1.89379 * (2/5)
    elif atom == "Sc":
        g_k = 4.756487 * (2/7)
    elif atom == "Ti":
        g_k = -0.78848 * (2/5)
    elif atom == "V":
        g_k = 5.1487057 * (2/7)
    elif atom == "Mn":
        g_k = 3.4532 * (2/5)
    else:
        print(f"No nuclear g-value is found for atom: {atom}")
    return g_k

gMn_alt = 3.46871790 * (2/5)

# Calculation
def get_hcf_fc_unrestricted(geometry, basis, active_space, unit='bohr', charge=0, spin=0):
    """
    Calculate hyperfine coupling constant (fermi-contact term) for an open-shell molecule with an unrestricted wavefunction
    """
    # Print molecular info
    print(f"geometry: {geometry}, basis: {basis}, active space:, {active_space}, charge: {charge}, spin (2s+1): {spin+1}")

    #PySCF UHF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    res = mc.kernel(mf.mo_coeff)
    
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    
    # Print info on Slowquant calculation
    print("Method: fuccsd, Layers: 2, Orbital Optimization: True")

    # Slowquant
    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccsd",
        {"n_layers":2},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_1step("bfgs", orbital_optimization=True)

    print(WF.energy_elec_RDM)
    
    # FC
    r""" a_{iso}^K = \frac{f_k}{2\pi M} \bigg\{\bigg [[A^K_{\alpha}]_I - [A^K_{\beta}]_I\bigg] + \bigg[[A^K_{\alpha}]_A \Gamma^{[1]}_{\alpha} - [A^K_{\beta}]_A \Gamma^{[1]}_{\beta}\bigg] \bigg\}"""
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
        m = spin * (1/2)
        f_k = calculate_constant()
        g_k = nuclear_g_factor(atom=atom[0])
        print("HFC without factor:", hfc)
        print("HFC:", f_k*g_k/m*hfc, "MHz")
 
def molecule():
    info = my_read_xyz_file(inp=sys.argv[1:])
    geometry = info[1]
    basis = info[0][0]
    spin = info[0][1]
    charge = info[0][2]
    active_space = info[0][3]

    get_hcf_fc_unrestricted(geometry=geometry, basis=basis, spin=spin, charge=charge, active_space=active_space, unit="angstrom")

molecule()
