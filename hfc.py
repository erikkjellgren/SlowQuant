
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

thetas = [0.0, -0.09222608518562322, 0.0, 0.0, -0.0902584279081604, 0.20277837232733795, 0.8812281245435793, 0.3287000015796323, 0.1749524558931268, 0.3792265792365249, -0.047142340989378376, 0.0, 0.0, -0.04736600943758396, 0.0, 1.0186268416214441, -0.40079631103459, 1.177746919858944, -0.6941994056018352, 1.1150615162132618, -2.569024876792717, -0.5146038930645087, 0.6022183211373238, 1.6105259988079677, -1.6862529180777948, -1.9968572631981223, -0.5790582649557015, -1.612720758692883, 0.12953940874657546, 1.126846269739397, -0.6315287748632721, 2.1645956081819637, 0.37342356281393424, 0.03153992602433851, -2.3469381715896502, 0.060567055481647585, 1.3167986208463711, 0.8607479569956811, 0.0031301433266161336, -1.5236020277714921, 1.341678409089755, 0.015405113614515577, 1.3077053842247481, 0.15936108644527436, 0.2896280295805148, -0.6352620375964291, 0.7719985344458347, 4.33749176027118, 0.03395391676237366, -0.10416253044380405]
c_a_mo = 
c_b_mo = 







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
 