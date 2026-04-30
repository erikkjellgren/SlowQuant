import numpy as np
import pyscf
from pyscf import scf, mcscf, fci
from pyscf.data import nist
import sys
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS

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

name = 0.2595517444453521

thetas = np.load(f'tiled_mo_oscar/{name}_thetas.npy')
c_a_mo = np.load(f'tiled_mo_oscar/{name}_a_mo.npy')
c_b_mo = np.load(f'tiled_mo_oscar/{name}_b_mo.npy')
rdma = np.load(f'tiled_mo_oscar/{name}_a_rdm.npy')
rdmb = np.load(f'tiled_mo_oscar/{name}_b_rdm.npy')
num_active_orbs = np.load(f'tiled_mo_oscar/{name}_num_active.npy')
num_inactive_orbs = np.load(f'tiled_mo_oscar/{name}_num_inactive.npy')
# Calculation

"""
Calculate hyperfine coupling constant (fermi-contact term) for an open-shell molecule with an unrestricted wavefunction
"""
# Print molecular info
# print(f"geometry: {geometry}, basis: {basis}, active space:, {active_space}, charge: {charge}, spin (2s+1): {spin+1}")
#PySCF UHF
mol = pyscf.M(atom="N   0.0  0.0     0.0; H   0.0  0.0 1.0362", basis="aug-cc-pvtz-j.nw", unit="angstrom", spin=2)
mol.build()

# FC
r""" a_{iso}^K = \frac{f_k}{2\pi M} \bigg\{\bigg [[A^K_{\alpha}]_I - [A^K_{\beta}]_I\bigg] + \bigg[[A^K_{\alpha}]_A \Gamma^{[1]}_{\alpha} - [A^K_{\beta}]_A \Gamma^{[1]}_{\beta}\bigg] \bigg\}"""
for atom in mol._atom:
    print(atom[0])
    amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
    mo_basis_a = amp_basis@c_a_mo
    mo_basis_b = amp_basis@c_b_mo
    h1mo_a = np.outer(np.conj(mo_basis_a), mo_basis_a)[:num_inactive_orbs + num_active_orbs, :num_inactive_orbs + num_active_orbs]
    h1mo_b = np.outer(np.conj(mo_basis_b), mo_basis_b)[:num_inactive_orbs + num_active_orbs, :num_inactive_orbs + num_active_orbs] 
    rdma_tmp = np.eye(num_inactive_orbs + num_active_orbs)
    rdmb_tmp = np.eye(num_inactive_orbs + num_active_orbs)
    rdma_tmp[num_inactive_orbs: , num_inactive_orbs:] = rdma
    rdmb_tmp[num_inactive_orbs: , num_inactive_orbs:] = rdmb

    hfc = np.trace(h1mo_a@rdma_tmp  - h1mo_b@rdmb_tmp)

    m = 2 * (1/2)
    f_k = calculate_constant()
    g_k = nuclear_g_factor(atom=atom[0])
    print("HFC without factor:", hfc)
    print("HFC:", f_k*g_k/m*hfc, "MHz")


