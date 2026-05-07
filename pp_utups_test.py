import numpy as np
import os
import pickle
import pyscf
from pyscf import scf, mcscf, fci
from pyscf.data import nist
import sys
# from qiskit.quantum_info import SparsePauliOp
# from qiskit_nature.second_q.operators import FermionicOp
# from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.operators import a_op
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.qiskit_interface.interface import FermionicMapper, FermionicOperator, FermionicOp
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.unitary_coupled_cluster.unrestricted_operators import one_elec_op_0i_0a_HFC
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
# from slowquant.unitary_coupled_cluster.fermionic_operator import get_folded_operator


mol = pyscf.M(atom="N   0.0  0.0           0.0; H   0.0  0.0 1.0362", basis="6-31g", unit="angstrom", spin=2)
# mol = pyscf.M(atom="N   0.0  0.0           0.0; H1   0.0  0.0 0.9948; H2 0.0  0.0  -0.8199", basis="sto-3g", unit="angstrom", spin=1)
uhf = pyscf.scf.UHF(mol).run()

mc = mcscf.UCASCI(uhf, (2,0), 4)

h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
g_eri = mol.intor("int2e")

#Slowquant
WF = UnrestrictedWaveFunctionUPS(
    mol.nelectron,
    ((2,0),4),
    mc.mo_coeff,
    h_core,
    g_eri,
    "utups",
    {"n_layers":2},
    include_active_kappa=True,
)

# WF.thetas = (2*np.pi*np.random.random(len(WF.thetas)) - np.pi).tolist()   
WF.run_wf_optimization_1step("bfgs", orbital_optimization=True, tol=1e-8, maxiter=5000)

print("")
print("E0:", WF.energy_elec_RDM) # Vigtigt at printe den her så man har noget at grep efter

name = str(np.random.random())
print(name)
np.save(f"{name}_mc_mo_coeff.npy", np.array(mc.mo_coeff))
np.save(f"{name}_thetas.npy", np.array(WF.thetas))
np.save(f"{name}_a_mo.npy", np.array(WF.c_a_mo))
np.save(f"{name}_b_mo.npy", np.array(WF.c_b_mo))

np.save(f"{name}_a_rdm", np.array(WF.rdm1aa))
np.save(f"{name}_b_rdm", np.array(WF.rdm1bb))

np.save(f"{name}_num_inactive_orbs", WF.num_inactive_orbs)
np.save(f"{name}_num_active_orbs", WF.num_active_orbs)

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

spin = 1
mapper = JordanWignerMapper()
# Fermi-contact operator, with RDM formulation:
# r""" a_{iso}^K = \frac{f_k}{2\pi M} \bigg\{\bigg [[A^K_{\alpha}]_I - [A^K_{\beta}]_I\bigg] + \bigg[[A^K_{\alpha}]_A \Gamma^{[1]}_{\alpha} - [A^K_{\beta}]_A \Gamma^{[1]}_{\beta}\bigg] \bigg\}"""
# And as an operator without rdms.
for atom in mol._atom:
    atom_name = atom[0]
    print(atom_name)
    print("classic operator")
    amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
    mo_basis_a = amp_basis@WF.c_a_mo
    mo_basis_b = amp_basis@WF.c_b_mo
    h1mo_a = np.outer(np.conj(mo_basis_a), mo_basis_a)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs]
    h1mo_b = np.outer(np.conj(mo_basis_b), mo_basis_b)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs]
    rdma = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
    rdmb = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
    rdma[WF.num_inactive_orbs: , WF.num_inactive_orbs:] = WF.rdm1aa
    rdmb[WF.num_inactive_orbs: , WF.num_inactive_orbs:] = WF.rdm1bb

    h1mo_a_active = h1mo_a[WF.num_inactive_orbs:, WF.num_inactive_orbs: ]
    h1mo_b_active = h1mo_b[WF.num_inactive_orbs:, WF.num_inactive_orbs: ]
    rdma_active = rdma[WF.num_inactive_orbs:, WF.num_inactive_orbs: ]
    rdmb_active = rdmb[WF.num_inactive_orbs:, WF.num_inactive_orbs: ]
    hfc = np.trace(h1mo_a@rdma  - h1mo_b@rdmb)
    hfc_active = np.trace(h1mo_a_active@rdma_active  - h1mo_b_active@rdmb_active)
    m = spin * (1/2)
    f_k = calculate_constant()
    g_k = nuclear_g_factor(atom=atom[0])
    print("HFC without factor:", hfc)
    print("HFC:", f_k*g_k/m*hfc, "MHz")
    print("HFC without factor (active):", hfc_active)
    print("HFC (active):", f_k*g_k/m*hfc_active, "MHz")


    print("quantum operator")
    np.save(f"{name}_{atom_name}_h1mo_a", np.array(h1mo_a))
    np.save(f"{name}_{atom_name}_h1mo_b", np.array(h1mo_b))
    operator = one_elec_op_0i_0a_HFC(h1mo_a, h1mo_b, num_inactive_orbs=WF.num_inactive_orbs, num_active_orbs=WF.num_active_orbs)
    operator_active = operator.get_folded_operator(WF.num_inactive_orbs, WF.num_active_orbs, WF.num_virtual_orbs)
    
    print("HFC without factor:", expectation_value(WF.ci_coeffs, [operator], WF.ci_coeffs, WF.ci_info))
    print("HFC:", f_k*g_k/m*expectation_value(WF.ci_coeffs, [operator], WF.ci_coeffs, WF.ci_info))
    print("HFC without factor (active):", expectation_value(WF.ci_coeffs, [operator_active], WF.ci_coeffs, WF.ci_info, do_folding=False)-operator_active.operators[()])
    print("HFC (active):", f_k*g_k/m*(expectation_value(WF.ci_coeffs, [operator_active], WF.ci_coeffs, WF.ci_info, do_folding=False)-operator_active.operators[()]))

    # np.save(f"{name}_{atom_name}_inactive_operator", operator_active.operators[()])
    # print(operator_active.operators)
    # print(operator_active[0,3])
    # for o in operator_active.operators:
    #     print(o)

    operator_active.operators.pop(())
    mapped_op = mapper.map(FermionicOp(operator_active.get_qiskit_form(WF.num_active_orbs), WF.num_active_spin_orbs))
    # print(mapped_op)
    t_mo_operator = {}
    for c, p in zip(mapped_op.coeffs, mapped_op.paulis):
        t_mo_operator[p.to_label()] = c.real
    # with open(f"{name}_{atom_name}_operator_with_inactive", "wb") as f:
    with open(f"{name}_{atom_name}_operator_active", "wb") as f:
        pickle.dump(t_mo_operator, f)
    
    # t_mo_operator.pop("IIIIIIII")
    # print(t_mo_operator)
    # print("inaktiv:", operator_active.operators[()])
    # with open(f"{name}_{atom_name}_operator_without_inactive", "wb") as f:
    #     pickle.dump(t_mo_operator, f)
