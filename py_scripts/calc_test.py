from pyscf import gto, scf
import numpy as np
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import (
    UnrestrictedWaveFunctionUPS,
)

"""Test unrestricted calculation for OH radical. Using utUPS"""
mol = gto.M(
atom="O 0 0 0; H 0 0 1",
basis="STO3G",
spin=1,
charge=0,
)

mf = scf.UHF(mol)
mf.kernel()


WF = UnrestrictedWaveFunctionUPS(mol.nelectron,
    ((2, 1), 3),
    mf.mo_coeff,
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
    ansatz="utups",
    ansatz_options={"n_layers":2},
    include_active_kappa=True,
    )

WF.run_wf_optimization_1step("SLSQP", True)

# spin = 1
# for atom in mol._atom:
#     atom_name = atom[0]
#     print(atom_name)
#     amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
#     mo_basis_a = amp_basis@WF.c_a_mo
#     mo_basis_b = amp_basis@WF.c_b_mo
#     h1mo_a = np.outer(np.conj(mo_basis_a), mo_basis_a)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs]
#     h1mo_b = np.outer(np.conj(mo_basis_b), mo_basis_b)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs]
#     rdma = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
#     rdmb = np.eye(WF.num_inactive_orbs + WF.num_active_orbs)
#     rdma[WF.num_inactive_orbs: , WF.num_inactive_orbs:] = WF.rdm1aa
#     rdmb[WF.num_inactive_orbs: , WF.num_inactive_orbs:] = WF.rdm1bb

#     h1mo_a_active = h1mo_a[WF.num_inactive_orbs:, WF.num_inactive_orbs: ]
#     h1mo_b_active = h1mo_b[WF.num_inactive_orbs:, WF.num_inactive_orbs: ]
#     rdma_active = rdma[WF.num_inactive_orbs:, WF.num_inactive_orbs: ]
#     rdmb_active = rdmb[WF.num_inactive_orbs:, WF.num_inactive_orbs: ]
#     hfc = np.trace(h1mo_a@rdma  - h1mo_b@rdmb)
#     hfc_active = np.trace(h1mo_a_active@rdma_active  - h1mo_b_active@rdmb_active)
#     m = spin * (1/2)
#     f_k = calculate_constant()
#     g_k = nuclear_g_factor(atom=atom[0])
#     print("HFC without factor:", hfc)
#     print("HFC:", f_k*g_k/m*hfc, "MHz")
#     print("HFC without factor (active):", hfc_active)
#     print("HFC (active):", f_k*g_k/m*hfc_active, "MHz")


def nuclear_g_factor(atom):
    if atom == "H":
        g_k = 2.79284734 * (2/1)
    elif atom == "O":
        g_k = -1.89379 * (2/5)
    else:
        print(f"No nuclear g-value is found for atom: {atom}")
    return g_k

a_iso = []
spin = 1
for atom in mol._atom:
    atom_name = atom[0]
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
    f_k = 400.1186763101158
    g_k = nuclear_g_factor(atom=atom[0])

    print("m", m)
    print("f", f_k)
    print("g", g_k)
    print("hfc", hfc)

    a_iso.append(f_k*g_k/m*hfc)

print(hfc)