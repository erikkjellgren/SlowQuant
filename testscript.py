import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
import pyscf
import matplotlib.pyplot as plt

# Define molecule and basis
# geometry = """
#     O 0.000     0.000    0.119;
#     H 0.000     0.757   -0.477;
#     H 0.000    -0.757   -0.477;
# """
# basis = "6-31G"
# cas = (8, 6) #(8e, 6o)

geometry = """
    H 0.000     0.000    0.000;
    H 0.000    -0.7414    0.000;
"""
basis = "6-31G"
cas = (2, 4)

# Initialize SlowQuant and PySCF
SQobj = sq.SlowQuant()
SQobj.set_molecule(geometry, distance_unit="angstrom")
SQobj.set_basis_set(basis)
SQobj.init_hartree_fock()
SQobj.hartree_fock.run_restricted_hartree_fock()
h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
g_eri = SQobj.integral.electron_repulsion_tensor

mol = pyscf.M(atom=geometry, basis=basis)
myhf = mol.RHF().run()

HF_electronic_energy = myhf.e_tot - mol.energy_nuc()

hf_orbitals = SQobj.hartree_fock.mo_coeff
c_orthonormal_hf = hf_orbitals
num_elec = SQobj.molecule.number_electrons
num_spin_orbs = SQobj.molecule.number_bf * 2
num_orbs = SQobj.molecule.number_bf

layer_depths = list(range(1, 2))
energies_tups_hf = []
energies_oo_tups_hf = []
energies_tups_hf_roto = []
energies_hf = [HF_electronic_energy] * len(layer_depths)

# for L in layer_depths:
#     print("Rotosolve")
#     print("Layer depth: {}".format(L))
#     # tUPS with HF orbitals
#     WF_hf = WaveFunctionUPS(num_elec=num_elec, cas=cas, mo_coeffs=c_orthonormal_hf, h_ao=h_core, g_ao=g_eri, ansatz="tups", ansatz_options={"n_layers": L, "do_qnp": False, "skip_last_singles": False, "assume_hf_reference": True}, include_active_kappa=True)
#
#     energies_tups_hf.append(WF_hf.energy_elec)
#
#     WF_hf.run_wf_optimization_1step("rotosolve", False)
#     roto_energy = WF_hf.energy_elec
#     print("Roto vanilla Energy:", roto_energy)

for L in layer_depths:
    print("Rotosolve2D")
    print("Layer depth: {}".format(L))
    # tUPS with HF orbitals
    WF_hf = WaveFunctionUPS(num_elec=num_elec, cas=cas, mo_coeffs=c_orthonormal_hf, h_ao=h_core, g_ao=g_eri, ansatz="qnp", ansatz_options={"n_layers": L, "do_qnp": False, "skip_last_singles": False, "assume_hf_reference": True}, include_active_kappa=True)

    energies_tups_hf.append(WF_hf.energy_elec)
    WF_hf.run_wf_optimization_1step("rotosolve_2d", False)
    roto_energy = WF_hf.energy_elec
    print("Roto 2D Energy:", roto_energy)

# # Plotting
# plt.figure(figsize=(12, 8))
# plt.plot(layer_depths, energies_tups_hf, label='tUPS-HF', marker='o', color='b')
# plt.plot(layer_depths, energies_tups_hf_roto, label='tUPS-HF-Rotosolve', marker='o', color='y')
#
# plt.xlabel('Layer Depth (L)')
# plt.ylabel('Electronic Energy (Hartree)')
# plt.title('Electronic Energies of H2O with Different Methods (CAS(8,6))')
# plt.legend()
# plt.grid(True)
# plt.show()
