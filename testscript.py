import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
import pyscf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
from functools import partial
from pyscf import gto, scf, fci
import pandas as pd


# Define molecule and basis.
geometry = """
    O 0.000     0.000    0.119;
    H 0.000     0.757   -0.477;
    H 0.000    -0.757   -0.477;
"""
basis = "sto-3g"
cas = (8, 6) #(8e, 6o)

# Initialize SlowQuant and PySCF.
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

myfci = fci.FCI(myhf)
FCI_energy, FCI_wf = myfci.kernel()
FCI_electronic_energy = FCI_energy - mol.energy_nuc()

hf_orbitals = SQobj.hartree_fock.mo_coeff
c_orthonormal_hf = hf_orbitals
num_elec = SQobj.molecule.number_electrons

# Ansatz and optimization configuration calculating number of energy evaluations and corresponding energies.
def run_conf(ansatz="tups", depth=1, opt_method="rotosolve_1d", param_option=None, tol=10**-12, max_func_evals=1000):

    print(f"Running antsatz {ansatz}, L={depth}, opt_method: {opt_method}, param_option: {param_option}...")

    # Using HF starting orbitals.
    WF_hf = WaveFunctionUPS(num_elec=num_elec, cas=cas, mo_coeffs=c_orthonormal_hf, h_ao=h_core, g_ao=g_eri,ansatz=ansatz,
                            ansatz_options={"n_layers": depth, "do_qnp": False, "skip_last_singles": False,
                                            "assume_hf_reference": True
                            },
                            include_active_kappa=True,
            )
    if param_option and opt_method == "rotosolve_2d": 
        WF_hf.run_wf_optimization_1step(opt_method, optimization_options={"param_option": param_option}, tol=tol, max_func_evals=max_func_evals)
    elif param_option:
        raise TypeError(f"Only the Rotosolve2D optimizer currently has parameter options. "
                        f"Expected None but got {param_option}.")
    else:
        WF_hf.run_wf_optimization_1step(opt_method, tol=tol, max_func_evals=max_func_evals)

    final_energy = WF_hf.energy_elec
    print(f"{ansatz} {opt_method} {param_option} Electronic Energy:", final_energy)

    if len(WF_hf.eval_hist) != len(WF_hf.energy_hist):
        raise AssertionError(f"len(WF_hf.eval_hist) == {len(WF_hf.eval_hist)} and "
                             f"len(WF_hf.energy_hist) == {len(WF_hf.energy_hist)}. "
                              "They should be equal!")

    return WF_hf.eval_hist, WF_hf.energy_hist, WF_hf.rnd_seed


# Define tUPS method and make it ready for optimization.
run_tups = partial(run_conf, ansatz="tups", depth=2)

# Different configurations to be run.
confs = [
    ("slsqp", None, "SLSQP"),
    #("rotosolve", None, "Rotosolve1D"),
    #("rotosolve_2d", "random_pairs", "2D_random_pairs"),
    #("rotosolve_2d", "ordered_sweep", "2D_ordered_sweep"),
    #("rotosolve_2d", "shuffled_sweep", "2D_shuffled_sweep"),
    #("rotosolve_2d", "simple_sweep", "2D_simple_sweep"),
    #("rotosolve_2d", "priority_sweep", "2D_priority_sweep"),
    #("rotosolve_2d", "priority_pairs", "2D_priority_pairs"),
]

# Colors for the plot.
colors = [
    "blue", 
    "red", 
    "green",   
    "orange", 
    "purple",
    "brown",
    "pink",
    "gray", 
    "cyan",
    "lime"
]


plt.figure(figsize=(8, 5))

evals_list = []
energies_list = []

rnd_seeds = []
# Used as minimum convergence value.
min_energy = 0.0


# Calculate energies for each optimization configuration.
for conf in confs:
    evals = np.array([0])
    energies = np.array([HF_electronic_energy])
    eval_hist, energy_hist, rnd_seed = run_tups(opt_method=conf[0], param_option=conf[1])

    evals = np.append(np.array([0]), eval_hist)
    energies = np.append(np.array([HF_electronic_energy]), np.array(energy_hist))

    local_min_energy = np.min(energies)
    if local_min_energy < min_energy:
        min_energy = local_min_energy
    
    evals_list.append(evals)
    energies_list.append(energies)
    rnd_seeds.append(int(rnd_seed) if rnd_seed else rnd_seed)

# Use min energy. If False FCI energy will be used instead.
use_min_energy = True

min_energy = -84.1367361343229
# Adding graphs to the plot.
plt.figure(figsize=(8, 5))
for i, (evals, energies) in enumerate(zip(evals_list, energies_list)):
    if use_min_energy:
        energies = energies - min_energy
    else:
        energies = energies - FCI_electronic_energy
    plt.plot(evals, energies, marker='o', markersize=3, linestyle='-', color=colors[i], label=confs[i][2])


# # Convert to a DataFrame
# max_len = max(len(e) for e in evals)  # Find the longest array
# data = {
#     f"Evals_{i}": np.pad(e, (0, max_len - len(e)), constant_values=np.nan) for i, e in enumerate(evals)
# }
# data.update({
#     f"Energies_{i}": np.pad(en, (0, max_len - len(en)), constant_values=np.nan) for i, en in enumerate(energies)
# })

# df = pd.DataFrame(data)

# # Save to CSV
# df.to_csv("output123.csv", index=False)

# print("CSV file saved as 'output123.csv'")


# Plot settings.
plt.xscale('linear')
plt.yscale('log')


print("Evals list: ", evals_list)
print("Energies list: ", energies_list)
print("Random seeds used: ", rnd_seeds)

ylabel = "Energy difference to FCI [Ha]"
# Define Y-axis scale dependent on reference energy.
plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
if use_min_energy:
    plt.gca().set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ylabel = "Energy difference to minimum convergence point [Ha]"
else:
    plt.gca().set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ylabel = "Energy difference to FCI [Ha]"

plt.xlabel("# Energy Evaluations")
plt.ylabel(ylabel)
plt.title("Optimizer Comparison - H2O (8e, 6o) tUPS, L=2")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.savefig("./plots/Optimizer Comparison - H2O (8e, 6o) tUPS, L=2, FCI log comparison run77")
plt.show()
