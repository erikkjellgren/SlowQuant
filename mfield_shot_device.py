import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyscf
from pyscf import mcscf, scf, gto, x2c, lib, gto
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
import basis_set_exchange as bse
import struct
 
 
# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value_energy
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a, get_HcoreB
from slowquant.unitary_coupled_cluster.generalized_density_matrix import get_orbital_gradient_generalized_real_imag, get_orbital_gradient_expvalue_real_imag, get_nonsplit_gradient_expvalue, get_gradient_finite_diff, get_electronic_energy_generalized
 
from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)
 
from slowquant.molecularintegrals.integralfunctions import DHF_one_electron_transform, DHF_two_electron_transform
 
 
# qWF imports:
from qiskit_aer.primitives import Sampler, SamplerV2
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, InterleavedQubitMapper
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer.noise import NoiseModel
 
 
 
def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", active_space)
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, cart = False)
    mol.build()
 
    mf = scf.GHF(mol)
 
    # Modyfing the Hcore
    # B in T
    B = np.array([.2, .5, .3]) ; B = (B * (2.3505 * 1e5) / np.linalg.norm(B)).tolist() ; orig = (0, 0, 0)
 
    # Integrals:
    hcore = mf.get_hcore().astype(complex)
    hcoreB_corr = get_HcoreB(mf, mol, B = B, orig = orig)
    g_eri = mol.intor("int2e")
 
    # qWF settings:
    mapper = JordanWignerMapper()
    backend = FakeTorino()
    sampler = SamplerV2(options={"backend_options":{"noise_model":NoiseModel.from_backend(backend)}})
    shots_list = [200000, 300000]
    n_runs = 10
    M0 = True
    post_select = True
    method = "fUCCSD"
    spin_consv = False
    active_k = True
    nl = 1
 
    # Load data:
    data = np.load("data_mfield_H3_def2svp_2_1_6_UCCSD_04.npz")
    c_mo, thetas_real, thetas_imag = data["c_mo"], data["thetas_real"], data["thetas_imag"]
 
    results = []
 
    for shots in shots_list:
        for i in range(n_runs):
            QI = QuantumInterface(
                    sampler,
                    method, # Ansatz
                    mapper,
                    ansatz_options = {"n_layers": nl, "is_spin_conserving" : spin_consv},
                    shots = shots,
                    do_M_ansatz0=M0, # default is false
                    do_postselection=post_select,
                )
            qWF = GeneralizedWaveFunctionCircuit(
                mol.nelectron,
                active_space,
                c_mo,
                hcore + hcoreB_corr,
                g_eri,
                QI,
                include_active_kappa = active_k,
            )
            if i == 0:
                QI.get_info()
                print("Depth of circuit:", qWF.QI.circuit.depth())

            qWF.set_thetas_initial(thetas_real, thetas_imag)
            energy = np.real(qWF.energy_elec)
            print("shots:", shots, "run:", i, "energy:", energy)
            results.append({"shots": shots, "run": i, "energy": energy})
 
        # Save after each number of shots
        df = pd.DataFrame(results)
        df.to_csv("shot_study_raw.csv", index=False)
 
    # Reference energy (Hartree):
    E_ref = -3.669832005739848

    df["deviation"] = df["energy"] - E_ref

    # Statistics:
    stats = df.groupby("shots").agg(
        mean = ("energy", "mean"),
        MD = ("deviation", "mean"),
        MAD = ("deviation", lambda x: x.abs().mean()),
        SD = ("deviation", "std"),
    ).reset_index()
    stats.to_csv("shot_study_stats.csv", index=False)
    print(stats)


    # Plot:
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.17
    rng = np.random.default_rng(0)  # only used to spread out the points horizontally

    for i, shots in enumerate(stats["shots"]):
        row = stats.iloc[i]
        first = (i == 0)  # only add legend labels once

        ax.bar(i - 0.32, row["MD"], width, color="#1f22cc", edgecolor="black",
            label="Mean deviation (MD)" if first else None)
        ax.bar(i - 0.13, row["MAD"], width, color="#b3adff", edgecolor="black", hatch="//",
            label="Mean absolute deviation (MAD)" if first else None)
        ax.bar(i + 0.06, row["SD"], width, color="#5ce8f0", edgecolor="black", hatch="..",
            label="Standard deviation (SD)" if first else None)

        deviations = df[df["shots"] == shots]["deviation"]
        ax.scatter(i + 0.32 + rng.uniform(-0.09, 0.09, len(deviations)), deviations,
                s=32, facecolors="#a9c4e8", edgecolors="#0d2a5c", linewidths=1.0, alpha=0.8, zorder=3,
                label="Individual deviations" if first else None)

        # Separator between shot counts
        ax.axvline(i + 0.5, color="#8aa8c4", linestyle="-.", linewidth=1.2)

    ax.axhline(0, color="#0d2a5c", linestyle="--", linewidth=1.5)
    # Leave some space above and below the data (and always show 0)
    low = min(0, df["deviation"].min(), stats["MD"].min())
    high = max(0, df["deviation"].max(), stats["MAD"].max(), stats["SD"].max())
    pad = 0.05 * (high - low)
    ax.set_ylim(low - pad, high + pad)

    ax.set_xticks(np.arange(len(stats["shots"])))
    ax.set_xticklabels(stats["shots"], rotation=25, ha="right")
    ax.set_xlim(-0.5, len(stats["shots"]) - 0.5)
    ax.set_xlabel("Number of shots")
    ax.set_ylabel("Deviation from reference energy (Hartree)")

    ax.grid(axis="y", color="#dddddd")
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(axis="y", which="minor", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_runs = df.groupby("shots").size().max()
    ax.text(1.0, -0.1, f"N = {n_runs} runs per shot count", transform=ax.transAxes,
            ha="right", va="top", color="#444444")

    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 3, 0]  # bars first, points last
    ax.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower center",
            bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)

    fig.tight_layout()
    fig.savefig("shot_study.png", dpi=300)

    plt.show()
 
    return df, stats
 
 
 
def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "def2-svp"
    #basis = ""
    active_space = ((2, 1), 6)
    #active_space = (2, 4)
    charge = 0
    spin = 1
 
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
 

h3()
