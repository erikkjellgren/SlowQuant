import json
import resource
import time
from contextlib import contextmanager

import basis_set_exchange as bse
import numpy as np
import pyscf
from pyscf import scf, tdscf
from pyscf.data import nist
from pyscf.prop import nmr, ssc
from pyscf.scf import hf

from slowquant.unitary_coupled_cluster.DHF_integral_direct import DHFCholesky, DHFDirect
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct_DHF import LinearResponseDirect
from slowquant.unitary_coupled_cluster.printing import info_block, key_value, roots_table, value_table

settings = {
    "atom": "H 1.0215 -0.0252 0.5645; H 1.1785 -0.5748 1.0355",
    "unit": "angstrom",
    "basis": "dyall-v2z",
    "charge": 0,
    "spin": 0,
    "nucmod": "gaussian",
    "max_memory_MB": 4000,
    "scf_conv_tol": 1e-11,
    "scf_direct_scf_tol": 1e-10,
    "scf_max_cycle": 100,
    "pyscf_shieldings": True,
    "pyscf_sscc": True,
    "pyscf_td": "TDA",
    "pyscf_td_max_cycle": 100,
    "lr_n_roots": 10,
    "mode": "classical",
    "cas": ((1, 1), 4),
    "ansatz": "fUCCSD",
    "ansatz_options": {"n_layers": 1, "is_spin_conserving": False},
    "include_active_kappa": True,
    "restart_npz": None,
    "initial_theta_amplitude": 0.0,
    "seed": 42,
    "backend": "cholesky",
    "screen_tol": 1e-10,
    "cd_tol": 1e-8,
    "cd_span": 1e-2,
    "cd_jk": False,
    "ssss": "full",
    "oo_method": "1step",
    "oo_tol": 1e-6,
    "oo_maxiter": 100,
    "theta_gtol": 1e-9,
    "theta_maxiter": 1000,
    "circuit_shots": None,
    "circuit_exact": None,
    "circuit_theta_method": "l-bfgs-b",
    "circuit_orbital_maxiter": 50,
    "lr_excitations": "SD",
    "lr_symmetrize_qG": False,
    "lr_tol": 1e-5,
    "lr_max_iter": 100,
    "shieldings": True,
    "sscc": True,
    "nuclei": None,
    "output_npz": "wf_4c.npz",
}

timings = {}


@contextmanager
def timed(label):
    t0 = time.time()
    yield
    timings[label] = time.time() - t0
    print(key_value(f"Done: {label}", f"{timings[label]:20.1f} s"), flush=True)


s = settings
info_block("Settings", [(k, str(v)) for k, v in s.items()])
hf.remove_overlap_zero_eigenvalue = False
elements = sorted({a.split()[0] for a in s["atom"].split(";") if a.strip()})
with timed("molecule"):
    mol = pyscf.M(
        atom=s["atom"],
        basis=bse.get_basis(s["basis"], elements=elements, fmt="nwchem"),
        unit=s["unit"],
        charge=s["charge"],
        spin=s["spin"],
        nucmod=s["nucmod"],
        max_memory=s["max_memory_MB"],
        verbose=0,
    )
info_block(
    "Molecule",
    [
        ("Basis set", s["basis"]),
        ("Basis functions", f"{mol.nao} spherical AOs, {mol.nao_2c()} large-component spinors, {2 * mol.nao_2c()} 4c functions"),
        ("Electrons / charge / spin", f"{mol.nelectron} / {mol.charge} / {mol.spin}"),
        ("Nuclear repulsion [Eh]", f"{mol.energy_nuc():20.10f}"),
    ],
)

with timed("PySCF DHF"):
    mf = scf.dhf.DHF(mol)
    mf.conv_tol = s["scf_conv_tol"]
    mf.direct_scf_tol = s["scf_direct_scf_tol"]
    mf.max_cycle = s["scf_max_cycle"]
    mf.kernel()
print(key_value("PySCF DHF energy [Eh]", f"{mf.e_tot:20.10f}  (converged: {mf.converged})"))
if not mf.converged:
    print(" WARNING: PySCF DHF did not converge; increase scf_max_cycle or loosen scf_conv_tol (e.g. 1e-9).")
if s["pyscf_shieldings"]:
    with timed("PySCF DHF shieldings"):
        sig = nmr.dhf.NMR(mf).kernel()

if s["pyscf_sscc"]:
    with timed("PySCF DHF spin-spin couplings"):
        so = ssc.dhf.SSC(mf)
        so.mb = "RKB"
        so.nuc_pair = [(i, j) for i in range(mol.natm) for j in range(i)]
        e11 = so.kernel()
        k_dhf = np.zeros((mol.natm, mol.natm))
        au = nist.HARTREE2J / nist.PLANCK * (0.5 * nist.E_MASS / nist.PROTON_MASS) ** 2
        for (i, j), t in zip(so.nuc_pair, e11):
            k_dhf[i, j] = k_dhf[j, i] = au * np.trace(t) / 3

if s["pyscf_td"] and s["lr_n_roots"] > 0:
    with timed(f"PySCF DHF {s['pyscf_td']}"):
        td = getattr(tdscf.dhf, s["pyscf_td"])(mf)
        td.nstates = s["lr_n_roots"]
        td.max_cycle = s["pyscf_td_max_cycle"]
        td.kernel()
    roots_table(f"PySCF DHF {s['pyscf_td']} excitation energies", td.e)
    if not np.all(td.converged):
        print(f" WARNING: PySCF {s['pyscf_td']} not converged for roots {np.where(~np.asarray(td.converged))[0]}")

C = np.array(mf.mo_coeff, dtype=complex)
restart = np.load(s["restart_npz"]) if s["restart_npz"] else None
if restart is not None:
    C = restart["c_mo"]

with timed("integral backend"):
    if s["backend"] == "cholesky":
        direct = DHFCholesky(
            mol, tol=s["cd_tol"], span=s["cd_span"], cd_jk=s["cd_jk"], screen_tol=s["screen_tol"], ssss=s["ssss"]
        )
        print(key_value("Cholesky vectors", f"{len(direct.V)} ({direct.V.nbytes / 1e9:.2f} GB)"))
    else:
        direct = DHFDirect(mol, screen_tol=s["screen_tol"], ssss=s["ssss"])

rng = np.random.default_rng(s["seed"])
with timed("wave function setup"):
    if s["mode"] == "circuit":
        from qiskit_aer.primitives import Sampler
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        from slowquant.qiskit_interface.generalized_circuit_wavefunction_DHF import GeneralizedWaveFunctionCircuitDHF
        from slowquant.qiskit_interface.generalized_interface import QuantumInterface
        from slowquant.qiskit_interface.linear_response.generalized_direct_DHF_circuit import (
            LinearResponseCircuitDirect,
        )

        QI = QuantumInterface(
            Sampler(), s["ansatz"], JordanWignerMapper(), ansatz_options=dict(s["ansatz_options"]), shots=s["circuit_shots"]
        )
        WF = GeneralizedWaveFunctionCircuitDHF(
            s["cas"], C, mol, QI, include_active_kappa=s["include_active_kappa"], direct=direct, exact=s["circuit_exact"]
        )
        n = len(WF.thetas) // 2
        if restart is not None:
            WF.set_thetas(restart["circuit_parameters"][:n], restart["circuit_parameters"][n:])
        elif s["initial_theta_amplitude"] > 0:
            a = s["initial_theta_amplitude"]
            WF.set_thetas_initial(rng.uniform(-a, a, n), rng.uniform(-a, a, n))
    else:
        WF = GeneralizedWaveFunctionUPS(
            s["cas"], C, mol, [], False, s["ansatz"], dict(s["ansatz_options"]),
            include_active_kappa=s["include_active_kappa"],
        )
        WF.set_integral_backend(direct)
        n = len(WF.thetas)
        if restart is not None:
            WF.set_thetas(list(restart["thetas_real"]), list(restart["thetas_imag"]))
        elif s["initial_theta_amplitude"] > 0:
            a = s["initial_theta_amplitude"]
            WF.set_thetas(list(rng.uniform(-a, a, n)), list(rng.uniform(-a, a, n)))
info_block(
    "Wave function",
    [
        ("Mode / ansatz", f"{s['mode']} / {s['ansatz']}"),
        ("Active space (electrons, spinors)", str(s["cas"])),
        ("Inactive / virtual / positronic", f"{WF.num_inactive_spin_orbs} / {WF.num_virtual_spin_orbs} / {WF.num_spin_orbs_NES}"),
        ("Thetas", len(WF.thetas)),
        ("Orbital rotations e-e / e-p", f"{len(WF.kappa_spin_idx)} / {len(WF.kappa_spin_idx_ep)}"),
    ],
)

with timed("wave function optimization"):
    if s["mode"] == "circuit":
        WF.run_wf_optimization_2step_DHF(
            tol=s["oo_tol"],
            maxiter=s["oo_maxiter"],
            theta_method=s["circuit_theta_method"],
            theta_gtol=s["theta_gtol"],
            theta_maxiter=s["theta_maxiter"],
            orbital_maxiter=s["circuit_orbital_maxiter"],
        )
    else:
        oo = WF.run_wf_optimization_onestep_direct_DHF if s["oo_method"] == "1step" else WF.run_wf_optimization_direct_DHF
        oo(tol=s["oo_tol"], maxiter=s["oo_maxiter"], theta_gtol=s["theta_gtol"], theta_maxiter=s["theta_maxiter"])
e_tot = WF.energy_elec + mol.energy_nuc()
print(key_value("E(WF) [Eh]", f"{e_tot:20.10f}"))
print(key_value("E(WF) - E(DHF) [Eh]", f"{e_tot - mf.e_tot:20.10f}"))
out = {"c_mo": WF.c_mo, "energy": e_tot, "settings": json.dumps(s, default=str)}
if s["mode"] == "circuit":
    out["circuit_parameters"] = np.array(WF.thetas)
else:
    out["thetas_real"] = np.array(WF.thetas_real)
    out["thetas_imag"] = np.array(WF.thetas_imag)
np.savez(s["output_npz"], **out)
print(key_value("MO coefficients and thetas saved", s["output_npz"]))

if s["shieldings"] or s["sscc"] or s["lr_n_roots"] > 0:
    with timed("linear response setup"):
        if s["mode"] == "circuit":
            LR = LinearResponseCircuitDirect(WF, s["lr_excitations"], symmetrize_qG=s["lr_symmetrize_qG"])
        else:
            LR = LinearResponseDirect(WF, s["lr_excitations"], symmetrize_qG=s["lr_symmetrize_qG"])
    if s["lr_n_roots"] > 0:
        with timed("excitation energies"):
            w = LR.calc_excitation_energies(s["lr_n_roots"], tol=s["lr_tol"], max_iter=s["lr_max_iter"])
    if s["shieldings"] or s["sscc"]:
        with timed("properties"):
            res = LR.get_properties(
                shieldings=s["shieldings"], sscc=s["sscc"], tol=s["lr_tol"], max_iter=s["lr_max_iter"], nuclei=s["nuclei"]
            )

if s["lr_n_roots"] > 0:
    value_table(
        f"Comparison: excitation energies (first {s['lr_n_roots']})",
        [("Root", 4, "d"), (f"PySCF DHF {s['pyscf_td']} [Eh]", 20, ".8f"), ("This WF, LR [Eh]", 18, ".8f"), ("Difference [Eh]", 16, ".8f")],
        [
            (i + 1, a, b, b - a)
            for i in range(s["lr_n_roots"])
            for a, b in [(td.e[i] if s["pyscf_td"] and i < len(td.e) else np.nan, w[i] if i < len(w) else np.nan)]
        ],
    )
labels = [f"{mol.atom_symbol(i)}{i + 1}" for i in range(mol.natm)]
if s["shieldings"] or s["pyscf_shieldings"]:
    value_table(
        "Comparison: isotropic shieldings (RMB-GIAO)",
        [("Nucleus", 8, ""), ("PySCF DHF [ppm]", 18, ".6f"), ("This WF [ppm]", 18, ".6f"), ("Difference [ppm]", 17, ".6f")],
        [
            (lab, a, b, b - a)
            for i, lab in enumerate(labels)
            for a, b in [(np.trace(sig[i]).real / 3 if s["pyscf_shieldings"] else np.nan, res["shieldings"][i] if s["shieldings"] else np.nan)]
        ],
    )
if s["sscc"] or s["pyscf_sscc"]:
    value_table(
        "Comparison: isotropic reduced spin-spin couplings (RKB)",
        [("Pair", 12, ""), ("PySCF DHF [Hz]", 18, ".6f"), ("This WF [Hz]", 18, ".6f"), ("Difference [Hz]", 16, ".6f")],
        [
            (f"{labels[i]}-{labels[j]}", a, b, b - a)
            for i in range(mol.natm)
            for j in range(i + 1, mol.natm)
            for a, b in [(k_dhf[i, j] if s["pyscf_sscc"] else np.nan, res["sscc"][i, j] if s["sscc"] else np.nan)]
        ],
    )

total = sum(timings.values())
value_table(
    "Timings",
    [("Step", 34, ""), ("Time [s]", 12, ".1f"), ("Share [%]", 10, ".1f")],
    [(k, v, 100 * v / total) for k, v in timings.items()] + [("total", total, 100.0)],
    f"Peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB (max_memory {s['max_memory_MB']} MB)",
)
