import json
import resource
import time
from contextlib import contextmanager

import basis_set_exchange as bse
import numpy as np
from pyscf import gto, scf, tdscf

from slowquant.unitary_coupled_cluster.generalized_integral_manager import IntegralManager
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.integral_direct import GHFCholesky, GHFDirect
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct import LinearResponseDirect
from slowquant.unitary_coupled_cluster.printing import info_block, key_value, roots_table, value_table

settings = {
    "atom": "H 0 0 0; I 0 0 1.61",
    "unit": "angstrom",
    "basis": "x2c-SVPall",
    "charge": 0,
    "spin": 0,
    "max_memory_MB": 4000,
    "x2c": True,
    "ecp": False,
    "B_tesla": None,
    "gauge_origin": None,
    "scf_conv_tol": 1e-10,
    "scf_max_cycle": 100,
    "pyscf_td": "TDA",
    "pyscf_td_max_cycle": 100,
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
    "incore_memory_MB": 256,
    "cd_tol": 1e-8,
    "cd_span": 1e-2,
    "cd_jk": False,
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
    "lr_symmetrize_qG": True,
    "lr_n_roots": 10,
    "lr_tol": 1e-6,
    "lr_max_iter": 100,
    "lr_max_space": 400,
    "polarizability": True,
    "polarizability_omega": 0.0,
    "output_npz": "wf_generalized.npz",
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
elements = sorted({a.split()[0] for a in s["atom"].split(";") if a.strip()})
with timed("molecule"):
    mol = gto.M(
        atom=s["atom"],
        basis=bse.get_basis(s["basis"], elements=elements, fmt="nwchem"),
        unit=s["unit"],
        charge=s["charge"],
        spin=s["spin"],
        max_memory=s["max_memory_MB"],
        verbose=0,
    )
info_block(
    "Molecule",
    [
        ("Basis set", s["basis"]),
        ("Basis functions", f"{mol.nao} spatial AOs, {2 * mol.nao} spin AOs"),
        ("Electrons / charge / spin", f"{mol.nelectron} / {mol.charge} / {mol.spin}"),
        ("Nuclear repulsion [Eh]", f"{mol.energy_nuc():20.10f}"),
    ],
)

with timed("PySCF GHF"):
    mf = scf.GHF(mol).x2c() if s["x2c"] else scf.GHF(mol)
    if s["B_tesla"] is not None:
        hB = IntegralManager(mol, B=np.array(s["B_tesla"]) * 4.254e-6, orig=s["gauge_origin"]).magnetic_field_H
        mf.get_hcore = lambda *args: hB
    mf.conv_tol = s["scf_conv_tol"]
    mf.max_cycle = s["scf_max_cycle"]
    mf.kernel()
print(key_value(f"PySCF GHF{'-X2C' if s['x2c'] else ''} energy [Eh]", f"{mf.e_tot:20.10f}  (converged: {mf.converged})"))

if s["pyscf_td"]:
    with timed(f"PySCF {s['pyscf_td']}"):
        td = getattr(tdscf.ghf, s["pyscf_td"])(mf)
        td.nstates = s["lr_n_roots"]
        td.max_cycle = s["pyscf_td_max_cycle"]
        td.kernel()
        nocc = mol.nelectron
        Co, Cv = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
        r = mol.intor("int1e_r", comp=3)
        mu = np.array([np.kron(np.eye(2), x) for x in r])
        mu_ov = np.einsum("pi,xpq,qa->xia", Co.conj(), mu, Cv)
        t = np.array([np.einsum("xia,ia->x", mu_ov, x + (y if isinstance(y, np.ndarray) else 0)) for x, y in td.xy])
        f_td = 2 / 3 * td.e * np.sum(np.abs(t) ** 2, axis=1)
    roots_table(f"PySCF GHF {s['pyscf_td']} excitation energies", td.e, [("f", f_td, ".6f")])
    if not np.all(td.converged):
        print(f" WARNING: PySCF {s['pyscf_td']} not converged for roots {np.where(~np.asarray(td.converged))[0]}")

C = np.array(mf.mo_coeff, dtype=complex)
restart = np.load(s["restart_npz"]) if s["restart_npz"] else None
if restart is not None:
    C = restart["c_mo"]

with timed("integral backend"):
    if s["backend"] == "cholesky":
        direct = GHFCholesky(
            mol, tol=s["cd_tol"], span=s["cd_span"], cd_jk=s["cd_jk"], screen_tol=s["screen_tol"],
            incore_memory=s["incore_memory_MB"],
        )
    else:
        direct = GHFDirect(mol, screen_tol=s["screen_tol"], incore_memory=s["incore_memory_MB"])

rng = np.random.default_rng(s["seed"])
ham = dict(do_x2c=s["x2c"], ecp=s["ecp"], B=s["B_tesla"], orig=s["gauge_origin"])
with timed("wave function setup"):
    if s["mode"] == "circuit":
        from qiskit_aer.primitives import Sampler
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        from slowquant.qiskit_interface.generalized_circuit_wavefunction_direct import (
            GeneralizedWaveFunctionCircuitDirect,
        )
        from slowquant.qiskit_interface.generalized_interface import QuantumInterface
        from slowquant.qiskit_interface.linear_response.generalized_direct_circuit import (
            LinearResponseCircuitDirect,
        )

        QI = QuantumInterface(
            Sampler(), s["ansatz"], JordanWignerMapper(), ansatz_options=dict(s["ansatz_options"]), shots=s["circuit_shots"]
        )
        WF = GeneralizedWaveFunctionCircuitDirect(
            s["cas"], C, mol, QI, include_active_kappa=s["include_active_kappa"], direct=direct,
            exact=s["circuit_exact"], **ham,
        )
        n = len(WF.thetas) // 2
        if restart is not None:
            WF.set_thetas(restart["circuit_parameters"][:n], restart["circuit_parameters"][n:])
        elif s["initial_theta_amplitude"] > 0:
            a = s["initial_theta_amplitude"]
            WF.set_thetas_initial(rng.uniform(-a, a, n), rng.uniform(-a, a, n))
    else:
        WF = GeneralizedWaveFunctionUPS(
            s["cas"], C, mol, s["ansatz"], ansatz_options=dict(s["ansatz_options"]),
            include_active_kappa=s["include_active_kappa"], **ham,
        )
        WF.set_integral_backend(direct)
        n = len(WF.thetas)
        if restart is not None:
            WF.set_thetas(list(restart["thetas_real"]), list(restart["thetas_imag"]))
        elif s["initial_theta_amplitude"] > 0:
            a = s["initial_theta_amplitude"]
            WF.set_thetas(list(rng.uniform(-a, a, n)), list(rng.uniform(-a, a, n)))

with timed("wave function optimization"):
    if s["mode"] == "circuit":
        WF.run_wf_optimization_2step_direct(
            tol=s["oo_tol"],
            maxiter=s["oo_maxiter"],
            theta_method=s["circuit_theta_method"],
            theta_gtol=s["theta_gtol"],
            theta_maxiter=s["theta_maxiter"],
            orbital_maxiter=s["circuit_orbital_maxiter"],
            e_scf=mf.e_tot,
        )
    else:
        oo = WF.run_wf_optimization_onestep_direct if s["oo_method"] == "1step" else WF.run_wf_optimization_direct
        oo(
            tol=s["oo_tol"], maxiter=s["oo_maxiter"], theta_gtol=s["theta_gtol"], theta_maxiter=s["theta_maxiter"],
            e_scf=mf.e_tot,
        )
e_tot = WF.energy_elec + mol.energy_nuc()
out = {"c_mo": WF.c_mo, "energy": e_tot, "settings": json.dumps(s, default=str)}
if s["mode"] == "circuit":
    out["circuit_parameters"] = np.array(WF.thetas)
else:
    out["thetas_real"] = np.array(WF.thetas_real)
    out["thetas_imag"] = np.array(WF.thetas_imag)
np.savez(s["output_npz"], **out)
print(key_value("MO coefficients and thetas saved", s["output_npz"]))

if s["lr_n_roots"] > 0 or s["polarizability"]:
    with timed("linear response setup"):
        if s["mode"] == "circuit":
            LR = LinearResponseCircuitDirect(WF, s["lr_excitations"], symmetrize_qG=s["lr_symmetrize_qG"])
        else:
            LR = LinearResponseDirect(WF, s["lr_excitations"], symmetrize_qG=s["lr_symmetrize_qG"])
    if s["lr_n_roots"] > 0:
        with timed("excitation energies"):
            LR.calc_excitation_energies(
                s["lr_n_roots"], tol=s["lr_tol"], max_iter=s["lr_max_iter"], max_space=s["lr_max_space"]
            )
        with timed("oscillator strengths"):
            LR.get_oscillator_strength()
        roots_table("Excitation energies and oscillator strengths (this WF)", LR.excitation_energies, [("f", LR.oscillator_strengths, ".6f")])
    if s["polarizability"]:
        with timed("polarizability"):
            LR.get_polarizability(omega=s["polarizability_omega"], tol=s["lr_tol"], max_iter=s["lr_max_iter"])

if s["lr_n_roots"] > 0:
    value_table(
        f"Comparison: excitation energies (first {s['lr_n_roots']})",
        [("Root", 4, "d"), (f"PySCF GHF {s['pyscf_td']} [Eh]", 20, ".8f"), ("This WF, LR [Eh]", 18, ".8f"),
         ("f PySCF", 10, ".6f"), ("f WF", 10, ".6f")],
        [
            (
                i + 1,
                td.e[i] if s["pyscf_td"] and i < len(td.e) else np.nan,
                LR.excitation_energies[i],
                f_td[i] if s["pyscf_td"] and i < len(f_td) else np.nan,
                LR.oscillator_strengths[i],
            )
            for i in range(s["lr_n_roots"])
        ],
    )

total = sum(timings.values())
value_table(
    "Timings",
    [("Step", 34, ""), ("Time [s]", 12, ".1f"), ("Share [%]", 10, ".1f")],
    [(k, v, 100 * v / total) for k, v in timings.items()] + [("total", total, 100.0)],
    f"Peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB (max_memory {s['max_memory_MB']} MB)",
)
