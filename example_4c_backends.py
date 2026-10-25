import time

import basis_set_exchange as bse
import numpy as np
import pyscf
from pyscf import scf
from pyscf.scf import hf

from slowquant.unitary_coupled_cluster.DHF_integral_direct import DHFCholesky, DHFDensityFit, DHFDirect, DHFHybrid
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct_DHF import LinearResponseDirect

hf.remove_overlap_zero_eigenvalue = False

ACTIVE_KAPPA = True  # False: ~8x fewer OO iterations, but a slightly different (less flexible) wave function
BACKEND = "cd_pivots"  # "cholesky" (stored vectors), "cd_pivots" (pivots only), "df_ssss", "df", "df_ssss+cd_pivots", "direct" (exact)

# mol = pyscf.M(  # H2 has 2 electrons: use CAS ((1, 1), 6) here, not ((2, 2), 8)
#     atom="H 1.0215 -0.0252 0.5645; H 1.1785 -0.5748 1.0355",
#     basis=bse.get_basis("dyall-v2z", elements=["H"], fmt="nwchem"),
#     unit="angstrom",
#     nucmod="gaussian",
#     max_memory=4000,
#     verbose=0,
# )

# mol = pyscf.M(
#     atom="H 0  0  0; F 0  0  0.9168",
#     basis=bse.get_basis("dyall-v2z", elements=["H", "F"], fmt="nwchem"),
#     unit="angstrom",
#     nucmod="gaussian",
#     max_memory=4000,
#     verbose=0,
# )

# mol = pyscf.M(
#     atom="H 0  0  0; I 0  0  1.60916",
#     basis=bse.get_basis("dyall-cv2z", elements=["H", "I"], fmt="nwchem"),
#     unit="angstrom",
#     nucmod="gaussian",
#     max_memory=4000,
#     verbose=0,
# )

mol = pyscf.M(
    atom="H 0  0  0; Br 0  0  1.41443",
    basis=bse.get_basis("dyall-cv4z", elements=["H", "Br"], fmt="nwchem"),
    unit="angstrom",
    nucmod="gaussian",
    max_memory=4000,
    verbose=0,
)

t = time.time()
mf = scf.dhf.DHF(mol)
mf.conv_tol = 1e-11
mf.direct_scf_tol = 1e-10
mf.kernel()
print(f"DHF {time.time() - t:.1f} s  E = {mf.e_tot:.10f}")

WF = GeneralizedWaveFunctionUPS(
    ((2, 2), 8),
    np.array(mf.mo_coeff, dtype=complex),
    mol,
    [],
    False,
    "fUCCSD",
    {"n_layers": 1, "is_spin_conserving": False},
    include_active_kappa=ACTIVE_KAPPA,
)

t = time.time()
backends = {
    "cholesky": lambda: DHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10),
    "cd_pivots": lambda: DHFCholesky(mol, tol=1e-8, store=False, screen_tol=1e-10),
    "df_ssss": lambda: DHFDensityFit(mol, beta=2.0, blocks="S", df_jk=True, screen_tol=1e-10),
    "df": lambda: DHFDensityFit(mol, beta=1.6, blocks="LS", screen_tol=1e-10),
    "df_ssss+cd_pivots": lambda: DHFHybrid(
        DHFDensityFit(mol, beta=1.6, blocks="S", df_jk=True, screen_tol=1e-10), DHFCholesky(mol, tol=1e-8, store=False, screen_tol=1e-10)
    ),
    "direct": lambda: DHFDirect(mol, screen_tol=1e-10),
}
WF.set_integral_backend(backends[BACKEND]())
print(f"{WF._direct.describe()} ({time.time() - t:.1f} s)")

if len(WF.thetas) > 0:
    np.random.seed(70)
    real = np.random.uniform(-0.05, 0.05, len(WF.thetas_real))
    np.random.seed(80)
    imag = np.random.uniform(-0.05, 0.05, len(WF.thetas_imag))
    WF.set_thetas(real, imag)

t = time.time()
WF.run_wf_optimization_onestep_direct_DHF(tol=1e-6, maxiter=1000, e_scf=mf.e_tot)
print(f"OO {time.time() - t:.1f} s  E_elec = {WF.energy_elec:.10f}")

t = time.time()
LR = LinearResponseDirect(WF, "SD")
res = LR.get_properties(shieldings=True, sscc=False, tol=1e-5)
print(f"Shieldings {time.time() - t:.1f} s:", res["shieldings"])
