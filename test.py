from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
import pyscf

mol = pyscf.M(atom="N 0 0 0; N 0 0 1.1", basis="augccpvtz", unit="angstrom")
rhf = pyscf.scf.RHF(mol).run()

WF = WaveFunctionUPS(
    mol.nelectron,
    (6, 6),
    rhf.mo_coeff,
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
    "fUCCSD",
    include_active_kappa=True,
)
print(WF.num_orbs)
WF.run_wf_optimization_2step("BFGS", True)
