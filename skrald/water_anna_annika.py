import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
# from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
# from slowquant.unitary_coupled_cluster.linear_response import naive
def unrestricted(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """Calculate hyperfine coupling constant (fermi-contact term) for a molecule"""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    res = mc.kernel(mf.mo_coeff)
    # Slowquant
    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccd",
        {"n_layers": 1},
        include_active_kappa=True,
    )
    
    WF.run_wf_optimization_1step("l-bfgs-b", True)
    print(WF.energy_elec_RDM)

def water():
    geometry = """O  0.0   0.0  0.11779 
    H  0.0   0.75545  -0.47116;
    H  0.0  -0.75545  -0.47116 
"""
    basis = "6-31g"
    active_space = ((2,2),4)
    charge = 0
    spin = 0

    unrestricted(geometry=geometry, basis=basis, active_space=active_space, unit='angstrom', charge=charge, spin=spin)


water()