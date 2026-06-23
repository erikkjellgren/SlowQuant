import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c



# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive, naive
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value, generalized_propagate_state
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a, generalized_one_elec_op_0i_0a
from slowquant.unitary_coupled_cluster.operators import a_op_spin
from slowquant.molecularintegrals.integralfunctions import generalized_one_electron_transform

def fuccsd_test(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.GHF(mol)

    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF

    mf.max_cycle = 50000

    # mf.scf()
    mf.kernel() 
    coeff=np.array(mf.mo_coeff, dtype=complex)
    # coeff = np.array([[ 3.64900899e-01+0.j, -3.39809858e-01+0.j, -4.23034160e-01+0.j,
    #     7.04634114e-01+0.j,  7.04634114e-01+0.j, -3.65594923e-01+0.j],
    #     [ 2.64900899e-01+0.j, -3.39809857e-01+0.j, -4.23034161e-01+0.j,
    #     -7.04634113e-01+0.j, -9.04634113e-01+0.j, -3.65594924e-01+0.j],
    #     [ 3.36265397e-01+0.j, -1.80510098e-01+0.j,  7.95339678e-01+0.j,
    #     -5.19507828e-10+0.j, -5.64343523e-10+0.j,  8.44426270e-01+0.j],
    #     [ 2.64900899e-01+0.j,  3.39809858e-01+0.j, -4.23034160e-01+0.j,
    #     7.04634114e-01+0.j, -7.04634114e-01+0.j,  3.65594923e-01+0.j],
    #     [ 2.64900899e-01+0.j,  3.39809857e-01+0.j, -4.23034161e-01+0.j,
    #     -7.04634113e-01+0.j,  7.04634113e-01+0.j,  2.65594924e-01+0.j],
    #     [ 3.36265397e-01+0.j,  1.80510098e-01+0.j,  7.95339678e-01+0.j,
    #     -5.19561803e-10+0.j,  5.64258719e-10+0.j, -8.44426270e-01+0.j]])
    e_nuc=mf.energy_nuc()


    WF =GeneralizedWaveFunctionUPS(
        # mol.nelectron,
        active_space,
        coeff,
        mol,
        "fuccs",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )


    np.random.seed(42)
    nye_vinkler_real = np.random.uniform(-0.05, 0.05, len(WF.thetas)).tolist()
    nye_vinkler_imag = [0.0] * len(WF.thetas)
    WF.set_thetas(nye_vinkler_real, nye_vinkler_imag)

    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)
    E_uccsd=WF._energy_elec
    # WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    print("E_opt fuccsd:", WF._energy_elec)
    print("E_opt fuccsd: (+nuc!)", WF._energy_elec + e_nuc)
    return E_uccsd






def gtups_test(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.GHF(mol)

    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF

    mf.max_cycle = 50000

    # mf.scf()
    mf.kernel()

    coeff_2=np.array(mf.mo_coeff, dtype=complex)

    # coeff_2 = np.array([[ 3.64900899e-01+0.j, -3.39809858e-01+0.j, -4.23034160e-01+0.j,
    #     7.04634114e-01+0.j,  7.04634114e-01+0.j, -3.65594923e-01+0.j],
    #     [ 2.64900899e-01+0.j, -3.39809857e-01+0.j, -4.23034161e-01+0.j,
    #     -7.04634113e-01+0.j, -9.04634113e-01+0.j, -3.65594924e-01+0.j],
    #     [ 3.36265397e-01+0.j, -1.80510098e-01+0.j,  7.95339678e-01+0.j,
    #     -5.19507828e-10+0.j, -5.64343523e-10+0.j,  8.44426270e-01+0.j],
    #     [ 2.64900899e-01+0.j,  3.39809858e-01+0.j, -4.23034160e-01+0.j,
    #     7.04634114e-01+0.j, -7.04634114e-01+0.j,  3.65594923e-01+0.j],
    #     [ 2.64900899e-01+0.j,  3.39809857e-01+0.j, -4.23034161e-01+0.j,
    #     -7.04634113e-01+0.j,  7.04634113e-01+0.j,  2.65594924e-01+0.j],
    #     [ 3.36265397e-01+0.j,  1.80510098e-01+0.j,  7.95339678e-01+0.j,
    #     -5.19561803e-10+0.j,  5.64258719e-10+0.j, -8.44426270e-01+0.j]])
    e_nuc=mf.energy_nuc()

    WF =GeneralizedWaveFunctionUPS(
        active_space,
        coeff_2,
        mol,
        "gtups",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    np.random.seed(42)
    nye_vinkler_real = np.random.uniform(-0.05, 0.05, len(WF.thetas)).tolist()
    nye_vinkler_imag = [0.0] * len(WF.thetas)
    WF.set_thetas(nye_vinkler_real, nye_vinkler_imag)

    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)
    # WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)


    E_gtups=WF._energy_elec
    print("E_opt gtups:", WF._energy_elec)
    print("E_opt gtups: (+nuc!)", WF._energy_elec + e_nuc)

    return E_gtups



def h2():
    geometry = """H 0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "631-g"
    active_space = ((1, 1), 4) #spin orbitaler or spinor basis
    # active_space = (2, 4)
    charge = 0
    spin = 0

    E_uccsd=fuccsd_test(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

    E_gtups=gtups_test(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

    print('Energy diff:', E_uccsd-E_gtups)



    
def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    basis = "sto-3g"
    # basis = "631-g"
    #active_space = ((2, 1), 6)
    active_space = ((2,1), 6)
    #active_space = (2, 4)
    charge = 0
    spin = 1
    
    E_uccsd=fuccsd_test(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

    E_gtups=gtups_test(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

    print('Energy diff:', E_uccsd-E_gtups)



# h3()
h2()

