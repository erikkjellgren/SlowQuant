import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c

def URL(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.03599967994):
    from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
    import slowquant.unitary_coupled_cluster.linear_response.unrestricted_naive as unaive
    np.set_printoptions(threshold=np.inf)
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.UHF(mol)

    mf.conv_tol_grad = 1e-8 #gradient tolerance form PYSCF
    mf.conv_tol =1e-10
    mf.max_cycle = 50000

    mf.kernel()
    coeff=np.array(mf.mo_coeff)
    # print(coeff, flush=True)
    e_nuc=mf.energy_nuc()
    print(e_nuc)


    "Non-relativistic integrals"
    h_1e = mol.intor("int1e_kin")  
    h_nuc=mol.intor("int1e_nuc")
    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")


    WF =UnrestrictedWaveFunctionUPS(
        active_space,
        coeff,
        mol,
        "fUCCSD",
        {"n_layers": 1},
        include_active_kappa=True,
    )

    ny_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas))
    # ny_theta_imag = [0.0] * len(WF.thetas)
    WF.thetas=ny_theta_real

    WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 5000)

    print("E_opt: (+nuc!)", WF._energy_elec + e_nuc, flush=True)
    # print("Optimized Thetas ", WF.thetas, flush=True)
    # print("Optimized MO coefficients", WF.c_mo, flush=True)

    LR = unaive.LinearResponseUPS(WF, excitations="sd")

    LR.calc_excitation_energies()
    print('Exci. LR ideal', LR.excitation_energies, flush=True)
    print('Osc. LR ideal',LR.get_oscillator_strength())



def GLR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.03599967994):
    from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
    from slowquant.unitary_coupled_cluster.linear_response import generalized_naive

    np.set_printoptions(threshold=np.inf)
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.GHF(mol)

    mf.conv_tol_grad = 1e-8 #gradient tolerance form PYSCF
    mf.conv_tol =1e-10
    mf.max_cycle = 50000

    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)
    # print(coeff, flush=True)
    e_nuc=mf.energy_nuc()
    # print(e_nuc)
    
    "Non-relativistic integrals"
    h_1e = mol.intor("int1e_kin")  
    h_nuc=mol.intor("int1e_nuc")
    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    WF =GeneralizedWaveFunctionUPS(
        # mol.nelectron,
        active_space,
        coeff,
        mol,
        "fUCCSD",
        False, #Do x2c
        False,
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    ny_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas))
    # ny_theta_imag = np.random.uniform(-0.05,0.05,len(WF.thetas)) 
    ny_theta_imag = [0.0] * len(WF.thetas)
    WF.set_thetas(ny_theta_real, ny_theta_imag)

    WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    print("E_opt: (+nuc!)", WF._energy_elec + e_nuc, flush=True)
    # print("Optimized Thetas ", WF.thetas, flush=True)
    # print("Optimized MO coefficients", WF.c_mo, flush=True)

    LR = generalized_naive.LinearResponse(WF, excitations="sd")

    LR.calc_excitation_energies()
    print('Exci. LR ideal', LR.excitation_energies, flush=True)
    print('Osc. LR ideal',LR.get_oscillator_strength(mol.intor("int1e_r")),flush=True) #forskel på denne og strengths??




def h7():
    geometry = """  H   1.152382   0.000000   0.000000
                    H   0.718499   0.900969   0.000000
                    H  -0.256328   1.123490   0.000000
                    H  -1.038362   0.500000   0.000000
                    H  -1.038362  -0.500000   0.000000
                    H  -0.256328  -1.123490   0.000000
                    H   0.718499  -0.900969   0.000000  """
    basis = "6-31g"
    active_space = ((4, 3), 14)
    charge = 0
    spin = 1


    URL(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
def h5():
    geometry = """  H   0.850651   0.000000   0.000000
                    H   0.262866   0.809017   0.000000
                    H  -0.688191   0.500000   0.000000
                    H  -0.688191  -0.500000   0.000000
                    H   0.262866  -0.809017   0.000000  """
    basis = "6-31g"
    active_space = ((3, 2), 5)
    charge = 0
    spin = 1
    URL(
            geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
        )


def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    basis = "def2SVP"
    basis = "sto-3g"
    # basis = "631-g"
    active_space = ((2, 1), 3)
    charge = 0
    spin = 1
    URL(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

h7()




