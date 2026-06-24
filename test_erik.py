import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
import slowquant.qiskit_interface.linear_response.generalized_naive as q_generalized_naive




def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.03599967994):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    # mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, nucmod=1)
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.GHF(mol)

    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF

    mf.max_cycle = 50000

    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)

    e_nuc=mf.energy_nuc()
    print(e_nuc)

    WF =GeneralizedWaveFunctionUPS(
        # mol.nelectron,
        active_space,
        coeff,
        mol,
        "fUCCSD",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )
    # WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    print("c_mo is real?", np.allclose(WF.c_mo.imag, 0))


    # np.random.seed(42)
    # nye_vinkler_real = np.random.uniform(-0.05, 0.05, len(WF.thetas)).tolist()
    # nye_vinkler_imag = [0.0] * len(WF.thetas)
    # WF.set_thetas(nye_vinkler_real, nye_vinkler_imag)


    # WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    print("E_opt: (+nuc!)", WF._energy_elec + e_nuc)

    LR = generalized_naive.LinearResponse(WF, excitations="sd")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
    print(LR.get_formatted_oscillator_strength)


    "Non-relativistic integrals"
    h_1e = mol.intor("int1e_kin")  
    h_nuc=mol.intor("int1e_nuc")
    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    #Mapper
    mapper = JordanWignerMapper()
    #Sampler
    primitive = Sampler(run_options={"shots": None})
    QI = QuantumInterface(primitive, "fUCCSD", mapper, ansatz_options=({"n_layers": 1, "is_spin_conserving" : False}))
    qWF = GeneralizedWaveFunctionCircuit(
        mol.nelectron,
        active_space,
        WF.c_mo,
        h_core,
        g_eri,
        QI,
        include_active_kappa=True,
    )
    # qWF.thetas = WF.thetas
    qWF.set_thetas(WF.thetas_real, WF.thetas_imag)
    qLR = q_generalized_naive.quantumLR(qWF, "SD")

    qLR.run(do_rdm=True)

    LR.calc_excitation_energies()
    print(LR.excitation_energies)
    # print(LR.get_oscillator_strengths(mol.intor("int1e_r")))
    # print(LR.get_oscillator_strength(mol.intor("int1e_r"))) #forskel på denne og strengths??

    excitation_energies = qLR.get_excitation_energies()
    print(excitation_energies)





def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    basis = "631-g"
    # basis = "STO-3G"
    # basis = "6-311-g**"
    active_space = ((2, 1), 6)
    # active_space = ((1,0), 2)
    #active_space = (2, 4)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

h3()