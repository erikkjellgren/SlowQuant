import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from qiskit_aer.primitives import SamplerV2,Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
import slowquant.qiskit_interface.linear_response.generalized_naive as q_generalized_naive




def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.03599967994):
    np.set_printoptions(threshold=np.inf)
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.GHF(mol)

    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF

    mf.max_cycle = 50000

    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)
    # print(coeff, flush=True)
    e_nuc=mf.energy_nuc()
    print(e_nuc)
    
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
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    ny_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas))
    # ny_theta_imag = np.random.uniform(-0.05,0.05,len(WF.thetas)) 
    ny_theta_imag = [0.0] * len(WF.thetas)
    WF.set_thetas(ny_theta_real, ny_theta_imag)

    WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    print("E_opt: (+nuc!)", WF._energy_elec + e_nuc, flush=True)
    print("Optimized Thetas ", WF.thetas, flush=True)
    print("Optimized MO coefficients", WF.c_mo, flush=True)

    LR = generalized_naive.LinearResponse(WF, excitations="sd")

    LR.calc_excitation_energies()
    print('Exci. LR ideal', LR.excitation_energies, flush=True)
    print('Osc. LR ideal',LR.get_oscillator_strength(mol.intor("int1e_r")),flush=True) #forskel på denne og strengths??

    # #Mapper
    # mapper = JordanWignerMapper()
    # #Sampler
    # primitive = Sampler(run_options={"shots": None})
    # QI = QuantumInterface(primitive, "fUCCSD", mapper, ansatz_options=({"n_layers": 1, "is_spin_conserving" : False}))
    # qWF = GeneralizedWaveFunctionCircuit(
    #     mol.nelectron,
    #     active_space,
    #     WF.c_mo,
    #     h_core,
    #     g_eri,
    #     QI,
    #     include_active_kappa=True,
    # )
    # # qWF.thetas = WF.thetas
    # qWF.set_thetas_initial(WF.thetas_real, WF.thetas_imag)
    # qLR = q_generalized_naive.quantumLR(qWF, "sd")

    # qLR.run(do_rdm=True)

    # LR.calc_excitation_energies()
    # print('Exci. LR ideal', LR.excitation_energies, flush=True)
    # print('Osc. LR ideal',LR.get_oscillator_strength(mol.intor("int1e_r")),flush=True) #forskel på denne og strengths??

    # excitation_energies = qLR.get_excitation_energies()
    # print('Exci. qLR ideal ',excitation_energies,flush=True)
    # qLR.get_normed_excitation_vectors()
    # qLR.get_transition_dipole(mol.intor("int1e_r"),)
    # print('Osc. qLR ideal',qLR.get_oscillator_strength(mol.intor("int1e_r")),flush=True)




def noisy(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.03599967994):
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
        False, #Do ecp
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )


    ny_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas))
    # ny_theta_imag = np.random.uniform(-0.05,0.05,len(WF.thetas)) 
    ny_theta_imag = [0.0] * len(WF.thetas)

    WF.set_thetas(ny_theta_real, ny_theta_imag)
    print('Theats noisy',WF.thetas, flush=True)
    # print(WF.thetas)

    WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    print("E_opt: (+nuc!)", WF._energy_elec + e_nuc, flush=True)
    print("Optimized Thetas ", WF.thetas, flush=True)
    print("Optimized MO coefficients", WF.c_mo, flush=True)

    LR = generalized_naive.LinearResponse(WF, excitations="sd")
    # LR.calc_excitation_energies()

    "Non-relativistic integrals"
    h_1e = mol.intor("int1e_kin")  
    h_nuc=mol.intor("int1e_nuc")
    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    #Mapper
    mapper = JordanWignerMapper()
    #Sampler
    # primitive = Sampler(run_options={"shots": 0})
    primitive = SamplerV2()
    QI = QuantumInterface(primitive, "fUCCSD", mapper, ansatz_options=({"n_layers": 1, "is_spin_conserving" : False}),  shots=15000, 
        do_M_ansatz0=True, do_postselection=True
)
#     QI = QuantumInterface(primitive, "fUCCSD", mapper, ansatz_options=({"n_layers": 1, "is_spin_conserving" : False}))

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
    qWF.set_thetas_initial(WF.thetas_real, WF.thetas_imag)
    qLR = q_generalized_naive.quantumLR(qWF, "SD")

    qLR.run(do_rdm=True)

    # LR.calc_excitation_energies()
    # print('Exci. LR ', LR.excitation_energies, flush=True)
    # print('Osc. LR ',LR.get_oscillator_strength(mol.intor("int1e_r")), flush=True) #forskel på denne og strengths??

    excitation_energies = qLR.get_excitation_energies()
    print('Exci. qLR noisy ',excitation_energies, flush=True)
    qLR.get_normed_excitation_vectors()
    qLR.get_transition_dipole(mol.intor("int1e_r"))
    print('Osc. qLR noisy',qLR.get_oscillator_strength(mol.intor("int1e_r")), flush=True)




def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    # basis = "def2SVP"
    # basis = "sto-3g"
    basis = "631-g"
    active_space = ((2, 1), 12)
    charge = 0
    spin = 1
    noisy(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

# h3()

def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "sto-3g"
    # active_space = ((1, 1), 8) #spin orbitaler or spinor basis
    active_space = ((1, 1), 4) #spin orbitaler or spinor basis
    # active_space = (2, 4)
    charge = 0
    spin = 0

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )

h2()




#Do post selection from Erik: 
# if isinstance(mapper, JordanWignerMapper) and do_generalized:
        # for bitint, val in dist.items():
        #     bitstr = format(bitint, f"0{num_qubits}b")
        #     if bitstr.count("1") == num_elec:
        #         new_dist[int(bitstr, 2)] = val
        #         prob_sum += val
# in https://github.com/erikkjellgren/SlowQuant/blob/generalized_orbitals/slowquant/qiskit_interface/util.py#L803