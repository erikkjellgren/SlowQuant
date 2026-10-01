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
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino



def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.03599967994):
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
        False,
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    ny_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas))
    # # ny_theta_imag = np.random.uniform(-0.05,0.05,len(WF.thetas)) 
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




def noisy(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.03599967994):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    # mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, nucmod=1)
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    mf = scf.GHF(mol)

    # mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.conv_tol_grad = 1e-8 #gradient tolerance form PYSCF
    mf.conv_tol =1e-10

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


    # ny_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas))
    # # ny_theta_imag = np.random.uniform(-0.05,0.05,len(WF.thetas)) 
    # ny_theta_imag = [0.0] * len(WF.thetas)

    # WF.set_thetas(ny_theta_real, ny_theta_imag)
    # print('Theats noisy',WF.thetas, flush=True)
    # # print(WF.thetas)
    # thetas=[(-0.00010488168937845798+0j), (7.2875253376076166e-06+0j), (-0.0004950240603016265+0j), (7.30578925582349e-06+0j), (-9.071896704829317e-06+0j), (0.00015991303249703666+0j), (-1.9343423495704733e-06+0j), (-0.0053600356313713545+0j), (-1.2626882043608773e-06+0j), (-0.09468452879757139+0j), (-0.0011679906026483865+0j), (-0.10243455643056505+0j), (-0.018675144902787735+0j), (0.0009552715950226049+0j), (0.021590905370059987+0j), (0.0027348864152116903+0j), (-0.014215773316071446+0j), (-0.003288693445104935+0j)]
    # thetas = np.array([(9.708612783713446e-05+0j), (-2.4254712211440956e-06+0j), (8.681142179202589e-05+0j), (-2.4118912249890074e-05+0j), (1.8778490084742975e-07+0j), (-2.3660012040101745e-05+0j), (-1.0451467092454949e-07+0j), (0.00535141938200827+0j), (6.20196172742991e-06+0j), (0.09468494046414426+0j), (0.0009769404383873693+0j), (0.10243707156076338+0j), (-0.018380252529241203+0j), (0.00537126824366951+0j), (0.02131388044545002+0j), (0.004746813660911727+0j), (0.01296395575330517+0j), (-0.004947308105548308+0j)])

    # WF.set_thetas(np.real(thetas) , np.imag(thetas))


    WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    # print("E_opt: (+nuc!)", WF._energy_elec + e_nuc, flush=True)
    # print("E_opt: (+nuc!)", WF._energy_elec, flush=True)
    print("Optimized Thetas ", WF.thetas, flush=True)
    print("Optimized MO coefficients", WF.c_mo, flush=True)

    # LR.calc_excitation_energies()

    "Non-relativistic integrals"
    h_1e = mol.intor("int1e_kin")  
    h_nuc=mol.intor("int1e_nuc")
    h_core=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    #Mapper
    mapper = JordanWignerMapper()
    #Sampler
    backend = FakeTorino()

    # primitive = Sampler(run_options={"shots": 0})
    # primitive = SamplerV2()
    sampler = Sampler(backend_options={"noise_model":NoiseModel.from_backend(backend)})

    # QI = QuantumInterface(primitive, "fUCCSD", mapper, ansatz_options=({"n_layers": 1, "is_spin_conserving" : False}),  shots=1000, 
    #     do_M_ansatz0=True)
    QI = QuantumInterface(sampler, "fUCCSD", mapper, ansatz_options=({"n_layers": 1, "is_spin_conserving" : False}))

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

    print(qWF.energy_elec)

    qLR = q_generalized_naive.quantumLR(qWF, "SD")
    print('her', flush=True)
    qLR.run(do_rdm=True)

    # LR = generalized_naive.LinearResponse(WF, excitations="sd")
    # LR.calc_excitation_energies()
    # print('Exci. LR state vector', LR.excitation_energies, flush=True)
    # print('Osc. LR state vector',LR.get_oscillator_strength(mol.intor("int1e_r")), flush=True) #forskel på denne og strengths??

    excitation_energies = qLR.get_excitation_energies()
    print('Exci. qLR noisy M0 on',excitation_energies, flush=True)
    qLR.get_normed_excitation_vectors()
    qLR.get_transition_dipole(mol.intor("int1e_r"))
    print('Osc. qLR noisy M0 on',qLR.get_oscillator_strength(mol.intor("int1e_r")), flush=True)

    # QI.update_mitigation_flags(do_M_mitigation = False, do_postselection = False, do_M_ansatz0 = False, do_M_ansatz0_plus = False)

    # print('Exci. qLR noisy M0 off',excitation_energies, flush=True)
    # qLR.get_normed_excitation_vectors()
    # qLR.get_transition_dipole(mol.intor("int1e_r"))
    # print('Osc. qLR noisy M0 off',qLR.get_oscillator_strength(mol.intor("int1e_r")), flush=True)



def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    basis = "def2SVP"
    # basis = "sto-3g"
    # basis = "631-g"
    active_space = ((2, 1), 6)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

h3()

