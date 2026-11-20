from pyscf import gto, scf
import numpy as np
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import (
    GeneralizedWaveFunctionUPS,
)
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from scipy.linalg import expm


def test_circuits_gradient_h32() -> None:
    """Test generalized calculation for H3."""
    mol = gto.M(
                atom = """H  0.0   0.0  0.0;
                          H  0.0  0.0  0.74""",
                basis = "sto-3g",
                charge = 0,
                spin = 0,
    )

    mf = scf.GHF(mol)
    mf.conv_tol = 1e-8        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 1000

    mf.kernel()

    # small random anti-Hermitian
    eps = 0.1  # controls "step size"
    X_anti = np.random.randn(mf.mo_coeff.shape[0],mf.mo_coeff.shape[0]) + 1j*np.random.randn(mf.mo_coeff.shape[0],mf.mo_coeff.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    U_step = expm(A_mat)

    coeff_u = mf.mo_coeff @ U_step

    WF =GeneralizedWaveFunctionUPS(
            ((1,1), 4),
            coeff_u,
            mol,
            "fUCCSD",
            False, #Do x2c
            {"n_layers": 0, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-8, maxiter = 2000)

    WF2 =GeneralizedWaveFunctionUPS(
            ((1,1), 4),
            WF.c_mo,
            mol,
            "fUCCSD",
            False, #Do x2c
            {"n_layers": 1, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    
    np.random.seed(42)
    nye_theta_real = np.random.uniform(-0.05, 0.05, len(WF2.thetas)).tolist()
    nye_theta_imag = [0.1] * len(WF2.thetas)
    WF2.set_thetas(nye_theta_real, nye_theta_imag)

    WF2.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-8, maxiter = 2000)

    mapper = JordanWignerMapper()

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    QI = QuantumInterface(
        Sampler(run_options={"shots": None}),
        "fUCCSD", # Ansatz
        mapper,
        ansatz_options = {"n_layers": 1, "is_spin_conserving" : False},
        ISA=False, # default is false
        do_M_mitigation=False, # default is false
        do_M_ansatz0=False, # default is false
        do_postselection=False, # default is false
    )

    qWF = GeneralizedWaveFunctionCircuit(
        mol.nelectron,
        ((1,1),4),
        WF2.c_mo,
        h_core,
        g_eri,
        QI,
        include_active_kappa = True,
    )

    qWF.set_thetas_initial(np.add(WF2.thetas_real, 0.002), np.add(WF2.thetas_imag, 0.002))

    qWF.run_wf_optimization_1step("bfgs", orbital_optimization=True, tol=1e-8)


    assert abs(WF2.energy_elec - qWF.energy_elec) < 10**-8


def test_circuits_h3() -> None:
    """Test generalized calculation for H3."""
    mol = gto.M(
                atom = """H  0.000000   0.000000       0.000000;
                            H  1.000000   0.000000       0.000000;
                            H  0.500000   0.8660254038   0.000000""",
                basis = "sto-3g",
                charge = 0,
                spin = 1,
    )

    mf = scf.GHF(mol)
    mf.conv_tol = 1e-8        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 1000

    mf.kernel()

    # small random anti-Hermitian
    eps = 0.1  # controls "step size"
    X_anti = np.random.randn(mf.mo_coeff.shape[0],mf.mo_coeff.shape[0]) + 1j*np.random.randn(mf.mo_coeff.shape[0],mf.mo_coeff.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    U_step = expm(A_mat)

    coeff_u = mf.mo_coeff @ U_step

    WF =GeneralizedWaveFunctionUPS(
            ((2,1), 6),
            coeff_u,
            mol,
            "fUCCSD",
            False, #Do x2c
            {"n_layers": 0, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-8, maxiter = 2000)

    WF2 =GeneralizedWaveFunctionUPS(
            ((2,1), 6),
            WF.c_mo,
            mol,
            "fUCCSD",
            False, #Do x2c
            {"n_layers": 1, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    
    np.random.seed(42)
    nye_theta_real = np.random.uniform(-0.05, 0.05, len(WF2.thetas)).tolist()
    nye_theta_imag = [0.1] * len(WF2.thetas)
    WF2.set_thetas(nye_theta_real, nye_theta_imag)

    WF2.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-8, maxiter = 2000)

    mapper = JordanWignerMapper()

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    QI = QuantumInterface(
        Sampler(run_options={"shots": None}),
        "fUCCSD", # Ansatz
        mapper,
        ansatz_options = {"n_layers": 1, "is_spin_conserving" : False},
        ISA=False, # default is false
        do_M_mitigation=False, # default is false
        do_M_ansatz0=False, # default is false
        do_postselection=False, # default is false
    )

    qWF = GeneralizedWaveFunctionCircuit(
        mol.nelectron,
        ((2,1),6),
        WF2.c_mo,
        h_core,
        g_eri,
        QI,
        include_active_kappa = True,
    )

    qWF.set_thetas_initial(WF2.thetas_real, WF2.thetas_imag)

    assert abs(WF2.energy_elec - qWF.energy_elec) < 10**-8



