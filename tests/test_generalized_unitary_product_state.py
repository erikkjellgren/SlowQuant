from pyscf import gto, scf
import numpy as np
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import (
    GeneralizedWaveFunctionUPS,
)

def test_ups_h3() -> None:
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
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)

    WF =GeneralizedWaveFunctionUPS(
            ((2,1), 6),
            coeff,
            mol,
            "fUCCSD",
            False, #Do x2c
            {"n_layers": 1, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    assert abs(WF.energy_elec - -2.9519207103291136) < 10**-8


def test_ups_h2() -> None:
    """Test generalized calculation for H2."""
    mol = gto.M(
                atom = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74""",
                basis = "631-g",
                charge = 0,
                spin = 0,
    )

    mf = scf.GHF(mol)
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)

    WF =GeneralizedWaveFunctionUPS(
            ((1, 1), 4),
            coeff,
            mol,
            "fUCCSD",
            False, #Do x2c
            {"n_layers": 1, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    assert abs(WF.energy_elec - -1.8613387609037015) < 10**-8







def test_gtups_h3() -> None:
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
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)

    WF =GeneralizedWaveFunctionUPS(
            ((2,1), 6),
            coeff,
            mol,
            "gtups",
            False, #Do x2c
            {"n_layers": 1, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    
    np.random.seed(42)
    nye_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas)).tolist()
    nye_theta_imag = [0.1] * len(WF.thetas)
    WF.set_thetas(nye_theta_real, nye_theta_imag)

    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    assert abs(WF.energy_elec - -2.9519207119698083) < 10**-8



def test_gtups_h2() -> None:
    """Test generalized calculation for H2."""
    mol = gto.M(
                atom = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74""",
                basis = "631-g",
                charge = 0,
                spin = 0,
    )

    mf = scf.GHF(mol)
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)

    WF =GeneralizedWaveFunctionUPS(
            ((1, 1), 4),
            coeff,
            mol,
            "gtups",
            False, #Do x2c
            {"n_layers": 1, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    assert abs(WF.energy_elec - -1.8613387616579782) < 10**-8



    


def test_gtups_h3_no_oo() -> None:
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
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)

    WF =GeneralizedWaveFunctionUPS(
            ((2,1), 6),
            coeff,
            mol,
            "gtups",
            False, #Do x2c
            {"n_layers": 1, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    
    np.random.seed(42)
    nye_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas)).tolist()
    nye_theta_imag = [0.1] * len(WF.thetas)
    WF.set_thetas(nye_theta_real, nye_theta_imag)

    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=False, tol=1e-10, maxiter = 2000)

    assert abs(WF.energy_elec - -2.933418352608597) < 10**-8
