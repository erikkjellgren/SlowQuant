# type: ignore
import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.linear_response import (
    allprojected,
    allselfconsistent,
    allstatetransfer,
    naive,
    projected,
    projected_statetransfer,
    selfconsistent,
    statetransfer,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.linear_response.solvers import Davidson


def test_lih_naive():
    """Test LiH energies for naive q LR methods."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SDTQ",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    # naive
    LR_naive = naive.LinearResponse(WF, excitations="SDTQ")
    LR_naive._construct_hessian_metric_blocks()

    A = LR_naive.A
    B = LR_naive.B
    Sigma = LR_naive.Sigma
    Delta = LR_naive.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ab = A @ trial
        Bb = B @ trial.conj()
        Sb = Sigma @ trial

        sigma_plus = Ab + Bb
        sigma_minus = Ab - Bb
        tau_minus = Sb
        return sigma_plus, sigma_minus, tau_minus

    d = Davidson()
    num_exc = 13

    start_guess = np.zeros((A.shape[0], num_exc))
    start_guess[np.argsort(np.diag(A))[:num_exc], np.arange(num_exc)] = 1.0
    trial = d._orthonormalize(start_guess)

    sp, sm, tm = right_transform(trial)

    _sp, _sm, _tm = LR_naive._right_transform(trial)

    assert np.allclose(sp, _sp, atol=1e-12)
    assert np.allclose(sm, _sm, atol=1e-12)
    assert np.allclose(tm, _tm, atol=1e-12)

def test_lih_projected():
    """Test LiH energies for naive q LR methods."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SDTQ",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    # naive
    LR_naive = projected.LinearResponse(WF, excitations="SDTQ")
    LR_naive._construct_hessian_metric_blocks()

    A = LR_naive.A
    B = LR_naive.B
    Sigma = LR_naive.Sigma
    Delta = LR_naive.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ab = A @ trial
        Bb = B @ trial.conj()
        Sb = Sigma @ trial

        sigma_plus = Ab + Bb
        sigma_minus = Ab - Bb
        tau_minus = Sb
        return sigma_plus, sigma_minus, tau_minus

    d = Davidson()
    num_exc = 13

    start_guess = np.zeros((A.shape[0], num_exc))
    start_guess[np.argsort(np.diag(A))[:num_exc], np.arange(num_exc)] = 1.0
    trial = d._orthonormalize(start_guess)

    sp, sm, tm = right_transform(trial)

    _sp, _sm, _tm = LR_naive._right_transform(trial)

    assert np.allclose(sp, _sp, atol=1e-12)
    assert np.allclose(sm, _sm, atol=1e-12)
    assert np.allclose(tm, _tm, atol=1e-12)

def test_lih_allprojected():
    """Test LiH energies for naive q LR methods."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SDTQ",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    # naive
    LR_naive = allprojected.LinearResponse(WF, excitations="SDTQ")
    LR_naive._construct_hessian_metric_blocks()

    A = LR_naive.A
    B = LR_naive.B
    Sigma = LR_naive.Sigma
    Delta = LR_naive.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ab = A @ trial
        Bb = B @ trial.conj()
        Sb = Sigma @ trial

        sigma_plus = Ab + Bb
        sigma_minus = Ab - Bb
        tau_minus = Sb
        return sigma_plus, sigma_minus, tau_minus

    d = Davidson()
    num_exc = 13

    start_guess = np.zeros((A.shape[0], num_exc))
    start_guess[np.argsort(np.diag(A))[:num_exc], np.arange(num_exc)] = 1.0
    trial = d._orthonormalize(start_guess)

    sp, sm, tm = right_transform(trial)

    _sp, _sm, _tm = LR_naive._right_transform(trial)

    assert np.allclose(sp, _sp, atol=1e-12)
    assert np.allclose(sm, _sm, atol=1e-12)
    assert np.allclose(tm, _tm, atol=1e-12)
