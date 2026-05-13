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
from slowquant.unitary_coupled_cluster.linear_response.solvers import PairedDavidson


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
    WF = WaveFunctionUCC(
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SDTQ",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    # naive
    LR = naive.LinearResponse(WF, excitations="SDTQ")
    LR._construct_hessian_metric_blocks()

    A = LR.A
    B = LR.B
    Sigma = LR.Sigma
    Delta = LR.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ab = A @ trial
        Bb = B @ trial.conj()
        Sb = Sigma @ trial

        sigma_plus = Ab + Bb
        sigma_minus = Ab - Bb
        tau_plus = Sb
        tau_minus = Sb
        return sigma_plus, sigma_minus, tau_plus, tau_minus

    d = PairedDavidson()
    num_exc = 13

    start_guess = np.random.rand(A.shape[0], num_exc)
    start_guess = np.vstack((start_guess, start_guess.conj()))
    trial = d._orthonormalize(start_guess)

    sp, sm, tp, tm = right_transform(trial[:A.shape[0], :])

    _sp, _sm, _tp, _tm = LR._right_transform(trial)

    assert np.allclose(sp, _sp, atol=1e-12)
    assert np.allclose(sm, _sm, atol=1e-12)
    assert np.allclose(tp, _tp, atol=1e-12)
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
    WF = WaveFunctionUCC(
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SDTQ",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    # naive
    LR = projected.LinearResponse(WF, excitations="SDTQ")
    LR._construct_hessian_metric_blocks()

    A = LR.A
    B = LR.B
    Sigma = LR.Sigma
    Delta = LR.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ab = A @ trial
        Bb = B @ trial.conj()
        Sb = Sigma @ trial

        sigma_plus = Ab + Bb
        sigma_minus = Ab - Bb
        tau_plus = Sb
        tau_minus = Sb
        return sigma_plus, sigma_minus, tau_plus, tau_minus

    d = PairedDavidson()
    num_exc = 13

    start_guess = np.random.rand(A.shape[0], num_exc)
    start_guess = np.vstack((start_guess, start_guess.conj()))
    trial = d._orthonormalize(start_guess)

    sp, sm, tp, tm = right_transform(trial[:A.shape[0], :])

    _sp, _sm, _tp, _tm = LR._right_transform(trial)

    assert np.allclose(sp, _sp, atol=1e-12)
    assert np.allclose(sm, _sm, atol=1e-12)
    assert np.allclose(tp, _tp, atol=1e-12)
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
    WF = WaveFunctionUCC(
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SDTQ",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    # naive
    LR = allprojected.LinearResponse(WF, excitations="SDTQ")
    LR._construct_hessian_metric_blocks()

    A = LR.A
    B = LR.B
    Sigma = LR.Sigma
    Delta = LR.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ab = A @ trial
        Bb = B @ trial.conj()
        Sb = Sigma @ trial

        sigma_plus = Ab + Bb
        sigma_minus = Ab - Bb
        tau_plus = Sb
        tau_minus = Sb
        return sigma_plus, sigma_minus, tau_plus, tau_minus

    d = PairedDavidson()
    num_exc = 13

    start_guess = np.random.rand(A.shape[0], num_exc)
    start_guess = np.vstack((start_guess, start_guess.conj()))
    trial = d._orthonormalize(start_guess)

    sp, sm, tp, tm = right_transform(trial[:A.shape[0], :])

    _sp, _sm, _tp, _tm = LR._right_transform(trial)

    assert np.allclose(sp, _sp, atol=1e-12)
    assert np.allclose(sm, _sm, atol=1e-12)
    assert np.allclose(tp, _tp, atol=1e-12)
    assert np.allclose(tm, _tm, atol=1e-12)
