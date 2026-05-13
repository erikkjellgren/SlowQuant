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


def test_lih_naive_explicit():
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
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    threshold = 10 ** (-5)

    # naive
    LR = naive.LinearResponse(WF, excitations="SD")
    LR._construct_hessian_metric_blocks()

    A = LR.A
    B = LR.B
    Sigma = LR.Sigma
    Delta = LR.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        trial_plus = trial[: trial.shape[0] // 2]
        trial_minus = trial[trial.shape[0] // 2 :]

        Abp = A @ trial_plus
        Abm = A @ trial_minus
        Bbp = B @ trial_plus.conj()
        Bbm = B @ trial_minus.conj()
        Sbp = Sigma @ trial_minus
        Sbm = Sigma @ trial_plus

        sigma_plus = Abp + Bbp
        sigma_minus = Abm - Bbm
        tau_plus = Sbp
        tau_minus = Sbm
        return sigma_plus, sigma_minus, tau_plus, tau_minus

    d = PairedDavidson()
    eigvals, eigvecs = d.solve(
        right_transform,
        (np.diag(A), np.diag(Sigma)),
        max_iteration=50,
        tolerance=1e-8,
        n_roots=3,
        # is_silent=True,
    )

    solutions = np.array(
        [
            0.12957563,
            0.17886086,
            0.17886086,
        ]
    )
    assert np.allclose(eigvals, solutions, atol=threshold)

    LR.normed_response_vectors = eigvecs
    LR.Z_q_normed = LR.normed_response_vectors[: len(LR.q_ops), :]
    LR.Z_G_normed = LR.normed_response_vectors[len(LR.q_ops) : len(LR.q_ops) + len(LR.G_ops), :]
    LR.Y_q_normed = LR.normed_response_vectors[len(LR.q_ops) + len(LR.G_ops) : 2 * len(LR.q_ops) + len(LR.G_ops), :]
    LR.Y_G_normed = LR.normed_response_vectors[2 * len(LR.q_ops) + len(LR.G_ops) :, :]
    LR.excitation_energies = eigvals
    LR.get_oscillator_strength()

    solutions = np.array(
        [
            0.049796,
            0.241266,
            0.241266,
        ]
    )
    assert np.allclose(LR.oscillator_strengths, solutions, atol=threshold)

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
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    threshold = 10 ** (-5)

    # naive
    LR = naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies(3, {"max_iterations": 50, "tolerance": 1e-8})

    solutions = np.array(
        [
            0.12957563,
            0.17886086,
            0.17886086,
        ]
    )
    assert np.allclose(LR.excitation_energies, solutions, atol=threshold)

    LR.get_oscillator_strength()

    solutions = np.array(
        [
            0.049796,
            0.241266,
            0.241266,
        ]
    )
    assert np.allclose(LR.oscillator_strengths, solutions, atol=threshold)

def test_lih_projected_explicit():
    """Test LiH energies for projected q LR methods."""
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
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    threshold = 10 ** (-5)

    # naive
    LR = projected.LinearResponse(WF, excitations="SD")
    LR._construct_hessian_metric_blocks()

    A = LR.A
    B = LR.B
    Sigma = LR.Sigma
    Delta = LR.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        trial_plus = trial[: trial.shape[0] // 2]
        trial_minus = trial[trial.shape[0] // 2 :]

        Abp = A @ trial_plus
        Abm = A @ trial_minus
        Bbp = B @ trial_plus.conj()
        Bbm = B @ trial_minus.conj()
        Sbp = Sigma @ trial_minus
        Sbm = Sigma @ trial_plus

        sigma_plus = Abp + Bbp
        sigma_minus = Abm - Bbm
        tau_plus = Sbp
        tau_minus = Sbm
        return sigma_plus, sigma_minus, tau_plus, tau_minus

    d = PairedDavidson()
    eigvals, eigvecs = d.solve(
        right_transform,
        (np.diag(A), np.diag(Sigma)),
        max_iteration=50,
        tolerance=1e-8,
        n_roots=3,
        # is_silent=True,
    )

    solutions = np.array(
        [
            0.12957561,
            0.17886086,
            0.17886086,
        ]
    )
    assert np.allclose(eigvals, solutions, atol=threshold)

    LR.normed_response_vectors = eigvecs
    LR.Z_q_normed = LR.normed_response_vectors[: len(LR.q_ops), :]
    LR.Z_G_normed = LR.normed_response_vectors[len(LR.q_ops) : len(LR.q_ops) + len(LR.G_ops), :]
    LR.Y_q_normed = LR.normed_response_vectors[len(LR.q_ops) + len(LR.G_ops) : 2 * len(LR.q_ops) + len(LR.G_ops), :]
    LR.Y_G_normed = LR.normed_response_vectors[2 * len(LR.q_ops) + len(LR.G_ops) :, :]
    LR.excitation_energies = eigvals
    LR.get_oscillator_strength()

    solutions = np.array(
        [
            0.049799,
            0.241266,
            0.241266,
        ]
    )
    assert np.allclose(LR.oscillator_strengths, solutions, atol=threshold)

def test_lih_projected():
    """Test LiH energies for projected q LR methods."""
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
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    threshold = 10 ** (-5)

    # naive
    LR = projected.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies(3, {"max_iterations": 50, "tolerance": 1e-8})

    solutions = np.array(
        [
            0.12957561,
            0.17886086,
            0.17886086,
        ]
    )
    assert np.allclose(LR.excitation_energies, solutions, atol=threshold)

    LR.get_oscillator_strength()

    solutions = np.array(
        [
            0.049799,
            0.241266,
            0.241266,
        ]
    )
    assert np.allclose(LR.oscillator_strengths, solutions, atol=threshold)

def test_lih_allprojected_explicit():
    """Test LiH energies for projected q LR methods."""
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
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    threshold = 10 ** (-5)

    # naive
    LR = allprojected.LinearResponse(WF, excitations="SD")
    LR._construct_hessian_metric_blocks()

    A = LR.A
    B = LR.B
    Sigma = LR.Sigma
    Delta = LR.Delta

    def right_transform(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        trial_plus = trial[: trial.shape[0] // 2]
        trial_minus = trial[trial.shape[0] // 2 :]

        Abp = A @ trial_plus
        Abm = A @ trial_minus
        Bbp = B @ trial_plus.conj()
        Bbm = B @ trial_minus.conj()
        Sbp = Sigma @ trial_minus
        Sbm = Sigma @ trial_plus

        sigma_plus = Abp + Bbp
        sigma_minus = Abm - Bbm
        tau_plus = Sbp
        tau_minus = Sbm
        return sigma_plus, sigma_minus, tau_plus, tau_minus

    d = PairedDavidson()
    eigvals, eigvecs = d.solve(
        right_transform,
        (np.diag(A), np.diag(Sigma)),
        max_iteration=50,
        tolerance=1e-8,
        n_roots=3,
        # is_silent=True,
    )

    solutions = np.array(
        [
            0.12973291,
            0.18092743,
            0.18092743,
        ]
    )
    assert np.allclose(eigvals, solutions, atol=threshold)

    LR.normed_response_vectors = eigvecs
    LR.Z_q_normed = LR.normed_response_vectors[: len(LR.q_ops), :]
    LR.Z_G_normed = LR.normed_response_vectors[len(LR.q_ops) : len(LR.q_ops) + len(LR.G_ops), :]
    LR.Y_q_normed = LR.normed_response_vectors[len(LR.q_ops) + len(LR.G_ops) : 2 * len(LR.q_ops) + len(LR.G_ops), :]
    LR.Y_G_normed = LR.normed_response_vectors[2 * len(LR.q_ops) + len(LR.G_ops) :, :]
    LR.excitation_energies = eigvals
    LR.get_oscillator_strength()

    solutions = np.array(
        [
            0.049950,
            0.250975,
            0.250975,
        ]
    )
    assert np.allclose(LR.oscillator_strengths, solutions, atol=threshold)

def test_lih_allprojected():
    """Test LiH energies for projected q LR methods."""
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
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    threshold = 10 ** (-5)

    # naive
    LR = allprojected.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies(3, {"max_iterations": 50, "tolerance": 1e-8})

    solutions = np.array(
        [
            0.12973291,
            0.18092743,
            0.18092743,
        ]
    )
    assert np.allclose(LR.excitation_energies, solutions, atol=threshold)

    LR.get_oscillator_strength()

    solutions = np.array(
        [
            0.049950,
            0.250975,
            0.250975,
        ]
    )
    assert np.allclose(LR.oscillator_strengths, solutions, atol=threshold)
