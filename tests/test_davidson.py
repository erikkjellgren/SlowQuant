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
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("L-BFGS-B", True)

    threshold = 10 ** (-5)

    # naive
    LR_naive = naive.LinearResponse(WF, excitations="SD")

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
