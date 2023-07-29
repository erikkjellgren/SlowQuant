import numpy as np

from slowquant.hartreefock.diis import DIIS
from slowquant.logger import _Logger


def functional(rho):
    A = -3 / 4 * (3 / np.pi) ** (1 / 3)
    e_x = A * rho ** (1 / 3)
    v_x = e_x + (A / 3 * rho ** (1 / 3))
    return e_x, v_x


def run_ksdft(
    S: np.ndarray,
    T: np.ndarray,
    V: np.ndarray,
    ERI: np.ndarray,
    grid_points: np.ndarray,
    grid_weights: np.ndarray,
    bf_amplitudes: np.ndarray,
    # functional,
    number_electrons: int,
    log: _Logger,
    dE_threshold: float,
    rmsd_threshold: float,
    max_scf_iterations: int,
    use_diis: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    r"""Run restricted Hartree-Fock calculation.

    Reference: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303
    """
    # Logging
    log.add_to_log(f"{'Iter':^4}    {'E_HF':^18}    {'DeltaE':^13}    {'RMS_D':^12}")

    # Core Hamiltonian
    Hcore = T + V

    # Diagonalizing overlap matrix
    Lambda_S, L_S = np.linalg.eigh(S)
    # Symmetric orthogonal inverse overlap matrix
    S_sqrt = np.dot(np.dot(L_S, np.diag(Lambda_S ** (-1 / 2))), np.transpose(L_S))

    # Initial Density
    F0prime = np.dot(np.dot(np.transpose(S_sqrt), Hcore), np.transpose(S_sqrt))
    _, C0prime = np.linalg.eigh(F0prime)

    C0 = np.transpose(np.dot(S_sqrt, C0prime))

    # Only using occupied MOs
    C0 = C0[0 : int(number_electrons / 2)]
    D0 = 2 * np.dot(np.transpose(C0), C0)

    # Initial Energy
    E0 = np.einsum('ij,ij->', D0, Hcore)

    # Init DIIS
    if use_diis:
        number_bf = len(S)
        diis_acceleration = DIIS(number_bf)

    # SCF iterations
    for iteration in range(1, max_scf_iterations + 1):
        # New Fock Matrix
        J = np.einsum('pqrs,sr->pq', ERI, D0)

        rho = density = np.einsum('kp,kq,pq->k', bf_amplitudes, bf_amplitudes, D0)
        e_xc, v_xc = functional(rho)
        E_xc = np.einsum('k,k,k->', e_xc, rho, grid_weights)
        V_xc = np.einsum('kp,kq,k,k->pq', bf_amplitudes, bf_amplitudes, v_xc, grid_weights)

        F = Hcore + J + V_xc

        # Do DIIS acceleration
        if use_diis:
            F = diis_acceleration.get_extrapolated_fock_matrix(F, D0, S)

        Fprime = np.dot(np.dot(np.transpose(S_sqrt), F), S_sqrt)
        _, Cprime = np.linalg.eigh(Fprime)

        C = np.dot(S_sqrt, Cprime)

        CT = np.transpose(C)
        CTocc = CT[0 : int(number_electrons / 2)]
        Cocc = np.transpose(CTocc)

        D = 2 * np.dot(Cocc, CTocc)

        # New SCF Energy
        E = 0.5 * np.einsum('ij,ij->', D, 2 * Hcore + J) + E_xc

        # Convergance
        dE = E - E0
        rmsd = np.einsum('ij->', (D - D0) ** 2) ** 0.5

        # Logging
        log.add_to_log(f'{iteration:>4}    {E: 18.12f}    {dE: 1.6e}    {rmsd:1.6e}')

        D0 = D
        E0 = E
        if np.abs(dE) < dE_threshold and rmsd < rmsd_threshold:
            break
    else:
        log.add_to_log(
            f'Restricted Hartree-Fock did not meet convergence requirements in {max_scf_iterations} iterations',
            is_warning=True,
        )

    return E, C, F, D
