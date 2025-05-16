import numpy as np

from slowquant.hartreefock.diis import DIIS
from slowquant.logger import _Logger


def run_unrestricted_hartree_fock(
    S: np.ndarray,
    T: np.ndarray,
    V: np.ndarray,
    ERI: np.ndarray,
    number_electrons_alpha: int,
    number_electrons_beta: int,
    lumo_homo_mix_coeff: float,
    log: _Logger,
    dE_threshold: float,
    rmsd_threshold: float,
    max_scf_iterations: int,
    use_diis: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Run unrestricted Hartree-Fock calculation."""
    # Logging
    log.add_to_log(
        f"{'Iter':^4}    {'E_UHF':^18}    {'DeltaE':^13}    {'RMS_D_alpha':^12}    {'RMS_D_beta':^12}"
    )

    # Core Hamiltonian
    Hcore = T + V

    # Diagonalizing overlap matrix
    Lambda_S, L_S = np.linalg.eigh(S)
    # Symmetric orthogonal inverse overlap matrix
    S_sqrt = np.dot(np.dot(L_S, np.diag(Lambda_S ** (-1 / 2))), np.transpose(L_S))

    # Initial Density
    F0prime = np.dot(np.dot(np.transpose(S_sqrt), Hcore), np.transpose(S_sqrt))
    _, C0prime = np.linalg.eigh(F0prime)

    C0_alpha = np.transpose(np.dot(S_sqrt, C0prime))
    C0_beta = np.transpose(np.dot(S_sqrt, C0prime))
    # Break spatial symmetry if system is closed shell
    if number_electrons_alpha == number_electrons_beta:
        C_HOMO = C0_alpha[number_electrons_alpha - 1]
        C_LUMO = C0_alpha[number_electrons_alpha]
        C0_alpha[number_electrons_alpha - 1] = (
            1 / ((1 + lumo_homo_mix_coeff**2)) ** 0.5 * (C_HOMO + lumo_homo_mix_coeff * C_LUMO)
        )
        C0_alpha[number_electrons_alpha] = (
            1 / ((1 + lumo_homo_mix_coeff**2)) ** 0.5 * (-lumo_homo_mix_coeff * C_HOMO + C_LUMO)
        )
    # Only using occupied MOs
    C0_alpha = C0_alpha[0 : int(number_electrons_alpha)]
    C0_beta = C0_beta[0 : int(number_electrons_beta)]
    D0_alpha = np.dot(np.transpose(C0_alpha), C0_alpha)
    D0_beta = np.dot(np.transpose(C0_beta), C0_beta)

    # Initial Energy
    E0 = np.einsum("ij,ij->", D0_alpha, Hcore) + np.einsum("ij,ij->", D0_beta, Hcore)

    # Init DIIS
    if use_diis:
        number_bf = len(S)
        diis_acceleration_alpha = DIIS(number_bf)
        diis_acceleration_beta = DIIS(number_bf)

    # SCF iterations
    for iteration in range(1, max_scf_iterations + 1):
        # New Fock Matrix
        J_alpha = np.einsum("pqrs,sr->pq", ERI, D0_alpha)
        K_alpha = np.einsum("psqr,sr->pq", ERI, D0_alpha)
        J_beta = np.einsum("pqrs,sr->pq", ERI, D0_beta)
        K_beta = np.einsum("psqr,sr->pq", ERI, D0_beta)
        F_alpha = Hcore + J_alpha + J_beta - K_alpha
        F_beta = Hcore + J_beta + J_alpha - K_beta

        # Do DIIS acceleration
        if use_diis:
            F_alpha = diis_acceleration_alpha.get_extrapolated_fock_matrix(F_alpha, D0_alpha, S)
            F_beta = diis_acceleration_beta.get_extrapolated_fock_matrix(F_beta, D0_beta, S)

        Fprime_alpha = np.dot(np.dot(np.transpose(S_sqrt), F_alpha), S_sqrt)
        Fprime_beta = np.dot(np.dot(np.transpose(S_sqrt), F_beta), S_sqrt)
        _, Cprime_alpha = np.linalg.eigh(Fprime_alpha)
        _, Cprime_beta = np.linalg.eigh(Fprime_beta)

        C_alpha = np.dot(S_sqrt, Cprime_alpha)
        C_beta = np.dot(S_sqrt, Cprime_beta)

        CT_alpha = np.transpose(C_alpha)
        CTocc_alpha = CT_alpha[0 : int(number_electrons_alpha)]
        Cocc_alpha = np.transpose(CTocc_alpha)
        CT_beta = np.transpose(C_beta)
        CTocc_beta = CT_beta[0 : int(number_electrons_beta)]
        Cocc_beta = np.transpose(CTocc_beta)

        D_alpha = np.dot(Cocc_alpha, CTocc_alpha)
        D_beta = np.dot(Cocc_beta, CTocc_beta)

        # New SCF Energy
        E = 0.5 * np.einsum("ij,ij->", D_alpha, Hcore + F_alpha) + 0.5 * np.einsum(
            "ij,ij->", D_beta, Hcore + F_beta
        )

        # Convergance
        dE = E - E0
        rmsd_alpha = np.einsum("ij->", (D_alpha - D0_alpha) ** 2) ** 0.5
        rmsd_beta = np.einsum("ij->", (D_beta - D0_beta) ** 2) ** 0.5

        # Logging
        log.add_to_log(
            f"{iteration:>4}    {E: 18.12f}    {dE: 1.6e}    {rmsd_alpha:1.6e}     {rmsd_beta:1.6e}"
        )

        D0_alpha = D_alpha
        D0_beta = D_beta
        E0 = E
        if np.abs(dE) < dE_threshold and rmsd_alpha < rmsd_threshold and rmsd_beta < rmsd_threshold:
            break
    else:
        log.add_to_log(
            f"Unrestricted Hartree-Fock did not meet convergence requirements in {max_scf_iterations} iterations",
            is_warning=True,
        )

    return E, C_alpha, C_beta, F_alpha, F_beta, D_alpha, D_beta
