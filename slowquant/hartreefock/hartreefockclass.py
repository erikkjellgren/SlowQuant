import numpy as np

import slowquant.hartreefock.hartreefock_in_memory as hf_in_mem
import slowquant.hartreefock.unrestricted_hartreefock_in_memory as uhf_in_mem
from slowquant.logger import _Logger


class _HartreeFock:
    def __init__(self, molecule_obj, integral_obj) -> None:
        """Initialize Hartree-Fock class.

        Args:
            molecule_obj: Molecule object.
            integral_obj: Integral object.
        """
        self.mol_obj = molecule_obj
        self.int_obj = integral_obj
        self.de_threshold = 10**-9
        self.rmsd_threshold = 10**-9
        self.max_scf_iterations = 100
        self.uhf_lumo_homo_mix_coeff = 0.15
        self.use_diis = True
        self.logger = _Logger()
        # Attributes generated from calculations
        self.E_hf: float
        self.E_uhf: float
        self.mo_coeff: np.ndarray
        self.rdm1: np.ndarray
        self.fock_matrix: np.ndarray
        self.mo_coeff_alpha: np.ndarray
        self.rdm1_alpha: np.ndarray
        self.fock_matrix_alpha: np.ndarray
        self.mo_coeff_beta: np.ndarray
        self.rdm1_beta: np.ndarray
        self.fock_matrix_beta: np.ndarray

    @property
    def log(self) -> str:
        """Get log."""
        return self.logger.log

    def run_restricted_hartree_fock(self) -> None:
        """Run restricted Hartree-Fock calculation."""
        E, C, F, D = hf_in_mem.run_hartree_fock(
            self.int_obj.overlap_matrix,
            self.int_obj.kinetic_energy_matrix,
            self.int_obj.nuclear_attraction_matrix,
            self.int_obj.electron_repulsion_tensor,
            self.mol_obj.number_electrons,
            self.logger,
            self.de_threshold,
            self.rmsd_threshold,
            self.max_scf_iterations,
            self.use_diis,
        )
        self.E_hf = E
        self.mo_coeff = C
        self.rdm1 = D
        self.fock_matrix = F

    def run_unrestricted_hartree_fock(self) -> None:
        """Run restricted Hartree-Fock calculation."""
        E, C_alpha, C_beta, F_alpha, F_beta, D_alpha, D_beta = uhf_in_mem.run_unrestricted_hartree_fock(
            self.int_obj.overlap_matrix,
            self.int_obj.kinetic_energy_matrix,
            self.int_obj.nuclear_attraction_matrix,
            self.int_obj.electron_repulsion_tensor,
            self.mol_obj.number_electrons,
            self.uhf_lumo_homo_mix_coeff,
            self.logger,
            self.de_threshold,
            self.rmsd_threshold,
            self.max_scf_iterations,
            self.use_diis,
        )
        self.E_uhf = E
        self.mo_coeff_alpha = C_alpha
        self.mo_coeff_beta = C_beta
        self.rdm1_alpha = D_alpha
        self.rdm1_beta = D_beta
        self.fock_matrix_alpha = F_alpha
        self.fock_matrix_beta = F_beta
