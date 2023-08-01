import numpy as np

from slowquant.dft.dft_in_memory import run_ksdft
from slowquant.logger import _Logger


class _KSDFT:
    def __init__(self, molecule_obj, integral_obj, grid_obj) -> None:
        """Initialize Hartree-Fock class.

        Args:
            molecule_obj: Molecule object.
            integral_obj: Integral object.
        """
        self.mol_obj = molecule_obj
        self.int_obj = integral_obj
        self.grid_obj = grid_obj
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

    def run_restricted_ksdft(self) -> None:
        """Run restricted Hartree-Fock calculation."""
        grid_points, grid_weights = self.grid_obj.sg1_grid
        E, C, F, D = run_ksdft(
            self.int_obj.overlap_matrix,
            self.int_obj.kinetic_energy_matrix,
            self.int_obj.nuclear_attraction_matrix,
            self.int_obj.electron_repulsion_tensor,
            grid_points,
            grid_weights,
            self.mol_obj.get_basis_function_amplitude(grid_points),
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
