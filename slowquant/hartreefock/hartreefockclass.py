import numpy as np

import slowquant.hartreefock.hartreefock_in_memory as hf_in_mem
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
        self.logger = _Logger()
        # Attributes generated from calculations
        self.E_hf: float | None = None
        self.mo_coeff: np.ndarray | None = None
        self.RDM1: np.ndarray | None = None
        self.fock_matrix: np.ndarray | None = None

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
        )
        self.E_hf = E
        self.mo_coeff = C
        self.RDM1 = D
        self.fock_matrix = F
