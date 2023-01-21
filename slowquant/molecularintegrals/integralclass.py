import numpy as np

from slowquant.molecularintegrals.integralfunctions import overlap_integral_driver
from slowquant.molecule.moleculeclass import _Molecule


class _Integral:
    def __init__(self, molecule_object_: _Molecule) -> None:
        """Initialize integral class.

        Args:
            molecule_object_: Molecule class object.
        """
        self.molecule_object = molecule_object_

    def get_overlap_matrix(self) -> np.ndarray:
        """Compute overlap integral matrix.

        Returns:
            Overlap integral matrix.
        """
        return overlap_integral_driver(self.molecule_object)
