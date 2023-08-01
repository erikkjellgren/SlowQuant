import numpy as np

from slowquant.molecularintegrals.integralclass import _Integral
from slowquant.molecule.moleculeclass import _Molecule
from slowquant.properties.propertiesfunctions import dipole_moment


class _Properties:
    def __init__(self, molecule_object: _Molecule, integral_object: _Integral) -> None:
        """Initialize properties class.

        Args:
            molecule_object: Molecule object.
            integral_object: Integral object.
        """
        self.mol_obj = molecule_object
        self.int_obj = integral_object

    def get_dipole_moment(self, rdm1: np.ndarray) -> np.ndarray:
        """Get dipole moment with repect to center of mass.

        Args:
            rdm1: One-electron reduced density matrix.

        Returns:
            Dipole moment (x, y, z).
        """
        dipole_integrals = []
        dipole_integrals.append(self.int_obj.get_multipole_matrix(np.array([1, 0, 0])))
        dipole_integrals.append(self.int_obj.get_multipole_matrix(np.array([0, 1, 0])))
        dipole_integrals.append(self.int_obj.get_multipole_matrix(np.array([0, 0, 1])))
        return dipole_moment(
            dipole_integrals,
            rdm1,
            self.mol_obj.atom_charges,
            self.mol_obj.atom_coordinates,
            self.mol_obj.center_of_mass,
        )
