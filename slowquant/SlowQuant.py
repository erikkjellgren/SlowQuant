# pylint: disable=C0103
from slowquant.molecularintegrals.integralclass import _Integral
from slowquant.molecule.moleculeclass import _Molecule


class SlowQuant:
    def __init__(self) -> None:
        """Initialize SlowQuant."""
        self.molecule: _Molecule | None = None
        self.integral: _Integral | None = None

    def set_molecule(self, molecule_file: str, distance_unit: str = "bohr") -> None:
        """Initialize molecule instance.

        Args:
            molecule_file: Filename of file containing molecular coordinates.
            distance_unit: Distance unit used coordinate file. Angstrom or Bohr (default). Internal representatio is Bohr.
        """
        self.molecule = _Molecule(molecule_file, distance_unit)

    def initialize_integrals(self):
        """Initialize integrals instance."""
        self.integral = _Integral(self.molecule)
