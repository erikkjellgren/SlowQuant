# pylint: disable=C0103
from slowquant.dft.dftclass import _KSDFT
from slowquant.grid.gridclass import _Grid
from slowquant.hartreefock.hartreefockclass import _HartreeFock
from slowquant.logger import _Logger
from slowquant.molecularintegrals.integralclass import _Integral
from slowquant.molecule.moleculeclass import _Molecule
from slowquant.properties.propertiesclass import _Properties


class SlowQuant:
    def __init__(self) -> None:
        """Initialize SlowQuant."""
        self.molecule: _Molecule | None = None
        self.integral: _Integral
        self.hartree_fock: _HartreeFock
        self.properties: _Properties
        self.grid: _Grid
        self.logger = _Logger()

    def set_molecule(
        self,
        molecule_file: str,
        molecular_charge: int = 0,
        distance_unit: str = 'bohr',
        basis_set: str | None = None,
    ) -> None:
        """Initialize molecule instance.

        Args:
            molecule_file: Filename of file containing molecular coordinates.
            molecular_charge: Total charge of molecule.
            distance_unit: Distance unit used coordinate file. Angstrom or Bohr (default). Internal representatio is Bohr.
            basis_set: Name of atom-centered basis set.
        """
        self.molecule = _Molecule(molecule_file, molecular_charge, distance_unit)
        if basis_set is not None:
            self.molecule._set_basis_set(basis_set)  # pylint: disable=W0212
            self.integral = _Integral(self.molecule)

    def set_basis_set(self, basis_set: str) -> None:
        """Set basis set.

        Args:
            basis_set: Name of basis set.
        """
        if self.molecule is not None:
            self.molecule._set_basis_set(basis_set)  # pylint: disable=W0212
            self.integral = _Integral(self.molecule)
            self.grid = _Grid(self.molecule)
        else:
            self.logger.add_to_log('Cannot set basis set, molecule is not defined.', is_warning=True)

    def init_hartree_fock(self) -> None:
        """Initialize Hartree-Fock module."""
        self.hartree_fock = _HartreeFock(self.molecule, self.integral)

    def init_properties(self) -> None:
        """Initialize Hartree-Fock module."""
        self.properties = _Properties(self.molecule, self.integral)

    def init_kohn_sham_dft(self) -> None:
        """Initialize Kohn-Sham DFT module."""
        self.ksdft = _KSDFT(self.molecule, self.integral, self.grid)
