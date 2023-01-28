import numpy as np

from slowquant.molecule.basis_reader import read_basis
from slowquant.molecule.constants import atom_to_properties
from slowquant.molecule.moleculefunctions import (
    contracted_normalization,
    primitive_normalization,
)


class _Molecule:
    def __init__(self, molecule_file: str, molecular_charge_: int = 0, distance_unit: str = "bohr") -> None:
        """Initialize molecule instance.

        Args:
            molecule_file: Filename of file containing molecular coordinates.
            molecular_charge_: Total charge of molecule.
            distance_unit: Distance unit used coordinate file.
                           Angstrom or Bohr (default).
                           Internal representation is Bohr.
        """
        self.molecular_charge = molecular_charge_
        self.shells: list[Shell] | None = None
        self.number_bf = 0
        if distance_unit.lower() == "angstrom":
            unit_factor = 1.889725989
        elif distance_unit.lower() == "au" or distance_unit.lower() == "bohr":
            unit_factor = 1.0
        else:
            raise ValueError(
                "distance_unit not valid can be 'angstrom' or 'bohr'. Was given: {distance_unit}"
            )

        if ".xyz" in molecule_file:
            with open(molecule_file, "r", encoding="UTF-8") as file:
                self.atoms = []
                for i, line in enumerate(file):
                    if i < 2:
                        continue
                    self.atoms.append(
                        Atom(
                            line.split()[0],
                            np.array(
                                [
                                    float(line.split()[1]) * unit_factor,
                                    float(line.split()[2]) * unit_factor,
                                    float(line.split()[3]) * unit_factor,
                                ]
                            ),
                            int(atom_to_properties(line.split()[0], "charge")),
                            atom_to_properties(line.split()[0], "mass"),
                        )
                    )
        elif ";" in molecule_file:
            lines = molecule_file.split(";")
            self.atoms = []
            for line in lines:
                if len(line) == 0 or line == "\n":
                    # If last line in input got an ";",
                    # then last line in the reading will be empty, or just a newline.
                    continue
                self.atoms.append(
                    Atom(
                        line.split()[0],
                        np.array(
                            [
                                float(line.split()[1]) * unit_factor,
                                float(line.split()[2]) * unit_factor,
                                float(line.split()[3]) * unit_factor,
                            ]
                        ),
                        int(atom_to_properties(line.split()[0], "charge")),
                        atom_to_properties(line.split()[0], "mass"),
                    )
                )
        else:
            raise ValueError(
                "Does only support:\n    .xyz files for molecule coordinates.\n    A string with the elements and coordinates (; delimited)."
            )

    def _set_basis_set(self, basis_set: str) -> None:
        """Set basis set.

        Args:
            basis_set: Name of basis set.
        """
        self.shells = []
        self.number_bf = 0
        for atom in self.atoms:
            bfs_exponents, bfs_contraction_coefficients, bfs_angular_moments = read_basis(
                atom.atom_name, basis_set
            )
            for bf_exponents, bf_contraction_coefficients, bf_angular_moments in zip(
                bfs_exponents, bfs_contraction_coefficients, bfs_angular_moments
            ):
                self.shells.append(
                    Shell(
                        atom.coordinate,
                        bf_exponents,
                        bf_contraction_coefficients,
                        bf_angular_moments,
                        self.number_bf,
                    )
                )
                self.number_bf += len(bf_angular_moments)

    @property
    def atom_coordinates(self) -> np.ndarray:
        """Get atom coordinates.

        Returns:
            Atom coordinates.
        """
        coords = np.zeros((len(self.atoms), 3))
        for i, atom in enumerate(self.atoms):
            coords[i, :] = atom.coordinate
        return coords

    @property
    def atom_charges(self) -> np.ndarray:
        """Get atom charges.

        Returns:
            Atom charges.
        """
        charges = np.zeros(len(self.atoms))
        for i, atom in enumerate(self.atoms):
            charges[i] = atom.nuclear_charge
        return charges

    @property
    def number_electrons(self) -> int:
        """Get number of electrons.

        Returns:
            Number of electrons.
        """
        n_elec = -self.molecular_charge
        for atom in self.atoms:
            n_elec += atom.nuclear_charge
        return n_elec


class Atom:
    def __init__(
        self,
        name: str,
        coordinate_: float,
        charge: int,
        mass: float,
    ) -> None:
        """Initialize atom instance.

        Args:
            name: Atom name.
            coordinate_: Atom coordinate.
            charge: Atom nuclear charge.
            mass: Atomic mass.
        """
        self.atom_name = name
        self.coordinate = coordinate_
        self.nuclear_charge = charge
        self.atomic_mass = mass


class Shell:
    def __init__(
        self,
        center_: np.ndarray,
        exponents_: np.ndarray,
        contraction_coefficients_: np.ndarray,
        angular_moments_: np.ndarray,
        bf_idx_: int,
    ):
        """Initialize shell instance.

        Args:
            center_: x,y,z-coordinate for shell location.
            exponents_: Gaussian exponents.
            contraction_coefficients_: Contraction coefficients.
            angular_moments_: Angular moments of the form x,y,z.
            bf_idx_: Starting index of basis-function in shell.
        """
        self.center = center_
        self.exponents = exponents_
        self.contraction_coefficients = contraction_coefficients_
        self.normalization = np.zeros((len(angular_moments_), len(exponents_)))
        for bf_i, angular_moment in enumerate(angular_moments_):
            for i, exponent in enumerate(exponents_):
                self.normalization[bf_i, i] = primitive_normalization(exponent, angular_moment)
            self.normalization[bf_i, :] *= contracted_normalization(
                exponents_, contraction_coefficients_ * self.normalization[bf_i, :], angular_moment
            )
        self.angular_moments = angular_moments_
        self.bf_idx = np.array((range(bf_idx_, bf_idx_ + len(angular_moments_))))
