import os

import numpy as np


def read_basis(atom_name: str, basis_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read basis set from file.

    Args:
        atom_name: Name of atom.
        basis_set: Name of basisset

    Returns:
        Basisset information, exponents, contraction coefficients and angular moments.
    """
    this_file_location = os.path.dirname(os.path.abspath(__file__))
    basis_set = basis_set.replace("*", "_st_")
    if len(atom_name) == 1:
        atom_name = f"{atom_name} "
    with open(f"{this_file_location}/basisset/{basis_set.lower()}.basis", "r", encoding="UTF-8") as basisfile:
        for line in basisfile:
            if line[0:2] == atom_name.lower():
                # Found the correct atom.
                break
        else:
            raise ValueError(
                "Could not find request atom in basisset.\nBasisset: {basisset}\nAtom: {atom_name}"
            )
        exponents = []
        coefficients = []
        angular_moments = []
        number_primitives = -1
        for line in basisfile:
            if line[0] == "*":
                continue
            if line[0] != " ":
                # Found new atom in file.
                break
            if "s" in line or "p" in line or "d" in line or "f" in line:
                angular = line.split()[1]
                number_primitives = int(line.split()[0])
                exponents_primitive: list[float] = []
                coefficients_primitive: list[float] = []
                continue
            if number_primitives > 0:
                exponents_primitive.append(  # pylint: disable=possibly-used-before-assignment
                    float(line.split()[0].replace("D", "e"))
                )
                coefficients_primitive.append(  # pylint: disable=possibly-used-before-assignment
                    float(line.split()[1].replace("D", "e"))
                )
                number_primitives -= 1
            if number_primitives == 0:
                if angular == "s":  # pylint: disable=possibly-used-before-assignment
                    angulars = [[0, 0, 0]]
                elif angular == "p":
                    angulars = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                elif angular == "d":
                    angulars = [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
                elif angular == "f":
                    angulars = [
                        [3, 0, 0],
                        [2, 1, 0],
                        [2, 0, 1],
                        [1, 2, 0],
                        [1, 1, 1],
                        [1, 0, 2],
                        [0, 3, 0],
                        [0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 3],
                    ]
                else:
                    raise ValueError(f"None implemented angular moment: {angular}")
                exponents.append(np.array(exponents_primitive))
                coefficients.append(np.array(coefficients_primitive))
                angular_moments.append(np.array(angulars))
                number_primitives = -1
    return exponents, coefficients, angular_moments
