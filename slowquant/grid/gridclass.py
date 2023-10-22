import numpy as np

from slowquant.grid.gridfunctions import becke_reweight, calculate_sg1_grid
from slowquant.molecule.moleculeclass import _Molecule


class _Grid:
    def __init__(
        self, molecule_object: _Molecule, number_radial_points: int = 64, number_angular_points: int = 0
    ) -> None:
        self._mol_obj = molecule_object
        self.number_radial_points = number_radial_points
        if not number_angular_points in [0, 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194]:
            raise ValueError(f"Number of angular points: {number_angular_points}; is not available.")
        self.number_angular_points = number_angular_points
        self._sg1_grid: np.ndarray | None = None

    def _generate_sg1_grid(self) -> None:
        atomic_grids = []
        for atom in self._mol_obj.atoms:
            atomic_grids.append(
                calculate_sg1_grid(
                    atom.coordinate,
                    atom.nuclear_charge,
                    self.number_radial_points,
                    self.number_angular_points,
                )
            )
        reweighted_atomic_grids = becke_reweight(
            self._mol_obj.atom_coordinates, self._mol_obj.atom_charges, atomic_grids
        )
        grid = reweighted_atomic_grids[0]
        for i in range(1, len(reweighted_atomic_grids)):
            grid = np.vstack((grid, reweighted_atomic_grids[i]))
        self._sg1_grid_points = grid[:, :3]
        self._sg1_grid_weights = grid[:, 3]

    @property
    def sg1_grid(self) -> np.ndarray:
        if self._sg1_grid is None:
            self._generate_sg1_grid()
        return self._sg1_grid_points, self._sg1_grid_weights
