import numpy as np

from slowquant.logger import _Logger
from slowquant.molecularintegrals.electronrepulsion import (
    electron_repulsion_integral_driver,
    get_cauchy_schwarz_matrix,
)
from slowquant.molecularintegrals.kineticenergy import kinetic_energy_integral_driver
from slowquant.molecularintegrals.multipole import multipole_integral_driver
from slowquant.molecularintegrals.nuclearattraction import (
    nuclear_attraction_integral_driver,
)
from slowquant.molecule.moleculeclass import _Molecule


class _Integral:
    def __init__(self, molecule_object_: _Molecule) -> None:
        """Initialize integral class.

        Args:
            molecule_object_: Molecule class object.
        """
        self.molecule_object = molecule_object_
        self.store_1e_int = True
        self.store_2e_int = True
        self.logger = _Logger()
        if (self.molecule_object.number_bf) ** 4 * 8 / 10**9 > 6:
            self.logger.add_to_log(
                f"Storing 2-electron integrals in memory will use approx {self.molecule_object.number_bf**4*8/10**9:3.1f} GB",
                is_warning=True,
            )
        self.force_recompute = False
        self._overlap_int: np.ndarray | None = None
        self._kineticenergy_int: np.ndarray | None = None
        self._nuclearattraction_int: np.ndarray | None = None
        self._electronrepulsion_int: np.ndarray | None = None
        # Integral screening
        self._cauchy_schwarz_matrix = get_cauchy_schwarz_matrix(self.molecule_object)
        self.cauchy_schwarz_threshold = 10**-12

    @property
    def overlap_matrix(self) -> np.ndarray:
        """Compute overlap integral matrix.

        Returns:
            Overlap integral matrix.
        """
        if self.store_1e_int:
            if self._overlap_int is None or self.force_recompute:
                self._overlap_int = multipole_integral_driver(self.molecule_object, np.array([0, 0, 0]))
            return self._overlap_int
        # Overlap integral is a special case of multipole integral,
        # where the multipole moments are all zero.
        return multipole_integral_driver(self.molecule_object, np.array([0, 0, 0]))

    @property
    def kinetic_energy_matrix(self) -> np.ndarray:
        """Compute kinetic-energy integral matrix.

        Returns:
            Kinetic-energy integral matrix.
        """
        if self.store_1e_int:
            if self._kineticenergy_int is None or self.force_recompute:
                self._kineticenergy_int = kinetic_energy_integral_driver(self.molecule_object)
            return self._kineticenergy_int
        return kinetic_energy_integral_driver(self.molecule_object)

    @property
    def nuclear_attraction_matrix(self) -> np.ndarray:
        """Compute nuclear-attraction integral matrix.

        Returns:
            Nuclear-attraction integral matrix.
        """
        if self.store_1e_int:
            if self._nuclearattraction_int is None or self.force_recompute:
                self._nuclearattraction_int = nuclear_attraction_integral_driver(self.molecule_object)
            return self._nuclearattraction_int
        return nuclear_attraction_integral_driver(self.molecule_object)

    @property
    def electron_repulsion_tensor(self) -> np.ndarray:
        """Compute electron-repulsion integral tensor.

        Returns:
            Electron-repulsion integral tensor.
        """
        if self.store_2e_int:
            if self._electronrepulsion_int is None or self.force_recompute:
                self._electronrepulsion_int = electron_repulsion_integral_driver(
                    self.molecule_object, self._cauchy_schwarz_matrix, self.cauchy_schwarz_threshold
                )
            return self._electronrepulsion_int
        return electron_repulsion_integral_driver(
            self.molecule_object, self._cauchy_schwarz_matrix, self.cauchy_schwarz_threshold
        )

    def get_multipole_matrix(self, multipole_order: np.ndarray) -> np.ndarray:
        """Compute multipole integral matrix.

        Args:
            multipole_order: Cartesian multipole orders (x, y, z).

        Returns:
            Multipole integral matrix.
        """
        return multipole_integral_driver(self.molecule_object, multipole_order)
