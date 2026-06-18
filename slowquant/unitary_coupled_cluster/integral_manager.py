import copy

import numpy as np
import pyscf

from slowquant.molecule.polarizable_embedding import PolarizableEmbedding
from slowquant.SlowQuant import SlowQuant


class IntegralManager:
    __slots__ = (
        "PE",
        "_electric_dipole",
        "_electron_electron_repulsion",
        "_h_ao",
        "_kinetic_energy",
        "_nuclear_electron_attraction",
        "_v_PE_multipole_ao",
        "int_obj",
    )

    def __init__(self, integral_obj: SlowQuant | pyscf.gto.mole.Mole, potfile: str = "") -> None:
        """Initilize the integral manager.

        Args:
            integral_obj: Integral generator object, can either be from SlowQuant or PySCF.
        """
        self.int_obj = copy.deepcopy(integral_obj)
        self._kinetic_energy: np.ndarray | None = None
        self._nuclear_electron_attraction: np.ndarray | None = None
        self._electron_electron_repulsion: np.ndarray | None = None
        self._electric_dipole: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._h_ao: np.ndarray | None = None
        self._v_PE_multipole_ao: np.ndarray | None = None
        self.PE = None
        if potfile:
            if not isinstance(self.int_obj, pyscf.gto.mole.Mole):
                raise TypeError("Polarizable Embedding is only implemented with PySCF integrals.")
            self.PE = PolarizableEmbedding(potfile, self.int_obj)

    @property
    def num_elec(self) -> int:
        """Number of electrons."""
        if isinstance(self.int_obj, SlowQuant):
            return self.int_obj.molecule.number_electrons
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            return self.int_obj.nelectron
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")

    @property
    def kinetic_energy(self) -> np.ndarray:
        """Electron kinetic energy integrals."""
        if isinstance(self._kinetic_energy, np.ndarray):
            return self._kinetic_energy
        if isinstance(self.int_obj, SlowQuant):
            kin_int = self.int_obj.integral.kinetic_energy_matrix
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            kin_int = self.int_obj.intor("int1e_kin")
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._kinetic_energy = kin_int
        return kin_int

    @property
    def nuclear_electron_attraction(self) -> np.ndarray:
        """Nuclear-electron attraction integrals."""
        if isinstance(self._nuclear_electron_attraction, np.ndarray):
            return self._nuclear_electron_attraction
        if isinstance(self.int_obj, SlowQuant):
            nuc_el_int = self.int_obj.integral.nuclear_attraction_matrix
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            nuc_el_int = self.int_obj.intor("int1e_nuc")
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._nuclear_electron_attraction = nuc_el_int
        return nuc_el_int

    @property
    def electron_electron_repulsion(self) -> np.ndarray:
        """Electron-electron repulsion integrals."""
        if isinstance(self._electron_electron_repulsion, np.ndarray):
            return self._electron_electron_repulsion
        if isinstance(self.int_obj, SlowQuant):
            e2_int = self.int_obj.integral.electron_repulsion_tensor
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            e2_int = self.int_obj.intor("int2e")
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._electron_electron_repulsion = e2_int
        return e2_int

    @property
    def nuclear_nuclear_repulsion(self) -> float:
        """Nuclear-nuclear repulsion."""
        if isinstance(self.int_obj, SlowQuant):
            return self.int_obj.molecule.nuclear_repulsion
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            return self.int_obj.energy_nuc()
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")

    @property
    def electric_dipole(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Electric dipole integrals."""
        if isinstance(self._electric_dipole, tuple):
            return self._electric_dipole
        if isinstance(self.int_obj, SlowQuant):
            dipole_integrals = (
                self.int_obj.integral.get_multipole_matrix(np.array([1, 0, 0])),
                self.int_obj.integral.get_multipole_matrix(np.array([0, 1, 0])),
                self.int_obj.integral.get_multipole_matrix(np.array([0, 0, 1])),
            )
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            x, y, z = self.int_obj.intor("int1e_r", comp=3)
            dipole_integrals = (x, y, z)
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._electric_dipole = dipole_integrals
        return dipole_integrals

    @property
    def v_PE_multipole_ao(self) -> np.ndarray:
        if isinstance(self._v_PE_multipole_ao, np.ndarray):
            return self._v_PE_multipole_ao
        if self.PE is None:
            raise ValueError("Polarizable Embedding is not activated.")
        if not isinstance(self.int_obj, pyscf.gto.mole.Mole):
            raise NotImplementedError
        fakemol = pyscf.gto.fakemol_for_charges(self.PE.coordinates)
        mol = self.int_obj
        v_PE_multipole_ao = np.zeros((mol.nao, mol.nao), dtype=np.float64)
        # charges
        if 0 in self.PE.multipoles:
            v_PE_multipole_ao += -np.sum(
                pyscf.df.incore.aux_e2(mol, fakemol, "int3c2e") * self.PE.multipoles[0].ravel(), axis=2
            )
        # dipoles
        if 1 in self.PE.multipoles:
            field_integrals = pyscf.df.incore.aux_e2(mol, fakemol, "int3c2e_ip1").transpose(1, 2, 3, 0)
            v_dip = -np.sum(field_integrals * self.PE.multipoles[1], axis=(2, 3))
            v_PE_multipole_ao += v_dip + v_dip.T
        # quadrupoles
        if 2 in self.PE.multipoles:
            v_quad = -0.5 * np.sum(
                (
                    pyscf.df.incore.aux_e2(mol, fakemol, "int3c2e_ipip1")
                    + pyscf.df.incore.aux_e2(mol, fakemol, "int3c2e_ipvip1")
                ).transpose(1, 2, 3, 0)
                * self.PE.multipoles[2],
                axis=(2, 3),
            )
            v_PE_multipole_ao += v_quad + v_quad.T
        self._v_PE_multipole_ao = v_PE_multipole_ao
        return self._v_PE_multipole_ao

    def v_PE_induction_ao(self, density_ao: np.ndarray) -> np.ndarray:
        if self.PE is None:
            raise ValueError("Polarizable Embedding is not activated.")
        if not isinstance(self.int_obj, pyscf.gto.mole.Mole):
            raise NotImplementedError
        fakemol = pyscf.gto.fakemol_for_charges(self.PE.coordinates)
        mol = self.int_obj
        # contribution from induced dipoles
        field_integrals = pyscf.df.incore.aux_e2(mol, fakemol, "int3c2e_ip1").transpose(1, 2, 3, 0)
        electronic_field = 2.0 * np.einsum("mn,mnpx->px", density_ao, field_integrals)
        rhs_field = electronic_field + self.PE.nuclear_field + self.PE.multipole_field
        induced_dipoles = self.PE.solve_induced_dipoles(rhs_field)
        v_PE_induction_ao = -np.einsum("px,mnpx->mn", induced_dipoles, field_integrals)
        v_PE_induction_ao += v_PE_induction_ao.T
        return v_PE_induction_ao

    @property
    def h_ao(self) -> np.ndarray:
        """One-electron core hamiltonian in AO."""
        if isinstance(self._h_ao, np.ndarray):
            return self._h_ao
        if isinstance(self.int_obj, SlowQuant):
            h_core = self.nuclear_electron_attraction + self.kinetic_energy
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            h_core = self.nuclear_electron_attraction + self.kinetic_energy
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._h_ao = h_core
        if self.PE:
            self._h_ao += self.v_PE_multipole_ao
        return h_core
