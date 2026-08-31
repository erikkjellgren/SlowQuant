import copy

import numpy as np
import pyscf
from pyscf.data import nist

from slowquant.SlowQuant import SlowQuant


class IntegralManager:
    __slots__ = (
        "_angular_momentum_giao",
        "_atom_coordinates",
        "_atom_charges",
        "_electric_dipole",
        "_electron_electron_repulsion",
        "_electron_electron_repulsion_giao",
        "_overlap_giao",
        "_h_ao",
        "_kinetic_energy",
        "_nuclear_electron_attraction",
        "int_obj",
    )

    def __init__(self, integral_obj: SlowQuant | pyscf.gto.mole.Mole) -> None:
        """Initilize the integral manager.

        Args:
            integral_obj: Integral generator object, can either be from SlowQuant or PySCF.
        """
        self.int_obj = copy.deepcopy(integral_obj)
        self._kinetic_energy: np.ndarray | None = None
        self._nuclear_electron_attraction: np.ndarray | None = None
        self._electron_electron_repulsion: np.ndarray | None = None
        self._electric_dipole: np.ndarray | None = None
        self._h_ao: np.ndarray | None = None
        self._atom_coordinates: np.ndarray | None = None
        self._atom_charges: np.ndarray | None = None
        self._angular_momentum_giao: np.ndarray | None = None
        self._overlap_giao: np.ndarray | None = None
        self._electron_electron_repulsion_giao: np.ndarray | None = None

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
    def electric_dipole(self) -> np.ndarray:
        """Electric dipole integrals."""
        if isinstance(self._electric_dipole, np.ndarray):
            return self._electric_dipole
        if isinstance(self.int_obj, SlowQuant):
            dipole_integrals = np.stack((
                self.int_obj.integral.get_multipole_matrix(np.array([1, 0, 0])),
                self.int_obj.integral.get_multipole_matrix(np.array([0, 1, 0])),
                self.int_obj.integral.get_multipole_matrix(np.array([0, 0, 1])),
            ))
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            dipole_integrals = self.int_obj.intor("int1e_r", comp=3)
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._electric_dipole = dipole_integrals
        return dipole_integrals

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
        return h_core

    def diamagnetic_shielding(self, atom_coord, common_orig = (0,0,0)) -> np.ndarray:
        """Diamagnetic shielding integrals."""
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("Diamagnetic shielding integrals not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            self.int_obj.set_common_orig(common_orig)
            self.int_obj.set_rinv_origin(atom_coord)
            dia_shield = self.int_obj.intor('int1e_cg_a11part', comp=9)
            dia_shield[::4] -= (dia_shield[0] + dia_shield[4] + dia_shield[8])
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        return dia_shield
    
    def orbital_paramagnetic(self, atom_coord) -> np.ndarray:
        """Paramagnetic spin orbit integrals."""
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("Paramagnetic spin orbit integrals not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            self.int_obj.set_rinv_orig(atom_coord)
            orbital_paramagnetic = self.int_obj.intor('int1e_prinvxp', 3)
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        return orbital_paramagnetic
    
    def angular_momentum(self, common_orig = (0,0,0)) -> np.ndarray:
        """Angular moment integrals."""
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("Angular momentum integrals not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            self.int_obj.set_common_origin(common_orig)
            angular_momentum = self.int_obj.intor('int1e_cg_irxp', 3) / 2
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        return angular_momentum
    
    @property
    def atom_coordinates(self) -> np.ndarray:
        """Atom coordinates."""
        if isinstance(self._atom_coordinates, np.ndarray):
            return self._atom_coordinates
        if isinstance(self.int_obj, SlowQuant):
            atom_coordinates = self.int_obj.molecule.atom_coordinates
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            atom_coordinates = self.int_obj.atom_coords()
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._atom_coordinates = atom_coordinates
        return atom_coordinates
    
    @property
    def atom_charges(self) -> np.ndarray:
        """Atom charges."""
        if isinstance(self._atom_charges, np.ndarray):
            return self._atom_charges
        if isinstance(self.int_obj, SlowQuant):
            atom_charges = self.int_obj.molecule.atom_charges
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            atom_charges = self.int_obj.atom_charges()
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._atom_charges = atom_charges
        return atom_charges
    
    def orbital_diamagnetic(self, atom1_coord, atom2_coord) -> np.ndarray:
        """Diamagnetic spin orbit integrals.
        vec{r}vec{r}/(|r-orig1|^3 |r-orig2|^3)
        Ref. JCP, 73, 5718"""
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("Orbital diamagnetic integrals not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            NUMINT_GRIDS = 30
            t, w = np.polynomial.legendre.leggauss(NUMINT_GRIDS)
            a = (1+t)/(1-t) * .8
            w *= 2/(1-t)**2 * .8
            fakemol = pyscf.gto.Mole()
            fakemol._atm = np.asarray([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
            fakemol._bas = np.asarray([[0, 1, NUMINT_GRIDS, 1, 0, 3, 3+NUMINT_GRIDS, 0]],
                                        dtype=np.int32)
            p_cart2sph_factor = 0.488602511902919921
            fakemol._env = np.hstack((atom2_coord, a**2, a**2*w*4/np.pi**.5/p_cart2sph_factor))
            fakemol._built = True

            pmol = self.int_obj + fakemol
            pmol.set_rinv_origin(atom1_coord)
            # <nabla i, j | k>  k is a fictitious basis for numerical integraion
            mat1 = pmol.intor(self.int_obj._add_suffix('int3c1e_iprinv'), comp=3,
                            shls_slice=(0, self.int_obj.nbas, 0, self.int_obj.nbas, self.int_obj.nbas, pmol.nbas))
            # <i, j | nabla k>
            mat  = pmol.intor(self.int_obj._add_suffix('int3c1e_iprinv'), comp=3,
                            shls_slice=(self.int_obj.nbas, pmol.nbas, 0, self.int_obj.nbas, 0, self.int_obj.nbas))
            mat += mat1.transpose(0,3,1,2) + mat1.transpose(0,3,2,1)
            orbital_diamagnetic = mat.reshape(9, *mat.shape[-2:])
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        return orbital_diamagnetic
    
    def fermi_contact(self, atom_coord) -> np.ndarray:
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("Fermi contact integrals not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            amp_basis = self.int_obj.eval_gto("GTOval_sph", coords=[atom_coord])[0]
            fermi_contact = np.array([np.outer(np.conj(amp_basis), amp_basis)]) * nist.G_ELECTRON * 2/3 * np.pi
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        return fermi_contact
    
    def spin_dipolar_fermi_contact(self, atom_coord) -> np.ndarray:
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("Spi dipolar integrals not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            self.int_obj.set_rinv_origin(atom_coord)
            a01p = self.int_obj.intor('int1e_sa01sp', 12).reshape(3, 4, self.int_obj.nao, self.int_obj.nao) * nist.G_ELECTRON / 4
            spin_dip_fermi_cont = -(a01p[:,:3] + a01p[:,:3].transpose(0,1,3,2)).reshape(9, self.int_obj.nao, self.int_obj.nao)
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        return spin_dip_fermi_cont

    @property
    def angular_momentum_giao(self) -> np.ndarray:
        """Angular moment integrals in GIAOs."""
        if isinstance(self._angular_momentum_giao, np.ndarray):
            return self._angular_momentum_giao
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("Angular momentum integrals in GIAOs not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            angular_momentum_giao = self.int_obj.intor("int1e_giao_irjxp", 3) / 2 + self.int_obj.intor("int1e_igkin", 3) + self.int_obj.intor("int1e_ignuc", 3)
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._angular_momentum_giao = angular_momentum_giao
        return angular_momentum_giao

    @property
    def overlap_giao(self) -> np.ndarray:
        """First derivative of the overlap integrals in GIAOs."""
        if isinstance(self._overlap_giao, np.ndarray):
            return self._overlap_giao
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("First derivative of the overlap integrals in GIAOs not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            overlap_giao = self.int_obj.intor("int1e_igovlp", 3)
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._overlap_giao = overlap_giao
        return overlap_giao

    @property
    def electron_electron_repulsion_giao(self) -> np.ndarray:
        """First derivative of the electron-electron repulsion integrals in GIAOs."""
        if isinstance(self._electron_electron_repulsion_giao, np.ndarray):
            return self._electron_electron_repulsion_giao
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("First derivative of the electron-electron repulsion integrals in GIAOs not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            electron_electron_repulsion_giao = self.int_obj.intor("int2e_ig1", 3)
            electron_electron_repulsion_giao = electron_electron_repulsion_giao + electron_electron_repulsion_giao.transpose(0,3,4,1,2)
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._electron_electron_repulsion_giao = electron_electron_repulsion_giao
        return electron_electron_repulsion_giao

    def diamagnetic_shielding_giao(self, atom_coord) -> np.ndarray:
        """Diamagnetic shielding integrals in GIAOs."""
        if isinstance(self.int_obj, SlowQuant):
            raise ValueError("Diamagnetic shielding integrals in GIAOs not implemented for integral object, {type(self.int_obj)}. Use integral object, {pyscf.gto.mole.Mole}")
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            self.int_obj.set_rinv_origin(atom_coord)
            self.int_obj.set_common_orig(atom_coord)
            print(self.int_obj.intor('int1e_giao_a11part', comp=9)[0])
            print(self.int_obj.intor('int1e_a01gp', comp=9)[0])
            dia_shield = self.int_obj.intor('int1e_giao_a11part', comp=9)
            dia_shield[::4] -= (dia_shield[0] + dia_shield[4] + dia_shield[8])
            dia_shield += self.int_obj.intor('int1e_a01gp', comp=9)
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        return dia_shield