import copy

import numpy as np
import pyscf

from slowquant.SlowQuant import SlowQuant


class IntegralManager:
    __slots__ = (
        "_electric_dipole",
        "_electron_electron_repulsion",
        "_h_ao",
        "_kinetic_energy",
        "_nuclear_electron_attraction",
        "_overlap",
        "_magnetic_field_H",
        "_magnetic_field_H_z",
        "int_obj",
        "x2c",
        "ecp",
        "B",
        "orig",

    )

    def __init__(self, integral_obj: SlowQuant | pyscf.gto.mole.Mole, x2c: bool = False, ecp = False, B = None, orig = None) -> None:
        """Initilize the integral manager.

        Args:
            integral_obj: Integral generator object, can either be from SlowQuant or PySCF.
        """
        self.int_obj = copy.deepcopy(integral_obj)
        self._kinetic_energy: np.ndarray | None = None
        self._nuclear_electron_attraction: np.ndarray | None = None
        self._electron_electron_repulsion: np.ndarray | None = None
        self._overlap: np.ndarray | None = None
        self._electric_dipole: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._h_ao: np.ndarray | None = None
        self._magnetic_field_H: np.ndarray | None = None
        self._magnetic_field_H_z: np.ndarray | None = None
        self.x2c = x2c
        self.ecp = ecp
        self.B = B
        self.orig = orig

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
    def h_ao(self) -> np.ndarray:
        """One-electron core hamiltonian in AO."""
        if isinstance(self._h_ao, np.ndarray):
            return self._h_ao
        if isinstance(self.int_obj, SlowQuant):
            h_core = self.nuclear_electron_attraction + self.kinetic_energy
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            if self.x2c:
                mf = pyscf.scf.GHF(self.int_obj).x2c()
                h_core = mf.get_hcore()
            elif self.ecp:
                h_core = self.nuclear_electron_attraction + self.kinetic_energy + self.int_obj.intor("ECPscalar")
            else:
                h_core = self.nuclear_electron_attraction + self.kinetic_energy
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._h_ao = h_core
        return h_core

    @property
    def overlap(self) -> np.ndarray:
        """Overlap integral in AO."""
        if isinstance(self._overlap, np.ndarray):
            return self._overlap
        if isinstance(self.int_obj, SlowQuant):
            overlap_int = self.int_obj.integral.overlap_matrix
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            overlap_int = self.int_obj.intor("int1e_ovlp")
        else:
            raise ValueError("Got unknown integral object, {type(self.int_obj)}")
        self._overlap = overlap_int
        return overlap_int

    @property
    def magnetic_field_H(self) -> np.ndarray:
        # Building the magnetic field Hamiltonian: 
        nao = self.int_obj.nao

        if self.orig == None:
            self.orig = (0.0, 0.0, 0.0)

        self.int_obj.set_common_origin(self.orig)

        L = self.int_obj.intor('int1e_cg_irxp')

        L_spinor = np.zeros((3, 2 * nao, 2 * nao), dtype=complex)
        L_spinor[:, :nao, :nao] = L
        L_spinor[:, nao:, nao:] = L
    
        rr = self.int_obj.intor('int1e_rr').reshape(3, 3, nao, nao)
        dia_ao = 1/8 * (  (self.B[1]**2 + self.B[2]**2) * rr[0, 0]  + (self.B[0]**2 + self.B[2]**2) * rr[1, 1] 
                        + (self.B[0]**2 + self.B[1]**2) * rr[2, 2] 
                        - 2 * self.B[0]*self.B[1] * rr[0,1] - 2 * self.B[1]*self.B[2] * rr[1,2] - 2 * self.B[0]*self.B[2] * rr[0,2]) 
    
        H_dia = np.zeros((2 * nao, 2 * nao), dtype=complex)
        H_dia[:nao, :nao] = dia_ao
        H_dia[nao:, nao:] = dia_ao
    
        ovlp = self.int_obj.intor("int1e_ovlp")
    
        sigmaSB = np.zeros((2 * nao, 2 * nao), dtype=complex)
        sigmaSB[:nao, :nao] =  ovlp * self.B[2]
        sigmaSB[nao:, nao:] = -ovlp * self.B[2]
        sigmaSB[:nao, nao:] =  ovlp * self.B[0] - 1j * ovlp * self.B[1]
        sigmaSB[nao:, :nao] =  ovlp * self.B[0] + 1j * ovlp * self.B[1]

        mf = pyscf.scf.GHF(self.int_obj)
        hcoreB = mf.get_hcore().astype(complex)

        g_e = 2.00231930436256            # Electronic g-factor

        hcoreB -= 0.5 * 1j * np.einsum('k,kij->ij', self.B, L_spinor)   # Correct
        hcoreB += H_dia                                                 # Correct 
        hcoreB += 0.5 * g_e/2 * sigmaSB                                         # Correct
    
        return hcoreB


    @property
    def magnetic_field_H_z(self) -> np.ndarray:
        # Building the magnetic field Hamiltonian: 
        nao = self.int_obj.nao

        if self.orig == None:
            self.orig = (0.0, 0.0, 0.0)

        self.int_obj.set_common_origin(self.orig)

        Lz_ao = self.int_obj.intor('int1e_cg_irxp')[2]

        Lz_spinor = np.zeros((2 * nao, 2 * nao), dtype=complex)
        Lz_spinor[:nao, :nao] = Lz_ao
        Lz_spinor[nao:, nao:] = Lz_ao

        rr = self.int_obj.intor('int1e_rr').reshape(3, 3, nao, nao)
        dia_ao = (self.B[2]**2 / 8.0) * (rr[0, 0] + rr[1, 1]) 

        H_dia = np.zeros((2 * nao, 2 * nao), dtype=complex)
        H_dia[:nao, :nao] = dia_ao
        H_dia[nao:, nao:] = dia_ao

        sigmaS = np.zeros((2 * nao, 2 * nao), dtype=complex)
        sigmaS[:nao, :nao] =  self.int_obj.intor("int1e_ovlp")
        sigmaS[nao:, nao:] = -self.int_obj.intor("int1e_ovlp")

        g_e = 2.00231930436256            # Electronic g-factor

        mf = pyscf.scf.GHF(self.int_obj)
        hcoreB = mf.get_hcore().astype(complex)

        hcoreB -= 0.5 * self.B[2] * 1j * Lz_spinor       # Correct for field in the z direction
        hcoreB += H_dia                     # Correct for field in the z direction
        hcoreB += 0.5 * g_e/2 * self.B[2] * sigmaS         # Correct for field in the z direction? Should there be a spin exchange contribution for cGHF?

        return hcoreB
