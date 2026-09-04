import copy

import numpy as np
import pyscf

from slowquant.SlowQuant import SlowQuant
from pyscf import lib
c = lib.param.LIGHT_SPEED
from pyscf.prop.ssc.dhf import sa01sa01_integral


class IntegralManager:
    __slots__ = (
        "_electric_dipole",
        "_electron_electron_repulsion",
        "_h_ao",
        "_kinetic_energy",
        "_nuclear_electron_attraction",
        "_overlap",
        "int_obj",
        "_h_B",
        "_h_B_RMB_GIAO",
        "_h_m",
        "_h_Bm_RMB_GIAO",
        "_S_B",
        "_g_B",
        "_S_m",
        "_h_m_RMB",
        "_h_mm_RMB",
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
        self._overlap: np.ndarray | None = None
        self._electric_dipole: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._h_ao: np.ndarray | None = None
        self._h_B: np.ndarray | None = None
        self._h_B_RMB_GIAO: np.ndarray | None = None
        self._h_m: np.ndarray | None = None
        self._h_Bm_RMB_GIAO: np.ndarray | None = None
        self._S_B: np.ndarray | None = None
        self._g_B: np.ndarray | None = None
        self._S_m: np.ndarray | None = None
        self._h_mm_RMB: np.ndarray | None = None
        self._h_m_RMB: np.ndarray | None = None

    @property
    def num_elec(self) -> int:
        """Number of electrons."""
        if isinstance(self.int_obj, SlowQuant):
            return self.int_obj.molecule.number_electrons
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            return self.int_obj.nelectron
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")

    # Not for 4-component!
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
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._kinetic_energy = kin_int
        return kin_int

    # Not for 4-component!
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
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
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
            e2_int = np.array([self.int_obj.intor("int2e_spinor"), self.int_obj.intor('int2e_spsp1spsp2_spinor')*(0.0625/c**4),
                      self.int_obj.intor('int2e_spsp2_spinor')*(0.25/c**2), self.int_obj.intor('int2e_spsp1_spinor')*(0.25/c**2)],dtype=np.complex128)

        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
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
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")

    # NOT MADE for 4-component yet!!
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
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
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
            mf = pyscf.scf.DHF(self.int_obj)
            h_core = mf.get_hcore()
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
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._overlap = overlap_int
        return overlap_int

    @property
    def h_B(self) -> np.ndarray:
        """h_B integral in AO."""
        if isinstance(self._h_B, np.ndarray):
            return self._h_B
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            n2c = self.int_obj.nao_2c()
            n4c = 2 * n2c

            h_B = np.zeros((3, n4c, n4c), dtype=complex)

            t1 = self.int_obj.intor('int1e_cg_sa10sp_spinor', 3)

            for b in range(3):
                h_B[b, :n2c, n2c:] =   0.5*t1[b]
                h_B[b, n2c:, :n2c] =   0.5*t1[b].conj().T
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._h_B = h_B
        return h_B

    @property
    def h_m(self) -> np.ndarray:
        """h_m integral in AO."""
        if isinstance(self._h_m, np.ndarray):
            return self._h_m
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            n2c = self.int_obj.nao_2c()
            n4c = n2c * 2
            natm = self.int_obj.natm

            h_m = np.zeros((natm, 3, n4c, n4c), dtype=complex)

            for I in range(natm):
                self.int_obj.set_rinv_origin(self.int_obj.atom_coord(I))
                t01 = self.int_obj.intor('int1e_sa01sp_spinor', 3)  #TRUE

                for m in range(3):
                    h_m[I, m, :n2c, n2c:] = 0.5 * t01[m]
                    h_m[I, m, n2c:, :n2c] = 0.5 * t01[m].conj().T
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._h_m = h_m
        return h_m

    @property
    def h_Bm_RMB_GIAO(self) -> np.ndarray:
        """h_Bm RMB GIAO integral in AO."""
        if isinstance(self._h_Bm_RMB_GIAO, np.ndarray):
            return self._h_Bm_RMB_GIAO
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            n2c = self.int_obj.nao_2c()
            n4c = n2c * 2
            natm = self.int_obj.natm

            h_Bm = np.zeros((natm, 3, 3, n4c, n4c), dtype=np.complex128)

            for I in range(natm):
                self.int_obj.set_rinv_origin(self.int_obj.atom_coord(I))
                t11 = self.int_obj.intor('int1e_giao_sa10sa01_spinor', 9).reshape(3,3,n2c,n2c)
                t11 += self.int_obj.intor('int1e_spgsa01_spinor', 9).reshape(3,3,n2c,n2c)
                for x in range(3):
                    for y in range(3):
                        # LS
                        h_Bm[I,x,y,:n2c,n2c:] = t11[x,y].conj().T * .5
                        # SL
                        h_Bm[I,x,y,n2c:,:n2c] = t11[x,y] * .5
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._h_Bm_RMB_GIAO = h_Bm
        return h_Bm

    @property
    def h_B_RMB_GIAO(self) -> np.ndarray:
        """h_B RMB GIAO integral in AO."""
        if isinstance(self._h_B_RMB_GIAO, np.ndarray):
            return self._h_B_RMB_GIAO
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            # LL 
            vg = self.int_obj.intor('int1e_gnuc_spinor', 3)

            # SS
            wg = self.int_obj.intor('int1e_spgnucsp_spinor', 3)
            v1 = self.int_obj.intor('int1e_giao_sa10nucsp_spinor', 3)

            # LS / SL and SS
            tg = self.int_obj.intor('int1e_spgsp_spinor', 3) # LS, SL, SS
            t1 = self.int_obj.intor('int1e_giao_sa10sp_spinor', 3) # LS, SL, SS

            t1cc = []

            # Complex conjugate transpose sum:
            for i in range(3):
                t1cc.append(t1[i] + t1[i].conj().T)

            # Construct integrals:
            n2c = self.int_obj.nao_2c()
            n4c = 2 * n2c

            h_B = np.zeros((3, n4c, n4c), dtype=complex)

            for i in range(3):
                # LL
                h_B[i, :n2c, :n2c] += vg[i]
                # LS
                h_B[i, :n2c, n2c:] =  tg[i].conj().T * .5 + t1cc[i] * .5
                # SL
                h_B[i, n2c:, :n2c] =  tg[i] * .5 + t1cc[i] * .5 
                # SS
                h_B[i, n2c:, n2c:] += wg[i]*(.25/c**2) - tg[i]*.5 - t1cc[i] * .5 + (v1[i]+v1[i].conj().T) * (.25/c**2)
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._h_B_RMB_GIAO = h_B
        return h_B

    @property
    def S_B(self) -> np.ndarray:
        """S_B integral in AO."""
        if isinstance(self._S_B, np.ndarray):
            return self._S_B
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            # LL
            ll =self.int_obj.intor('int1e_govlp_spinor', 3)

            # SS
            ss_1 = self.int_obj.intor('int1e_spgsp_spinor', 3)
            ss_2 = self.int_obj.intor('int1e_giao_sa10sp_spinor', 3)

            # Complec conjugation transpose combination:
            ss_2_c = []
            for i in range(3):
                ss_2_c.append(ss_2[i] + ss_2[i].conj().T)

            # Construct integrals:
            n2c = self.int_obj.nao_2c()
            n4c = 2 * n2c

            S_B = np.zeros((3, n4c, n4c), dtype=complex)

            for i in range(3):
                # LL
                S_B[i, :n2c,:n2c] = ll[i]
                # SS
                S_B[i, n2c:,n2c:] = 1/(4*c**2) * (ss_1[i] + ss_2_c[i])
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._S_B = S_B
        return S_B

    @property
    def g_B(self) -> np.ndarray:
        """g_B integral in AO."""
        if isinstance(self._g_B, np.ndarray):
            return self._g_B
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            # Integrals:
            # LLLL
            ll1 = self.int_obj.intor('int2e_g1_spinor', 3)

            LLLL = []
            for i in range(3):
                LLLL.append(ll1[i]
                    + np.einsum("abcd->cdab", ll1[i])
                    )

            # SSSS
            ss1 = self.int_obj.intor('int2e_spgsp1spsp2_spinor', 3) * .0625 / c**4
            ss2 = self.int_obj.intor('int2e_giao_sa10sp1spsp2_spinor', 3) * .0625 / c**4

            SSSS =[]
            for i in range(3):
                SSSS.append(ss1[i]           
                    + np.einsum('abcd->cdab', ss1[i])

                    + ss2[i]
                    + np.einsum('abcd->badc', ss2[i].conj()) #'abcd->bacd'
                    + np.einsum('abcd->cdab', ss2[i])
                    + np.einsum('abcd->dcba', ss2[i].conj()) #'abcd->dcab'
                )

            #np.einsum('abcd,cdab,abcd,bacd,cdab,dcab->abcd', ss1[i], ss1[i], ss2[i], ss2[i], ss2[i], ss2[i])

            # LLSS and SSLL
            ls1 = self.int_obj.intor('int2e_spgsp1_spinor', 3) * .25 / c**2
            ls2 = self.int_obj.intor('int2e_g1spsp2_spinor', 3) * .25 / c**2
            ls3 = self.int_obj.intor('int2e_giao_sa10sp1_spinor', 3) * .25 / c**2

            LLSS = []
            SSLL = []

            for i in range(3):
                SSLL.append(ls1[i]
                            
                            + np.einsum('abcd->cdab', ls2[i])

                            + ls3[i]
                            + np.einsum('abcd->badc', ls3[i].conj())  #'abcd->bacd'
                            )

            #SLSL.append(np.einsum("abcd,cdab,abcd,bacd->abcd", ls1[i], ls2[i], ls3[i], ls3[i]))

                LLSS.append(np.einsum("abcd->cdab", SSLL[i]))


            # Saving all integrals
            g_B = []

            for i in range(3):
                tmp = np.array([LLLL[i], SSSS[i], LLSS[i], SSLL[i]])
                g_B.append(tmp)
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._g_B = g_B
        return g_B

    @property
    def S_m(self) -> np.ndarray:
        """S_B integral in AO."""
        if isinstance(self._S_m, np.ndarray):
            return self._S_m
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            natm = self.int_obj.natm
            n2c = self.int_obj.nao_2c()
            n4c = n2c * 2
            S_m = np.zeros((natm, 3, n4c, n4c), dtype=complex)

            for I in range(natm):
                self.int_obj.set_rinv_origin(self.int_obj.atom_coord(I))
                a01int = self.int_obj.intor('int1e_sa01sp_spinor', 3)

                tm = a01int[I] + a01int[I].conj().T

                S_m[I][n2c:, n2c:] = tm * (.25/c**2)  # sign? and complex conjugate contribution? factors?

        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._S_m = S_m
        return S_m

    @property
    def h_m_RMB(self) -> np.ndarray:
        """h_m integral in AO."""
        if isinstance(self._h_m_RMB, np.ndarray):
            return self._h_m_RMB
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            n2c = self.int_obj.nao_2c()
            n4c = n2c * 2
            natm = self.int_obj.natm

            h_m = np.zeros((natm, 3, n4c, n4c), dtype=complex)

            for I in range(natm):
                self.int_obj.set_rinv_origin(self.int_obj.atom_coord(I))
                t01 = self.int_obj.intor('int1e_sa01sp_spinor', 3)  #TRUE

                for m in range(3):
                    h_m[I, m, :n2c, n2c:] = 0.5 * t01[m]
                    h_m[I, m, n2c:, :n2c] = 0.5 * t01[m].conj().T
        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._h_m_RMB = h_m
        return h_m

    @property
    def h_mm_RMB(self) -> np.ndarray:
        """h_Bm RMB GIAO integral in AO."""
        if isinstance(self._h_mm_RMB, np.ndarray):
            return self._h_mm_RMB
        elif isinstance(self.int_obj, pyscf.gto.mole.Mole):
            n2c = self.int_obj.nao_2c()
            n4c = n2c * 2
            natm = self.int_obj.natm

            h_mm = np.zeros((natm, natm, 3, 3, n4c, n4c), dtype=np.complex128)

            for I in range(natm):
                for J in range(I, natm):
                    orig1 = self.int_obj.atom_coord(I)
                    orig2 = self.int_obj.atom_coord(J)
                    a01a01 = sa01sa01_integral(self.int_obj, orig1, orig2)  # (3, 3, n2c, n2c)

                    block = np.zeros((3, 3, n4c, n4c), dtype=np.complex128)
                    block[:, :, n2c:, :n2c] =  0.5 * a01a01
                    block[:, :, :n2c, n2c:] =  0.5 * a01a01.conj().transpose(0, 1, 3, 2)

                    h_mm[I, J] = block
                    h_mm[J, I] = block  # symmetric

        else:
            raise ValueError(f"Got unknown integral object, {type(self.int_obj)}")
        self._h_mm_RMB = h_mm
        return h_mm


    