import numpy as np


class Integrals:
    def __init__(
        self,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        c_mo: np.ndarray,
        num_inactive_orbs: int,
        num_active_orbs: int,
        num_virtual_orbs: int,
    ) -> None:
        self.num_inactive_orbs = num_inactive_orbs
        self.num_active_orbs = num_active_orbs
        self.num_virtual_orbs = num_virtual_orbs
        self.num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
        self.c_mo = np.copy(c_mo)
        self._h_ao = h_ao
        self._g_ao = g_ao
        self.opt_paths = {}

    @property
    def c_mo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._c_mo

    @c_mo.setter
    def c_mo(self, mo_coeffs: np.ndarray) -> None:
        self._h_ii = None
        self._h_ij = None
        self._h_vw = None
        self._h_ii_Ci = None
        self._h_iv = None
        self._h_ia = None
        self._h_va = None
        self._h_ab = None
        self._g_vwxy = None
        self._g_iijj = None
        self._g_ijji = None
        self._g_iivw = None
        self._g_iviw = None
        self._g_iijj_Cij = None
        self._g_ijji_Cij = None
        self._g_iivw_Ci = None
        self._g_iviw_Ci = None
        self._g_ivjw = None
        self._g_ijvw = None
        self._g_ijkk_Ck = None
        self._g_ijkj_Cj = None
        self._g_ikkj_Ck = None
        self._g_iijv_Ci = None
        self._g_iija_Ci = None
        self._g_ijjv_Cj = None
        self._g_ijja_Cj = None
        self._g_iiva_Ci = None
        self._g_ivia_Ci = None
        self._g_iiab_Ci = None
        self._g_iaib_Ci = None
        self._g_ivwx = None
        self._g_iavw = None
        self._g_ivaw = None
        self._g_vwxa = None
        self._g_vwab = None
        self._g_vawb = None
        self._g_ivwa = None
        self._c_mo = (
                mo_coeffs[:, : self.num_inactive_orbs],
                mo_coeffs[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs],
                mo_coeffs[:, self.num_inactive_orbs + self.num_active_orbs:],
                np.copy(mo_coeffs),
                )

    def h_PQ(self, name, path, sector1, sector2):
        if name not in self.opt_paths:
            path_info = np.einsum_path(path, self.c_mo[sector1], self.c_mo[sector2], self._h_ao)
            self.opt_paths[name] = path_info[0]
        return np.einsum(path, self.c_mo[sector1], self.c_mo[sector2], self._h_ao, optimize=self.opt_paths[name])

    def g_PQRS(self, name, path, sector1, sector2, sector3, sector4):
        if name not in self.opt_paths:
            path_info = np.einsum_path(path, self.c_mo[sector1], self.c_mo[sector2], self.c_mo[sector3], self.c_mo[sector4], self._g_ao)
            self.opt_paths[name] = path_info[0]
        return np.einsum(path, self.c_mo[sector1], self.c_mo[sector2], self.c_mo[sector3], self.c_mo[sector4], self._g_ao, optimize=self.opt_paths[name])

    @property
    def h_ii(self) -> np.ndarray:
        if self._h_ii is None:
            self._h_ii = self.h_PQ('h_ii', "Pi,Qi,PQ->i", 0, 0)
        return self._h_ii

    @property
    def h_ij(self) -> np.ndarray:
        if self._h_ij is None:
            self._h_ij = self.h_PQ('h_ij', "Pi,Qj,PQ->ij", 0, 0)
        return self._h_ij

    @property
    def h_ii_Ci(self) -> np.ndarray:
        if self._h_ii_Ci is None:
            self._h_ii_Ci = self.h_PQ('h_ii_Ci', "Pi,Qi,PQ->", 0, 0)
        return self._h_ii_Ci

    @property
    def h_vw(self) -> np.ndarray:
        if self._h_vw is None:
            self._h_vw = self.h_PQ('h_vw', "Pv,Qw,PQ->vw", 1, 1)
        return self._h_vw

    @property
    def h_iv(self) -> np.ndarray:
        if self._h_iv is None:
            self._h_iv = self.h_PQ('h_iv', "Pi,Qv,PQ->iv", 0, 1)
        return self._h_iv

    @property
    def h_ia(self) -> np.ndarray:
        if self._h_ia is None:
            self._h_ia = self.h_PQ('h_ia', "Pi,Qa,PQ->ia", 0, 2)
        return self._h_ia

    @property
    def h_va(self) -> np.ndarray:
        if self._h_va is None:
            self._h_va = self.h_PQ('h_va', "Pv,Qa,PQ->va", 1, 2)
        return self._h_va

    @property
    def h_ab(self) -> np.ndarray:
        if self._h_ab is None:
            self._h_ab = self.h_PQ('h_ab', "Pa,Qb,PQ->ab", 2, 2)
        return self._h_ab

    @property
    def g_vwxy(self) -> np.ndarray:
        if self._g_vwxy is None:
            self._g_vwxy = self.g_PQRS('g_vwxy', "Pv,Qw,Rx,Sy,PQRS->vwxy", 1, 1, 1, 1)
        return self._g_vwxy

    @property
    def g_iijj(self) -> np.ndarray:
        if self._g_iijj is None:
            self._g_iijj = self.g_PQRS('g_iijj', "Pi,Qi,Rj,Sj,PQRS->ij", 0, 0, 0, 0)
        return self._g_iijj

    @property
    def g_ijji(self) -> np.ndarray:
        if self._g_ijji is None:
            self._g_ijji = self.g_PQRS('g_ijji', "Pi,Qj,Rj,Si,PQRS->ij", 0, 0, 0, 0)
        return self._g_ijji

    @property
    def g_iivw(self) -> np.ndarray:
        if self._g_iivw is None:
            self._g_iivw = self.g_PQRS('g_iivw', "Pi,Qi,Rv,Sw,PQRS->ivw", 0, 0, 1, 1)
        return self._g_iivw

    @property
    def g_iviw(self) -> np.ndarray:
        if self._g_iviw is None:
            self._g_iviw = self.g_PQRS('g_iviw', "Pi,Qv,Ri,Sw,PQRS->ivw", 0, 1, 0, 1)
        return self._g_iviw

    @property
    def g_iijj_Cij(self) -> np.ndarray:
        if self._g_iijj_Cij is None:
            self._g_iijj_Cij = self.g_PQRS('g_iijj_Cij', 'Pi,Qi,Rj,Sj,PQRS->', 0, 0, 0, 0)
        return self._g_iijj_Cij

    @property
    def g_ijji_Cij(self) -> np.ndarray:
        if self._g_ijji_Cij is None:
            self._g_ijji_Cij = self.g_PQRS('g_ijji_Cij', "Pi,Qj,Rj,Si,PQRS->", 0, 0, 0, 0)
        return self._g_ijji_Cij

    @property
    def g_iivw_Ci(self) -> np.ndarray:
        if self._g_iivw_Ci is None:
            self._g_iivw_Ci = self.g_PQRS('g_iivw_Ci', "Pi,Qi,Rv,Sw,PQRS->vw", 0, 0, 1, 1)
        return self._g_iivw_Ci

    @property
    def g_iviw_Ci(self) -> np.ndarray:
        if self._g_iviw_Ci is None:
            self._g_iviw_Ci = self.g_PQRS('g_iviw_Ci', "Pi,Qv,Ri,Sw,PQRS->vw", 0, 1, 0, 1)
        return self._g_iviw_Ci

    @property
    def g_ijvw(self) -> np.ndarray:
        if self._g_ijvw is None:
            self._g_ijvw = self.g_PQRS('g_ijvw', "Pi,Qj,Rv,Sw,PQRS->ijvw", 0, 0, 1, 1)
        return self._g_ijvw

    @property
    def g_ijkk_Ck(self) -> np.ndarray:
        if self._g_ijkk_Ck is None:
            self._g_ijkk_Ck = self.g_PQRS('g_ijkk_Ck', "Pi,Qj,Rk,Sk,PQRS->ij", 0, 0, 0, 0)
        return self._g_ijkk_Ck

    @property
    def g_ijkj_Cj(self) -> np.ndarray:
        if self._g_ijkj_Cj is None:
            self._g_ijkj_Cj = self.g_PQRS('g_ijkj_Cj', "Pi,Qj,Rk,Sj,PQRS->ik", 0, 0, 0, 0)
        return self._g_ijkj_Cj

    @property
    def g_ivjw(self) -> np.ndarray:
        if self._g_ivjw is None:
            self._g_ivjw = self.g_PQRS('g_ivjw', "Pi,Qv,Rj,Sw,PQRS->ivjw", 0, 1, 0, 1)
        return self._g_ivjw

    @property
    def g_ikkj_Ck(self) -> np.ndarray:
        if self._g_ikkj_Ck is None:
            self._g_ikkj_Ck = self.g_PQRS('g_ikkj_Ck', "Pi,Qk,Rk,Sj,PQRS->ij", 0, 0, 0, 0)
        return self._g_ikkj_Ck

    @property
    def g_iijv_Ci(self) -> np.ndarray:
        if self._g_iijv_Ci is None:
            self._g_iijv_Ci = self.g_PQRS('g_iijv_Ci', "Pi,Qi,Rj,Sv,PQRS->jv", 0, 0, 0, 1)
        return self._g_iijv_Ci

    @property
    def g_ijjv_Cj(self) -> np.ndarray:
        if self._g_ijjv_Cj is None:
            self._g_ijjv_Cj = self.g_PQRS('g_ijjv_Cj', "Pi,Qj,Rj,Sv,PQRS->iv", 0, 0, 0, 1)
        return self._g_ijjv_Cj

    @property
    def g_iija_Ci(self) -> np.ndarray:
        if self._g_iija_Ci is None:
            self._g_iija_Ci = self.g_PQRS('g_iija_Ci', "Pi,Qi,Rj,Sa,PQRS->ja", 0, 0, 0, 2)
        return self._g_iija_Ci

    @property
    def g_ijja_Cj(self) -> np.ndarray:
        if self._g_ijja_Cj is None:
            self._g_ijja_Cj = self.g_PQRS('g_ijja_Cj', "Pi,Qj,Rj,Sa,PQRS->ia", 0, 0, 0, 2)
        return self._g_ijja_Cj

    @property
    def g_iiva_Ci(self) -> np.ndarray:
        if self._g_iiva_Ci is None:
            self._g_iiva_Ci = self.g_PQRS('g_iiva_Ci', "Pi,Qi,Rv,Sa,PQRS->va", 0, 0, 1, 2)
        return self._g_iiva_Ci

    @property
    def g_ivia_Ci(self) -> np.ndarray:
        if self._g_ivia_Ci is None:
            self._g_ivia_Ci = self.g_PQRS('g_ivia_Ci', "Pi,Qv,Ri,Sa,PQRS->va", 0, 1, 0, 2)
        return self._g_ivia_Ci

    @property
    def g_iiab_Ci(self) -> np.ndarray:
        if self._g_iiab_Ci is None:
            self._g_iiab_Ci = self.g_PQRS('g_iiab_Ci', "Pi,Qi,Ra,Sb,PQRS->ab", 0, 0, 2, 2)
        return self._g_iiab_Ci

    @property
    def g_iaib_Ci(self) -> np.ndarray:
        if self._g_iaib_Ci is None:
            self._g_iaib_Ci = self.g_PQRS('g_iaib_Ci', "Pi,Qa,Ri,Sb,PQRS->ab", 0, 2, 0, 2)
        return self._g_iaib_Ci

    @property
    def g_ivwx(self) -> np.ndarray:
        if self._g_ivwx is None:
            self._g_ivwx = self.g_PQRS('g_ivwx', "Pi,Qv,Rw,Sx,PQRS->ivwx", 0, 1, 1, 1)
        return self._g_ivwx

    @property
    def g_iavw(self) -> np.ndarray:
        if self._g_iavw is None:
            self._g_iavw = self.g_PQRS('g_iavw', "Pi,Qa,Rv,Sw,PQRS->iavw", 0, 2, 1, 1)
        return self._g_iavw

    @property
    def g_ivwa(self) -> np.ndarray:
        if self._g_ivwa is None:
            self._g_ivwa = self.g_PQRS('g_ivwa', "Pi,Qv,Rw,Sa,PQRS->ivwa", 0, 1, 1, 2)
        return self._g_ivwa

    @property
    def g_vwxa(self) -> np.ndarray:
        if self._g_vwxa is None:
            self._g_vwxa = self.g_PQRS('g_vwxa', "Pv,Qw,Rx,Sa,PQRS->vwxa", 1, 1, 1, 2)
        return self._g_vwxa

    @property
    def g_vwab(self) -> np.ndarray:
        if self._g_vwab is None:
            self._g_vwab = self.g_PQRS('g_vwab', "Pv,Qw,Ra,Sb,PQRS->vwab", 1, 1, 2, 2)
        return self._g_vwab

    @property
    def g_vawb(self) -> np.ndarray:
        if self._g_vawb is None:
            self._g_vawb = self.g_PQRS('g_vawb', "Pv,Qa,Rw,Sb,PQRS->vawb", 1, 2, 1, 2)
        return self._g_vawb

    def build_fock_matrix_integrals(self):
        g_pQiS = np.einsum('Pp,Ri,PQRS->pQiS', self.c_mo[3], self.c_mo[0])
