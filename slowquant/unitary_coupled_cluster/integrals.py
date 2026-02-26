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

    @property
    def c_mo(self) -> np.ndarray:
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
        self._g_ivwi = None
        self._g_pvii_Ci = None
        self._g_piiv_Ci = None
        self._g_iijj_Cij = None
        self._g_ijji_Cij = None
        self._g_iivw_Ci = None
        self._g_ivwi_Ci = None
        self._g_ivwj = None
        self._g_ijvw = None
        self._g_ijkk_Ck = None
        self._g_ijkj_Cj = None
        self._g_ivjw = None
        self._g_ikkj_Ck = None
        self._g_iijv_Ci = None
        self._g_iija_Ci = None
        self._g_ijjv_Cj = None
        self._g_ijja_Cj = None
        self._g_iiva_Ci = None
        self._g_ivai_Ci = None
        self._g_iiab_Ci = None
        self._g_iabi_Ci = None
        self._g_ivwx = None
        self._g_iavw = None
        self._g_ivaw = None
        self._g_vwxa = None
        self._g_vwab = None
        self._g_vawb = None
        self._g_ivwa = None
        self._c_mo = np.copy(mo_coeffs)

    @property
    def h_ii(self) -> np.ndarray:
        if self._h_ii is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._h_ii = np.einsum("Pi,Qi,PQ->i", Ci, Ci, self._h_ao, optimize=['einsum_path', (0, 2), (0, 1)])
        return self._h_ii

    @property
    def h_ij(self) -> np.ndarray:
        if self._h_ij is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._h_ij = np.einsum("Pi,Qj,PQ->ij", Ci, Ci, self._h_ao, optimize=['einsum_path', (0, 2), (0, 1)])
        return self._h_ij

    @property
    def h_ii_Ci(self) -> np.ndarray:
        if self._h_ii_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._h_ii_Ci = np.einsum("Pi,Qi,PQ->", Ci, Ci, self._h_ao, optimize=['einsum_path', (0, 2), (0, 1)])
        return self._h_ii_Ci

    @property
    def h_vw(self) -> np.ndarray:
        if self._h_vw is None:
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._h_vw = np.einsum(
                "Pv,Qw,PQ->vw", Ca, Ca, self._h_ao, optimize=["einsum_path", (0, 2), (0, 1)]
            )
        return self._h_vw

    @property
    def h_iv(self) -> np.ndarray:
        if self._h_iv is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._h_iv = np.einsum(
                "Pi,Qv,PQ->iv", Ci, Ca, self._h_ao, optimize=['einsum_path', (1, 2), (0, 1)]
            )
        return self._h_iv

    @property
    def h_ia(self) -> np.ndarray:
        if self._h_ia is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._h_ia = np.einsum(
                "Pi,Qa,PQ->ia", Ci, Cv, self._h_ao, optimize=["einsum_path", (0, 2), (0, 1)]
            )
        return self._h_ia

    @property
    def h_va(self) -> np.ndarray:
        if self._h_va is None:
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._h_va = np.einsum(
                "Pv,Qa,PQ->va", Ca, Cv, self._h_ao, optimize=["einsum_path", (0, 2), (0, 1)]
            )
        return self._h_va

    @property
    def h_ab(self) -> np.ndarray:
        if self._h_ab is None:
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._h_ab = np.einsum(
                "Pa,Qb,PQ->ab", Cv, Cv, self._h_ao, optimize=["einsum_path", (0, 2), (0, 1)]
            )
        return self._h_ab

    @property
    def g_vwxy(self) -> np.ndarray:
        if self._g_vwxy is None:
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_vwxy = np.einsum(
                "Pv,Qw,Rx,Sy,PQRS->vwxy",
                Ca,
                Ca,
                Ca,
                Ca,
                self._g_ao,
                optimize=["einsum_path", (0, 4), (0, 3), (0, 2), (0, 1)],
            )
        return self._g_vwxy

    @property
    def g_iijj(self) -> np.ndarray:
        if self._g_iijj is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_iijj = np.einsum(
                "Pi,Qi,Rj,Sj,PQRS->ij",
                Ci,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (0, 1), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iijj

    @property
    def g_ijji(self) -> np.ndarray:
        if self._g_ijji is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_ijji = np.einsum(
                "Pi,Qj,Rj,Si,PQRS->ij",
                Ci,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (0, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ijji

    @property
    def g_iivw(self) -> np.ndarray:
        if self._g_iivw is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_iivw = np.einsum(
                "Pi,Qi,Rv,Sw,PQRS->ivw",
                Ci,
                Ci,
                Ca,
                Ca,
                self._g_ao,
                optimize=["einsum_path", (0, 1), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iivw

    @property
    def g_ivwi(self) -> np.ndarray:
        if self._g_ivwi is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_ivwi = np.einsum(
                "Pi,Qv,Rw,Si,PQRS->ivw",
                Ci,
                Ca,
                Ca,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (0, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ivwi

    @property
    def g_iijj_Cij(self) -> np.ndarray:
        if self._g_iijj_Cij is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_iijj_Cij = np.einsum(
                "Pi,Qi,Rj,Sj,PQRS->",
                Ci,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (0, 1), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iijj_Cij

    @property
    def g_ijji_Cij(self) -> np.ndarray:
        if self._g_ijji_Cij is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_ijji_Cij = np.einsum(
                "Pi,Qj,Rj,Si,PQRS->",
                Ci,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (0, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ijji_Cij

    @property
    def g_iivw_Ci(self) -> np.ndarray:
        if self._g_iivw_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_iivw_Ci = np.einsum(
                "Pi,Qi,Rv,Sw,PQRS->vw",
                Ci,
                Ci,
                Ca,
                Ca,
                self._g_ao,
                optimize=["einsum_path", (0, 1), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iivw_Ci

    @property
    def g_ivwi_Ci(self) -> np.ndarray:
        if self._g_ivwi_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_ivwi_Ci = np.einsum(
                "Pi,Qv,Rw,Si,PQRS->vw",
                Ci,
                Ca,
                Ca,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (0, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ivwi_Ci

    @property
    def g_ijvw(self) -> np.ndarray:
        if self._g_ijvw is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_ijvw = np.einsum(
                "Pi,Qj,Rv,Sw,PQRS->ijvw",
                Ci,
                Ci,
                Ca,
                Ca,
                self._g_ao,
                optimize=['einsum_path', (2, 4), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ijvw

    @property
    def g_ijkk_Ck(self) -> np.ndarray:
        if self._g_ijkk_Ck is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_ijkk_Ck = np.einsum(
                "Pi,Qj,Rk,Sk,PQRS->ij",
                Ci,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=['einsum_path', (2, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ijkk_Ck

    @property
    def g_ijkj_Cj(self) -> np.ndarray:
        if self._g_ijkj_Cj is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_ijkj_Cj = np.einsum(
                "Pi,Qj,Rk,Sj,PQRS->ik",
                Ci,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=['einsum_path', (1, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ijkj_Cj

    @property
    def g_ivjw(self) -> np.ndarray:
        if self._g_ivjw is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_ivjw = np.einsum(
                "Pi,Qv,Rj,Sw,PQRS->ivjw",
                Ci,
                Ca,
                Ci,
                Ca,
                self._g_ao,
                optimize=['einsum_path', (1, 4), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ivjw

    @property
    def g_ikkj_Ck(self) -> np.ndarray:
        if self._g_ikkj_Ck is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_ikkj_Ck = np.einsum(
                "Pi,Qk,Rk,Sj,PQRS->ij",
                Ci,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=['einsum_path', (1, 2), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ikkj_Ck

    @property
    def g_iijv_Ci(self) -> np.ndarray:
        if self._g_iijv_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_iijv_Ci = np.einsum(
                "Pi,Qi,Rj,Sv,PQRS->jv",
                Ci,
                Ci,
                Ci,
                Ca,
                self._g_ao,
                optimize=['einsum_path', (0, 1), (2, 3), (1, 2), (0, 1)],
            )
        return self._g_iijv_Ci

    @property
    def g_ijjv_Cj(self) -> np.ndarray:
        if self._g_ijjv_Cj is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_ijjv_Cj = np.einsum(
                "Pi,Qj,Rj,Sv,PQRS->iv",
                Ci,
                Ci,
                Ci,
                Ca,
                self._g_ao,
                optimize=['einsum_path', (1, 2), (2, 3), (1, 2), (0, 1)],
            )
        return self._g_ijjv_Cj

    @property
    def g_iija_Ci(self) -> np.ndarray:
        if self._g_iija_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_iija_Ci = np.einsum(
                "Pi,Qi,Rj,Sa,PQRS->ja",
                Ci,
                Ci,
                Ci,
                Cv,
                self._g_ao,
                optimize=['einsum_path', (0, 1), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iija_Ci

    @property
    def g_ijja_Cj(self) -> np.ndarray:
        if self._g_ijja_Cj is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_ijja_Cj = np.einsum(
                "Pi,Qj,Rj,Sa,PQRS->ia",
                Ci,
                Ci,
                Ci,
                Cv,
                self._g_ao,
                optimize=['einsum_path', (1, 2), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ijja_Cj

    @property
    def g_iiva_Ci(self) -> np.ndarray:
        if self._g_iiva_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_iiva_Ci = np.einsum(
                "Pi,Qi,Rv,Sa,PQRS->va",
                Ci,
                Ci,
                Ca,
                Cv,
                self._g_ao,
                optimize=['einsum_path', (0, 1), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iiva_Ci

    @property
    def g_ivai_Ci(self) -> np.ndarray:
        if self._g_ivai_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_ivai_Ci = np.einsum(
                "Pi,Qv,Ra,Si,PQRS->va",
                Ci,
                Ca,
                Cv,
                Ci,
                self._g_ao,
                optimize=['einsum_path', (0, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_ivai_Ci

    @property
    def g_iiab_Ci(self) -> np.ndarray:
        if self._g_iiab_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_iiab_Ci = np.einsum(
                "Pi,Qi,Ra,Sb,PQRS->ab",
                Ci,
                Ci,
                Cv,
                Cv,
                self._g_ao,
                optimize=['einsum_path', (0, 1), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iiab_Ci

    @property
    def g_iabi_Ci(self) -> np.ndarray:
        if self._g_iabi_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_iabi_Ci = np.einsum(
                "Pi,Qa,Rb,Si,PQRS->ab",
                Ci,
                Cv,
                Cv,
                Ci,
                self._g_ao,
                optimize=['einsum_path', (0, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iabi_Ci

    @property
    def g_ivwx(self) -> np.ndarray:
        if self._g_ivwx is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_ivwx = np.einsum(
                "Pi,Qv,Rw,Sx,PQRS->ivwx",
                Ci,
                Ca,
                Ca,
                Ca,
                self._g_ao,
                optimize=['einsum_path', (1, 4), (1, 3), (1, 2), (0, 1)],
            )
        return self._g_ivwx

    @property
    def g_iavw(self) -> np.ndarray:
        if self._g_iavw is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_iavw = np.einsum(
                "Pi,Qa,Rv,Sw,PQRS->iavw",
                Ci,
                Cv,
                Ca,
                Ca,
                self._g_ao,
                optimize=['einsum_path', (2, 4), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_iavw

    @property
    def g_ivwa(self) -> np.ndarray:
        if self._g_ivwa is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_ivwa = np.einsum(
                "Pi,Qv,Rw,Sa,PQRS->ivwa",
                Ci,
                Ca,
                Ca,
                Cv,
                self._g_ao,
                optimize=['einsum_path', (1, 4), (1, 3), (0, 2), (0, 1)],
            )
        return self._g_ivwa

    @property
    def g_vwxa(self) -> np.ndarray:
        if self._g_vwxa is None:
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_vwxa = np.einsum(
                "Pv,Qw,Rx,Sa,PQRS->vwxa",
                Ca,
                Ca,
                Ca,
                Cv,
                self._g_ao,
                optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)],
            )
        return self._g_vwxa

    @property
    def g_vwab(self) -> np.ndarray:
        if self._g_vwab is None:
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_vwab = np.einsum(
                "Pv,Qw,Ra,Sb,PQRS->vwab",
                Ca,
                Ca,
                Cv,
                Cv,
                self._g_ao,
                optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)],
            )
        return self._g_vwab

    @property
    def g_vawb(self) -> np.ndarray:
        if self._g_vawb is None:
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            Cv = self.c_mo[:, self.num_inactive_orbs + self.num_active_orbs:]
            self._g_vawb = np.einsum(
                "Pv,Qa,Rw,Sb,PQRS->vawb",
                Ca,
                Cv,
                Ca,
                Cv,
                self._g_ao,
                optimize=['einsum_path', (0, 4), (1, 3), (0, 2), (0, 1)],
            )
        return self._g_vawb
