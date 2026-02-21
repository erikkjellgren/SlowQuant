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
        self._c_mo = np.copy(c_mo)
        self._h_ao = h_ao
        self._g_ao = g_ao
        self._h_pv = None
        self._h_pi = None
        self._h_ii = None
        self._h_vw = None
        self._h_ii_Ci = None
        self._g_pijj = None
        self._g_piij = None
        self._g_pivw = None
        self._g_pvwi = None
        self._g_pvii = None
        self._g_piiv = None
        self._g_vwxy = None
        self._g_pvwx = None
        self._g_iijj = None
        self._g_ijji = None
        self._g_iivw = None
        self._g_vwii = None
        self._g_viiw = None
        self._g_ivwi = None
        self._g_pvii_Ci = None
        self._g_piiv_Ci = None
        self._g_iijj_Cij = None
        self._g_ijji_Cij = None
        self._g_iivw_Ci = None
        self._g_ivwi_Ci = None

    @property
    def c_mo(self) -> np.ndarray:
        return self._c_mo

    @c_mo.setter
    def c_mo(self, mo_coeffs: np.ndarray) -> None:
        self._h_pv = None
        self._h_pi = None
        self._h_ii = None
        self._h_vw = None
        self._h_ii_Ci = None
        self._g_pijj = None
        self._g_piij = None
        self._g_pivw = None
        self._g_pvwi = None
        self._g_pvii = None
        self._g_piiv = None
        self._g_vwxy = None
        self._g_pvwx = None
        self._g_iijj = None
        self._g_ijji = None
        self._g_iivw = None
        self._g_vwii = None
        self._g_viiw = None
        self._g_ivwi = None
        self._g_pvii_Ci = None
        self._g_piiv_Ci = None
        self._g_iijj_Cij = None
        self._g_ijji_Cij = None
        self._g_iivw_Ci = None
        self._g_ivwi_Ci = None
        self._c_mo = np.copy(mo_coeffs)

    @property
    def h_pv(self) -> np.ndarray:
        if self._h_pv is None:
            Cg = self.c_mo[:, :]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._h_pv = np.einsum(
                "Pp,Qv,PQ->pv", Cg, Ca, self._h_ao, optimize=["einsum_path", (0, 2), (0, 1)]
            )
        return self._h_pv

    @property
    def h_pi(self) -> np.ndarray:
        if self._h_pi is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._h_pi = np.einsum(
                "Pp,Qi,PQ->pi", Cg, Ci, self._h_ao, optimize=["einsum_path", (0, 2), (0, 1)]
            )
        return self._h_pi

    @property
    def h_ii(self) -> np.ndarray:
        if self._h_ii is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._h_ii = np.einsum("Pi,Qi,PQ->i", Ci, Ci, self._h_ao)
        return self._h_ii

    @property
    def h_ii_Ci(self) -> np.ndarray:
        if self._h_ii_Ci is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._h_ii_Ci = np.einsum("Pi,Qi,PQ->", Ci, Ci, self._h_ao)
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
    def g_pijj(self) -> np.ndarray:
        if self._g_pijj is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_pijj = np.einsum(
                "Pp,Qi,Rj,Sj,PQRS->pij",
                Cg,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (2, 3), (2, 3), (1, 2), (0, 1)],
            )
        return self._g_pijj

    @property
    def g_piij(self) -> np.ndarray:
        if self._g_piij is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            self._g_piij = np.einsum(
                "Pp,Qi,Ri,Sj,PQRS->pij",
                Cg,
                Ci,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (1, 2), (2, 3), (1, 2), (0, 1)],
            )
        return self._g_piij

    @property
    def g_pivw(self) -> np.ndarray:
        if self._g_pivw is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_pivw = np.einsum(
                "Pp,Qi,Rv,Sw,PQRS->pivw",
                Cg,
                Ci,
                Ca,
                Ca,
                self._g_ao,
                optimize=["einsum_path", (1, 4), (1, 3), (1, 2), (0, 1)],
            )
        return self._g_pivw

    @property
    def g_pvwi(self) -> np.ndarray:
        if self._g_pvwi is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_pvwi = np.einsum(
                "Pp,Qv,Rw,Si,PQRS->pvwi",
                Cg,
                Ca,
                Ca,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (3, 4), (1, 3), (1, 2), (0, 1)],
            )
        return self._g_pvwi

    @property
    def g_pvii(self) -> np.ndarray:
        if self._g_pvii is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_pvii = np.einsum(
                "Pp,Qv,Ri,Si,PQRS->pvi",
                Cg,
                Ca,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (2, 3), (2, 3), (1, 2), (0, 1)],
            )
        return self._g_pvii

    @property
    def g_piiv(self) -> np.ndarray:
        if self._g_piiv is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_piiv = np.einsum(
                "Pp,Qi,Ri,Sv,PQRS->piv",
                Cg,
                Ci,
                Ci,
                Ca,
                self._g_ao,
                optimize=["einsum_path", (1, 2), (2, 3), (1, 2), (0, 1)],
            )
        return self._g_piiv

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
    def g_pvwx(self) -> np.ndarray:
        if self._g_pvwx is None:
            Cg = self.c_mo[:, :]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_pvwx = np.einsum(
                "Pp,Qv,Rw,Sx,PQRS->pvwx",
                Cg,
                Ca,
                Ca,
                Ca,
                self._g_ao,
                optimize=["einsum_path", (1, 4), (1, 3), (1, 2), (0, 1)],
            )
        return self._g_pvwx

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
    def g_vwii(self) -> np.ndarray:
        if self._g_vwii is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_vwii = np.einsum(
                "Pv,Qw,Ri,Si,PQRS->vwi",
                Ca,
                Ca,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (2, 3), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_vwii

    @property
    def g_viiw(self) -> np.ndarray:
        if self._g_viiw is None:
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_viiw = np.einsum(
                "Pv,Qi,Ri,Sw,PQRS->viw",
                Ca,
                Ci,
                Ci,
                Ca,
                self._g_ao,
                optimize=["einsum_path", (1, 2), (2, 3), (0, 2), (0, 1)],
            )
        return self._g_viiw

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
    def g_pvii_Ci(self) -> np.ndarray:
        if self._g_pvii_Ci is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_pvii_Ci = np.einsum(
                "Pp,Qv,Ri,Si,PQRS->pv",
                Cg,
                Ca,
                Ci,
                Ci,
                self._g_ao,
                optimize=["einsum_path", (2, 3), (2, 3), (1, 2), (0, 1)],
            )
        return self._g_pvii_Ci

    @property
    def g_piiv_Ci(self) -> np.ndarray:
        if self._g_piiv_Ci is None:
            Cg = self.c_mo[:, :]
            Ci = self.c_mo[:, : self.num_inactive_orbs]
            Ca = self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs]
            self._g_piiv_Ci = np.einsum(
                "Pp,Qi,Ri,Sv,PQRS->pv",
                Cg,
                Ci,
                Ci,
                Ca,
                self._g_ao,
                optimize=["einsum_path", (1, 2), (2, 3), (1, 2), (0, 1)],
            )
        return self._g_piiv_Ci

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
