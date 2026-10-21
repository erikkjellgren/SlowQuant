"""Molekylære egenskaber for generaliserede (spinor) bølgefunktioner i SlowQuant.

Samler operator-bygning (NR / X2C, med/uden picture change, spin-free / spin-orbit)
og beregning af egenskaber ét sted.

Eksempel:
    props = GeneralizedProperties(WF, mol, mf, x2c=True, picture_change=True, spin_free=False)
    props.dipole_moment()
    props.electric_field_gradient()
    props.excitation_energies()
    props.oscillator_strengths()
    props.polarizability()
"""

import numpy as np
import scipy.linalg
from scipy.linalg import solve

from slowquant.molecularintegrals.integralfunctions import generalized_one_electron_transform
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import (
    generalized_expectation_value,
)
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_one_elec_op_0i_0a
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive


# --------------------------------------------------------------------------- #
# Hjælpefunktioner
# --------------------------------------------------------------------------- #
def _block_diag(mat: np.ndarray) -> np.ndarray:
    """Skalar AO-matrix -> spinor-basis (alpha/beta blokdiagonal)."""
    return scipy.linalg.block_diag(mat, mat)


def _sigma_dot(prp4: np.ndarray) -> np.ndarray:
    """Sætter (qx, qy, qz, q0) sammen til en 2x2 Pauli-blokmatrix."""
    qx, qy, qz, q0 = prp4
    return np.block(
        [
            [q0 + 1j * qz, qy + 1j * qx],
            [-qy + 1j * qx, q0 - 1j * qz],
        ]
    )


def _symmetrize_efg(ao_efg: np.ndarray, dim: int) -> np.ndarray:
    """Hermitisk i AO-indeks og symmetrisk i (alpha, beta)."""
    ao_efg = 0.5 * (ao_efg + ao_efg.conj().transpose(0, 2, 1))
    ao_efg = ao_efg.reshape(3, 3, dim, dim)
    return 0.5 * (ao_efg + ao_efg.transpose(1, 0, 2, 3))


# --------------------------------------------------------------------------- #
# Property-klassen
# --------------------------------------------------------------------------- #
class GeneralizedProperties:
    """Properties for the generalized wave function"""

    def __init__(
        self,
        wf,
        mol,
        mf=None,
        *,
        x2c: bool = False,
        picture_change: bool = False,
        spin_free: bool = False,
        c: float = 137.036,
        verbose: bool = True,
    ):
        if x2c and picture_change and mf is None:
            raise ValueError("mf (med with_x2c) skal gives når x2c=True og picture_change=True.")
        self.wf = wf
        self.mol = mol
        self.mf = mf
        self.x2c = x2c
        self.picture_change = picture_change
        self.spin_free = spin_free
        self.c = c
        self.verbose = verbose
        self._lr = None
        self._dip_ao = None

    # ------------------------------------------------------------------ #
    # Generelle hjælpere
    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _is_spinor(self, ao: np.ndarray) -> bool:
        """True hvis integralet allerede er i spinor-basis (2*nao x 2*nao).

        Afgøres ud fra formen, så x2c-flaget til generalized_one_electron_transform
        altid passer til integralet (NR og spin-free giver nao x nao).
        """
        return ao.shape[-1] == 2 * self.mol.nao

    def ao_to_operator(self, ao: np.ndarray):
        """AO-integral -> MO -> andenkvantiseret operator i det aktive rum."""
        mo = generalized_one_electron_transform(self.wf.c_mo, ao, x2c=self._is_spinor(ao))
        return generalized_one_elec_op_0i_0a(
            mo, self.wf.num_inactive_spin_orbs, self.wf.num_active_spin_orbs
        )

    def expectation_value(self, ao: np.ndarray, real: bool = True, tol: float = 1e-8):
        """<0|O|0> for en én-elektron operator givet i AO-basis."""
        op = self.ao_to_operator(ao)
        val = generalized_expectation_value(
            self.wf.ci_coeffs, [op], self.wf.ci_coeffs, self.wf.ci_info
        )
        if real:
            if abs(np.imag(val)) > tol:
                self._log(f"Advarsel: imaginær del af forventningsværdi = {np.imag(val):.2e}")
            return float(np.real(val))
        return val

    # ------------------------------------------------------------------ #
    # Operator-bygning
    # ------------------------------------------------------------------ #
    def dipole_integrals(self) -> np.ndarray:
        """Dipol-integraler, shape (3, n, n). Caches."""
        if self._dip_ao is None:
            self._dip_ao = self.build_one_electron_operator("int1e_r", "int1e_sprsp")
        return self._dip_ao

    def build_one_electron_operator(self, int_LL: str, int_SS: str) -> np.ndarray:
        """Generel (3-komponent) én-elektron operator, fx dipol."""
        mol, c1 = self.mol, 0.5 / self.c

        if not self.x2c:
            self._log("Operator: NR")
            return mol.intor_symmetric(int_LL)

        if not self.picture_change:
            self._log("Operator: X2C uden picture change")
            r = mol.intor_symmetric(int_LL)
            return np.array([_block_diag(x) for x in r])

        xmol = self.mf.with_x2c.get_xmol()[0]
        nao = xmol.nao
        r = xmol.intor_symmetric(int_LL)
        sprsp = xmol.intor_symmetric(int_SS).reshape(3, 4, nao, nao)

        if self.spin_free:
            self._log("Operator: X2C picture change (spin-free)")
            return self.mf.with_x2c.picture_change((r, sprsp[:, 3] * c1**2))

        self._log("Operator: X2C picture change (spin-orbit)")
        r_so = np.array([_block_diag(x) for x in r])
        sprsp_so = np.array([_sigma_dot(x * c1**2) for x in sprsp])
        return self.mf.with_x2c.picture_change((r_so, sprsp_so))

    def efg_integrals(self, atom_idx: int) -> np.ndarray:
        """EFG-integraler ved kerne atom_idx, shape (3, 3, n, n)."""
        mol, c1 = self.mol, 0.5 / self.c
        origin = mol.atom_coord(atom_idx)
        nao_c = mol.nao

        def _ll(m):
            return (
                m.intor("int1e_ipiprinv")
                + m.intor("int1e_ipiprinv").transpose(0, 2, 1)
                + 2 * m.intor("int1e_iprinvip")
            )  # (9, nao, nao)

        if not self.x2c:
            self._log("EFG: NR")
            with mol.with_rinv_origin(origin):
                return _ll(mol).reshape(3, 3, nao_c, nao_c)

        if not self.picture_change:
            self._log("EFG: X2C uden picture change")
            with mol.with_rinv_origin(origin):
                efg_ao = _ll(mol)
            ao_efg = np.array([_block_diag(x) for x in efg_ao])
            return _symmetrize_efg(ao_efg, 2 * nao_c)

        xmol = self.mf.with_x2c.get_xmol()[0]
        nao_x = xmol.nao
        with xmol.with_rinv_origin(origin):  # midlertidig, ændrer ikke xmol permanent
            efg_ao = _ll(xmol)
            if self.spin_free:
                ss = (
                    xmol.intor("int1e_ipipsprinvsp")
                    + xmol.intor("int1e_ipipsprinvsp").transpose(0, 2, 1)
                    + 2 * xmol.intor("int1e_ipsprinvspip")
                ).reshape(9, 4, nao_x, nao_x)
            else:
                ss = (
                    xmol.intor("int1e_ipipsprinvsp") + xmol.intor("int1e_ipsprinvspip")
                ).reshape(9, 4, nao_x, nao_x)

        if self.spin_free:
            self._log("EFG: X2C picture change (spin-free)")
            ao_efg = self.mf.with_x2c.picture_change((efg_ao, ss[:, 3] * c1**2))
            return _symmetrize_efg(ao_efg, nao_c)

        self._log("EFG: X2C picture change (spin-orbit)")
        f2_LL = np.array([_block_diag(x) for x in efg_ao])
        f2_SS = []
        for k in range(9):
            A = _sigma_dot(ss[k])
            f2_SS.append((A + A.conj().T) * c1**2)  # A + A† efter _sigma_dot
        ao_efg = self.mf.with_x2c.picture_change((f2_LL, np.array(f2_SS)))

        if self.verbose:
            dim = 2 * nao_c
            op = ao_efg.reshape(3, 3, dim, dim)
            op = 0.5 * (op + op.transpose(1, 0, 2, 3))
            anti = 0.5 * (op - op.conj().transpose(0, 1, 3, 2))
            print("Anti-hermitisk del (før hermitisering):", np.abs(anti).max())

        return _symmetrize_efg(ao_efg, 2 * nao_c)

    # ------------------------------------------------------------------ #
    # Grundtilstandsegenskaber
    # ------------------------------------------------------------------ #
    def dipole_moment(self) -> dict:
        """Elektronisk, nuklear og total dipol (a.u.)."""
        dip_ao = self.dipole_integrals()
        elec = np.array([self.expectation_value(dip_ao[i]) for i in range(3)])
        nuc = np.einsum("i,ij->j", self.mol.atom_charges(), self.mol.atom_coords())
        total = -elec + nuc
        self._log(
            "Dipolmoment (a.u.):  x: {:.6f}  y: {:.6f}  z: {:.6f}".format(*total)
        )
        return {"electronic": elec, "nuclear": nuc, "total": total}

    def _efg_nuclear(self, A: int) -> np.ndarray:
        coords, charges = self.mol.atom_coords(), self.mol.atom_charges()
        efg = np.zeros((3, 3))
        for B in range(self.mol.natm):
            if B == A:
                continue
            R = coords[B] - coords[A]
            r = np.linalg.norm(R)
            efg += charges[B] * (3 * np.outer(R, R) / r**5 - np.eye(3) / r**3)
        return efg

    def electric_field_gradient(self, atoms=None) -> dict:
        """EFG-tensor (a.u.) ved de angivne atomer (default: alle).

        Returnerer {atom_idx: {"electronic", "nuclear", "total"}}.
        """
        if atoms is None:
            atoms = range(self.mol.natm)
        elif isinstance(atoms, int):
            atoms = [atoms]

        results = {}
        for A in atoms:
            ints = self.efg_integrals(A)
            elec = np.zeros((3, 3))
            for a in range(3):
                for b in range(a, 3):  # tensoren er symmetrisk
                    elec[a, b] = elec[b, a] = self.expectation_value(ints[a, b])
            elec -= np.eye(3) * np.trace(elec) / 3  # sporløs
            elec *= -1  # elektronladning
            nuc = self._efg_nuclear(A)
            total = elec + nuc
            results[A] = {"electronic": elec, "nuclear": nuc, "total": total}

            if self.verbose:
                t = total
                print(f"EFG ved atom {A} ({self.mol.atom_symbol(A)}):")
                print(f"  xx={t[0,0]:.6f}  xy={t[0,1]:.6f}  xz={t[0,2]:.6f}")
                print(f"  yy={t[1,1]:.6f}  yz={t[1,2]:.6f}")
                print(f"  zz={t[2,2]:.6f}")
                print(f"  Spor: {np.trace(t):.2e}   Symmetrisk: {np.allclose(t, t.T)}")
        return results

    # ------------------------------------------------------------------ #
    # Lineær respons
    # ------------------------------------------------------------------ #
    def linear_response(self, excitations: str = "sd"):
        """LR-objekt; bygges og diagonaliseres kun én gang."""
        if self._lr is None:
            self._lr = generalized_naive.LinearResponse(self.wf, excitations=excitations)
            self._lr.calc_excitation_energies()
        return self._lr

    def excitation_energies(self) -> np.ndarray:
        return self.linear_response().excitation_energies

    def oscillator_strengths(self) -> np.ndarray:
        dip_ao = self.dipole_integrals()
        return self.linear_response().get_oscillator_strength(
            dip_ao, x2c=self._is_spinor(dip_ao)
        )

    def polarizability(self, omega: float = 0.0) -> np.ndarray:
        """Fuld 3x3 polarisabilitetstensor. omega != 0 kræver LR.metric."""
        lr = self.linear_response()
        V = lr.get_property_gradient(self.dipole_integrals())  # (n_param, 3)
        E = lr.hessian
        if omega != 0.0:
            E = E - omega * lr.metric
        X = solve(E, V)
        alpha = np.real(np.einsum("ix,iy->xy", V.conj(), X))
        self._log(
            "Polarisabilitet (a.u.):  xx: {:.4f}  yy: {:.4f}  zz: {:.4f}".format(
                *np.diag(alpha)
            )
        )
        return alpha