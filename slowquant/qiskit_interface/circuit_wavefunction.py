import copy
import time
from functools import partial
from typing import Any

import numpy as np
import pyscf
import scipy
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import (
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSamplerV1,
    BaseSamplerV2,
)
from qiskit.quantum_info import SparsePauliOp

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.SlowQuant import SlowQuant
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_electronic_energy,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.integral_manager import IntegralManager
from slowquant.unitary_coupled_cluster.operators import Epq, hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.optimizers import Optimizers


class WaveFunctionCircuit:
    def __init__(
        self,
        active_space: tuple[int, int] | tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,
        integral_generator: SlowQuant | pyscf.gto.mole.Mole,
        quantum_interface: QuantumInterface,
        wavefunction_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize circuit based UPS wave function.

        Wavefunction Options:
            * do_pp [bool]: Make perfect-pairing reference determinant.
                            Will also reorder the orbitalers to match the determinant.
                            (default: False)
            * include_active_kappa [bool]: Include active-active orbital rotations.
                                           (default: True)
            * resolve_unpaired_idx [str]: Specify how to resolve spatial index with respect to occupation of reference determinant.
                                          'both', include spatial index in both occ and unocc idx list.
                                          'occ', include spatial index only in occ idx list.
                                          'unocc', include spatial index only in unocc idx list.
                                          (default: 'both')
            * reference_determinant [str]: Specify a reference determinant for the active space part.
                                          1 specifying occupied orbital and 0 specifying unoccupied orbital.

        Args:
            active_space: (num_active_elec, num_active_orbs) or ((num_active_elec_alpha, num_active_elec_beta), num_active_orbs),
                 orbitals are counted in spatial basis.
            mo_coeffs: Initial orbital coefficients.
            integral_generator: Integral generator object.
            quantum_interface: QuantumInterface.
            wavefunction_options: Wavefunction options.
        """
        if wavefunction_options is None:
            wavefunction_options = {}
        self.wavefunction_options = copy.deepcopy(wavefunction_options)
        valid_options = (
            "do_pp",
            "resolve_unpaired_idx",
            "include_active_kappa",
            "reference_determinant",
        )
        for option in wavefunction_options:
            if option not in valid_options:
                raise ValueError(
                    f"Got unknown option for UPS wave function, {option}. Valid options are: {valid_options}"
                )
        # Default options
        self.wavefunction_options.setdefault("resolve_unpaired_idx", "both")
        self.wavefunction_options.setdefault("include_active_kappa", True)
        self.wavefunction_options.setdefault("do_pp", False)
        if len(active_space) != 2:
            raise ValueError(f"cas must have two elements, got {len(active_space)} elements.")
        if isinstance(active_space[0], int):
            if active_space[0] % 2 == 0:
                cas = ((active_space[0] // 2, active_space[0] // 2), active_space[1])
            else:
                # Uneven number of electrons mean one electron must be unpaired.
                cas = ((active_space[0] // 2 + 1, active_space[0] // 2), active_space[1])
        else:
            cas = ((active_space[0][0], active_space[0][1]), active_space[1])
        # Init stuff
        self.int_gen = IntegralManager(integral_generator)
        if np.sum(cas[0]) > self.int_gen.num_elec:
            raise ValueError("More active electrons than total number electrons.")
        if (np.sum(cas[0]) % 2 == 0 and self.int_gen.num_elec % 2 == 1) or (
            np.sum(cas[0]) % 2 == 1 and self.int_gen.num_elec % 2 == 0
        ):
            raise ValueError("Specified CAS gives odd number of electrons in inactive space.")
        self.num_inactive_spin_orbs = self.int_gen.num_elec - int(np.sum(cas[0]))
        self.num_inactive_orbs = self.num_inactive_spin_orbs // 2
        self.num_active_orbs = cas[1]
        self.num_active_spin_orbs = 2 * self.num_active_orbs
        self.num_active_orbs = self.num_active_spin_orbs // 2
        self.num_virtual_orbs = (
            len(self.int_gen.kinetic_energy) - self.num_inactive_orbs - self.num_active_orbs
        )
        if self.num_virtual_orbs < 0:
            raise ValueError(
                "Number of inactive + number of active orbitals is larger than total number of orbitals."
            )
        self.num_virtual_spin_orbs = 2 * self.num_virtual_orbs
        self.num_orbs = self.num_inactive_orbs + self.num_active_orbs + self.num_virtual_orbs
        self.num_spin_orbs = 2 * self.num_orbs
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_elec = self.num_active_elec_alpha + self.num_active_elec_beta
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._h_mo = None
        self._g_mo = None
        self._energy_elec: float | None = None
        self.num_energy_evals = 0
        self._c_mo = mo_coeffs
        # Reference wave function
        self._pp = self.wavefunction_options["do_pp"]
        if self.wavefunction_options["do_pp"]:
            if "reference_determinant" in self.wavefunction_options.keys():
                raise ValueError(
                    "Both 'do_pp' and 'reference_determinant' are requested in 'wavefunction_options'."
                )
            if self.num_active_elec_alpha != self.num_active_elec_beta:
                raise ValueError(
                    "perfect-pairing is only defined for equal number of alpha and beta electrons."
                )
            # Obtain pp determinant
            pp_det = ""
            spin_orb = 0
            elec_count = self.num_active_elec
            while spin_orb < self.num_active_spin_orbs:
                if (
                    elec_count >= 2
                    and (self.num_active_spin_orbs - spin_orb) >= 4
                    and elec_count <= (self.num_active_spin_orbs - spin_orb - 2)
                ):
                    pp_det += "1100"
                    elec_count -= 2
                    spin_orb += 4
                elif elec_count == 0:
                    pp_det += "0"
                    spin_orb += 1
                elif elec_count != 0:
                    pp_det += "1"
                    spin_orb += 1
                    elec_count -= 1
            if len(pp_det) != self.num_active_spin_orbs or pp_det.count("1") != self.num_active_elec:
                raise ValueError("Perfect pairing determinant violates orbital or electron numbers")

            # Swap mo coefficients to resembles pp layout
            hf_det = "1" * self.num_active_elec + "0" * (self.num_active_spin_orbs - self.num_active_elec)
            hole = [i for i, (h, p) in enumerate(zip(hf_det, pp_det)) if h == "1" and p == "0"]
            part = [i for i, (h, p) in enumerate(zip(hf_det, pp_det)) if h == "0" and p == "1"]
            hole_spatial = sorted(set(i // 2 + self.num_inactive_orbs for i in hole))
            part_spatial = sorted(set(i // 2 + self.num_inactive_orbs for i in part))
            pp_mo_coeffs = mo_coeffs.copy()
            pp_mo_coeffs[:, hole_spatial + part_spatial] = pp_mo_coeffs[:, part_spatial + hole_spatial]
            # Note self._c_mo is set previously but overwritten here.
            self._c_mo = pp_mo_coeffs

            # Assign weight to reference
            ref_det = pp_det
        elif "reference_determinant" in self.wavefunction_options.keys():
            ref_det = self.wavefunction_options["reference_determinant"]
            if len(ref_det) != self.num_active_spin_orbs:
                raise ValueError(
                    f"Reference determinant is {len(ref_det)} spin orbitals and the active space is {self.num_active_spin_orbs} spin orbitals."
                )
            ref_alpha = 0
            ref_beta = 0
            for i, idx in ref_det:
                if i % 2 == 0 and idx == "1":
                    ref_alpha += 1
                elif idx == "1":
                    ref_beta += 1
            if ref_alpha != self.num_active_elec_alpha or ref_beta != self.num_active_elec_beta:
                raise ValueError(
                    "Number of electrons ({ref_alpha}, {ref_beta}) is different from the active space ({self.num_active_elec_alpha, self.num_active_elec_beta})."
                )
        else:
            ref_det = ""
            for i in range(self.num_active_orbs):
                if i < self.num_active_elec_alpha:
                    ref_det += "1"
                else:
                    ref_det += "0"
                if i < self.num_active_elec_beta:
                    ref_det += "1"
                else:
                    ref_det += "0"
        # Construct spin orbital indices
        self.inactive_spin_idx = [x for x in range(self.num_inactive_spin_orbs)]
        self.active_spin_idx = [x + self.num_inactive_spin_orbs for x in range(self.num_active_spin_orbs)]
        self.virtual_spin_idx = [
            x + self.num_inactive_spin_orbs + self.num_active_spin_orbs
            for x in range(self.num_virtual_spin_orbs)
        ]
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        for i, orb_idx in enumerate(self.active_spin_idx):
            if ref_det[i] == "1":
                self.active_occ_spin_idx.append(orb_idx)
            else:
                self.active_unocc_spin_idx.append(orb_idx)
        self.active_spin_idx_shifted = [x - self.num_inactive_spin_orbs for x in self.active_spin_idx]
        self.active_occ_spin_idx_shifted = [x - self.num_inactive_spin_orbs for x in self.active_occ_spin_idx]
        self.active_unocc_spin_idx_shifted = [
            x - self.num_inactive_spin_orbs for x in self.active_unocc_spin_idx
        ]
        # Construct spatial idx
        self.inactive_idx = [x for x in range(self.num_inactive_orbs)]
        self.active_idx = [x + self.num_inactive_orbs for x in range(self.num_active_orbs)]
        self.virtual_idx = [
            x + self.num_inactive_orbs + self.num_active_orbs for x in range(self.num_virtual_orbs)
        ]
        self.active_occ_idx = []
        self.active_unocc_idx = []
        for i, orb_idx in enumerate(self.active_idx):
            if ref_det[2 * i] == "1" and ref_det[2 * i + 1] == "1":
                self.active_occ_idx.append(orb_idx)
            elif ref_det[2 * i] == "0" and ref_det[2 * i + 1] == "0":
                self.active_unocc_idx.append(orb_idx)
            elif self.wavefunction_options["resolve_unpaired_idx"] == "both":
                self.active_occ_idx.append(orb_idx)
                self.active_unocc_idx.append(orb_idx)
            elif self.wavefunction_options["resolve_unpaired_idx"] == "occ":
                self.active_occ_idx.append(orb_idx)
            elif self.wavefunction_options["resolve_unpaired_idx"] == "unocc":
                self.active_unocc_idx.append(orb_idx)
            else:
                raise ValueError(
                    f"Got unknown option for resolve_unpaired_idx, {wavefunction_options['resolve_unpaired_idx']}, excpected 'both', 'occ' or 'unocc'."
                )
        self.active_idx_shifted = [x - self.num_inactive_orbs for x in self.active_idx]
        self.active_occ_idx_shifted = [x - self.num_inactive_orbs for x in self.active_occ_idx]
        self.active_unocc_idx_shifted = [x - self.num_inactive_orbs for x in self.active_unocc_idx]
        # Find non-redundant kappas
        self._kappa = []
        kappa_idx = []
        kappa_no_activeactive_idx = []
        kappa_no_activeactive_idx_dagger = []
        self._kappa_old = []
        # kappa can be optimized in spatial basis
        # Loop over all q>p orb combinations and find redundant kappas
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                # find redundant kappas
                if p in self.inactive_idx and q in self.inactive_idx:
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    continue
                if not self.wavefunction_options["include_active_kappa"]:
                    if p in self.active_idx and q in self.active_idx:
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    kappa_no_activeactive_idx.append((p, q))
                    kappa_no_activeactive_idx_dagger.append((q, p))
                # the rest is non-redundant
                self._kappa.append(0.0)
                self._kappa_old.append(0.0)
                kappa_idx.append((p, q))
        # HF like orbital rotation indices
        kappa_hf_like_idx = []
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.virtual_idx:
                    kappa_hf_like_idx.append((p, q))
                elif p in self.inactive_idx and q in self.active_unocc_idx:
                    kappa_hf_like_idx.append((p, q))
                elif p in self.active_occ_idx and q in self.virtual_idx:
                    kappa_hf_like_idx.append((p, q))
        self.kappa_idx = np.array(kappa_idx, dtype=np.int64)
        self.kappa_no_activeactive_idx = np.array(kappa_no_activeactive_idx, dtype=np.int64)
        self.kappa_no_activeactive_idx_dagger = np.array(kappa_no_activeactive_idx_dagger, dtype=np.int64)
        self.kappa_hf_like_idx = np.array(kappa_hf_like_idx, dtype=np.int64)
        hf_det = "1" * self.num_active_elec + "0" * (self.num_active_spin_orbs - self.num_active_elec)
        if ref_det == hf_det:
            # If the refernce determinant is just Hartree-Fock,
            # then reconstruct it inside QI to allow for other mappers than JW.
            self.ref_det = None
        else:
            self.ref_det = ref_det
        # Setup Quantum Interface
        self.QI = quantum_interface
        self.QI.construct_circuit(
            self.active_occ_idx_shifted,
            self.active_unocc_idx_shifted,
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            self.num_active_orbs,
            (self.num_active_elec_alpha, self.num_active_elec_beta),
            ref_det=self.ref_det,
        )

    @property
    def kappa(self) -> list[float]:
        """Get orbital rotation parameters."""
        return self._kappa.copy()

    @kappa.setter
    def kappa(self, k: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._kappa = k.copy()
        # Move current expansion point.
        self._c_mo = self.c_mo
        self._kappa_old = self.kappa

    @property
    def c_mo(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        kappa_mat = np.zeros_like(self._c_mo)
        if len(self.kappa) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa) - np.array(self._kappa_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(self.kappa, self._kappa_old, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = one_electron_integral_transform(self.c_mo, self.int_gen.h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_mo, self.int_gen.electron_electron_repulsion)
        return self._g_mo

    @property
    def thetas(self) -> list[float]:
        """Getter for ansatz parameters.

        Returns:
            Ansatz parameters.
        """
        return self.QI.parameters

    @thetas.setter
    def thetas(self, parameters: list[float]) -> None:
        """Setter for ansatz paramters.

        Args:
            parameters: New ansatz paramters.
        """
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._energy_elec = None
        self.QI.parameters = parameters

    def change_primitive(self, primitive: BaseSamplerV1 | BaseSamplerV2, verbose: bool = True) -> None:
        """Change the primitive expectation value calculator.

        Args:
            primitive: Primitive object.
            verbose: Print more info.
        """
        if verbose:
            print(
                "Using this function is only recommended for switching from ideal simulator to shot-noise or quantum hardware.\n \
                Multiple switching back and forth can lead to un-expected outcomes and is an experimental feature.\n"
            )

        if isinstance(primitive, (BaseEstimatorV1, BaseEstimatorV2)):
            raise ValueError("Estimator is not supported.")
        elif not isinstance(primitive, (BaseSamplerV1, BaseSamplerV2)):
            raise TypeError(f"Unsupported primitive, {type(primitive)}")
        self.QI._primitive = primitive
        if verbose:
            if self.QI.mitigation_flags.do_M_ansatz0:
                print("Reset RDMs, energies, QI metrics, and correlation matrix.")
            else:
                print("Reset RDMs, energies, and QI metrics.")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._energy_elec = None
        self.QI.total_device_calls = 0
        self.QI.total_shots_used = 0
        self.QI.total_paulis_evaluated = 0

        # Reset circuit and initiate re-transpiling
        ISA_old = self.QI.ISA
        self._reconstruct_circuit()  # Reconstruct circuit but keeping parameters
        self.QI._transpiled = False
        self.QI.ISA = ISA_old  # Redo ISA including transpilation if requested
        self.QI.shots = self.QI.shots  # Redo shots parameter check

        if verbose:
            self.QI.get_info()

    def _reconstruct_circuit(self) -> None:
        """Construct circuit again."""
        self.QI.construct_circuit(
            self.active_occ_idx_shifted,
            self.active_unocc_idx_shifted,
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            self.num_active_orbs,
            (self.num_active_elec_alpha, self.num_active_elec_beta),
            ref_det=self.ref_det,
        )

    @property
    def rdm1(self) -> np.ndarray:
        r"""Calculate one-electron reduced density matrix.

        The trace condition is enforced:

        .. math::
            \sum_i\Gamma^{[1]}_{ii} = N_e

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs), dtype=np.float64)
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_ = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_ = q - self.num_inactive_orbs
                    rdm1_op = Epq(p, q).get_folded_operator(
                        self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                    )
                    val = self.QI.quantum_expectation_value(rdm1_op)
                    self._rdm1[p_, q_] = val  # type: ignore [index]
                    self._rdm1[q_, p_] = val  # type: ignore [index]
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        r"""Calculate two-electron reduced density matrix.

        The trace condition is enforced:

        .. math::
            \sum_{ij}\Gamma^{[2]}_{iijj} = N_e(N_e-1)

        Returns:
            Two-electron reduced density matrix.
        """
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                ),
                dtype=np.float64,
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_ = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_ = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_ = r - self.num_inactive_orbs
                        if p == q:
                            s_lim = r + 1
                        elif p == r:
                            s_lim = q + 1
                        elif q < r:
                            s_lim = p
                        else:
                            s_lim = p + 1
                        for s in range(self.num_inactive_orbs, s_lim):
                            s_ = s - self.num_inactive_orbs
                            pdm2_op = (Epq(p, q) * Epq(r, s)).get_folded_operator(
                                self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                            )
                            val = self.QI.quantum_expectation_value(pdm2_op)
                            if q == r:
                                val -= self.rdm1[p_, s_]
                            self._rdm2[p_, q_, r_, s_] = val  # type: ignore
                            self._rdm2[r_, s_, p_, q_] = val  # type: ignore
                            self._rdm2[q_, p_, s_, r_] = val  # type: ignore
                            self._rdm2[s_, r_, q_, p_] = val  # type: ignore
        return self._rdm2

    @property
    def rdm3(self) -> np.ndarray:
        r"""Calculate three-electron reduced density matrix.

        The trace condition is enforced:

        .. math::
            \sum_{ijk}\Gamma^{[3]}_{iijjkk} = N_e(N_e-1)(N_e-2)

        Returns:
            Three-electron reduced density matrix.
        """
        if self._rdm3 is None:
            self._rdm3 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                ),
                dtype=np.float64,
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_ = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_ = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_ = r - self.num_inactive_orbs
                        for s in range(self.num_inactive_orbs, p + 1):
                            s_ = s - self.num_inactive_orbs
                            for t in range(self.num_inactive_orbs, r + 1):
                                t_ = t - self.num_inactive_orbs
                                for u in range(self.num_inactive_orbs, p + 1):
                                    u_ = u - self.num_inactive_orbs
                                    pdm3_op = (Epq(p, q) * Epq(r, s) * Epq(t, u)).get_folded_operator(
                                        self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                                    )
                                    val = self.QI.quantum_expectation_value(pdm3_op)
                                    if t == s:
                                        val -= self.rdm2[p_, q_, r_, u_]
                                    if r == q:
                                        val -= self.rdm2[p_, s_, t_, u_]
                                    if t == q:
                                        val -= self.rdm2[p_, u_, r_, s_]
                                    if t == s and r == q:
                                        val -= self.rdm1[p_, u_]
                                    self._rdm3[p_, q_, r_, s_, t_, u_] = val  # type: ignore
                                    self._rdm3[p_, q_, t_, u_, r_, s_] = val  # type: ignore
                                    self._rdm3[r_, s_, p_, q_, t_, u_] = val  # type: ignore
                                    self._rdm3[r_, s_, t_, u_, p_, q_] = val  # type: ignore
                                    self._rdm3[t_, u_, p_, q_, r_, s_] = val  # type: ignore
                                    self._rdm3[t_, u_, r_, s_, p_, q_] = val  # type: ignore
                                    self._rdm3[q_, p_, s_, r_, u_, t_] = val  # type: ignore
                                    self._rdm3[q_, p_, u_, t_, s_, r_] = val  # type: ignore
                                    self._rdm3[s_, r_, q_, p_, u_, t_] = val  # type: ignore
                                    self._rdm3[s_, r_, u_, t_, q_, p_] = val  # type: ignore
                                    self._rdm3[u_, t_, q_, p_, s_, r_] = val  # type: ignore
                                    self._rdm3[u_, t_, s_, r_, q_, p_] = val  # type: ignore
        return self._rdm3

    @property
    def rdm4(self) -> np.ndarray:
        r"""Calculate four-electron reduced density matrix.

        The trace condition is enforced:

        .. math::
            \sum_{ijkl}\Gamma^{[4]}_{iijjkkll} = N_e(N_e-1)(N_e-2)(N_e-3)

        Returns:
            Four-electron reduced density matrix.
        """
        if self._rdm4 is None:
            self._rdm4 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                ),
                dtype=np.float64,
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_ = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_ = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_ = r - self.num_inactive_orbs
                        for s in range(self.num_inactive_orbs, p + 1):
                            s_ = s - self.num_inactive_orbs
                            for t in range(self.num_inactive_orbs, r + 1):
                                t_ = t - self.num_inactive_orbs
                                for u in range(self.num_inactive_orbs, p + 1):
                                    u_ = u - self.num_inactive_orbs
                                    for m in range(self.num_inactive_orbs, t + 1):
                                        m_ = m - self.num_inactive_orbs
                                        for n in range(self.num_inactive_orbs, p + 1):
                                            n_ = n - self.num_inactive_orbs
                                            pdm4_op = (
                                                Epq(p, q) * Epq(r, s) * Epq(t, u) * Epq(m, n)
                                            ).get_folded_operator(
                                                self.num_inactive_orbs,
                                                self.num_active_orbs,
                                                self.num_virtual_orbs,
                                            )
                                            val = self.QI.quantum_expectation_value(pdm4_op)
                                            if r == q:
                                                val -= self.rdm3[p_, s_, t_, u_, m_, n_]
                                            if t == q:
                                                val -= self.rdm3[p_, u_, r_, s_, m_, n_]
                                            if m == q:
                                                val -= self.rdm3[p_, n_, r_, s_, t_, u_]
                                            if m == u:
                                                val -= self.rdm3[p_, q_, r_, s_, t_, n_]
                                            if t == s:
                                                val -= self.rdm3[p_, q_, r_, u_, m_, n_]
                                            if m == s:
                                                val -= self.rdm3[p_, q_, r_, n_, t_, u_]
                                            if m == u and r == q:
                                                val -= self.rdm2[p_, s_, t_, n_]
                                            if m == u and t == q:
                                                val -= self.rdm2[p_, n_, r_, s_]
                                            if t == s and m == u:
                                                val -= self.rdm2[p_, q_, r_, n_]
                                            if t == s and r == q:
                                                val -= self.rdm2[p_, u_, m_, n_]
                                            if t == s and m == q:
                                                val -= self.rdm2[p_, n_, r_, u_]
                                            if m == s and r == q:
                                                val -= self.rdm2[p_, n_, t_, u_]
                                            if m == s and t == q:
                                                val -= self.rdm2[p_, u_, r_, n_]
                                            if m == u and t == s and r == q:
                                                val -= self.rdm1[p_, n_]
                                            self._rdm4[p_, q_, r_, s_, t_, u_, m_, n_] = val  # type: ignore
                                            self._rdm4[p_, q_, r_, s_, m_, n_, t_, u_] = val  # type: ignore
                                            self._rdm4[p_, q_, t_, u_, r_, s_, m_, n_] = val  # type: ignore
                                            self._rdm4[p_, q_, t_, u_, m_, n_, r_, s_] = val  # type: ignore
                                            self._rdm4[p_, q_, m_, n_, r_, s_, t_, u_] = val  # type: ignore
                                            self._rdm4[p_, q_, m_, n_, t_, u_, r_, s_] = val  # type: ignore
                                            self._rdm4[r_, s_, p_, q_, t_, u_, m_, n_] = val  # type: ignore
                                            self._rdm4[r_, s_, p_, q_, m_, n_, t_, u_] = val  # type: ignore
                                            self._rdm4[r_, s_, t_, u_, p_, q_, m_, n_] = val  # type: ignore
                                            self._rdm4[r_, s_, t_, u_, m_, n_, p_, q_] = val  # type: ignore
                                            self._rdm4[r_, s_, m_, n_, p_, q_, t_, u_] = val  # type: ignore
                                            self._rdm4[r_, s_, m_, n_, t_, u_, p_, q_] = val  # type: ignore
                                            self._rdm4[t_, u_, p_, q_, r_, s_, m_, n_] = val  # type: ignore
                                            self._rdm4[t_, u_, p_, q_, m_, n_, r_, s_] = val  # type: ignore
                                            self._rdm4[t_, u_, r_, s_, p_, q_, m_, n_] = val  # type: ignore
                                            self._rdm4[t_, u_, r_, s_, m_, n_, p_, q_] = val  # type: ignore
                                            self._rdm4[t_, u_, m_, n_, p_, q_, r_, s_] = val  # type: ignore
                                            self._rdm4[t_, u_, m_, n_, r_, s_, p_, q_] = val  # type: ignore
                                            self._rdm4[m_, n_, p_, q_, r_, s_, t_, u_] = val  # type: ignore
                                            self._rdm4[m_, n_, p_, q_, t_, u_, r_, s_] = val  # type: ignore
                                            self._rdm4[m_, n_, r_, s_, p_, q_, t_, u_] = val  # type: ignore
                                            self._rdm4[m_, n_, r_, s_, t_, u_, p_, q_] = val  # type: ignore
                                            self._rdm4[m_, n_, t_, u_, p_, q_, r_, s_] = val  # type: ignore
                                            self._rdm4[m_, n_, t_, u_, r_, s_, p_, q_] = val  # type: ignore
                                            self._rdm4[q_, p_, s_, r_, u_, t_, n_, m_] = val  # type: ignore
                                            self._rdm4[q_, p_, s_, r_, n_, m_, u_, t_] = val  # type: ignore
                                            self._rdm4[q_, p_, u_, t_, s_, r_, n_, m_] = val  # type: ignore
                                            self._rdm4[q_, p_, u_, t_, n_, m_, s_, r_] = val  # type: ignore
                                            self._rdm4[q_, p_, n_, m_, s_, r_, u_, t_] = val  # type: ignore
                                            self._rdm4[q_, p_, n_, m_, u_, t_, s_, r_] = val  # type: ignore
                                            self._rdm4[s_, r_, q_, p_, u_, t_, n_, m_] = val  # type: ignore
                                            self._rdm4[s_, r_, q_, p_, n_, m_, u_, t_] = val  # type: ignore
                                            self._rdm4[s_, r_, u_, t_, q_, p_, n_, m_] = val  # type: ignore
                                            self._rdm4[s_, r_, u_, t_, n_, m_, q_, p_] = val  # type: ignore
                                            self._rdm4[s_, r_, n_, m_, q_, p_, u_, t_] = val  # type: ignore
                                            self._rdm4[s_, r_, n_, m_, u_, t_, q_, p_] = val  # type: ignore
                                            self._rdm4[u_, t_, q_, p_, s_, r_, n_, m_] = val  # type: ignore
                                            self._rdm4[u_, t_, q_, p_, n_, m_, s_, r_] = val  # type: ignore
                                            self._rdm4[u_, t_, s_, r_, q_, p_, n_, m_] = val  # type: ignore
                                            self._rdm4[u_, t_, s_, r_, n_, m_, q_, p_] = val  # type: ignore
                                            self._rdm4[u_, t_, n_, m_, q_, p_, s_, r_] = val  # type: ignore
                                            self._rdm4[u_, t_, n_, m_, s_, r_, q_, p_] = val  # type: ignore
                                            self._rdm4[n_, m_, q_, p_, s_, r_, u_, t_] = val  # type: ignore
                                            self._rdm4[n_, m_, q_, p_, u_, t_, s_, r_] = val  # type: ignore
                                            self._rdm4[n_, m_, s_, r_, q_, p_, u_, t_] = val  # type: ignore
                                            self._rdm4[n_, m_, s_, r_, u_, t_, q_, p_] = val  # type: ignore
                                            self._rdm4[n_, m_, u_, t_, q_, p_, s_, r_] = val  # type: ignore
                                            self._rdm4[n_, m_, u_, t_, s_, r_, q_, p_] = val  # type: ignore
        return self._rdm4

    def precalc_rdm_paulis(self, rdm_order: int) -> None:
        """Pre-calculate all Paulis used to construct RDMs up to a certain order.

        This utilizes the saving feature in QuantumInterface when using the Sampler primitive.
        If saving is turned up in QuantumInterface this function will do nothing but waste device time.

        Args:
            rdm_order: Max order RDM.
        """
        if not isinstance(
            self.QI._primitive,
            (BaseSamplerV1, BaseSamplerV2),
        ):
            raise TypeError(
                f"This feature is only supported for Sampler got {type(self.QI._primitive)} from QuantumInterface"
            )
        if rdm_order > 4:
            raise ValueError(f"Precalculation only supported up to order 4 got {rdm_order}")
        if rdm_order < 1:
            raise ValueError(f"Precalculation need at least an order of 1 got {rdm_order}")
        cumulated_paulis = None
        if rdm_order >= 1:
            self._rdm1 = None
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, p + 1):
                    rdm1_op = Epq(p, q).get_folded_operator(
                        self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                    )
                    mapped_op = self.QI.op_to_qbit(rdm1_op)
                    if cumulated_paulis is None:
                        cumulated_paulis = set(mapped_op.paulis)
                    else:
                        cumulated_paulis = cumulated_paulis.union(mapped_op.paulis)
        if rdm_order >= 2:
            self._rdm2 = None
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, p + 1):
                    for r in range(self.num_inactive_orbs, p + 1):
                        if p == q:
                            s_lim = r + 1
                        elif p == r:
                            s_lim = q + 1
                        elif q < r:
                            s_lim = p
                        else:
                            s_lim = p + 1
                        for s in range(self.num_inactive_orbs, s_lim):
                            pdm2_op = (Epq(p, q) * Epq(r, s)).get_folded_operator(
                                self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                            )
                            mapped_op = self.QI.op_to_qbit(pdm2_op)
                            cumulated_paulis = cumulated_paulis.union(mapped_op.paulis)  # type: ignore[union-attr]
        if rdm_order >= 3:
            self._rdm3 = None
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, p + 1):
                    for r in range(self.num_inactive_orbs, p + 1):
                        for s in range(self.num_inactive_orbs, p + 1):
                            for t in range(self.num_inactive_orbs, r + 1):
                                for u in range(self.num_inactive_orbs, p + 1):
                                    pdm3_op = (Epq(p, q) * Epq(r, s) * Epq(t, u)).get_folded_operator(
                                        self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                                    )
                                    mapped_op = self.QI.op_to_qbit(pdm3_op)
                                    cumulated_paulis = cumulated_paulis.union(mapped_op.paulis)  # type: ignore[union-attr]
        if rdm_order >= 4:
            self._rdm4 = None
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                for q in range(self.num_inactive_orbs, p + 1):
                    for r in range(self.num_inactive_orbs, p + 1):
                        for s in range(self.num_inactive_orbs, p + 1):
                            for t in range(self.num_inactive_orbs, r + 1):
                                for u in range(self.num_inactive_orbs, p + 1):
                                    for m in range(self.num_inactive_orbs, t + 1):
                                        for n in range(self.num_inactive_orbs, p + 1):
                                            pdm4_op = (
                                                Epq(p, q) * Epq(r, s) * Epq(t, u) * Epq(m, n)
                                            ).get_folded_operator(
                                                self.num_inactive_orbs,
                                                self.num_active_orbs,
                                                self.num_virtual_orbs,
                                            )
                                            mapped_op = self.QI.op_to_qbit(pdm4_op)
                                            cumulated_paulis = cumulated_paulis.union(mapped_op.paulis)  # type: ignore[union-attr]
        # Calling expectation value to put all Paulis in cliques
        # and compute distributions for the cliques.
        # The coefficients are set to one, so the Paulis cannot cancel out.
        _ = self.QI._sampler_quantum_expectation_value(
            SparsePauliOp(cumulated_paulis, np.ones(len(cumulated_paulis)))  # type: ignore[arg-type]
        )

    def check_orthonormality(self) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T
        """
        S_ortho = one_electron_integral_transform(self.c_mo, self.int_gen.overlap)
        one = np.identity(len(S_ortho))
        diff = np.abs(S_ortho - one)
        print("Max ortho-normal diff:", np.max(diff))

    @property
    def energy_elec(self) -> float:
        """Get electronic energy.

        Returns:
            Electronic energy.
        """
        if self._energy_elec is None:
            self._energy_elec = self._calc_energy_elec()
        return self._energy_elec

    def _get_hamiltonian(self, qiskit_form: bool = False) -> FermionicOperator | dict[str, float]:
        """Return electronic Hamiltonian as FermionicOperator.

        Returns:
            FermionicOperator.
        """
        H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)

        if qiskit_form:
            return H.get_qiskit_form(self.num_orbs)
        return H

    def _calc_energy_elec(self) -> float:
        """Run electronic energy simulation, regardless of self._energy_elec variable.

        Returns:
            Electronic energy.
        """
        H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
        energy_elec = self.QI.quantum_expectation_value(H)
        return energy_elec

    def run_wf_optimization(
        self,
        orbital_optimization: bool = False,
        tol: float = 1e-10,
        maxiter: int = 1000,
        optimization_options: dict[str, Any] | None = None,
    ) -> None:
        """Run variational optimization of wavefunction.

        Optimization options:
            * theta_optimization [bool]: Perform theta optimization.
                                         (default: True)
            * theta_optimizer [str]: Optimizer used for theta optimization.
                                     (default: BFGS)
            * orbital_optimizer [str]: Optimizer used for orbital optimization.
                                     (default: BFGS)
            * 1step_optimizer [str]: Optimizer used for 1step optimizer.
                                     (fallback: copy theta_optimizer)
            * opt_type [str]: Optimization type, can be '1step' or '2step'.
                              (default: 1step)
            * is_silent_subiterations [bool]: Silence sub iterations in 2step.
                                              (default: False)
            * print_std [bool]: Print standard deviation of the electronic Hamiltonian during optimization.
                                (default: False)

        Args:
            orbital_optimization: Perform orbital optimization.
            tol: Tolerance for finishing the optimization.
            maxiter: Maximum number of iterations.
            optimization_options: Additional optimization options.
        """
        if optimization_options is None:
            optimization_options = {}
        self._optimization_options = copy.deepcopy(optimization_options)
        valid_options = (
            "theta_optimization",
            "theta_optimizer",
            "orbital_optimizer",
            "opt_type",
            "is_silent_subiterations",
            "1step_optimizer",
            "print_std",
        )
        for option in self._optimization_options:
            if option not in valid_options:
                raise ValueError(
                    f"Got unknown option for optimization, {option}. Valid options are: {valid_options}"
                )
        self._optimization_options["tol"] = tol
        self._optimization_options["maxiter"] = int(maxiter)
        self._optimization_options["orbital_optimization"] = orbital_optimization
        self._optimization_options.setdefault("theta_optimization", True)
        self._optimization_options.setdefault("theta_optimizer", "BFGS")
        self._optimization_options.setdefault("orbital_optimizer", "BFGS")
        self._optimization_options.setdefault("opt_type", "1step")
        self._optimization_options.setdefault("is_silent_subiterations", False)
        self._optimization_options.setdefault("print_std", False)
        if len(self.kappa) == 0 and self._optimization_options["orbital_optimization"]:
            print("No kappa parameters turning off orbital optimization.")
        if len(self.thetas) == 0 and self._optimization_options["theta_optimization"]:
            print("No thetas parameters turning off theta optimization.")
        if self._optimization_options["opt_type"].lower() == "2step" and (
            not self._optimization_options["orbital_optimization"]
            or not self._optimization_options["theta_optimization"]
        ):
            if not self._optimization_options["orbital_optimization"]:
                print("Orbital optimization not requested changing optimizer type to 1step.")
                self._optimization_options["opt_type"] = "1step"
            elif not self._optimization_options["theta_optimization"]:
                print("theta optimization not requested changing optimizer type to 1step.")
                self._optimization_options["opt_type"] = "1step"
        if (
            self._optimization_options["opt_type"].lower() == "1step"
            and "1step_optimizer" not in self._optimization_options.keys()
        ):
            print(
                "'1step_optimizer' was not specifed. Using the optimizer specified as 'theta_optimizer': {self._optimization_options['theta_optimizer']}"
            )
            self._optimization_options["1step_optimizer"] = self._optimization_options["theta_optimizer"]
        if self._optimization_options["theta_optimization"] and isinstance(self.QI.ansatz, QuantumCircuit):
            if self._optimization_options["opt_type"].lower() == "1step" and self._optimization_options[
                "1step_optimizer"
            ].lower() not in ("cobyla", "cobyqa"):
                raise ValueError("Custom Ansatz in QI only works with COBYLA and COBYQA as optimizer.")
            elif self._optimization_options["opt_type"].lower() == "2step" and self._optimization_options[
                "theta_optimizer"
            ].lower() not in ("cobyla", "cobyqa"):
                raise ValueError("Custom Ansatz in QI only works with COBYLA and COBYQA as optimizer.")
        print("### Parameters information:")
        if self._optimization_options["orbital_optimization"]:
            print(f"### Number kappa: {len(self.kappa)}")
        if self._optimization_options["theta_optimization"]:
            print("### Number theta: {len(self.thetas)}")
        if self._optimization_options["opt_type"].lower() == "1step":
            self._run_wf_optimization_1step()
        elif self._optimization_options["opt_type"].lower() == "2step":
            self._run_wf_optimization_1step()
        else:
            raise ValueError(
                f"Got unknown 'opt_type', {[self._optimization_options['opt_type']]} excepted '1step' or '2step'."
            )

    def _run_wf_optimization_2step(
        self,
    ) -> None:
        """Run two step optimization of wave function."""
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        for full_iter in range(0, self._optimization_options["maxiter"]):
            full_start = time.time()

            # Do ansatz optimization
            if not self._optimization_options["is_silent_subiterations"]:
                print("--------Ansatz optimization")
                subheader = "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                if self._optimization_options["print_std"]:
                    subheader += " | Std(H)"
                print(subheader)

            energy_theta = partial(
                self._calc_energy_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            gradient_theta = partial(
                self._calc_gradient_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            optimizer = Optimizers(
                energy_theta,
                self._optimization_options["theta_optimizer"],
                grad=gradient_theta,
                maxiter=self._optimization_options["maxiter"],
                tol=self._optimization_options["tol"],
                is_silent=self._optimization_options["is_silent_subiterations"],
                energy_eval_callback=lambda: self.num_energy_evals,
                std_callback=(
                    (
                        lambda: self.QI.quantum_variance(
                            hamiltonian_0i_0a(
                                self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs
                            ).get_folded_operator(
                                self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                            )
                        )
                    )
                    if self._optimization_options["print_std"]
                    else None
                ),
            )
            res = optimizer.minimize(
                self.thetas,
                extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names},
            )
            self.thetas = res.x.tolist()

            if not self._optimization_options["is_silent_subiterations"]:
                print("--------Orbital optimization")
                subheader = "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                if self._optimization_options["print_std"]:
                    subheader += " | Std(H)"
                print(subheader)

            energy_oo = partial(
                self._calc_energy_optimization,
                theta_optimization=False,
                kappa_optimization=True,
            )
            gradient_oo = partial(
                self._calc_gradient_optimization,
                theta_optimization=False,
                kappa_optimization=True,
            )

            optimizer = Optimizers(
                energy_oo,
                self._optimization_options["orbital_optimizer"],
                grad=gradient_oo,
                maxiter=self._optimization_options["maxiter"],
                tol=self._optimization_options["tol"],
                is_silent=self._optimization_options["is_silent_subiterations"],
                energy_eval_callback=lambda: self.num_energy_evals,
                std_callback=(
                    (
                        lambda: self.QI.quantum_variance(
                            hamiltonian_0i_0a(
                                self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs
                            ).get_folded_operator(
                                self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                            )
                        )
                    )
                    if self._optimization_options["print_std"]
                    else None
                ),
            )
            res = optimizer.minimize([0.0] * len(self.kappa_idx))
            for i in range(len(self.kappa)):
                self._kappa[i] = 0.0
                self._kappa_old[i] = 0.0
            e_new = res.fun
            time_str = f"{time.time() - full_start:7.2f}"  # type: ignore
            e_str = f"{e_new:3.12f}"
            print(
                f"{str(full_iter + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)} | {str(self.num_energy_evals).center(11)}"
            )  # type: ignore
            if abs(e_new - e_old) < self._optimization_options["tol"]:
                break
            e_old = e_new
        self._energy_elec = e_new

    def _run_wf_optimization_1step(
        self,
    ) -> None:
        """Run one step optimization of wave function."""
        if (
            self._optimization_options["1step_optimizer"].lower() == "rotosolve"
            and self._optimization_options["orbital_optimization"]
        ):
            raise ValueError(
                "Cannot use RotoSolve together with orbital optimization in the one-step solver."
            )
        header = (
            "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
        )
        if self._optimization_options["print_std"]:
            header += " | Std(H)"
        print(header)
        if self._optimization_options["orbital_optimization"]:
            if self._optimization_options["theta_optimization"]:
                energy = partial(
                    self._calc_energy_optimization,
                    theta_optimization=True,
                    kappa_optimization=True,
                )
                gradient = partial(
                    self._calc_gradient_optimization,
                    theta_optimization=True,
                    kappa_optimization=True,
                )
            else:
                energy = partial(
                    self._calc_energy_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )
                gradient = partial(
                    self._calc_gradient_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )
        else:
            energy = partial(
                self._calc_energy_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            gradient = partial(
                self._calc_gradient_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
        if self._optimization_options["orbital_optimization"]:
            if self._optimization_options["theta_optimization"]:
                parameters = self.kappa + self.thetas
            else:
                parameters = self.kappa
        else:
            parameters = self.thetas
        optimizer = Optimizers(
            energy,
            self._optimization_options["1step_optimizer"],
            grad=gradient,
            maxiter=self._optimization_options["maxiter"],
            tol=self._optimization_options["tol"],
            energy_eval_callback=lambda: self.num_energy_evals,
            std_callback=(
                (
                    lambda: self.QI.quantum_variance(
                        hamiltonian_0i_0a(
                            self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs
                        ).get_folded_operator(
                            self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                        )
                    )
                )
                if self._optimization_options["print_std"]
                else None
            ),
        )
        res = optimizer.minimize(
            parameters, extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names}
        )
        if self._optimization_options["orbital_optimization"]:
            if self._optimization_options["theta_optimization"]:
                self.thetas = res.x[len(self.kappa) :].tolist()
            for i in range(len(self.kappa)):
                self._kappa[i] = 0.0
                self._kappa_old[i] = 0.0
        else:
            self.thetas = res.x.tolist()
        self._energy_elec = res.fun

    def _calc_energy_optimization(
        self, parameters: list[float], theta_optimization: bool, kappa_optimization: bool
    ) -> float:
        """Calculate electronic energy.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: Doing theta optimization.
            kappa_optimization: Doing kappa optimization.

        Returns:
            Electronic energy.
        """
        num_kappa = 0
        self.num_energy_evals += 1  # count one measurement
        if kappa_optimization:
            num_kappa = len(self.kappa_idx)
            self.kappa = parameters[:num_kappa]
        if theta_optimization:
            self.thetas = parameters[num_kappa:]
            # Build operator
            H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            return self.QI.quantum_expectation_value(H)
        # RDM is more expensive than evaluation of the Hamiltonian.
        # Thus only construct these if orbital-optimization is turned on,
        # since the RDMs will be reused in the oo gradient calculation.
        return get_electronic_energy(
            self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs, self.rdm1, self.rdm2
        )

    def _calc_gradient_optimization(
        self, parameters: list[float], theta_optimization: bool, kappa_optimization: bool
    ) -> np.ndarray:
        """Calculate electronic gradient.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: Doing theta optimization.
            kappa_optimization: Doing kappa optimization.

        Returns:
            Electronic gradient.
        """
        num_kappa = 0
        gradient = np.zeros(len(parameters))
        num_kappa = 0
        if kappa_optimization:
            num_kappa = len(self.kappa_idx)
            self.kappa = parameters[:num_kappa]
        if theta_optimization:
            self.thetas = parameters[num_kappa:]
        if kappa_optimization:
            gradient[:num_kappa] = get_orbital_gradient(
                self.h_mo,
                self.g_mo,
                self.kappa_idx,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.rdm1,
                self.rdm2,
            )
        if theta_optimization:
            H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            for i in range(len(parameters[num_kappa:])):
                R = self.QI.grad_param_R[self.QI.param_names[i]]
                e_vals_grad = _get_energy_evals_for_grad(H, self.QI, parameters[num_kappa:], i, R)
                grad = 0.0
                for j, mu in enumerate(list(range(1, 2 * R + 1))):
                    x_mu = (2 * mu - 1) / (2 * R) * np.pi
                    grad += e_vals_grad[j] * (-1) ** (mu - 1) / (4 * R * (np.sin(1 / 2 * x_mu)) ** 2)
                gradient[num_kappa + i] = grad
            self.num_energy_evals += 2 * np.sum(
                list(self.QI.grad_param_R.values())
            )  # Count energy measurements for all gradients
        return gradient


def _get_energy_evals_for_grad(
    operator: FermionicOperator,
    quantum_interface: QuantumInterface,
    parameters: list[float],
    idx: int,
    R: int,
) -> list[float]:
    """Get energy evaluations needed for the gradient calculation.

    The gradient formula is defined for x=0.
    The x_shift variable is used to shift the energy function, such that current parameter value is in zero.

    Args:
        operator: Operator which the derivative is with respect to.
        quantum_interface: Quantum interface class object.
        parameters: Parameters.
        idx: Parameter idx.
        R: Parameter to control we get the needed points.

    Returns:
        Energies in a few fixed points.
    """
    e_vals = []
    x = parameters.copy()
    x_shift = x[idx]
    for mu in range(1, 2 * R + 1):
        x_mu = (2 * mu - 1) / (2 * R) * np.pi
        x[idx] = x_mu + x_shift
        e_vals.append(quantum_interface.quantum_expectation_value(operator, custom_parameters=x))
    return e_vals
