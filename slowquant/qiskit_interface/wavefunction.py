import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import scipy
from qiskit.primitives import Estimator
from qiskit.primitives import Estimator as EstimatorNoNoise
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA
from qiskit_ibm_runtime import Estimator
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.qiskit_interface.operators import Epq, hamiltonian_full_space


class WaveFunction:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        qiskit_service,
        qiskit_options,
        qiskit_backend,
        qiskit_session,
        qiskit_mapper: str = "parity",
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UCC wave function.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            c_orthonormal: Initial orbital coefficients.
            h_ao: One-electron integrals in AO for Hamiltonian.
            g_ao: Two-electron integrals in AO.
            include_active_kappa: Include active-active orbital rotations.
        """
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        self._c_orthonormal = c_orthonormal
        self.h_ao = h_ao
        self.g_ao = g_ao
        self.inactive_spin_idx = []
        self.virtual_spin_idx = []
        self.active_spin_idx = []
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        self.active_spin_idx_shifted = []
        self.active_occ_spin_idx_shifted = []
        self.active_unocc_spin_idx_shifted = []
        self.num_elec = num_elec
        self.num_spin_orbs = num_spin_orbs
        self.num_orbs = num_spin_orbs // 2
        self._include_active_kappa = include_active_kappa
        self.num_active_elec = 0
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
        self.num_virtual_spin_orbs = 0
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._h_mo = None
        self._g_mo = None
        active_space = []
        orbital_counter = 0
        for i in range(num_elec - cas[0], num_elec):
            active_space.append(i)
            orbital_counter += 1
        for i in range(num_elec, num_elec + 2 * cas[1] - orbital_counter):
            active_space.append(i)
        for i in range(num_elec):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_occ_spin_idx.append(i)
                self.num_active_spin_orbs += 1
                self.num_active_elec += 1
            else:
                self.inactive_spin_idx.append(i)
                self.num_inactive_spin_orbs += 1
        for i in range(num_elec, num_spin_orbs):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_unocc_spin_idx.append(i)
                self.num_active_spin_orbs += 1
            else:
                self.virtual_spin_idx.append(i)
                self.num_virtual_spin_orbs += 1
        if len(self.active_spin_idx) != 0:
            active_shift = np.min(self.active_spin_idx)
            for active_idx in self.active_spin_idx:
                self.active_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_spin_idx:
                self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_spin_idx:
                self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)
        self.num_inactive_orbs = self.num_inactive_spin_orbs // 2
        self.num_active_orbs = self.num_active_spin_orbs // 2
        self.num_virtual_orbs = self.num_virtual_spin_orbs // 2
        # Contruct spatial idx
        self.inactive_idx: list[int] = []
        self.virtual_idx: list[int] = []
        self.active_idx: list[int] = []
        self.active_occ_idx: list[int] = []
        self.active_unocc_idx: list[int] = []
        for idx in self.inactive_spin_idx:
            if idx // 2 not in self.inactive_idx:
                self.inactive_idx.append(idx // 2)
        for idx in self.active_spin_idx:
            if idx // 2 not in self.active_idx:
                self.active_idx.append(idx // 2)
        for idx in self.virtual_spin_idx:
            if idx // 2 not in self.virtual_idx:
                self.virtual_idx.append(idx // 2)
        for idx in self.active_occ_spin_idx:
            if idx // 2 not in self.active_occ_idx:
                self.active_occ_idx.append(idx // 2)
        for idx in self.active_unocc_spin_idx:
            if idx // 2 not in self.active_unocc_idx:
                self.active_unocc_idx.append(idx // 2)
        # Find non-redundant kappas
        self.kappa = []
        self.kappa_idx = []
        self.kappa_idx_dagger = []
        self.kappa_redundant = []
        self.kappa_redundant_idx = []
        self._kappa_old = []
        self._kappa_redundant_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self.kappa_redundant.append(0.0)
                    self._kappa_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self.kappa_redundant.append(0.0)
                    self._kappa_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self.kappa_redundant.append(0.0)
                        self._kappa_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                if include_active_kappa:
                    if p in self.active_occ_idx and q in self.active_occ_idx:
                        self.kappa_redundant.append(0.0)
                        self._kappa_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                    if p in self.active_unocc_idx and q in self.active_unocc_idx:
                        self.kappa_redundant.append(0.0)
                        self._kappa_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                self.kappa.append(0.0)
                self._kappa_old.append(0.0)
                self.kappa_idx.append([p, q])
                self.kappa_idx_dagger.append([q, p])
        # HF like orbital rotation indecies
        self.kappa_hf_like_idx = []
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append([p, q])
                elif p in self.inactive_idx and q in self.active_unocc_idx:
                    self.kappa_hf_like_idx.append([p, q])
                elif p in self.active_occ_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append([p, q])
        # Setup Qiskit stuff
        self.qiskit_options = qiskit_options
        self.qiskit_service = qiskit_service
        self.qiskit_backend = qiskit_backend.lower()
        self.qiskit_session = qiskit_session
        if qiskit_mapper.lower() == "parity":
            self.qiskit_mapper = ParityMapper(
                num_particles=(self.num_active_elec // 2, self.num_active_elec // 2)
            )
        elif qiskit_mapper.lower() in ("jordanwigner", "jw"):
            self.qiskit_mapper = JordanWignerMapper()
        else:
            raise ValueError(f"Unknown mapper; {qiskit_mapper}")
        self.qiskit_ansatz = UCCSD(
            self.num_active_orbs,
            (self.num_active_elec // 2, self.num_active_elec // 2),
            self.qiskit_mapper,
            initial_state=HartreeFock(
                self.num_active_orbs,
                (self.num_active_elec // 2, self.num_active_elec // 2),
                self.qiskit_mapper,
            ),
        )
        self._ansatz_parameters = [0.0] * self.qiskit_ansatz.num_parameters
        if self.qiskit_backend == "noiseless":
            self.qiskit_estimator = EstimatorNoNoise()
        else:
            self.qiskit_estimator = Estimator(
                backend=self.qiskit_backend, options=self.qiskit_options, session=self.qiskit_session
            )

    @property
    def c_orthonormal(self) -> np.ndarray:
        """Get orthonormalization coefficients (MO coefficients).

        Returns:
            Orthonormalization coefficients.
        """
        return self._c_orthonormal

    @c_orthonormal.setter
    def c_orthonormal(self, c: np.ndarray) -> None:
        """Set orthonormalization coefficients.

        Args:
            c: Orthonormalization coefficients.
        """
        self._h_mo = None
        self._g_mo = None
        self._c_orthonormal = c

    @property
    def c_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        kappa_mat = np.zeros_like(self._c_orthonormal)
        if len(self.kappa) != 0:
            if np.max(np.abs(self.kappa)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        if len(self.kappa_redundant) != 0:
            if np.max(np.abs(self.kappa_redundant)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa_redundant, self.kappa_redundant_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        return np.matmul(self._c_orthonormal, scipy.linalg.expm(-kappa_mat))

    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = one_electron_integral_transform(self.c_trans, self.h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_trans, self.g_ao)
        return self._g_mo

    @property
    def ansatz_parameters(self) -> list[float]:
        return self._ansatz_parameters

    @ansatz_parameters.setter
    def ansatz_parameters(self, parameters: list[float]) -> None:
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._ansatz_parameters = parameters.copy()

    @property
    def rdm1(self) -> np.ndarray:
        """Calcuate one-electron reduced density matrix.

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    rdm1_op = (
                        Epq(p, q)
                        .get_folded_operator(
                            self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                        )
                        .get_qiskit_form(self.num_active_orbs)
                    )
                    job = self.qiskit_estimator.run(
                        circuits=self.qiskit_ansatz,
                        parameter_values=self.ansatz_parameters,
                        observables=self.qiskit_mapper.map(FermionicOp(rdm1_op, 2 * self.num_active_orbs)),
                    )
                    val = job.result().values[0]
                    self._rdm1[p_idx, q_idx] = val
                    self._rdm1[q_idx, p_idx] = val
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        """Calcuate two-electron reduced density matrix.

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
                )
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_idx = r - self.num_inactive_orbs
                        if p == q:
                            s_lim = r + 1
                        elif p == r:
                            s_lim = q + 1
                        elif q < r:
                            s_lim = p
                        else:
                            s_lim = p + 1
                        for s in range(self.num_inactive_orbs, s_lim):
                            s_idx = s - self.num_inactive_orbs
                            pdm2_op = (
                                (Epq(p, q) * Epq(r, s))
                                .get_folded_operator(
                                    self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                                )
                                .get_qiskit_form(self.num_active_orbs)
                            )
                            job = self.qiskit_estimator.run(
                                circuits=self.qiskit_ansatz,
                                parameter_values=self.ansatz_parameters,
                                observables=self.qiskit_mapper.map(
                                    FermionicOp(pdm2_op, 2 * self.num_active_orbs)
                                ),
                            )
                            val = job.result().values[0]
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val
        return self._rdm2

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            I = C_\text{MO}S\C_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = one_electron_integral_transform(self.c_trans, overlap_integral)
        one = np.identity(len(S_ortho))
        diff = np.abs(S_ortho - one)
        print("Max ortho-normal diff:", np.max(diff))

    def run_vqe(
        self,
    ) -> None:
        """Run VQE of wave function."""
        H = hamiltonian_full_space(self.h_mo, self.g_mo, self.num_orbs)
        energy = partial(
            calc_energy,
            operator=(
                H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            ).get_qiskit_form(self.num_active_orbs),
            ansatz=self.qiskit_ansatz,
            mapper=self.qiskit_mapper,
            reg=2 * self.num_active_orbs,
            estimator=self.qiskit_estimator,
        )

        global iteration
        global start
        iteration = 0  # type: ignore
        start = time.time()  # type: ignore

        def print_progress(x) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration
            global start
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{energy(x):3.12f}"
            print(f"{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            start = time.time()  # type: ignore

        def print_progress_SPSA(___, theta, f_val, _, __) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration
            global start
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{f_val:3.12f}"
            print(f"{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore
            start = time.time()  # type: ignore

        optimizer = SLSQP(callback=print_progress)
        res = optimizer.minimize(energy, self.ansatz_parameters)
        self.ansatz_parameters = res.x.tolist()


def calc_energy(parameters, operator, ansatz, mapper, reg, estimator):
    job = estimator.run(
        circuits=ansatz, parameter_values=parameters, observables=mapper.map(FermionicOp(operator, reg))
    )
    result = job.result()
    return result.values[0]
