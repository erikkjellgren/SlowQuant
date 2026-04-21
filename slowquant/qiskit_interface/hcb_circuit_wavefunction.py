import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import pyscf
import scipy
from qiskit import QuantumCircuit
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
from slowquant.qiskit_interface.hcb_interface import HCBQuantumInterface
from slowquant.SlowQuant import SlowQuant
from slowquant.unitary_coupled_cluster.hardcoreboson_operator import HardcorebosonOperator
from slowquant.unitary_coupled_cluster.integral_manager import IntegralManager
from slowquant.unitary_coupled_cluster.operators import hamiltonian_hcb_0i_0a
from slowquant.unitary_coupled_cluster.optimizers import Optimizers


class WaveFunctionHCBCircuit:
    def __init__(
        self,
        cas: Sequence[int],
        mo_coeffs: np.ndarray,
        integral_generator: SlowQuant | pyscf.gto.mole.Mole,
        quantum_interface: HCBQuantumInterface,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize circuit based UPS wave function.

        Args:
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            mo_coeffs: Initial orbital coefficients.
            integral_generator: Integral generator object.
            quantum_interface: QuantumInterface.
            include_active_kappa: Include active-active orbital rotations.
        """
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        if isinstance(quantum_interface.ansatz, QuantumCircuit):
            print(
                "WARNING: A QI with a custom Ansatz was passed. VQE will only work with COBYLA and COBYQA optimizer."
            )
        self.int_gen = IntegralManager(integral_generator)
        self._c_mo = mo_coeffs
        self.inactive_idx = []
        self.virtual_idx = []
        self.active_idx = []
        self.active_occ_idx = []
        self.active_unocc_idx = []
        self.num_orbs = len(self.int_gen.kinetic_energy)
        self.num_active_elec_pair = 0
        self.num_active_orbs = 0
        self.num_inactive_orbs = 0
        self.num_virtual_orbs = 0
        self._rdm1 = None
        self._rdm2 = None
        self._hr1 = None
        self._hr2 = None
        self._energy_elec: float | None = None
        self.num_energy_evals = 0  # number of energy measurements on quanutm
        active_space = []
        orbital_counter = 0
        num_elec_pair = self.int_gen.num_elec // 2
        for i in range(num_elec_pair - cas[0] // 2, num_elec_pair):
            active_space.append(i)
            orbital_counter += 1
        for i in range(num_elec_pair, num_elec_pair + cas[1] - orbital_counter):
            active_space.append(i)
        for i in range(num_elec_pair):
            if i in active_space:
                self.active_idx.append(i)
                self.active_occ_idx.append(i)
                self.num_active_orbs += 1
                self.num_active_elec_pair += 1
            else:
                self.inactive_idx.append(i)
                self.num_inactive_orbs += 1
        for i in range(num_elec_pair, self.num_orbs):
            if i in active_space:
                self.active_idx.append(i)
                self.active_unocc_idx.append(i)
                self.num_active_orbs += 1
            else:
                self.virtual_idx.append(i)
                self.num_virtual_orbs += 1
        # Find non-redundant kappas
        self._kappa = []
        kappa_idx = []
        kappa_no_activeactive_idx = []
        kappa_no_activeactive_idx_dagger = []
        kappa_redundant_idx = []
        self._kappa_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    kappa_redundant_idx.append((p, q))
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    kappa_redundant_idx.append((p, q))
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        kappa_redundant_idx.append((p, q))
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    kappa_no_activeactive_idx.append((p, q))
                    kappa_no_activeactive_idx_dagger.append((q, p))
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
        self.kappa_idx = np.array(kappa_idx, dtype=int)
        self.kappa_no_activeactive_idx = np.array(kappa_no_activeactive_idx, dtype=int)
        self.kappa_no_activeactive_idx_dagger = np.array(kappa_no_activeactive_idx_dagger, dtype=int)
        self.kappa_redundant_idx = np.array(kappa_redundant_idx, dtype=int)
        self.kappa_hf_like_idx = np.array(kappa_hf_like_idx, dtype=int)
        # Setup Qiskit stuff
        self.QI = quantum_interface
        self.QI.construct_circuit(self.num_active_orbs, self.num_active_elec_pair)

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
        self._hr1 = None
        self._hr2 = None
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
    def hr1(self) -> np.ndarray:
        if self._hr1 is None:
            self._hr1 = np.zeros_like(self.int_gen.h_ao)
            h_mo = one_electron_integral_transform(self.c_mo, self.int_gen.h_ao)
            g_mo = two_electron_integral_transform(self.c_mo, self.int_gen.electron_electron_repulsion)
            for p in range(self.num_orbs):
                for q in range(self.num_orbs):
                    if p == q:
                        self._hr1[p, q] = 2 * h_mo[p, p] + g_mo[p, p, p, p]
                    else:
                        self._hr1[p, q] = g_mo[p, q, p, q]
        return self._hr1

    @property
    def hr2(self) -> np.ndarray:
        if self._hr2 is None:
            self._hr2 = np.zeros_like(self.int_gen.h_ao)
            g_mo = two_electron_integral_transform(self.c_mo, self.int_gen.electron_electron_repulsion)
            for p in range(self.num_orbs):
                for q in range(self.num_orbs):
                    if p != q:
                        self._hr2[p, q] = 2 * g_mo[p, p, q, q] - g_mo[p, q, p, q]
        return self._hr2

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
        self.QI.construct_circuit(self.num_active_orbs, self.num_active_elec_pair)

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
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    rdm1_op = Epq(p, q).get_folded_operator(
                        self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                    )
                    val = self.QI.quantum_expectation_value(rdm1_op)
                    self._rdm1[p_idx, q_idx] = val  # type: ignore [index]
                    self._rdm1[q_idx, p_idx] = val  # type: ignore [index]
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
                            pdm2_op = (Epq(p, q) * Epq(r, s)).get_folded_operator(
                                self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                            )
                            val = self.QI.quantum_expectation_value(pdm2_op)
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore [index]
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val  # type: ignore [index]
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val  # type: ignore [index]
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val  # type: ignore [index]
        return self._rdm2

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
        if rdm_order > 2:
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
        # Calling expectation value to put all Paulis in cliques
        # and compute distributions for the cliques.
        # The coefficients are set to one, so the Paulis cannot cancel out.
        _ = self.QI._sampler_quantum_expectation_value(
            SparsePauliOp(cumulated_paulis, np.ones(len(cumulated_paulis)))  # type: ignore[arg-type]
        )

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = one_electron_integral_transform(self.c_mo, overlap_integral)
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

    def _calc_energy_elec(self) -> float:
        """Run electronic energy simulation, regardless of self._energy_elec variable.

        Returns:
            Electronic energy.
        """
        H = hamiltonian_hcb_0i_0a(self.hr1, self.hr2, self.num_inactive_orbs, self.num_active_orbs)
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
        energy_elec = self.QI.quantum_expectation_value(H)
        return energy_elec

    def run_wf_optimization_2step(
        self,
        optimizer_name: str,
        orbital_optimization: bool = False,
        tol: float = 1e-10,
        maxiter: int = 1000,
        is_silent_subiterations: bool = False,
        print_std: bool = False,
    ) -> None:
        """Run two step optimization of wave function.

        Args:
            optimizer_name: Name of optimizer.
            orbital_optimization: Perform orbital optimization.
            tol: Convergence tolerance.
            maxiter: Maximum number of iterations.
            is_silent_subiterations: Silence subiterations.
            print_std: Print standard deviation of the electronic Hamiltonian during optimization.
        """
        if isinstance(self.QI.ansatz, QuantumCircuit) and optimizer_name.lower() not in ("cobyla", "cobyqa"):
            raise ValueError("Custom Ansatz in QI only works with COBYLA and COBYQA as optimizer.")
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa)}")
        print(f"### Number theta: {len(self.thetas)}")
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        for full_iter in range(0, int(maxiter)):
            full_start = time.time()

            # Do ansatz optimization
            if not is_silent_subiterations:
                print("--------Ansatz optimization")
                subheader = "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                if print_std:
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
                optimizer_name,
                grad=gradient_theta,
                maxiter=maxiter,
                tol=tol,
                is_silent=is_silent_subiterations,
                energy_eval_callback=lambda: self.num_energy_evals,
                std_callback=(
                    (
                        lambda: self.QI.quantum_variance(
                            hamiltonian_hcb_0i_0a(
                                self.hr1, self.hr2, self.num_inactive_orbs, self.num_active_orbs
                            ).get_folded_operator(
                                self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                            )
                        )
                    )
                    if print_std
                    else None
                ),
            )
            res = optimizer.minimize(
                self.thetas,
                extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names},
            )
            self.thetas = res.x.tolist()

            if orbital_optimization and len(self.kappa) != 0:
                if not is_silent_subiterations:
                    print("--------Orbital optimization")
                    subheader = "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                    if print_std:
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
                    "l-bfgs-b",
                    grad=gradient_oo,
                    maxiter=maxiter,
                    tol=tol,
                    is_silent=is_silent_subiterations,
                    energy_eval_callback=lambda: self.num_energy_evals,
                    std_callback=(
                        (
                            lambda: self.QI.quantum_variance(
                                hamiltonian_hcb_0i_0a(
                                    self.hr1, self.hr2, self.num_inactive_orbs, self.num_active_orbs
                                ).get_folded_operator(
                                    self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                                )
                            )
                        )
                        if print_std
                        else None
                    ),
                )
                res = optimizer.minimize([0.0] * len(self.kappa_idx))
                for i in range(len(self.kappa)):
                    self._kappa[i] = 0.0
                    self._kappa_old[i] = 0.0
            else:
                # If there is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa) == 0:
                    print(
                        "WARNING: No orbital optimization performed, because there is no non-redundant orbital parameters"
                    )
                break

            e_new = res.fun
            time_str = f"{time.time() - full_start:7.2f}"  # type: ignore
            e_str = f"{e_new:3.12f}"
            print(
                f"{str(full_iter + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)} | {str(self.num_energy_evals).center(11)}"
            )  # type: ignore
            if abs(e_new - e_old) < tol:
                break
            e_old = e_new
        self._energy_elec = e_new

    def run_wf_optimization_1step(
        self,
        optimizer_name: str,
        orbital_optimization: bool = False,
        tol: float = 1e-10,
        maxiter: int = 1000,
        print_std: bool = False,
    ) -> None:
        """Run one step optimization of wave function.

        Args:
            optimizer_name: Name of optimizer.
            orbital_optimization: Perform orbital optimization.
            tol: Convergence tolerance.
            maxiter: Maximum number of iterations.
            print_std: Print standard deviation in sub-iteration headers.
        """
        if isinstance(self.QI.ansatz, QuantumCircuit) and optimizer_name.lower() not in ("cobyla", "cobyqa"):
            raise ValueError("Custom Ansatz in QI only works with COBYLA and COBYQA as optimizer.")
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa)}")
        print(f"### Number theta: {len(self.thetas)}")
        if optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa) != 0:
                raise ValueError(
                    "Cannot use RotoSolve together with orbital optimization in the one-step solver."
                )

        header = (
            "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
        )
        if print_std:
            header += " | Std(H)"
        print(header)
        if orbital_optimization:
            if len(self.thetas) > 0:
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
        if orbital_optimization:
            if len(self.thetas) > 0:
                parameters = self.kappa + self.thetas
            else:
                parameters = self.kappa
        else:
            parameters = self.thetas
        optimizer = Optimizers(
            energy,
            optimizer_name,
            grad=gradient,
            maxiter=maxiter,
            tol=tol,
            energy_eval_callback=lambda: self.num_energy_evals,
            std_callback=(
                (
                    lambda: self.QI.quantum_variance(
                        hamiltonian_hcb_0i_0a(
                            self.hr1, self.hr2, self.num_inactive_orbs, self.num_active_orbs
                        ).get_folded_operator(
                            self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
                        )
                    )
                )
                if print_std
                else None
            ),
        )
        res = optimizer.minimize(
            parameters, extra_options={"R": self.QI.grad_param_R, "param_names": self.QI.param_names}
        )
        if orbital_optimization:
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
            H = hamiltonian_hcb_0i_0a(self.hr1, self.hr2, self.num_inactive_orbs, self.num_active_orbs)
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
            H = hamiltonian_hcb_0i_0a(self.hr1, self.hr2, self.num_inactive_orbs, self.num_active_orbs)
            H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)
            for i in range(len(parameters[num_kappa:])):
                R = self.QI.grad_param_R[self.QI.param_names[i]]
                e_vals_grad = _get_energy_evals_for_grad(H, self.QI, parameters, i, R)
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
    operator: HardcorebosonOperator,
    quantum_interface: HCBQuantumInterface,
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
