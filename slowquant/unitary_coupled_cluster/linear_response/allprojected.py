import time
import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient_response,
    get_orbital_response_property_gradient,
    get_orbital_response_metric_sigma_right_transformed,
    get_orbital_hessian_diagonal,
    get_orbital_metric_diagonal,
    get_orbital_gradient_response_right_transformed,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.linear_response.solvers import (
    one_index_transform
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import (
    commutator,
    hamiltonian_0i_0a,
    hamiltonian_2i_2a,
    one_elec_op_0i_0a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


class LinearResponse(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: str,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        super().__init__(wave_function, excitations)

        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        if len(self.q_ops) != 0:
            grad = get_orbital_gradient_response(  # proj-q and naive-q lead to same working equations
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        H00_ket = propagate_state([self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        self._G_expect = np.zeros(len(self.G_ops))
        self._HG_expect = np.zeros(len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state([op], self.wf.ci_coeffs, *self.index_info)
            self._G_expect[i] = expectation_value(
                self.wf.ci_coeffs,
                [],
                G_ket,
                *self.index_info,
            )
            self._HG_expect[i] = expectation_value(
                H00_ket,
                [],
                G_ket,
                *self.index_info,
            )
            # <0| H G |0>
            grad[i] = self._HG_expect[i]
            # - E * <0| G |0>
            grad[i] -= self.wf.energy_elec * self._G_expect[i]
            # <0| Gd H |0>
            grad[i + len(self.G_ops)] = self._HG_expect[i]
            # - E * <0| Gd |0>
            grad[i + len(self.G_ops)] -= self.wf.energy_elec * self._G_expect[i]
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))

    def _construct_hessian_metric_blocks(self) -> None:
        H_2i_2a = hamiltonian_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )
        idx_shift = len(self.q_ops)

        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                # <0| [qId, H*qJ] |0> = <0| qId H qJ |0>, commutator implementation is faster.
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [commutator(qI.dagger, H_2i_2a * qJ)],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # - <0| [qId, qJ] |0> * E = - <0| qId qJ |0> * E, commutator implementation is faster.
                tmp = expectation_value(
                    self.wf.ci_coeffs,
                    [commutator(qI.dagger, qJ)],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                val -= tmp * self.wf.energy_elec
                self.A[i, j] = self.A[j, i] = val
                # Make Sigma
                # <0| [qId, qJ] |0> = <0| qId qJ |0>, commutator implementation is faster.
                self.Sigma[i, j] = self.Sigma[j, i] = tmp
        for j, qJ in enumerate(self.q_ops):
            Hq_ket = propagate_state([commutator(self.H_1i_1a, qJ)], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops):
                # Make A
                # <0| Gd [H, q] |0> = <0| Gd H q |0>, commutator implementation is faster.
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger],
                    Hq_ket,
                    *self.index_info,
                )
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = val
        for j, GJ in enumerate(self.G_ops):
            GJ_ket = propagate_state([GJ], self.wf.ci_coeffs, *self.index_info)
            HGJ_ket = propagate_state([self.H_0i_0a], GJ_ket, *self.index_info)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                # Make A
                # <0| GId H GJ |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    HGJ_ket,
                    *self.index_info,
                )
                # <0 | GId |0> * <0| GJ |0> * E
                val += self._G_expect[i] * self._G_expect[j] * self.wf.energy_elec
                # - <0| GId GJ |0> * E
                val -= (
                    expectation_value(
                        GI_ket,
                        [],
                        GJ_ket,
                        *self.index_info,
                    )
                    * self.wf.energy_elec
                )
                # - 1/2*<0| GId |0> * <0| H GJ |0>
                val -= 0.5 * self._G_expect[i] * self._HG_expect[j]
                # - 1/2*<0| GJ |0> * <0| GId H |0>
                val -= 0.5 * self._G_expect[j] * self._HG_expect[i]
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                # 1/2<0| GId H |0> * <0| GJd |0>
                val = 0.5 * self._HG_expect[i] * self._G_expect[j]
                # 1/2<0| GJd H |0> * <0| GId |0>
                val += 0.5 * self._HG_expect[j] * self._G_expect[i]
                # - <0| GId |0> * <0| GJd |0> * E
                val -= self._G_expect[i] * self._G_expect[j] * self.wf.energy_elec
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                # <0| GId GJ |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    GJ_ket,
                    *self.index_info,
                )
                # - <0| GId |0> * <0| GJ |0>
                val -= self._G_expect[i] * self._G_expect[j]
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = val

    def _right_transform(self, trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Right transform for Davidson solver.

        Args:
            trial: Trial vectors.

        Returns:
            sigma_plus, sigma_minus, tau_minus as defined in the Davidson solver.
        """
        num_q = len(self.q_ops)
        num_G = len(self.G_ops)
        num_ops = num_q + num_G
        n_roots = trial.shape[1]
        kappas_plus = trial[:num_q, :]
        Ss_plus = trial[num_q:num_ops, :]
        kappas_minus = trial[num_ops:num_ops+num_q, :]
        Ss_minus = trial[num_ops+num_q:, :]
        sigma_plus = np.zeros((num_ops, n_roots))
        sigma_minus = np.zeros((num_ops, n_roots))
        tau_plus = np.zeros((num_ops, n_roots))
        tau_minus = np.zeros((num_ops, n_roots))

        if num_q != 0:
            K_lowerp = np.zeros((self.wf.num_orbs, self.wf.num_orbs, n_roots))
            K_lowerm = np.zeros((self.wf.num_orbs, self.wf.num_orbs, n_roots))
            for kappa, (q, p) in zip(kappas_plus, self.wf.kappa_no_activeactive_idx):
                K_lowerp[p, q, :] = kappa
            for kappa, (q, p) in zip(kappas_minus, self.wf.kappa_no_activeactive_idx):
                K_lowerm[p, q, :] = kappa
            for root in range(n_roots):
                h_lowerp, g_lowerp = one_index_transform(K_lowerp[:, :, root], self.wf.h_mo, self.wf.g_mo)
                h_lowerm, g_lowerm = one_index_transform(K_lowerm[:, :, root], self.wf.h_mo, self.wf.g_mo)
                tH00p_lower = hamiltonian_0i_0a(
                    h_lowerp,
                    g_lowerp,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                )
                tH00m_lower = hamiltonian_0i_0a(
                    h_lowerm,
                    g_lowerm,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                )
                tH00lp_ket = propagate_state([tH00p_lower], self.wf.ci_coeffs, *self.index_info)
                tH00lm_ket = propagate_state([tH00m_lower], self.wf.ci_coeffs, *self.index_info)

                qsp = FermionicOperator({})
                for kappa, q in zip(kappas_plus[:, root], self.q_ops):
                    qsp += kappa * q
                qsm = FermionicOperator({})
                for kappa, q in zip(kappas_minus[:, root], self.q_ops):
                    qsm += kappa * q

                # (A+B)_qq @ b_q
                # (A-B)_qq @ b_q
                # Sigma_qq @ b_q
                sigma_plus[:num_q, root] += get_orbital_gradient_response_right_transformed(
                    h_lowerp,
                    g_lowerp,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )
                sigma_minus[:num_q, root] += get_orbital_gradient_response_right_transformed(
                    h_lowerm,
                    g_lowerm,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )
                for i, qi in enumerate(self.q_ops):
                    sigma_plus[i, root] += expectation_value(
                            self.wf.ci_coeffs,
                            [commutator(qi.dagger, qsp) * self.H_1i_1a],
                            self.wf.ci_coeffs,
                            *self.index_info,
                    )
                    sigma_minus[i, root] += expectation_value(
                            self.wf.ci_coeffs,
                            [commutator(qi.dagger, qsm) * self.H_1i_1a],
                            self.wf.ci_coeffs,
                            *self.index_info,
                    )
                # <0| [qid, qs] |0>
                val = get_orbital_response_metric_sigma_right_transformed(
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.kappa_no_activeactive_idx,
                    trial[:num_ops, root],
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                )
                sigma_plus[:num_q, root] -= self.wf.energy_elec * val
                tau_minus[:num_q, root] += val
                val = get_orbital_response_metric_sigma_right_transformed(
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.kappa_no_activeactive_idx,
                    trial[num_ops:, root],
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                )
                sigma_minus[:num_q, root] -= self.wf.energy_elec * val
                tau_plus[:num_q, root] += val

                # (A+B)_Gq @ b_q
                # (A-B)_Gq @ b_q
                for i, GI in enumerate(self.G_ops):
                    # <0| GId H qs |0>
                    sigma_plus[num_q + i, root] += expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger],
                        tH00lp_ket,
                        *self.index_info,
                    )
                    sigma_minus[num_q + i, root] += expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger],
                        tH00lm_ket,
                        *self.index_info,
                    )

                Gsp = FermionicOperator({})
                for S, G in zip(Ss_plus[:, root], self.G_ops):
                    Gsp += S * G
                Gsp_ket = propagate_state([Gsp], self.wf.ci_coeffs, *self.index_info)
                Gsm = FermionicOperator({})
                for S, G in zip(Ss_minus[:, root], self.G_ops):
                    Gsm += S * G
                Gsm_ket = propagate_state([Gsm], self.wf.ci_coeffs, *self.index_info)

                # (A+B)_qG @ b_G
                # (A-B)_qG @ b_G
                for i, qi in enumerate(self.q_ops):
                    # <0| Gsd H qi |0>
                    sigma_plus[i, root] += expectation_value(
                        Gsp_ket,
                        [commutator(self.H_1i_1a, qi)],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    sigma_minus[i, root] += expectation_value(
                        Gsm_ket,
                        [commutator(self.H_1i_1a, qi)],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )

        for root in range(n_roots):
            Gsp = FermionicOperator({})
            for S, G in zip(Ss_plus[:, root], self.G_ops):
                Gsp += S * G
            Gsp_expect = sum(Ss_plus[:, root] * self._G_expect)
            HGsp_expect = sum(Ss_plus[:, root] * self._HG_expect)
            Gsm = FermionicOperator({})
            for S, G in zip(Ss_minus[:, root], self.G_ops):
                Gsm += S * G
            Gsm_expect = sum(Ss_minus[:, root] * self._G_expect)
            HGsm_expect = sum(Ss_minus[:, root] * self._HG_expect)

            # Gsp |0>
            Gsp_ket = propagate_state([Gsp], self.wf.ci_coeffs, *self.index_info)
            # ( H - E ) Gsp |0>
            HGsp_ket = propagate_state([self.H_0i_0a], Gsp_ket, *self.index_info) - self.wf.energy_elec * Gsp_ket

            # Gsm |0>
            Gsm_ket = propagate_state([Gsm], self.wf.ci_coeffs, *self.index_info)
            # ( H - E ) Gsm |0>
            HGsm_ket = propagate_state([self.H_0i_0a], Gsm_ket, *self.index_info) - self.wf.energy_elec * Gsm_ket

            # (A+B)_GG @ b_G
            # (A-B)_GG @ b_G
            # Sigma_GG @ b_G
            for i, GI in enumerate(self.G_ops):
                # Gi |0>
                GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)

                # <0| GId ( H - E ) Gs |0>
                sigma_plus[num_q + i, root] += expectation_value(
                    GI_ket,
                    [],
                    HGsp_ket,
                    *self.index_info,
                )
                sigma_minus[num_q + i, root] += expectation_value(
                    GI_ket,
                    [],
                    HGsm_ket,
                    *self.index_info,
                )
                # ( 1 - h ) E <0| GId |0> <0| Gs |0>
                sigma_plus[num_q + i, root] += self.wf.energy_elec * self._G_expect[i] * (Gsp_expect - Gsp_expect.conjugate())
                sigma_minus[num_q + i, root] += self.wf.energy_elec * self._G_expect[i] * (Gsm_expect + Gsm_expect.conjugate())
                # - 0.5 ( 1 - h ) <0| GId |0> <0| H Gs |0>
                sigma_plus[num_q + i, root] -= 0.5 * self._G_expect[i] * (HGsp_expect - HGsp_expect.conjugate())
                sigma_minus[num_q + i, root] -= 0.5 * self._G_expect[i] * (HGsm_expect + HGsm_expect.conjugate())
                # - 0.5 ( 1 - h ) <0| Gs |0> <0| GId H |0>
                sigma_plus[num_q + i, root] -= 0.5 * (Gsp_expect - Gsp_expect.conjugate()) * self._HG_expect[i]
                sigma_minus[num_q + i, root] -= 0.5 * (Gsm_expect + Gsm_expect.conjugate()) * self._HG_expect[i]
                # <0| GId Gs |0>
                tau_plus[num_q + i, root] += expectation_value(
                    GI_ket,
                    [],
                    Gsm_ket,
                    *self.index_info,
                )
                tau_minus[num_q + i, root] += expectation_value(
                    GI_ket,
                    [],
                    Gsp_ket,
                    *self.index_info,
                )
                # - <0| GId |0> <0| Gs |0>
                tau_plus[num_q + i, root] -= self._G_expect[i] * Gsm_expect
                tau_minus[num_q + i, root] -= self._G_expect[i] * Gsp_expect

        return sigma_plus, sigma_minus, tau_plus, tau_minus

    def _compute_preconditioner(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the preconditioner for the Davidson solver.

        Returns:
            prec_A, prec_sigma: Preconditioner for A and Sigma blocks.
        """
        num_q = len(self.q_ops)
        num_G = len(self.G_ops)
        prec_A = np.zeros(num_q + num_G)
        prec_sigma = np.zeros(num_q + num_G)

        F_0i_0a = one_elec_op_0i_0a(
            self.wf.F_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )

        if len(self.q_ops) != 0:
            # Approximate q diagonal
            prec_A[:num_q] = get_orbital_hessian_diagonal(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            # Exact q diagonal
            prec_sigma[:num_q] = get_orbital_metric_diagonal(
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
            )
        for i, GI in enumerate(self.G_ops):
            GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
            # Approximate G diagonal
            prec_A[i + num_q] += expectation_value(
                GI_ket,
                [F_0i_0a],
                GI_ket,
                *self.index_info,
            )
            GIGI_expect = expectation_value(
                GI_ket,
                [],
                GI_ket,
                *self.index_info,
            )
            prec_A[i + num_q] -= self.wf.energy_elec * GIGI_expect
            prec_A[i + num_q] += self.wf.energy_elec * self._G_expect[i]**2
            prec_A[i + num_q] -= self._G_expect[i] * self._HG_expect[i]
            # Exact G diagonal
            prec_sigma[i + num_q] += GIGI_expect
            prec_sigma[i + num_q] -= self._G_expect[i]**2

        return prec_A, prec_sigma

    def property_gradient(self, integral: np.ndarray) -> np.ndarray:
        """Calculate top half of the property gradient.
        Bottom half can be found through conjugation and sign change.

        Args:
            integral: MO integral for which to calculate the gradient.

        Returns:
            Gradient of property.
        """
        num_q = len(self.q_ops)
        num_G = len(self.G_ops)
        num_ops = num_q + num_G
        gradient = np.zeros((num_ops))
        if num_q != 0:
            gradient[:num_q] = get_orbital_gradient_response_right_transformed(
                integral,
                None,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
        op = one_elec_op_0i_0a(
            integral,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        op_ket = propagate_state([op], self.wf.ci_coeffs, *self.index_info)
        for i, GI in enumerate(self.G_ops):
            GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
            gradient[num_q + i] = expectation_value(
                GI_ket,
                [],
                op_ket,
                *self.index_info,
            )
            gradient[num_q + i] -= self._G_expect[i] * expectation_value(
                self.wf.ci_coeffs,
                [],
                op_ket,
                *self.index_info,
            )
        return gradient.reshape(-1, 1)

    def get_transition_dipole(self, dipole_integrals: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Calculate transition dipole moment.

        Returns:
            Transition dipole moment.
        """
        number_excitations = len(self.excitation_energies)
        num_ops = len(self.q_ops) + len(self.G_ops)
        mux = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[2])
        mux_op = one_elec_op_0i_0a(
            mux,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        muy_op = one_elec_op_0i_0a(
            muy,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        muz_op = one_elec_op_0i_0a(
            muz,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        mux_ket = propagate_state([mux_op], self.wf.ci_coeffs, *self.index_info)
        muxd_ket = propagate_state([mux_op.dagger], self.wf.ci_coeffs, *self.index_info)
        muy_ket = propagate_state([muy_op], self.wf.ci_coeffs, *self.index_info)
        muyd_ket = propagate_state([muy_op.dagger], self.wf.ci_coeffs, *self.index_info)
        muz_ket = propagate_state([muz_op], self.wf.ci_coeffs, *self.index_info)
        muzd_ket = propagate_state([muz_op.dagger], self.wf.ci_coeffs, *self.index_info)
        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            q_part_x = 0.0
            q_part_y = 0.0
            q_part_z = 0.0
            if len(self.q_ops) != 0:
                q_part_x = get_orbital_response_property_gradient(
                    mux,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    num_ops,
                )
                q_part_y = get_orbital_response_property_gradient(
                    muy,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    num_ops,
                )
                q_part_z = get_orbital_response_property_gradient(
                    muz,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    num_ops,
                )
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            for i, G in enumerate(self.G_ops):
                G_ket = propagate_state([G], self.wf.ci_coeffs, *self.index_info)
                # <0| G |0>
                exp_G = exp_G_dagger = self._G_expect[i]
                # Z * <0| Gd |0> * <0| mux |0>
                g_part_x += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        mux_ket,
                        *self.index_info,
                    )
                )
                # - Z * <0| Gd mux |0>
                g_part_x -= self.Z_G_normed[i, state_number] * expectation_value(
                    G_ket,
                    [],
                    mux_ket,
                    *self.index_info,
                )
                # - Y * <0| G |0> * <0| mux |0>
                g_part_x -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        mux_ket,
                        *self.index_info,
                    )
                )
                # Y * <0| mux G |0>
                g_part_x += self.Y_G_normed[i, state_number] * expectation_value(
                    muxd_ket,
                    [],
                    G_ket,
                    *self.index_info,
                )
                # Z * <0| Gd |0> * <0| muy |0>
                g_part_y += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        muy_ket,
                        *self.index_info,
                    )
                )
                # - Z * <0| Gd muy |0>
                g_part_y -= self.Z_G_normed[i, state_number] * expectation_value(
                    G_ket,
                    [],
                    muy_ket,
                    *self.index_info,
                )
                # - Y * <0| G |0> * <0| muy |0>
                g_part_y -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        muy_ket,
                        *self.index_info,
                    )
                )
                # Y * <0| muy G |0>
                g_part_y += self.Y_G_normed[i, state_number] * expectation_value(
                    muyd_ket,
                    [],
                    G_ket,
                    *self.index_info,
                )
                # Z * <0| Gd |0> * <0| muz |0>
                g_part_z += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        muz_ket,
                        *self.index_info,
                    )
                )
                # - Z * <0| Gd muz |0>
                g_part_z -= self.Z_G_normed[i, state_number] * expectation_value(
                    G_ket,
                    [],
                    muz_ket,
                    *self.index_info,
                )
                # - Y * <0| G |0> * <0| muz |0>
                g_part_z -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        muz_ket,
                        *self.index_info,
                    )
                )
                # Y * <0| muz G |0>
                g_part_z += self.Y_G_normed[i, state_number] * expectation_value(
                    muzd_ket,
                    [],
                    G_ket,
                    *self.index_info,
                )
            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
