import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.linear_response.solvers import (
    get_orbital_rotation_gradient,
    get_orbital_metric_qq_block,
    one_index_transform,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import commutator, hamiltonian_0i_0a, hamiltonian_1i_1a, hamiltonian_2i_2a, double_commutator, one_elec_op_0i_0a
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
            grad = get_orbital_gradient_response(
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
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state([op], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = propagate_state([op.dagger], self.wf.ci_coeffs, *self.index_info)
            # <0 | H G |0>
            grad[i] = expectation_value(
                H00_ket,
                [],
                G_ket,
                *self.index_info,
            )
            # - <0| G H |0>
            grad[i] -= expectation_value(
                Gd_ket,
                [],
                H00_ket,
                *self.index_info,
            )
            # <0| Gd H |0>
            grad[i + len(self.G_ops)] = expectation_value(
                G_ket,
                [],
                H00_ket,
                *self.index_info,
            )
            # - <0| H Gd |0>
            grad[i + len(self.G_ops)] -= expectation_value(
                H00_ket,
                [],
                Gd_ket,
                *self.index_info,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))

    def _construct_hessian_metric_blocks(self) -> None:
        """Construct the Hessian and metric blocks for the Davidson solver."""
        idx_shift = len(self.q_ops)
        H00_ket = propagate_state([self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        if len(self.q_ops) != 0:
            # Do orbital-orbital blocks
            self.A[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_idx_dagger,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            self.B[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_idx_dagger,
                self.wf.kappa_no_activeactive_idx_dagger,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
            )
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                # <0| [GId, H, qJ] |0>
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [double_commutator(GI.dagger, self.H_1i_1a, qJ, do_symmetrized=True)],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                # <0| [GId, H, qJd] |0>
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [double_commutator(GI.dagger, self.H_1i_1a, qJ.dagger, do_symmetrized=True)],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val
        for j, GJ in enumerate(self.G_ops):
            GJH_ket = propagate_state([GJ], H00_ket, *self.index_info)
            GJdH_ket = propagate_state([GJ.dagger], H00_ket, *self.index_info)
            HGJd_ket = propagate_state([self.H_0i_0a, GJ.dagger], self.wf.ci_coeffs, *self.index_info)
            HGJ_ket = propagate_state([self.H_0i_0a, GJ], self.wf.ci_coeffs, *self.index_info)
            GJ_ket = propagate_state([GJ], self.wf.ci_coeffs, *self.index_info)
            GJd_ket = propagate_state([GJ.dagger], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                GId_ket = propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                # Make A
                # <0| GId H GJ |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    HGJ_ket,
                    *self.index_info,
                )
                # <0| GJ H GId |0>
                val += expectation_value(
                    HGJd_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                # - 1/2<0| GId GJ H |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        GI_ket,
                        [],
                        GJH_ket,
                        *self.index_info,
                    )
                )
                # - 1/2*<0| H GJ GId |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        GJdH_ket,
                        [],
                        GId_ket,
                        *self.index_info,
                    )
                )
                # - 1/2*<0| GJ GId H |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        GJd_ket,
                        [GI.dagger],
                        H00_ket,
                        *self.index_info,
                    )
                )
                # - 1/2*<0| H GId GJ |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        H00_ket,
                        [GI.dagger],
                        GJ_ket,
                        *self.index_info,
                    )
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                # <0| GId H GJd |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    HGJd_ket,
                    *self.index_info,
                )
                # - <0| GId GJd H |0>
                val -= expectation_value(
                    GI_ket,
                    [],
                    GJdH_ket,
                    *self.index_info,
                )
                # - <0| H GJd GId |0>
                val -= expectation_value(
                    GJH_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                # <0| GJd H GId |0>
                val += expectation_value(
                    HGJ_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                # <0| GId GJ |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    GJ_ket,
                    *self.index_info,
                )
                # - <0| GJ GId |0>
                val -= expectation_value(
                    GJd_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = val

    def _right_transform(self, trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        kappas = trial[:num_q, :]
        Ss = trial[num_q:, :]
        sigma_plus = np.zeros((num_ops, n_roots))
        sigma_minus = np.zeros((num_ops, n_roots))
        tau_minus = np.zeros((num_ops, n_roots))

        H00_ket = propagate_state([self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        if num_q != 0:
            K_plus = np.zeros((self.wf.num_orbs, self.wf.num_orbs, n_roots))
            K_minus = np.zeros((self.wf.num_orbs, self.wf.num_orbs, n_roots))
            for kappa, (q, p) in zip(kappas, self.wf.kappa_no_activeactive_idx):
                K_plus[p, q, :] = kappa
                K_plus[q, p, :] = kappa.conjugate()
                K_minus[p, q, :] = kappa
                K_minus[q, p, :] = - kappa.conjugate()

            for root in range(n_roots):
                h_plus, g_plus = one_index_transform(K_plus[:, :, root], self.wf.h_mo, self.wf.g_mo)
                tH_0i_0a = hamiltonian_0i_0a(
                    h_plus,
                    g_plus,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                )

                tH00_ket = propagate_state([tH_0i_0a], self.wf.ci_coeffs, *self.index_info)

                # (A+B)_qq @ b_q
                sigma_plus[:num_q, root] = get_orbital_rotation_gradient(
                    h_plus,
                    g_plus,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )

                qs = FermionicOperator({})
                for kappa, q in zip(kappas[:, root], self.q_ops):
                    qs += kappa * q

                # (A+B)_Gq @ b_q
                for i, GI in enumerate(self.G_ops):
                    sigma_plus[num_q + i, root] = expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger],
                        tH00_ket,
                        *self.index_info,
                    )
                    sigma_plus[num_q + i, root] += expectation_value(
                        tH00_ket,
                        [GI.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    qGd = commutator(qs, GI.dagger)
                    sigma_plus[num_q + i, root] += 0.5 * expectation_value(
                        self.wf.ci_coeffs,
                        [self.H_1i_1a * qGd],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    Gdqd = commutator(GI.dagger, qs.dagger)
                    sigma_plus[num_q + i, root] += 0.5 * expectation_value(
                        self.wf.ci_coeffs,
                        [Gdqd * self.H_1i_1a],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )

                Gs = FermionicOperator({})
                for S, G in zip(Ss[:, root], self.G_ops):
                    Gs += S * G + S.conjugate() * G.dagger
                Gs_ket = propagate_state([Gs], self.wf.ci_coeffs, *self.index_info)
                Gsd_ket = propagate_state([Gs.dagger], self.wf.ci_coeffs, *self.index_info)
                GsH11 = Gs * self.H_1i_1a

                # (A+B)_qG @ b_G
                for i, qi in enumerate(self.q_ops):
                    qidH11 = qi.dagger * self.H_1i_1a
                    sigma_plus[i, root] += expectation_value(
                        self.wf.ci_coeffs,
                        [qidH11],
                        Gs_ket,
                        *self.index_info,
                    )
                    sigma_plus[i, root] -= 0.5 * expectation_value(
                        Gsd_ket,
                        [qidH11],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    sigma_plus[i, root] -= 0.5 * expectation_value(
                        self.wf.ci_coeffs,
                        [qi.dagger * GsH11],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )

            for root in range(n_roots):
                h_minus, g_minus = one_index_transform(K_minus[:, :, root], self.wf.h_mo, self.wf.g_mo)
                tH_0i_0a = hamiltonian_0i_0a(
                    h_minus,
                    g_minus,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                )

                tH00_ket = propagate_state([tH_0i_0a], self.wf.ci_coeffs, *self.index_info)

                # (A-B)_qq @ b_q
                sigma_minus[:num_q, root] = get_orbital_rotation_gradient(
                    h_minus,
                    g_minus,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )

                qs = FermionicOperator({})
                for kappa, q in zip(kappas[:, root], self.q_ops):
                    qs += kappa * q

                # (A+B)_Gq @ b_q
                for i, GI in enumerate(self.G_ops):
                    sigma_minus[num_q + i, root] = expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger],
                        tH00_ket,
                        *self.index_info,
                    )
                    sigma_minus[num_q + i, root] -= expectation_value(
                        tH00_ket,
                        [GI.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    qGd = commutator(qs, GI.dagger)
                    sigma_minus[num_q + i, root] += 0.5 * expectation_value(
                        self.wf.ci_coeffs,
                        [self.H_1i_1a * qGd],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    Gdqd = commutator(GI.dagger, qs.dagger)
                    sigma_minus[num_q + i, root] -= 0.5 * expectation_value(
                        self.wf.ci_coeffs,
                        [Gdqd * self.H_1i_1a],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )

                Gs = FermionicOperator({})
                for S, G in zip(Ss[:, root], self.G_ops):
                    Gs += S * G - S.conjugate() * G.dagger
                Gs_ket = propagate_state([Gs], self.wf.ci_coeffs, *self.index_info)
                Gsd_ket = propagate_state([Gs.dagger], self.wf.ci_coeffs, *self.index_info)
                GsH11 = Gs * self.H_1i_1a

                # (A-B)_qG @ b_G
                for i, qi in enumerate(self.q_ops):
                    qidH11 = qi.dagger * self.H_1i_1a
                    sigma_minus[i, root] += expectation_value(
                        self.wf.ci_coeffs,
                        [qidH11],
                        Gs_ket,
                        *self.index_info,
                    )
                    sigma_minus[i, root] -= 0.5 * expectation_value(
                        Gsd_ket,
                        [qidH11],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    sigma_minus[i, root] -= 0.5 * expectation_value(
                        self.wf.ci_coeffs,
                        [qi.dagger * GsH11],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )

            for root in range(n_roots):
                tau_minus[:num_q, root] = get_orbital_metric_qq_block(
                    self.wf.kappa_no_activeactive_idx,
                    trial[:, root],
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                )

        for root in range(n_roots):

            Gs = FermionicOperator({})
            for S, G in zip(Ss[:, root], self.G_ops):
                Gs += S * G + S.conjugate() * G.dagger

            Gs_ket = propagate_state([Gs], self.wf.ci_coeffs, *self.index_info)
            Gsd_ket = propagate_state([Gs.dagger], self.wf.ci_coeffs, *self.index_info)
            GsH_ket = propagate_state([Gs], H00_ket, *self.index_info)
            GsdH_ket = propagate_state([Gs.dagger], H00_ket, *self.index_info)
            HGs_ket = propagate_state([self.H_0i_0a], Gs_ket, *self.index_info)
            HGsd_ket = propagate_state([self.H_0i_0a], Gsd_ket, *self.index_info)

            # (A+B)_GG @ b_G
            for i, GI in enumerate(self.G_ops):
                GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                GId_ket = propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                sigma_plus[num_q + i, root] += expectation_value(
                    GI_ket,
                    [],
                    HGs_ket,
                    *self.index_info,
                )
                sigma_plus[num_q + i, root] += expectation_value(
                    HGsd_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                sigma_plus[num_q + i, root] -= 0.5 * expectation_value(
                    GI_ket,
                    [],
                    GsH_ket,
                    *self.index_info,
                )
                sigma_plus[num_q + i, root] -= 0.5 * expectation_value(
                    GsdH_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                sigma_plus[num_q + i, root] -= 0.5 * expectation_value(
                    Gsd_ket,
                    [GI.dagger],
                    H00_ket,
                    *self.index_info,
                )
                sigma_plus[num_q + i, root] -= 0.5 * expectation_value(
                    H00_ket,
                    [GI.dagger],
                    Gs_ket,
                    *self.index_info,
                )

            Gs = FermionicOperator({})
            for S, G in zip(Ss[:, root], self.G_ops):
                Gs += S * G - S.conjugate() * G.dagger
            Gs_ket = propagate_state([Gs], self.wf.ci_coeffs, *self.index_info)
            Gsd_ket = propagate_state([Gs.dagger], self.wf.ci_coeffs, *self.index_info)
            GsH_ket = propagate_state([Gs], H00_ket, *self.index_info)
            GsdH_ket = propagate_state([Gs.dagger], H00_ket, *self.index_info)
            HGs_ket = propagate_state([self.H_0i_0a], Gs_ket, *self.index_info)
            HGsd_ket = propagate_state([self.H_0i_0a], Gsd_ket, *self.index_info)

            # (A-B)_GG @ b_G
            for i, GI in enumerate(self.G_ops):
                GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                GId_ket = propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                sigma_minus[num_q + i, root] += expectation_value(
                    GI_ket,
                    [],
                    HGs_ket,
                    *self.index_info,
                )
                sigma_minus[num_q + i, root] += expectation_value(
                    HGsd_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                sigma_minus[num_q + i, root] -= 0.5 * expectation_value(
                    GI_ket,
                    [],
                    GsH_ket,
                    *self.index_info,
                )
                sigma_minus[num_q + i, root] -= 0.5 * expectation_value(
                    GsdH_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                sigma_minus[num_q + i, root] -= 0.5 * expectation_value(
                    Gsd_ket,
                    [GI.dagger],
                    H00_ket,
                    *self.index_info,
                )
                sigma_minus[num_q + i, root] -= 0.5 * expectation_value(
                    H00_ket,
                    [GI.dagger],
                    Gs_ket,
                    *self.index_info,
                )

        for root in range(n_roots):
            Gs = FermionicOperator({})
            for S, G in zip(Ss[:, root], self.G_ops):
                Gs += S * G
            Gs_ket = propagate_state([Gs], self.wf.ci_coeffs, *self.index_info)
            Gsd_ket = propagate_state([Gs.dagger], self.wf.ci_coeffs, *self.index_info)
            # Sigma_GG @ b_G
            for i, GI in enumerate(self.G_ops):
                tau_minus[num_q + i, root] = expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger],
                    Gs_ket,
                    *self.index_info,
                )
                tau_minus[num_q + i, root] -= expectation_value(
                    Gsd_ket,
                    [GI.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )

        return sigma_plus, sigma_minus, tau_minus



    def get_transition_dipole(self) -> np.ndarray:
        """Calculate transition dipole moment.

        Returns:
            Transition dipole moment.
        """
        number_excitations = len(self.excitation_energies)
        dipole_integrals = self.wf.int_gen.electric_dipole
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
        transition_dipole_x = 0.0
        transition_dipole_y = 0.0
        transition_dipole_z = 0.0
        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            transfer_op = FermionicOperator({})
            for i, G in enumerate(self.G_ops):
                transfer_op += (
                    self.Z_G_normed[i, state_number] * G.dagger + self.Y_G_normed[i, state_number] * G
                )
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
                    number_excitations,
                )
                q_part_y = get_orbital_response_property_gradient(
                    muy,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
                q_part_z = get_orbital_response_property_gradient(
                    muz,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
            transfer_ket = propagate_state([transfer_op], self.wf.ci_coeffs, *self.index_info)
            transferd_ket = propagate_state([transfer_op.dagger], self.wf.ci_coeffs, *self.index_info)
            # <0| mux T |0>
            transition_dipole_x = expectation_value(
                muxd_ket,
                [],
                transfer_ket,
                *self.index_info,
            )
            # - <0| T mux |0>
            transition_dipole_x -= expectation_value(
                transferd_ket,
                [],
                mux_ket,
                *self.index_info,
            )
            # <0| muy T |0>
            transition_dipole_y = expectation_value(
                muyd_ket,
                [],
                transfer_ket,
                *self.index_info,
            )
            # - <0| T muy |0>
            transition_dipole_y -= expectation_value(
                transferd_ket,
                [],
                muy_ket,
                *self.index_info,
            )
            # <0| muz T |0>
            transition_dipole_z = expectation_value(
                muzd_ket,
                [],
                transfer_ket,
                *self.index_info,
            )
            # - <0| T muz |0>
            transition_dipole_z -= expectation_value(
                transferd_ket,
                [],
                muz_ket,
                *self.index_info,
            )
            transition_dipoles[state_number, 0] = q_part_x + transition_dipole_x
            transition_dipoles[state_number, 1] = q_part_y + transition_dipole_y
            transition_dipoles[state_number, 2] = q_part_z + transition_dipole_z
        return transition_dipoles
