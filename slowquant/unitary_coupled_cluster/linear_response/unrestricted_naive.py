from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.unrestricted_lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import a_op
from slowquant.unitary_coupled_cluster.unrestricted_density_matrix import (
    get_orbital_gradient_response_unrestricted,
    get_orbital_response_hessian_block_unrestricted,
    get_orbital_response_metric_sigma_unrestricted,
    get_orbital_response_property_gradient_unrestricted,
)
from slowquant.unitary_coupled_cluster.unrestricted_operators import (
    unrestricted_one_elec_op_full_space,
)
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS


class LinearResponseUPS(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: UnrestrictedWaveFunctionUPS,
        excitations: str,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        super().__init__(wave_function, excitations)

        # Screen for A_ii = 0
        finite_excitations_idx = []
        A = get_orbital_response_hessian_block_unrestricted(
            self.wf.haa_mo,
            self.wf.hbb_mo,
            self.wf.gaaaa_mo,
            self.wf.gbbbb_mo,
            self.wf.gaabb_mo,
            self.wf.gbbaa_mo,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.rdm1aa,
            self.wf.rdm1bb,
            self.wf.rdm2aaaa,
            self.wf.rdm2bbbb,
            self.wf.rdm2aabb,
            self.wf.rdm2bbaa,
        )
        # Man behøver ikke regne hele A, men det er bare lige nemt at gøre for qq
        for i, q in enumerate(self.q_ops):
            if abs(A[i, i]) > 10**-6:  # whatever rimeligt threshold
                finite_excitations_idx.append(True)
            else:
                finite_excitations_idx.append(False)

        for i, G in enumerate(self.G_ops):
            GI_ket = propagate_state([G], self.wf.ci_coeffs, *self.index_info)
            HGI_ket = propagate_state([self.H_0i_0a, G], self.wf.ci_coeffs, *self.index_info)
            # <0| GId H GJ |0>
            A = expectation_value(
                GI_ket,
                [],
                HGI_ket,
                *self.index_info,
            )
            if abs(A) > 10**-6:  # whatever rimeligt threshold
                finite_excitations_idx.append(True)
            else:
                finite_excitations_idx.append(False)

        finite_excitations_idx = np.array(finite_excitations_idx)
        print("finite", len(finite_excitations_idx), "q", len(self.q_ops), "G", len(self.G_ops  ))
        idx_shift = len(self.q_ops)
        G_shift = int(len(self.G_ops) / 2)
        q_shift = int(len(self.q_ops) / 2)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        if len(self.q_ops) != 0:
            grad = get_orbital_gradient_response_unrestricted(
                self.wf.haa_mo,
                self.wf.hbb_mo,
                self.wf.gaaaa_mo,
                self.wf.gbbbb_mo,
                self.wf.gaabb_mo,
                self.wf.gbbaa_mo,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1aa,
                self.wf.rdm1bb,
                self.wf.rdm2aaaa,
                self.wf.rdm2bbbb,
                self.wf.rdm2aabb,
                self.wf.rdm2bbaa,
            )
            if len(grad) != 0:
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
        # Start RDM version
        if len(self.q_ops) != 0:
            self.A[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block_unrestricted(
                self.wf.haa_mo,
                self.wf.hbb_mo,
                self.wf.gaaaa_mo,
                self.wf.gbbbb_mo,
                self.wf.gaabb_mo,
                self.wf.gbbaa_mo,
                self.wf.kappa_no_activeactive_idx_dagger,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1aa,
                self.wf.rdm1bb,
                self.wf.rdm2aaaa,
                self.wf.rdm2bbbb,
                self.wf.rdm2aabb,
                self.wf.rdm2bbaa,
            )
            # print(self.A)
            self.B[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block_unrestricted(
                self.wf.haa_mo,
                self.wf.hbb_mo,
                self.wf.gaaaa_mo,
                self.wf.gbbbb_mo,
                self.wf.gaabb_mo,
                self.wf.gbbaa_mo,
                self.wf.kappa_no_activeactive_idx_dagger,
                self.wf.kappa_no_activeactive_idx_dagger,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1aa,
                self.wf.rdm1bb,
                self.wf.rdm2aaaa,
                self.wf.rdm2bbbb,
                self.wf.rdm2aabb,
                self.wf.rdm2bbaa,
            )
            self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma_unrestricted(
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1aa,
                self.wf.rdm1bb,
            )
        # End RDM

        # Start manual version
        """
        from slowquant.unitary_coupled_cluster.unrestricted_operators import unrestricted_hamiltonian_full_space
        H = unrestricted_hamiltonian_full_space(
            self.wf.haa_mo,
            self.wf.hbb_mo,
            self.wf.gaaaa_mo,
            self.wf.gbbbb_mo,
            self.wf.gaabb_mo,
            self.wf.gbbaa_mo,
            self.wf.num_orbs,
        )
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops):
                # Make A
                # <0| qd H q |0>
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [qJ.dagger * H * qI],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # -<0| qd q H |0>
                val -= expectation_value(
                    self.wf.ci_coeffs,
                    [qJ.dagger * qI * H],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # -<0| H q qd |0>
                val -= expectation_value(
                    self.wf.ci_coeffs,
                    [H * qI * qJ.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # <0| q H qd |0>
                val += expectation_value(
                    self.wf.ci_coeffs,
                    [qI * H * qJ.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                self.A[i, j] = val
                # make B
                # <0| qd H qd |0>
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [qJ.dagger * H * qI.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # -<0| qd qd H |0>
                val -= expectation_value(
                    self.wf.ci_coeffs,
                    [qJ.dagger * qI.dagger * H],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # -<0| H qd qd |0>
                val -= expectation_value(
                    self.wf.ci_coeffs,
                    [H * qI.dagger * qJ.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # <0| qd H qd |0>
                val += expectation_value(
                    self.wf.ci_coeffs,
                    [qI.dagger * H * qJ.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                self.B[i, j] = val
                # make sigma
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [qJ.dagger * qI],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                val -= expectation_value(
                    self.wf.ci_coeffs,
                    [qI * qJ.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                self.Sigma[i, j] = val
        """
        # End manual version
        if len(self.q_ops) != 0:
            for j, qJ in enumerate(self.q_ops):
                Hq_ket = propagate_state([self.H_1i_1a * qJ], self.wf.ci_coeffs, *self.index_info)
                qdH_ket = propagate_state([qJ.dagger * self.H_1i_1a], self.wf.ci_coeffs, *self.index_info)
                for i, GI in enumerate(self.G_ops):
                    G_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                    Gd_ket = propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                    # Make A
                    # <0| Gd H q |0>
                    val = expectation_value(
                        G_ket,
                        [],
                        Hq_ket,
                        *self.index_info,
                    )
                    # - 1/2<0| H q Gd |0>
                    val -= (
                        1
                        / 2
                        * expectation_value(
                            qdH_ket,
                            [],
                            Gd_ket,
                            *self.index_info,
                        )
                    )
                    # - 1/2<0| H Gd q |0>
                    val -= (
                        1
                        / 2
                        * expectation_value(
                            self.wf.ci_coeffs,
                            [self.H_1i_1a * GI.dagger * qJ],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                    )
                    # i er G, j er q (j kaldes først i loopet)
                    if j % 2 == 0:
                        if i % 2 == 0:
                            # print(int(i-i/2), j, int(i-i/2)+idx_shift)
                            self.A[int(i - (i / 2)) + idx_shift, int(j - (j / 2))] = self.A[
                                int(j - (j / 2)), int(i - (i / 2)) + idx_shift
                            ] = val  # alpha alpha
                        else:
                            # print(i, int(i-(i/2+0.5)), int(i-(i/2+0.5)) + idx_shift + G_shift, j, int(j-(j/2+0.5)), int(j-(j/2+0.5)) + q_shift) #find ud af hvordan jeg kan få den til ikke at gå ud af matrice grænsen....
                            self.A[
                                int(i - (i / 2 + 0.5)) + idx_shift + G_shift, int(j - (j / 2 + 0.5))
                            ] = self.A[
                                int(j - (j / 2 + 0.5)) + q_shift, int(i - (i / 2 + 0.5)) + idx_shift
                            ] = val  # up right alpha beta/beta alpha
                            self.A[
                                int(i - (i / 2 + 0.5)) + idx_shift, int(j - (j / 2 + 0.5)) + q_shift
                            ] = self.A[
                                int(j - (j / 2 + 0.5)), int(i - (i / 2 + 0.5)) + idx_shift + G_shift
                            ] = val  # bottom left alpha beta/ beta alpha
                    else:
                        self.A[
                            int(i - (i / 2 + 0.5)) + idx_shift + G_shift, int(j - (j / 2 + 0.5)) + q_shift
                        ] = self.A[
                            int(j - (j / 2 + 0.5)) + q_shift, int(i - (i / 2 + 0.5)) + idx_shift + G_shift
                        ] = val  # beta beta
                    # self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val

                    # Make B
                    # <0| qd H Gd |0>
                    val = expectation_value(
                        Hq_ket,
                        [],
                        Gd_ket,
                        *self.index_info,
                    )
                    # - 1/2*<0| Gd qd H |0>
                    val -= (
                        1
                        / 2
                        * expectation_value(
                            G_ket,
                            [],
                            qdH_ket,
                            *self.index_info,
                        )
                    )
                    # - 1/2*<0| qd Gd H |0>
                    val -= (
                        1
                        / 2
                        * expectation_value(
                            self.wf.ci_coeffs,
                            [qJ.dagger * GI.dagger * self.H_1i_1a],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                    )
                    # i er G, j er q (j kaldes først i loopet)
                    if j % 2 == 0:
                        if i % 2 == 0:
                            self.B[int(i - (i / 2)) + idx_shift, int(j - (j / 2))] = self.B[
                                int(j - (j / 2)), int(i - (i / 2)) + idx_shift
                            ] = val  # alpha alpha
                        else:
                            self.B[
                                int(i - (i / 2 + 0.5)) + idx_shift + G_shift, int(j - (j / 2 + 0.5))
                            ] = self.B[
                                int(j - (j / 2 + 0.5)) + q_shift, int(i - (i / 2 + 0.5)) + idx_shift
                            ] = val  # up right alpha beta/beta alpha
                            self.B[
                                int(i - (i / 2 + 0.5)) + idx_shift, int(j - (j / 2 + 0.5)) + q_shift
                            ] = self.B[
                                int(j - (j / 2 + 0.5)), int(i - (i / 2 + 0.5)) + idx_shift + G_shift
                            ] = val  # bottom left alpha beta/ beta alpha
                    else:
                        self.B[
                            int(i - (i / 2 + 0.5)) + idx_shift + G_shift, int(j - (j / 2 + 0.5)) + q_shift
                        ] = self.B[
                            int(j - (j / 2 + 0.5)) + q_shift, int(i - (i / 2 + 0.5)) + idx_shift + G_shift
                        ] = val  # beta beta
                    # self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val
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
                # Forsøg på at inddele i blokke
                if i % 2 == 0:
                    if j % 2 == 0:
                        self.A[int(i - (i / 2)) + idx_shift, int(j - (j / 2)) + idx_shift] = self.A[
                            int(j - (j / 2)) + idx_shift, int(i - (i / 2)) + idx_shift
                        ] = val
                    else:
                        self.A[
                            int(i - (i / 2 + 0.5)) + idx_shift, int(j - (j / 2 + 0.5)) + idx_shift + G_shift
                        ] = self.A[
                            int(j - (j / 2 + 0.5)) + idx_shift, int(i - (i / 2 + 0.5)) + idx_shift + G_shift
                        ] = val
                else:
                    self.A[
                        int(i - (i / 2 + 0.5)) + idx_shift + G_shift,
                        int(j - (j / 2 + 0.5)) + idx_shift + G_shift,
                    ] = self.A[
                        int(j - (j / 2 + 0.5)) + idx_shift + G_shift,
                        int(i - (i / 2 + 0.5)) + idx_shift + G_shift,
                    ] = val
                # gammel måde gem
                # self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val

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
                if i % 2 == 0:
                    if j % 2 == 0:
                        self.B[int(i - (i / 2)) + idx_shift, int(j - (j / 2)) + idx_shift] = self.B[
                            int(j - (j / 2)) + idx_shift, int(i - (i / 2)) + idx_shift
                        ] = val
                    else:
                        self.B[
                            int(i - (i / 2 + 0.5)) + idx_shift, int(j - (j / 2 + 0.5)) + idx_shift + G_shift
                        ] = self.B[
                            int(j - (j / 2 + 0.5)) + idx_shift, int(i - (i / 2 + 0.5)) + idx_shift + G_shift
                        ] = val
                else:
                    self.B[
                        int(i - (i / 2 + 0.5)) + idx_shift + G_shift,
                        int(j - (j / 2 + 0.5)) + idx_shift + G_shift,
                    ] = self.B[
                        int(j - (j / 2 + 0.5)) + idx_shift + G_shift,
                        int(i - (i / 2 + 0.5)) + idx_shift + G_shift,
                    ] = val

                # self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
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
                if i % 2 == 0:
                    if j % 2 == 0:
                        self.Sigma[int(i - (i / 2)) + idx_shift, int(j - (j / 2)) + idx_shift] = self.Sigma[
                            int(j - (j / 2)) + idx_shift, int(i - (i / 2)) + idx_shift
                        ] = val
                    else:
                        self.Sigma[
                            int(i - (i / 2 + 0.5)) + idx_shift, int(j - (j / 2 + 0.5)) + idx_shift + G_shift
                        ] = self.Sigma[
                            int(j - (j / 2 + 0.5)) + idx_shift, int(i - (i / 2 + 0.5)) + idx_shift + G_shift
                        ] = val
                else:
                    self.Sigma[
                        int(i - (i / 2 + 0.5)) + idx_shift + G_shift,
                        int(j - (j / 2 + 0.5)) + idx_shift + G_shift,
                    ] = self.Sigma[
                        int(j - (j / 2 + 0.5)) + idx_shift + G_shift,
                        int(i - (i / 2 + 0.5)) + idx_shift + G_shift,
                    ] = val
                # self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = val
        self.A = self.A[np.outer(finite_excitations_idx, finite_excitations_idx)].reshape(
            (np.sum(finite_excitations_idx), np.sum(finite_excitations_idx))
        )
        self.B = self.B[np.outer(finite_excitations_idx, finite_excitations_idx)].reshape(
            (np.sum(finite_excitations_idx), np.sum(finite_excitations_idx))
        )
        self.Sigma = self.Sigma[np.outer(finite_excitations_idx, finite_excitations_idx)].reshape(
            (np.sum(finite_excitations_idx), np.sum(finite_excitations_idx))
        )
        self.Delta = np.zeros(
            (len(self.Sigma), len(self.Sigma))
        )  # Delta er defineret her fordi den ellers har forkert dimension i unrestricted_lr_baseclass.py

    def get_transition_dipole(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        """Calculate transition dipole moment.

        Args:
            dipole_integrals: Dipole integrals ordered as (x,y,z).

        Returns:
            Transition dipole moment.
        """
        if len(dipole_integrals) != 3:
            raise ValueError(f"Expected 3 dipole integrals got {len(dipole_integrals)}")
        number_excitations = len(self.excitation_energies)
        mux_aa = one_electron_integral_transform(self.wf.c_a_mo, dipole_integrals[0])
        mux_bb = one_electron_integral_transform(self.wf.c_b_mo, dipole_integrals[0])
        muy_aa = one_electron_integral_transform(self.wf.c_a_mo, dipole_integrals[1])
        muy_bb = one_electron_integral_transform(self.wf.c_b_mo, dipole_integrals[1])
        muz_aa = one_electron_integral_transform(self.wf.c_a_mo, dipole_integrals[2])
        muz_bb = one_electron_integral_transform(self.wf.c_b_mo, dipole_integrals[2])
        # should be optimized to unirestricted_one_elec_op_0i_0a
        mux_op = unrestricted_one_elec_op_full_space(
            mux_aa,
            mux_bb,
            self.wf.num_orbs,
        )
        muy_op = unrestricted_one_elec_op_full_space(
            muy_aa,
            muy_bb,
            self.wf.num_orbs,
        )
        muz_op = unrestricted_one_elec_op_full_space(
            muz_aa,
            muz_bb,
            self.wf.num_orbs,
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
            q_part_x = get_orbital_response_property_gradient_unrestricted(
                mux_aa,
                mux_bb,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1aa,
                self.wf.rdm1bb,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_y = get_orbital_response_property_gradient_unrestricted(
                muy_aa,
                muy_bb,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1aa,
                self.wf.rdm1bb,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_z = get_orbital_response_property_gradient_unrestricted(
                muz_aa,
                muz_bb,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1aa,
                self.wf.rdm1bb,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            # print(q_part_x)
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

    def get_property_gradient_unrestricted(self, property_integrals: np.ndarray) -> np.ndarray:
        """Calculate unrestricted property gradient.

        Args:
            property_integrals: Integrals in AO basis.

        Returns:
            Property gradient.
        """
        in_shape = property_integrals.shape[:-2]
        size_mo = self.wf.num_inactive_orbs + self.wf.num_active_orbs + self.wf.num_inactive_orbs
        property_integrals = property_integrals.reshape(-1, size_mo, size_mo)
        num_mo = len(property_integrals)
        mo_a = np.zeros((num_mo, size_mo, size_mo))
        mo_b = np.zeros((num_mo, size_mo, size_mo))
        for i, ao in enumerate(property_integrals):
            mo_a[i, :, :] += one_electron_integral_transform(self.wf.c_a_mo, ao)
            mo_b[i, :, :] += one_electron_integral_transform(self.wf.c_b_mo, ao)

        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), num_mo))

        # orbital rotation part
        V[:idx_shift_q, :] = get_orbital_response_property_gradient_unrestricted(
            mo_a,
            mo_b,
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.rdm1aa,
            self.wf.rdm1bb,
        )

        for idx, G in enumerate(self.G_ops):
            G_ket = propagate_state([G], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = propagate_state([G.dagger], self.wf.ci_coeffs, *self.index_info)
            # Inactive part
            for i in range(self.wf.num_inactive_orbs):
                E_ket_a = propagate_state(
                    [a_op(i, "alpha", True) * a_op(i, "alpha", False)], self.wf.ci_coeffs, *self.index_info
                )
                E_ket_b = propagate_state(
                    [a_op(i, "beta", True) * a_op(i, "beta", False)], self.wf.ci_coeffs, *self.index_info
                )
                # < 0 | GE | 0 >
                val_a = expectation_value(Gd_ket, [], E_ket_a, *self.index_info)
                val_b = expectation_value(Gd_ket, [], E_ket_b, *self.index_info)
                # -< 0 | EG | 0 >
                val_a -= expectation_value(
                    E_ket_a, [], Gd_ket, *self.index_info
                )  # Den skal ikek være dagger da indices er ii (aka det bliver det samme!)
                val_b -= expectation_value(E_ket_b, [], Gd_ket, *self.index_info)
                V[idx + idx_shift_q, :] += mo_a[:, i, i] * val_a
                V[idx + idx_shift_q, :] += mo_b[:, i, i] * val_b

            # active part
            for p in range(self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                for q in range(
                    self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs
                ):
                    E_ket_a = propagate_state(
                        [a_op(p, "alpha", True) * a_op(q, "alpha", False)],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    E_ket_b = propagate_state(
                        [a_op(p, "beta", True) * a_op(q, "beta", False)], self.wf.ci_coeffs, *self.index_info
                    )
                    Ed_ket_a = propagate_state(
                        [a_op(q, "alpha", True) * a_op(p, "alpha", False)],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    Ed_ket_b = propagate_state(
                        [a_op(q, "beta", True) * a_op(p, "beta", False)], self.wf.ci_coeffs, *self.index_info
                    )
                    # < 0 | GE | 0 >
                    val_a = expectation_value(Gd_ket, [], E_ket_a, *self.index_info)
                    val_b = expectation_value(Gd_ket, [], E_ket_b, *self.index_info)
                    # -< 0 | EG | 0 >
                    val_a -= expectation_value(Ed_ket_a, [], G_ket, *self.index_info)
                    val_b -= expectation_value(Ed_ket_b, [], G_ket, *self.index_info)
                    V[idx + idx_shift_q, :] += mo_a[:, p, q] * val_a
                    V[idx + idx_shift_q, :] += mo_b[:, p, q] * val_b
            # check if complex
            if np.allclose(mo_a, mo_a.transpose(0, -1, -2)) and np.allclose(mo_b, mo_b.transpose(0, -1, -2)):
                return np.vstack((V, -1 * V)).reshape(-1, *in_shape)
            return np.vstack((V, V)).reshape(-1, *in_shape)
