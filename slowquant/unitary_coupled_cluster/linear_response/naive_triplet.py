from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass_triplet import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_matrix import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_triplet_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_static_property_gradient,
)
from slowquant.unitary_coupled_cluster.operators import Tpq
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


class LinearResponseUCC(LinearResponseBaseClass):
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

        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
            t_rdm2=self.wf.t_rdm2,
        )
        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        grad = get_orbital_gradient_response(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
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
        # Do orbital-orbital blocks
        self.A[: len(self.q_ops), : len(self.q_ops)] = get_triplet_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.B[: len(self.q_ops), : len(self.q_ops)] = get_triplet_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
            rdms, self.wf.kappa_no_activeactive_idx
        )
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
                # - 1/2*<0| H q Gd |0>
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
                # - 1/2*<0| H Gd q |0>
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
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
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
                # - 1/2*<0| GId GJ H |0>
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

    def get_property_gradient(self, property_integrals: np.ndarray) -> np.ndarray:
        """Calculate property gradient.

        Args:
            property_integrals: Integrals in AO basis.

        Returns:
            Property gradient.
        """
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
        )
        in_shape = property_integrals.shape[:-2]
        size_mo = self.wf.num_inactive_orbs + self.wf.num_active_orbs + self.wf.num_virtual_orbs
        property_integrals = property_integrals.reshape(-1, size_mo, size_mo)
        num_mo = len(property_integrals)
        mo = np.zeros((num_mo, size_mo, size_mo))
        for i, ao in enumerate(property_integrals):
            mo[i,:,:] += one_electron_integral_transform(self.wf.c_mo, ao)

        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), num_mo))

        # Orbital rotation part
        V[: idx_shift_q, :] = (
            get_orbital_response_static_property_gradient(
            rdms, 
            mo, 
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        ))
        for idx, G in enumerate(self.G_ops):
            G_ket = propagate_state([G], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = propagate_state([G.dagger], self.wf.ci_coeffs, *self.index_info)
            # Inactive part
            for i in range(self.wf.num_inactive_orbs):
                T_ket = propagate_state([Tpq(i, i)], self.wf.ci_coeffs, *self.index_info)
                # < 0 | G T | 0 >
                val = expectation_value(
                    Gd_ket,
                    [],
                    T_ket,
                    *self.index_info
                    )
                # - < 0 | T G | 0 >
                val -= expectation_value(
                    T_ket,
                    [],
                    G_ket,
                    *self.index_info
                    )
                V[idx + idx_shift_q, :] += mo[:,i, i] * val
            # Active part
            for p in range(self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                for q in range(self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                    T_ket = propagate_state([Tpq(p, q)], self.wf.ci_coeffs, *self.index_info)
                    Td_ket = propagate_state([Tpq(q, p)], self.wf.ci_coeffs, *self.index_info)
                    # < 0 | G T | 0 >
                    val = expectation_value(
                        Gd_ket,
                        [],
                        T_ket,
                        *self.index_info
                        )
                    # - < 0 | E G | 0 >
                    val -= expectation_value(
                        Td_ket,
                        [],
                        G_ket,
                        *self.index_info
                        )
                    V[idx + idx_shift_q, :] += mo[:, p, q] * val
        if np.allclose(mo, mo.transpose(0, -1, -2)):
            return np.vstack((V, -1 * V)).reshape(-1, *in_shape)
        return np.vstack((V, V)).reshape(-1, *in_shape)
