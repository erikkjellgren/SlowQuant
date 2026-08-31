import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient_response,
    get_orbital_response_property_gradient_1e,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import (
    hamiltonian_2i_2a,
    Epq,
    Tpq,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


class LinearResponse(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: str,
        triplet: bool = False,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            triplet: If the linear response should be triplet spin-adapted.
        """
        super().__init__(wave_function, excitations, triplet)

        H_2i_2a = hamiltonian_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

        idx_shift = len(self.q_ops)
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
        UdH00_ket = propagate_state(["Ud", self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state(
                [op],
                self.wf.csf_coeffs,
                *self.index_info,
            )
            # <0| H U G |CSF>
            grad[i] = -expectation_value(
                UdH00_ket,
                [],
                G_ket,
                *self.index_info,
            )
            # <CSF| Gd Ud H |0>
            grad[i + len(self.G_ops)] = expectation_value(
                G_ket,
                [],
                UdH00_ket,
                *self.index_info,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [qI.dagger * H_2i_2a * qJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                val -= (
                    expectation_value(
                        self.wf.ci_coeffs, [qI.dagger * qJ], self.wf.ci_coeffs, *self.index_info
                    )
                    * self.wf.energy_elec
                )
                self.A[i, j] = self.A[j, i] = val
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = expectation_value(
                    self.wf.ci_coeffs,
                    [qI.dagger * qJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
        for j, qJ in enumerate(self.q_ops):
            UdHq_ket = propagate_state(["Ud", self.H_1i_1a * qJ], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops):
                G_ket = propagate_state([GI], self.wf.csf_coeffs, *self.index_info)
                # Make A
                # <CSF| Gd Ud H q |0>
                val = expectation_value(
                    G_ket,
                    [],
                    UdHq_ket,
                    *self.index_info,
                )
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = val
        for j, GJ in enumerate(self.G_ops):
            UdHUGJ_ket = propagate_state(
                ["Ud", self.H_0i_0a, "U", GJ],
                self.wf.csf_coeffs,
                *self.index_info,
            )
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                # <CSF| GId Ud H U GJ |CSF>
                val = expectation_value(
                    self.wf.csf_coeffs,
                    [GI.dagger],
                    UdHUGJ_ket,
                    *self.index_info,
                )
                if i == j:
                    val -= self.wf.energy_elec
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                if i == j:
                    self.Sigma[i + idx_shift, j + idx_shift] = 1

    def get_property_gradient(self, property_integrals: np.ndarray | tuple[np.ndarray]) -> np.ndarray:
        """Calculate property gradient.

        Args:
            property_integrals: Integrals in AO basis.

        Returns:
            Property gradient.
        """
        size_mo = self.wf.num_inactive_orbs + self.wf.num_active_orbs + self.wf.num_virtual_orbs
        num_mo = len(property_integrals)
        mo = np.zeros((num_mo, size_mo, size_mo))
        for i, ao in enumerate(property_integrals):
            mo[i, :, :] += one_electron_integral_transform(self.wf.c_mo, ao)

        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), num_mo))

        if len(self.q_ops) != 0:
            # Orbital response part
            V[:idx_shift_q, :] = get_orbital_response_property_gradient_1e(
                mo,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
            )
        
        if not self.triplet:
            E = Epq
        else:
            E = Tpq

        for idx, G in enumerate(self.G_ops):
            UG_ket = propagate_state(["U",G], self.wf.csf_coeffs, *self.index_info)
            # Inactive part
            for i in range(self.wf.num_inactive_orbs):
                Ed_ket = propagate_state([E(i, i)], self.wf.ci_coeffs, *self.index_info) 
                # - < 0 | E U G | CSF >
                val = - expectation_value(
                    Ed_ket, 
                    [], 
                    UG_ket, 
                    *self.index_info
                )
                V[idx + idx_shift_q, :] += mo[:, i, i] * val
            # Active part
            for v in range(self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                for w in range(
                    self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs
                ):
                    Ed_ket = propagate_state([E(w, v)], self.wf.ci_coeffs, *self.index_info)
                    # - < 0 | E U G | CSF >
                    val = - expectation_value(
                        Ed_ket, 
                        [], 
                        UG_ket, 
                        *self.index_info
                    )
                    V[idx + idx_shift_q, :] += mo[:, v, w] * val
        if np.allclose(mo, mo.transpose(0, -1, -2)):
            return np.vstack((V, -1 * V))
        return np.vstack((V, V))
