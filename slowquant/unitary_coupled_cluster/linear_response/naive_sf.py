import numpy as np
import scipy

from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid_flow_commutator,
    expectation_value_hybrid_flow_double_commutator,
    hamiltonian_hybrid_0i_0a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import ThetaPicker


class LinearResponseSFUCC(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        self.wf = wave_function
        self.theta_picker = ThetaPicker(
            self.wf.active_occ_spin_idx,
            self.wf.active_unocc_spin_idx,
        )
        self.G_ops: list[OperatorHybrid] = []
        num_spin_orbs = self.wf.num_spin_orbs
        excitations = excitations.lower()
        if "s" in excitations:
            for _, _, _, op_ in self.theta_picker.get_t1_generator_sf(num_spin_orbs):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(op)
        if "d" in excitations:
            for _, _, _, _, _, op_ in self.theta_picker.get_t2_generator_sf(num_spin_orbs):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(op)
        num_parameters = len(self.G_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))
        self.H_0i_0a = hamiltonian_hybrid_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )
        print("Gs", len(self.G_ops))

        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, self.H_0i_0a, op, self.wf.state_vector
            )
            grad[i + len(self.G_ops)] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op.dagger, self.H_0i_0a, self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        idx_shift = 0
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i + idx_shift, j + idx_shift] = self.A[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector, GI.dagger, self.H_0i_0a, GJ, self.wf.state_vector
                )
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector, GI.dagger, self.H_0i_0a, GJ.dagger, self.wf.state_vector
                )
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, GI.dagger, GJ, self.wf.state_vector
                )

    def calc_excitation_energies(self) -> None:
        """Calculate excitation energies."""
        size = len(self.A)
        E2 = np.zeros((size * 2, size * 2))
        E2[:size, :size] = self.A
        E2[:size, size:] = self.B
        E2[size:, :size] = self.B
        E2[size:, size:] = self.A
        (
            hess_eigval,
            _,
        ) = np.linalg.eig(E2)
        print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")
        # if np.min(hess_eigval) < 0:
        #    raise ValueError("Negative eigenvalue in Hessian.")

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.Sigma
        S[:size, size:] = self.Delta
        S[size:, :size] = -self.Delta
        S[size:, size:] = -self.Sigma
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")

        eigval, _ = scipy.linalg.eig(E2, S)
        sorting = np.argsort(eigval)
        # self.excitation_energies = np.real(eigval[sorting][size:])
        self.excitation_energies = eigval
