import copy
from collections.abc import Sequence

import numpy as np
import scipy

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_hessian_A,
    get_orbital_response_hessian_B,
    get_orbital_response_metric_sgima,
    get_orbital_response_property_gradient,
    get_orbital_response_vector_norm,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid_flow,
    expectation_value_hybrid_flow_commutator,
    expectation_value_hybrid_flow_double_commutator,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    OperatorPauli,
    energy_hamiltonian_pauli,
    epq_pauli,
    hamiltonian_pauli_1i_1a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import (
    WaveFunctionUCC,
    construct_one_rdm,
    construct_two_rdm,
)
from slowquant.unitary_coupled_cluster.util import ThetaPicker


class ResponseOperator:
    def __init__(self, occ_idx: Sequence[int], unocc_idx: Sequence[int], operator: OperatorHybrid) -> None:
        """Initialize response excitation operator.

        Args:
            occ_idx: Index of occupied orbitals.
            unocc_idx: Index of unoccupied orbitals.
            operator: Operator.
        """
        self.occ_idx = occ_idx
        self.unocc_idx = unocc_idx
        self.operator = operator


class LinearResponseUCC:
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
        is_spin_conserving: bool = False,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            is_spin_conserving: Use spin-conseving operators.
        """
        self.wf = copy.deepcopy(wave_function)
        self.theta_picker = ThetaPicker(
            self.wf.active_occ_spin_idx,
            self.wf.active_unocc_spin_idx,
            is_spin_conserving=is_spin_conserving,
        )

        self.G_ops: list[ResponseOperator] = []
        self.q_ops: list[ResponseOperator] = []
        num_spin_orbs = self.wf.num_spin_orbs
        num_elec = self.wf.num_elec
        excitations = excitations.lower()
        if 's' in excitations:
            for _, a, i, op_ in self.theta_picker.get_t1_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i,), (a,), op))
        if 'd' in excitations:
            for _, a, i, b, j, op_ in self.theta_picker.get_t2_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j), (a, b), op))
        if 't' in excitations:
            for _, a, i, b, j, c, k, op_ in self.theta_picker.get_t3_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k), (a, b, c), op))
        if 'q' in excitations:
            for _, a, i, b, j, c, k, d, l, op_ in self.theta_picker.get_t4_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l), (a, b, c, d), op))
        if '5' in excitations:
            for _, a, i, b, j, c, k, d, l, e, m, op_ in self.theta_picker.get_t5_generator(
                num_spin_orbs, num_elec
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l, m), (a, b, c, d, e), op))
        if '6' in excitations:
            for _, a, i, b, j, c, k, d, l, e, m, f, n, op_ in self.theta_picker.get_t6_generator(
                num_spin_orbs, num_elec
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l, m, n), (a, b, c, d, e, f), op))
        for i, a in self.wf.kappa_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs, self.wf.num_elec)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
            )
            self.q_ops.append(ResponseOperator((i), (a), op))

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.M = np.zeros((num_parameters, num_parameters))
        self.Q = np.zeros((num_parameters, num_parameters))
        self.V = np.zeros((num_parameters, num_parameters))
        self.W = np.zeros((num_parameters, num_parameters))
        H_1i_1a = convert_pauli_to_hybrid_form(
            hamiltonian_pauli_1i_1a(
                self.wf.h_core,
                self.wf.g_eri,
                self.wf.c_trans,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
                num_elec,
            ),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        H_en = convert_pauli_to_hybrid_form(
            energy_hamiltonian_pauli(
                self.wf.h_core,
                self.wf.g_eri,
                self.wf.c_trans,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
                num_elec,
            ),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        rdm1 = construct_one_rdm(self.wf)
        rdm2 = construct_two_rdm(self.wf)
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
            self.wf.num_virtual_spin_orbs // 2,
            rdm1,
            rdm2=rdm2,
        )
        idx_shift = len(self.q_ops)
        print('Gs', len(self.G_ops))
        print('qs', len(self.q_ops))
        grad = get_orbital_gradient_response(
            rdms,
            self.wf.h_core,
            self.wf.g_eri,
            self.wf.c_trans,
            self.wf.kappa_idx,
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
        )
        if len(grad) != 0:
            print('idx, max(abs(grad orb)):', np.argmax(np.abs(grad)), np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, H_1i_1a, op.operator, self.wf.state_vector
            )
            grad[i + len(self.G_ops)] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op.operator.dagger, H_1i_1a, self.wf.state_vector
            )
        if len(grad) != 0:
            print('idx, max(abs(grad active)):', np.argmax(np.abs(grad)), np.max(np.abs(grad)))
        # Do orbital-orbital blocks
        self.M[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_A(
            rdms,
            self.wf.h_core,
            self.wf.g_eri,
            self.wf.c_trans,
            self.wf.kappa_idx,
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
        )
        self.Q[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_B(
            rdms,
            self.wf.h_core,
            self.wf.g_eri,
            self.wf.c_trans,
            self.wf.kappa_idx,
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
        )
        self.V[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sgima(
            rdms, self.wf.kappa_idx
        )
        for j, opJ in enumerate(self.q_ops):
            qJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                # Make M
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, H_1i_1a, qJ], self.wf.state_vector
                ) - expectation_value_hybrid_flow(
                    self.wf.state_vector, [H_1i_1a, qJ, GI.dagger], self.wf.state_vector
                )
                self.M[i + idx_shift, j] = self.M[j, i + idx_shift] = val
                # Make Q
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [qJ.dagger, H_1i_1a, GI.dagger], self.wf.state_vector
                ) - expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, qJ.dagger, H_1i_1a], self.wf.state_vector
                )
                self.Q[i + idx_shift, j] = self.Q[j, i + idx_shift] = val
        for j, opJ in enumerate(self.G_ops):
            GJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                if i < j:
                    continue
                # Make M
                self.M[i + idx_shift, j + idx_shift] = self.M[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector, GI.dagger, H_en, GJ, self.wf.state_vector
                )
                # Make Q
                self.Q[i + idx_shift, j + idx_shift] = self.Q[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector, GI.dagger, H_en, GJ.dagger, self.wf.state_vector
                )
                # Make V
                self.V[i + idx_shift, j + idx_shift] = self.V[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, GI.dagger, GJ, self.wf.state_vector
                )

    def calc_excitation_energies(self) -> None:
        """Calculate excitation energies."""
        size = len(self.M)
        E2 = np.zeros((size * 2, size * 2))
        E2[:size, :size] = self.M
        E2[:size, size:] = self.Q
        E2[size:, :size] = np.conj(self.Q)
        E2[size:, size:] = np.conj(self.M)
        (
            hess_eigval,
            _,
        ) = np.linalg.eig(E2)
        print(f'Smallest Hessian eigenvalue: {np.min(hess_eigval)}')

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.V
        S[:size, size:] = self.W
        S[size:, :size] = -np.conj(self.W)
        S[size:, size:] = -np.conj(self.V)
        print(f'Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.V)))}')

        eigval, eigvec = scipy.linalg.eig(E2, S)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
        self.normed_response_vectors = np.zeros_like(self.response_vectors)
        for state_number in range(size):
            norm = self.get_excited_state_norm(state_number)
            if norm < 10**-10:
                continue
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def get_excited_state_norm(self, state_number: int) -> float:
        """Calculate the norm of excited state.

        Args:
            state_number: Which excited state, counting from zero.

        Returns:
            Norm of excited state.
        """
        number_excitations = len(self.excitation_energies)
        rdm1 = construct_one_rdm(self.wf)
        rdm2 = construct_two_rdm(self.wf)
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
            self.wf.num_virtual_spin_orbs // 2,
            rdm1,
            rdm2=rdm2,
        )
        q_part = get_orbital_response_vector_norm(
            rdms, self.wf.kappa_idx, self.response_vectors, state_number, number_excitations
        )
        shift = len(self.q_ops)
        transfer_op = OperatorHybrid({})
        for i, op in enumerate(self.G_ops):
            G = op.operator
            transfer_op += (
                self.response_vectors[i + shift, state_number] * G.dagger
                + self.response_vectors[i + shift + number_excitations, state_number] * G
            )
        return q_part + expectation_value_hybrid_flow_commutator(
            self.wf.state_vector, transfer_op, transfer_op.dagger, self.wf.state_vector
        )

    def get_transition_dipole(
        self, state_number: int, dipole_integrals: Sequence[np.ndarray]
    ) -> tuple[float, float, float]:
        """Calculate transition dipole moment.

        Args:
            state_number: Which excited state, counting from zero.
            dipole_integrals: Dipole integrals ordered as (x,y,z).

        Returns:
            Transition dipole moment.
        """
        if len(dipole_integrals) != 3:
            raise ValueError(f'Expected 3 dipole integrals got {len(dipole_integrals)}')
        number_excitations = len(self.excitation_energies)
        rdm1 = construct_one_rdm(self.wf)
        rdm2 = construct_two_rdm(self.wf)
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
            self.wf.num_virtual_spin_orbs // 2,
            rdm1,
            rdm2=rdm2,
        )
        shift = len(self.q_ops)
        transfer_op = OperatorHybrid({})
        for i, op in enumerate(self.G_ops):
            G = op.operator
            transfer_op += (
                self.normed_response_vectors[i + shift, state_number] * G.dagger
                + self.normed_response_vectors[i + shift + number_excitations, state_number] * G
            )
        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])
        mux_op = OperatorPauli({})
        muy_op = OperatorPauli({})
        muz_op = OperatorPauli({})
        for p in range(self.wf.num_spin_orbs // 2):
            for q in range(self.wf.num_spin_orbs // 2):
                Epq_op = epq_pauli(p, q, self.wf.num_spin_orbs, self.wf.num_elec)
                if abs(mux[p, q]) > 10**-10:
                    mux_op += mux[p, q] * Epq_op
                if abs(muy[p, q]) > 10**-10:
                    muy_op += muy[p, q] * Epq_op
                if abs(muz[p, q]) > 10**-10:
                    muz_op += muz[p, q] * Epq_op
        mux_op = convert_pauli_to_hybrid_form(
            mux_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        muy_op = convert_pauli_to_hybrid_form(
            muy_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        muz_op = convert_pauli_to_hybrid_form(
            muz_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        transition_dipole_x = 0.0
        transition_dipole_y = 0.0
        transition_dipole_z = 0.0
        q_part_x = get_orbital_response_property_gradient(
            rdms,
            mux,
            self.wf.kappa_idx,
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
            self.normed_response_vectors,
            state_number,
            number_excitations,
        )
        q_part_y = get_orbital_response_property_gradient(
            rdms,
            muy,
            self.wf.kappa_idx,
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
            self.normed_response_vectors,
            state_number,
            number_excitations,
        )
        q_part_z = get_orbital_response_property_gradient(
            rdms,
            muz,
            self.wf.kappa_idx,
            self.wf.num_inactive_spin_orbs // 2,
            self.wf.num_active_spin_orbs // 2,
            self.normed_response_vectors,
            state_number,
            number_excitations,
        )
        if mux_op.operators != {}:
            transition_dipole_x = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, mux_op, transfer_op, self.wf.state_vector
            )
        if muy_op.operators != {}:
            transition_dipole_y = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, muy_op, transfer_op, self.wf.state_vector
            )
        if muz_op.operators != {}:
            transition_dipole_z = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, muz_op, transfer_op, self.wf.state_vector
            )
        return q_part_x + transition_dipole_x, q_part_y + transition_dipole_y, q_part_z + transition_dipole_z

    def get_oscillator_strength(self, state_number: int, dipole_integrals: Sequence[np.ndarray]) -> float:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\left>\right|^2

        Args:
            state_number: Target excited state (zero being the first excited state).
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Rerturns:
            Oscillator Strength.
        """
        transition_dipole_x, transition_dipole_y, transition_dipole_z = self.get_transition_dipole(
            state_number, dipole_integrals
        )
        excitation_energy = self.excitation_energies[state_number]
        return (
            2
            / 3
            * excitation_energy
            * (transition_dipole_x**2 + transition_dipole_y**2 + transition_dipole_z**2)
        )

    def get_nice_output(self, dipole_integrals: Sequence[np.ndarray]) -> str:
        """Create table of excitation energies and oscillator strengths.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Nicely formatted table.
        """
        output = (
            'Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n'
        )
        for i, exc_energy in enumerate(self.excitation_energies):
            osc_strength = self.get_oscillator_strength(i, dipole_integrals)
            exc_str = f'{exc_energy:2.6f}'
            exc_str_ev = f'{exc_energy*27.2114079527:3.6f}'
            osc_str = f'{osc_strength:1.6f}'
            output += f'{str(i+1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n'
        return output
