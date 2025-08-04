from collections.abc import Sequence

import numpy as np
from qiskit.primitives import BaseSampler

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.qiskit_interface.linear_response.lr_baseclass import (
    get_num_CBS_elements,
    get_num_nonCBS,
    quantumLRBaseClass,
)
from slowquant.qiskit_interface.util import Clique
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    commutator,
    double_commutator,
    hamiltonian_2i_2a,
    one_elec_op_0i_0a,
)


class quantumLR(quantumLRBaseClass):
    def run(
        self,
        do_rdm: bool = True,
        do_gradients: bool = True,
    ) -> None:
        """Run simulation of naive LR matrix elements.

        Args:
            do_rdm: Use RDMs for QQ part.
            do_gradients: Calculate gradients w.r.t. orbital rotations and active space excitations.
        """
        idx_shift = self.num_q
        print("Gs", self.num_G)
        print("qs", self.num_q)

        if self.num_q != 0:
            if do_rdm:
                if isinstance(self.wf.QI._primitive, BaseSampler):
                    self.wf.precalc_rdm_paulis(2)
                # RDMs
                if do_gradients:
                    # Check gradients
                    grad = get_orbital_gradient_response(
                        self.wf.h_mo,
                        self.wf.g_mo,
                        self.wf.kappa_no_activeactive_idx,
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.rdm1,
                        self.wf.rdm2,
                    )
            elif do_gradients:
                grad = np.zeros(2 * self.num_q)
                for i, op in enumerate(self.q_ops):
                    grad[i] = self.wf.QI.quantum_expectation_value(
                        (self.H_1i_1a * op).get_folded_operator(*self.orbs)
                    )
                    grad[i + self.num_q] = self.wf.QI.quantum_expectation_value(
                        (op.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
                    )
        if do_gradients:
            if self.num_q != 0:
                print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
                if np.max(np.abs(grad)) > 10**-3:
                    print("WARNING: Large Gradient detected in q of ", np.max(np.abs(grad)))

            grad = np.zeros(2 * self.num_G)
            for i, op in enumerate(self.G_ops):
                grad[i] = self.wf.QI.quantum_expectation_value(
                    commutator(self.H_0i_0a, op).get_folded_operator(*self.orbs)
                )
                grad[i + self.num_G] = self.wf.QI.quantum_expectation_value(
                    commutator(op.dagger, self.H_0i_0a).get_folded_operator(*self.orbs)
                )
            if len(grad) != 0:
                print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
                if np.max(np.abs(grad)) > 10**-3:
                    print("WARNING: Large Gradient detected in G of ", np.max(np.abs(grad)))

        # qq
        if self.num_q != 0:
            if do_rdm:
                self.A[: self.num_q, : self.num_q] = get_orbital_response_hessian_block(
                    self.wf.h_mo,
                    self.wf.g_mo,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )
                self.B[: self.num_q, : self.num_q] = get_orbital_response_hessian_block(
                    self.wf.h_mo,
                    self.wf.g_mo,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )
                self.Sigma[: self.num_q, : self.num_q] = get_orbital_response_metric_sigma(
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                )
            else:
                self.H_2i_2a = hamiltonian_2i_2a(
                    self.wf.h_mo,
                    self.wf.g_mo,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.num_virtual_orbs,
                )
                for j, qJ in enumerate(self.q_ops):
                    for i, qI in enumerate(self.q_ops[j:], j):
                        # Make A
                        self.A[i, j] = self.A[j, i] = self.wf.QI.quantum_expectation_value(
                            (qI.dagger * self.H_2i_2a * qJ).get_folded_operator(*self.orbs)
                        ) - self.wf.QI.quantum_expectation_value(
                            (qI.dagger * qJ * self.H_2i_2a).get_folded_operator(*self.orbs)
                        )
                        # Make B
                        self.B[i, j] = self.B[j, i] = -(
                            self.wf.QI.quantum_expectation_value(
                                (qI.dagger * qJ.dagger * self.H_2i_2a).get_folded_operator(*self.orbs)
                            )
                        )
                        # Make Sigma
                        self.Sigma[i, j] = self.Sigma[j, i] = self.wf.QI.quantum_expectation_value(
                            (qI.dagger * qJ).get_folded_operator(*self.orbs)
                        )

            # Gq
            for j, qJ in enumerate(self.q_ops):
                for i, GI in enumerate(self.G_ops):
                    # Make A
                    val = (
                        self.wf.QI.quantum_expectation_value(
                            (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs)
                        )
                        - 1
                        / 2
                        * self.wf.QI.quantum_expectation_value(
                            (self.H_1i_1a * qJ * GI.dagger).get_folded_operator(*self.orbs)
                        )
                        - 1
                        / 2
                        * self.wf.QI.quantum_expectation_value(
                            (self.H_1i_1a * GI.dagger * qJ).get_folded_operator(*self.orbs)
                        )
                    )
                    self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                    # Make B
                    val = (
                        self.wf.QI.quantum_expectation_value(
                            (qJ.dagger * self.H_1i_1a * GI.dagger).get_folded_operator(*self.orbs)
                        )
                        - 1
                        / 2
                        * self.wf.QI.quantum_expectation_value(
                            (GI.dagger * qJ.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
                        )
                        - 1
                        / 2
                        * self.wf.QI.quantum_expectation_value(
                            (qJ.dagger * GI.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
                        )
                    )
                    self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = (
                    self.wf.QI.quantum_expectation_value(
                        double_commutator(
                            GI.dagger, self.H_0i_0a, GJ, do_symmetrized=True
                        ).get_folded_operator(*self.orbs)
                    )
                )
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = (
                    self.wf.QI.quantum_expectation_value(
                        double_commutator(GI.dagger, self.H_0i_0a, GJ.dagger).get_folded_operator(*self.orbs)
                    )
                )
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = (
                    self.wf.QI.quantum_expectation_value(
                        commutator(GI.dagger, GJ).get_folded_operator(*self.orbs)
                    )
                )

    def _get_qbitmap(
        self,
        cliques: bool = False,
        do_rdm: bool = False,
    ) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
        """Get qubit map of operators.

        Args:
            cliques: If using cliques.
            do_rdm: Use RDMs for QQ part.

        Returns:
            Qubit map of operators.
        """
        idx_shift = self.num_q
        print("Gs", self.num_G)
        print("qs", self.num_q)

        A = [[""] * self.num_params for _ in range(self.num_params)]
        B = [[""] * self.num_params for _ in range(self.num_params)]
        Sigma = [[""] * self.num_params for _ in range(self.num_params)]

        if not do_rdm:
            self.H_2i_2a = hamiltonian_2i_2a(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.num_virtual_orbs,
            )
            for j, qJ in enumerate(self.q_ops):
                for i, qI in enumerate(self.q_ops[j:], j):
                    # Make A
                    A[i][j] = A[j][i] = (
                        self.wf.QI.op_to_qbit(
                            (qI.dagger * self.H_2i_2a * qJ).get_folded_operator(*self.orbs)
                        ).paulis.to_labels()
                        + self.wf.QI.op_to_qbit(
                            (qI.dagger * qJ * self.H_2i_2a).get_folded_operator(*self.orbs)
                        ).paulis.to_labels()
                    )
                    # Make B
                    B[i][j] = B[j][i] = self.wf.QI.op_to_qbit(
                        (qI.dagger * qJ.dagger * self.H_2i_2a).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()
                    # Make Sigma
                    Sigma[i][j] = Sigma[j][i] = self.wf.QI.op_to_qbit(
                        (qI.dagger * qJ).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                val = (
                    self.wf.QI.op_to_qbit(
                        (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()
                    + self.wf.QI.op_to_qbit(
                        (self.H_1i_1a * qJ * GI.dagger).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()
                    + self.wf.QI.op_to_qbit(
                        (self.H_1i_1a * GI.dagger * qJ).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()
                )
                A[i + idx_shift][j] = A[j][i + idx_shift] = val
                # Make B
                val = (
                    self.wf.QI.op_to_qbit(
                        (qJ.dagger * self.H_1i_1a * GI.dagger).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()
                    + self.wf.QI.op_to_qbit(
                        (GI.dagger * qJ.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()
                    + self.wf.QI.op_to_qbit(
                        (qJ.dagger * GI.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()
                )
                B[i + idx_shift][j] = B[j][i + idx_shift] = val

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                A[i + idx_shift][j + idx_shift] = A[j + idx_shift][i + idx_shift] = self.wf.QI.op_to_qbit(
                    double_commutator(GI.dagger, self.H_1i_1a, GJ, do_symmetrized=True).get_folded_operator(
                        *self.orbs
                    )
                ).paulis.to_labels()
                # Make B
                B[i + idx_shift][j + idx_shift] = B[j + idx_shift][i + idx_shift] = self.wf.QI.op_to_qbit(
                    double_commutator(GI.dagger, self.H_1i_1a, GJ.dagger).get_folded_operator(*self.orbs)
                ).paulis.to_labels()
                # Make Sigma
                Sigma[i + idx_shift][j + idx_shift] = Sigma[j + idx_shift][i + idx_shift] = (
                    self.wf.QI.op_to_qbit(
                        commutator(GI.dagger, GJ).get_folded_operator(*self.orbs)
                    ).paulis.to_labels()
                )

        if cliques:
            for i in range(self.num_params):
                for j in range(self.num_params):
                    if not A[i][j] == "":
                        clique = Clique()
                        clique.add_paulis([str(x) for x in A[i][j]])
                        A[i][j] = [x.head for x in clique.cliques]  # type: ignore [call-overload]
                    if not B[i][j] == "":
                        clique = Clique()
                        clique.add_paulis([str(x) for x in B[i][j]])
                        B[i][j] = [x.head for x in clique.cliques]  # type: ignore [call-overload]
                    if not Sigma[i][j] == "":
                        clique = Clique()
                        clique.add_paulis([str(x) for x in Sigma[i][j]])
                        Sigma[i][j] = [x.head for x in clique.cliques]  # type: ignore [call-overload]

        print("Number of non-CBS Pauli strings in A: ", get_num_nonCBS(A))
        print("Number of non-CBS Pauli strings in B: ", get_num_nonCBS(B))
        print("Number of non-CBS Pauli strings in Sigma: ", get_num_nonCBS(Sigma))

        CBS, nonCBS = get_num_CBS_elements(A)
        print("In A    , number of: CBS elements: ", CBS, ", non-CBS elements ", nonCBS)
        CBS, nonCBS = get_num_CBS_elements(B)
        print("In B    , number of: CBS elements: ", CBS, ", non-CBS elements ", nonCBS)
        CBS, nonCBS = get_num_CBS_elements(Sigma)
        print("In Sigma, number of: CBS elements: ", CBS, ", non-CBS elements ", nonCBS)

        return A, B, Sigma

    def run_std(
        self,
        no_coeffs: bool = False,
        verbose: bool = True,
        cv: bool = True,
        save: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get standard deviation in matrix elements of LR equation.

        Args:
            no_coeffs:  Boolean to no include coefficients
            verbose:    Boolean to print more info
            cv:         Boolean to calculate coefficient of variance
            save:       Boolean to save operator-specific standard deviations

        Returns:
            Array of standard deviations for A, B and Sigma
        """
        idx_shift = self.num_q
        print("Gs", self.num_G)
        print("qs", self.num_q)

        self.H_2i_2a = hamiltonian_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

        A = np.zeros((self.num_params, self.num_params))
        B = np.zeros((self.num_params, self.num_params))
        Sigma = np.zeros((self.num_params, self.num_params))

        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                A[i, j] = A[j, i] = np.sqrt(
                    self.wf.QI.quantum_variance(
                        (qI.dagger * self.H_2i_2a * qJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                    )
                    + self.wf.QI.quantum_variance(
                        (qI.dagger * qJ * self.H_2i_2a).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                    )
                )
                # Make B
                B[i, j] = B[j, i] = np.sqrt(
                    self.wf.QI.quantum_variance(
                        (qI.dagger * qJ.dagger * self.H_2i_2a).get_folded_operator(*self.orbs),
                        no_coeffs=no_coeffs,
                    )
                )
                # Make Sigma
                Sigma[i, j] = Sigma[j, i] = np.sqrt(
                    self.wf.QI.quantum_variance(
                        (qI.dagger * qJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                    )
                )

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                val = np.sqrt(
                    self.wf.QI.quantum_variance(
                        (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                    )
                    + 1
                    / 2
                    * self.wf.QI.quantum_variance(
                        (self.H_1i_1a * qJ * GI.dagger).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                    )
                    + 1
                    / 2
                    * self.wf.QI.quantum_variance(
                        (self.H_1i_1a * GI.dagger * qJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                    )
                )
                A[i + idx_shift, j] = A[j, i + idx_shift] = val
                # Make B
                val = np.sqrt(
                    self.wf.QI.quantum_variance(
                        (qJ.dagger * self.H_1i_1a * GI.dagger).get_folded_operator(*self.orbs),
                        no_coeffs=no_coeffs,
                    )
                    + 1
                    / 2
                    * self.wf.QI.quantum_variance(
                        (GI.dagger * qJ.dagger * self.H_1i_1a).get_folded_operator(*self.orbs),
                        no_coeffs=no_coeffs,
                    )
                    + 1
                    / 2
                    * self.wf.QI.quantum_variance(
                        (qJ.dagger * GI.dagger * self.H_1i_1a).get_folded_operator(*self.orbs),
                        no_coeffs=no_coeffs,
                    )
                )
                B[i + idx_shift, j] = B[j, i + idx_shift] = val

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                A[i + idx_shift, j + idx_shift] = A[j + idx_shift, i + idx_shift] = np.sqrt(
                    self.wf.QI.quantum_variance(
                        double_commutator(
                            GI.dagger, self.H_0i_0a, GJ, do_symmetrized=True
                        ).get_folded_operator(*self.orbs),
                        no_coeffs=no_coeffs,
                    )
                )
                # Make B
                B[i + idx_shift, j + idx_shift] = B[j + idx_shift, i + idx_shift] = np.sqrt(
                    self.wf.QI.quantum_variance(
                        double_commutator(GI.dagger, self.H_0i_0a, GJ.dagger).get_folded_operator(*self.orbs),
                        no_coeffs=no_coeffs,
                    )
                )
                # Make Sigma
                Sigma[i + idx_shift, j + idx_shift] = Sigma[j + idx_shift, i + idx_shift] = np.sqrt(
                    self.wf.QI.quantum_variance(
                        commutator(GI.dagger, GJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                    )
                )

        if no_coeffs:
            cv = False
        self._analyze_std(A, B, Sigma, verbose=verbose, cv=cv, save=save)
        return A, B, Sigma

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

        mux = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[2])
        mux_op = one_elec_op_0i_0a(mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        muy_op = one_elec_op_0i_0a(muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        muz_op = one_elec_op_0i_0a(muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        transition_dipole_x = 0.0
        transition_dipole_y = 0.0
        transition_dipole_z = 0.0
        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            transfer_op = FermionicOperator({}, {})
            for i, G in enumerate(self.G_ops):
                transfer_op += (
                    self._Z_G_normed[i, state_number] * G.dagger + self._Y_G_normed[i, state_number] * G
                )
            q_part_x = 0.0
            q_part_y = 0.0
            q_part_z = 0.0
            if self.num_q != 0:
                q_part_x = get_orbital_response_property_gradient(
                    mux,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_excitation_vectors,
                    state_number,
                    number_excitations,
                )
                q_part_y = get_orbital_response_property_gradient(
                    muy,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_excitation_vectors,
                    state_number,
                    number_excitations,
                )
                q_part_z = get_orbital_response_property_gradient(
                    muz,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_excitation_vectors,
                    state_number,
                    number_excitations,
                )
            if self.num_G != 0:
                transition_dipole_x = self.wf.QI.quantum_expectation_value(
                    commutator(mux_op, transfer_op).get_folded_operator(*self.orbs)
                )
                transition_dipole_y = self.wf.QI.quantum_expectation_value(
                    commutator(muy_op, transfer_op).get_folded_operator(*self.orbs)
                )
                transition_dipole_z = self.wf.QI.quantum_expectation_value(
                    commutator(muz_op, transfer_op).get_folded_operator(*self.orbs)
                )
            transition_dipoles[state_number, 0] = q_part_x + transition_dipole_x
            transition_dipoles[state_number, 1] = q_part_y + transition_dipole_y
            transition_dipoles[state_number, 2] = q_part_z + transition_dipole_z

        return transition_dipoles
