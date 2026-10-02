import numpy as np

from slowquant.qiskit_interface.linear_response.lr_baseclass import (
    get_num_CBS_elements,
    get_num_nonCBS,
    quantumLRBaseClass,
)
from slowquant.qiskit_interface.util import Clique
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_triplet_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient_1e,
    get_orbital_response_property_gradient_2e,
)
from slowquant.unitary_coupled_cluster.operators import (
    commutator,
    double_commutator,
    hamiltonian_2i_2a,
    one_elec_op_0i_0a,
    hamiltonian_0i_0a,
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
                        (commutator(self.H_1i_1a, op)).get_folded_operator(*self.orbs)
                    )
                    grad[i + self.num_q] = self.wf.QI.quantum_expectation_value(
                        (commutator(op.dagger, self.H_1i_1a)).get_folded_operator(*self.orbs)
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
                if not self.triplet:
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
                else:
                    self.A[: len(self.q_ops), : len(self.q_ops)] = get_triplet_orbital_response_hessian_block(
                        self.wf.h_mo,
                        self.wf.g_mo,
                        self.wf.kappa_no_activeactive_idx_dagger,
                        self.wf.kappa_no_activeactive_idx,
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.rdm1,
                        self.wf.rdm2,
                        self.wf.t_rdm2,
                    )
                    self.B[: len(self.q_ops), : len(self.q_ops)] = get_triplet_orbital_response_hessian_block(
                        self.wf.h_mo,
                        self.wf.g_mo,
                        self.wf.kappa_no_activeactive_idx_dagger,
                        self.wf.kappa_no_activeactive_idx_dagger,
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.rdm1,
                        self.wf.rdm2,
                        self.wf.t_rdm2,
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
                            (double_commutator(qI.dagger, self.H_2i_2a, qJ)).get_folded_operator(*self.orbs)
                        )
                        # Make B
                        self.B[i, j] = self.B[j, i] = -(
                            self.wf.QI.quantum_expectation_value(
                                (double_commutator(qI.dagger, self.H_2i_2a, qJ.dagger)).get_folded_operator(
                                    *self.orbs
                                )
                            )
                        )
                        # Make Sigma
                        self.Sigma[i, j] = self.Sigma[j, i] = self.wf.QI.quantum_expectation_value(
                            (commutator(qI.dagger, qJ)).get_folded_operator(*self.orbs)
                        )

            # Gq
            for j, qJ in enumerate(self.q_ops):
                for i, GI in enumerate(self.G_ops):
                    # Make A
                    val = self.wf.QI.quantum_expectation_value(
                        (
                            double_commutator(GI.dagger, self.H_1i_1a, qJ, do_symmetrized=False)
                        ).get_folded_operator(*self.orbs)
                    )
                    self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                    # Make B
                    val = self.wf.QI.quantum_expectation_value(
                        (
                            double_commutator(GI.dagger, self.H_1i_1a, qJ.dagger, do_symmetrized=False)
                        ).get_folded_operator(*self.orbs)
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

    def get_property_gradient(self, int1e: np.ndarray, int2e: np.ndarray | None = None, spin: bool = False) -> np.ndarray:
        """Calculate property gradient.

        Args:
            int1e: one-electron property integrals in MO basis.
            int2e: two-electron property integrals in MO basis.
            spin: if the operator generated from the integrals contains spin.

        Returns:
            Property gradient.
        """

        if np.allclose(int1e, int1e.transpose(0, -1, -2)):
            # real integral
            fac = -1
        elif np.allclose(int1e, -1 * int1e.transpose(0, -1, -2)):
            # imaginary integral
            fac = 1
        else:
            raise ValueError("Wrong symmetry: int1e must be symmetric or antisymmetric")
        
        if int2e is not None:
            if len(int1e) != len(int2e):
                raise ValueError(f"Mismatched arrays: int1e and int2e must have the same length, got {len(int1e)} and {len(int2e)}")
            if self.triplet:
                raise ValueError("Not implemented: triplet response and int2e cannot be used simutaniously.")
            if not np.allclose(int2e, -1 * fac * int2e.transpose(0,2,1,4,3)):
                raise ValueError("Mismatched symmetry: int1e and int2e must either both be symmetric or antisymmetric")

        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), len(int1e)))

        if len(self.q_ops) != 0 and spin == self.triplet:
            # Orbital response part
            V[:idx_shift_q, :] = get_orbital_response_property_gradient_1e(
                int1e,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
            )

            if int2e is not None:
                V[:idx_shift_q, :] += get_orbital_response_property_gradient_2e(
                        int2e,
                        self.wf.kappa_no_activeactive_idx,
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.rdm1,
                        self.wf.rdm2,
                    )

        # Excitation response part
        for mu, int1e_mu in enumerate(int1e):
            if int2e is None:
                op = one_elec_op_0i_0a(int1e_mu, self.wf.num_inactive_orbs, self.wf.num_active_orbs, spin)
            else:
                op = hamiltonian_0i_0a(int1e_mu, int2e[mu], self.wf.num_inactive_orbs, self.wf.num_active_orbs)
            for idx, G in enumerate(self.G_ops):
                V[idx + idx_shift_q, mu] = self.wf.QI.quantum_expectation_value(commutator(G, op).get_folded_operator(*self.orbs))
        
        return np.vstack((V, fac * V))
