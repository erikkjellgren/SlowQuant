        if len(grad) != 0:
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Hessian gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(op.operator, H_en), self.wf.state_vector
            )
            grad[i + len(self.G_ops)] = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(H_en, op.operator.dagger), self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Hessian gradient detected in G of ", np.max(np.abs(grad)))

        #######
        ### Construct matrices
        #######

        else:
            for j, opJ in enumerate(self.q_ops):
                qJ = opJ.operator
                for i, opI in enumerate(self.q_ops):
                    qI = opI.operator
                    if i < j:
                        continue
                    if calculation_type in ("all_proj", "ST_proj"):
                        # Make M (A)
                        val = expectation_value_hybrid_flow(
                            self.wf.state_vector, [qI.dagger, H_2i_2a, qJ], self.wf.state_vector
                        ) - (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [qI.dagger, qJ], self.wf.state_vector
                            )
                            * self.wf.energy_elec
                        )
                        self.M[i, j] = self.M[j, i] = val
                        # Make Q (B) = 0
                        # Make V
                        self.V[i, j] = self.V[j, i] = expectation_value_hybrid_flow(
                            self.wf.state_vector, [qI.dagger, qJ], self.wf.state_vector
                        )
                        # Make W = 0
                    else:
                        raise NameError("Could not determine calculation_type got: {calculation_type}")
        # Gq/RQ and qG/QR
        # Remember: [G_i^d,[H,q_j]] = [[q_i^d,H],G_j] = [G_j,[H,q_i^d]]
        for j, opJ in enumerate(self.q_ops):
            qJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                if calculation_type == "sc":
                    # Make M (A)
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, H_1i_1a, qJ], self.wf.state_vector
                    )
                    # Make Q (B)
                    self.Q[j, i + idx_shift] = self.Q[i + idx_shift, j] = -expectation_value_hybrid_flow(
                        csf,
                        [GI.dagger, U.dagger, qJ.dagger, H_1i_1a],
                        self.wf.state_vector,
                    )
                    # Make V = 0
                    # Make W = 0
                elif calculation_type == "all_proj":
                    # Make M (A)
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        self.wf.state_vector, [GI.dagger, H_1i_1a, qJ], self.wf.state_vector
                    )
                    # Make Q (B) = 0
                    # Make V = 0
                    # Make W = 0
                elif calculation_type == "ST_proj":
                    # Make M (A)
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, H_1i_1a, qJ], self.wf.state_vector
                    )
                    # Make Q (B) = 0
                    # Make V = 0
                    # Make W = 0
                else:
                    raise NameError("Could not determine calculation_type got: {calculation_type}")
        # GG/RR
        for j, opJ in enumerate(self.G_ops):
            GJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                if i < j:
                    continue
                if calculation_type == "sc":
                    # Make M (A)
                    val = expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, H_en, U, GJ], csf
                    ) - expectation_value_hybrid_flow(
                        csf, [GI.dagger, GJ, U.dagger, H_en], self.wf.state_vector
                    )
                    self.M[i + idx_shift, j + idx_shift] = self.M[j + idx_shift, i + idx_shift] = val
                    # Make Q (B)
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[
                        j + idx_shift, i + idx_shift
                    ] = -expectation_value_hybrid_flow(
                        csf, [GI.dagger, GJ.dagger, U.dagger, H_en], self.wf.state_vector
                    )
                    # Make V
                    if i == j:
                        self.V[i + idx_shift, j + idx_shift] = self.V[j + idx_shift, i + idx_shift] = 1
                    # Make W = 0
                elif calculation_type in ("proj", "all_proj"):
                    # Make M (A)
                    val = (
                        expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, H_en, GJ], self.wf.state_vector
                        )
                        - (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector
                            )
                            * self.wf.energy_elec
                        )
                        - (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [GI.dagger], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(
                                self.wf.state_vector, [H_en, GJ], self.wf.state_vector
                            )
                        )
                        + (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [GI.dagger], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(  # ToDo: this can be done more efficiently by calculating once and storing in WF object
                                self.wf.state_vector, [GJ], self.wf.state_vector
                            )
                            * self.wf.energy_elec
                        )
                    )
                    self.M[i + idx_shift, j + idx_shift] = self.M[j + idx_shift, i + idx_shift] = val
                    # Make Q (B)
                    val = (
                        expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, H_en], self.wf.state_vector
                        )
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [GJ.dagger], self.wf.state_vector
                        )
                    ) - (
                        expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger], self.wf.state_vector)
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [GJ.dagger], self.wf.state_vector
                        )
                        * self.wf.energy_elec
                    )
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[j + idx_shift, i + idx_shift] = val
                    # Make V
                    self.V[i + idx_shift, j + idx_shift] = self.V[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_hybrid_flow(
                        self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector
                    ) - (
                        expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger], self.wf.state_vector)
                        * expectation_value_hybrid_flow(self.wf.state_vector, [GJ], self.wf.state_vector)
                    )
                    # Make W = 0
                else:
                    raise NameError("Could not determine calculation_type got: {calculation_type}")

    def calc_excitation_energies(self, do_working_equations: bool = False) -> None:
        """Calculate excitation energies."""
        size = len(self.M)
        # Make Hessian
        E2 = np.zeros((size * 2, size * 2))
        E2[:size, :size] = self.M
        E2[:size, size:] = self.Q
        E2[size:, :size] = np.conj(self.Q)
        E2[size:, size:] = np.conj(self.M)
        (
            hess_eigval,
            _,
        ) = np.linalg.eig(E2)
        print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.V
        S[:size, size:] = self.W
        S[size:, :size] = -np.conj(self.W)
        S[size:, size:] = -np.conj(self.V)
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.V)))}")
        if np.min(np.abs(np.diagonal(self.V))) < 0:
            raise ValueError("This value is bad. Abort.")

        eigval, eigvec = scipy.linalg.eig(E2, S)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
        self.normed_response_vectors = np.zeros_like(self.response_vectors)  # response_vector / <n|n>**1/2
        for state_number in range(size):
            norm = self.get_excited_state_norm(state_number, do_working_equations=do_working_equations)
            if norm < 10**-10:
                continue
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def analyse_response_vector(self, state_number: int, threshold: float = 0.05) -> str:
        """Analyse the response vector.

        Args:
            state_number: Excitation index counting from zero.
            threshold: Only print response vector elements that are larger than threshold*max(response vector element).

        Returns:
            Tabulized excitation assignment.
        """
        output = f"Response vector analysis for excitation {state_number+1}\n"
        output += (
            "Occupied idxs | Unoccupied idxs | Response vector element | Normalized response vector element\n"
        )
        excitations = len(self.q_ops) + len(self.G_ops)
        skip_threshold = threshold * np.max(np.abs(self.response_vectors[:, state_number]))
        for resp_val, normed_resp_val, operator in zip(
            self.response_vectors[:excitations, state_number],
            self.normed_response_vectors[:excitations, state_number],
            self.q_ops + self.G_ops,
        ):
            if abs(resp_val) < skip_threshold:
                continue
            resp_val_str = f"{resp_val:1.6f}"
            normed_resp_val_str = f"{normed_resp_val:1.6f}"
            output += f"{str(operator.occ_idx).center(13)} | {str(operator.unocc_idx).center(15)} | {resp_val_str.center(23)} | {normed_resp_val_str.center(34)}\n"
        for resp_val, normed_resp_val, operator in zip(
            self.response_vectors[excitations:, state_number],
            self.normed_response_vectors[excitations:, state_number],
            self.q_ops + self.G_ops,
        ):
            if abs(resp_val) < skip_threshold:
                continue
            resp_val_str = f"{resp_val:1.6f}"
            normed_resp_val_str = f"{normed_resp_val:1.6f}"
            output += f"{str(operator.unocc_idx).center(13)} | {str(operator.occ_idx).center(15)} | {resp_val_str.center(23)} | {normed_resp_val_str.center(34)}\n"
        return output

    def get_excited_state_overlap(self, state_number: int) -> float:
        """Calculate overlap of excitated state with the ground state.

        Args:
            state_number: Which excited state, counting from zero.

        Returns:
            Overlap between ground state and excited state.
        """
        number_excitations = len(self.excitation_energies)
        print("WARNING: This function [get_excited_state_overlap] might not be working.")

        for i, op in enumerate(self.q_ops + self.G_ops):
            G = op.operator
            if i == 0:
                transfer_op = (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
            else:
                transfer_op += (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
        return expectation_value_hybrid(self.wf.state_vector, transfer_op, self.wf.state_vector)

    def get_excited_state_norm(self, state_number: int, do_working_equations: bool = False) -> float:
        r"""
        Calculate the norm of excited state

        Only for naive G and q!

        .. math::
            <n|n> = <0|[Q,Q^<dagger]|0> with Q = \sum_k ( Z_k X_k^\dagger + Y_k X_k )

        Args:
            state_number: Which excited state, counting from zero.

        Returns:
            Norm of excited state.
        """
        if (
            sum(
                [
                    self.do_projected_operators,
                    self.do_selfconsistent_operators,
                    self.do_statetransfer_operators,
                    self.do_hermitian_statetransfer_operators,
                    self.do_all_projected_operators,
                    self.do_ST_projected_operators,
                ]
            )
            >= 1
            and not self.do_debugging
            and not do_working_equations
        ):
            print(
                "WARNING: Calculation of excited state norm only possible for naive operators. Only energies and response vectors are valid. Try do_working_equation or do_debugging."
            )
        # Slow version without RDMs
        number_excitations = len(self.excitation_energies)
        if not do_working_equations:
            for i, op in enumerate(self.q_ops + self.G_ops):
                G = op.operator
                if i == 0:
                    transfer_op = (
                        self.response_vectors[i, state_number] * G.dagger
                        + self.response_vectors[i + number_excitations, state_number] * G
                    )
                else:
                    transfer_op += (
                        self.response_vectors[i, state_number] * G.dagger
                        + self.response_vectors[i + number_excitations, state_number] * G
                    )
            norm = expectation_value_contracted(
                self.wf.state_vector,
                commutator_contract(transfer_op, transfer_op.dagger),
                self.wf.state_vector,
            )
        else:
            # Get sub-transfer operators:
            # t_Zq_op = \sum_i Z_i q_i^\dagger
            # t_Yq_op = \sum_i Y_i q_i
            # t_ZG_op = \sum_i Z_i G_i^\dagger
            # t_YG_op = \sum_i Y_i G_i
            idx_shift = len(self.q_ops)
            """ OLD
            t_Zq_op = OperatorHybrid({})
            t_Yq_op = OperatorHybrid({})
            for i, op in enumerate(self.q_ops):
                if i == 0:
                    t_Zq_op = self.response_vectors[i, state_number] * op.operator.dagger
                    t_Yq_op = self.response_vectors[i + number_excitations, state_number] * op.operator
                else:
                    t_Zq_op += self.response_vectors[i, state_number] * op.operator.dagger
                    t_Yq_op += self.response_vectors[i + number_excitations, state_number] * op.operator
            """
            q_part = get_orbital_response_vector_norm(
                self.wf.rdms, self.wf.kappa_idx, self.response_vectors, state_number, number_excitations
            )
            for i, op in enumerate(self.G_ops):
                if i == 0:
                    t_ZG_op = self.response_vectors[i + idx_shift, state_number] * op.operator.dagger
                    t_YG_op = (
                        self.response_vectors[i + idx_shift + number_excitations, state_number] * op.operator
                    )
                else:
                    t_ZG_op += self.response_vectors[i + idx_shift, state_number] * op.operator.dagger
                    t_YG_op += (
                        self.response_vectors[i + idx_shift + number_excitations, state_number] * op.operator
                    )
            # projected and all-projected
            if self.do_projected_operators or self.do_all_projected_operators:
                """OLD
                norm = (
                    # t_Zq_op = \sum_i Z_i q_i^\dagger
                    expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_Zq_op, t_Zq_op.dagger], self.wf.state_vector
                    )
                    # t_Yq_op = \sum_i Y_i q_i
                    - expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_Yq_op.dagger, t_Yq_op], self.wf.state_vector
                    )
                """
                norm = (
                    q_part
                    # t_ZG_op = \sum_i Z_i G_i^\dagger
                    + expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_ZG_op, t_ZG_op.dagger], self.wf.state_vector
                    )
                    - (
                        expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_ZG_op.dagger], self.wf.state_vector
                        )
                        * expectation_value_hybrid_flow(self.wf.state_vector, [t_ZG_op], self.wf.state_vector)
                    )
                    # t_YG_op = \sum_i Y_i G_i
                    + (
                        expectation_value_hybrid_flow(self.wf.state_vector, [t_YG_op], self.wf.state_vector)
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_YG_op.dagger], self.wf.state_vector
                        )
                    )
                    - expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_YG_op.dagger, t_YG_op], self.wf.state_vector
                    )
                )
            # ST, HST, and ST/projected
            elif (
                self.do_statetransfer_operators
                or self.do_hermitian_statetransfer_operators
                or self.do_ST_projected_operators
                or self.do_selfconsistent_operators
            ):
                """OLD
                norm = (
                    # t_Zq_op = \sum_i Z_i q_i^\dagger
                    expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_Zq_op, t_Zq_op.dagger], self.wf.state_vector
                    )
                    # t_Yq_op = \sum_i Y_i q_i
                    - expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_Yq_op.dagger, t_Yq_op], self.wf.state_vector
                    )
                """
                norm = (
                    q_part
                    # t_ZG_op = \sum_i Z_i G_i^\dagger
                    + expectation_value_hybrid_flow(self.csf, [t_ZG_op, t_ZG_op.dagger], self.csf)
                    # t_YG_op = \sum_i Y_i G_i
                    - expectation_value_hybrid_flow(self.csf, [t_YG_op.dagger, t_YG_op], self.csf)
                )
            # naive
            else:
                """OLD
                norm = (
                    expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_Zq_op, t_Zq_op.dagger], self.wf.state_vector
                    )
                    - expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_Yq_op.dagger, t_Yq_op], self.wf.state_vector
                    )
                """
                norm = (
                    q_part
                    + expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_ZG_op, t_ZG_op.dagger], self.wf.state_vector
                    )
                    - expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_ZG_op.dagger, t_ZG_op], self.wf.state_vector
                    )
                    + expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_YG_op, t_YG_op.dagger], self.wf.state_vector
                    )
                    - expectation_value_hybrid_flow(
                        self.wf.state_vector, [t_YG_op.dagger, t_YG_op], self.wf.state_vector
                    )
                )

        return norm

    def get_transition_dipole(
        self, state_number: int, dipole_integrals: Sequence[np.ndarray], do_working_equations: bool = False
    ) -> tuple[float, float, float]:
        r"""Calculate transition dipole moment
        Only for naive G and q!

        .. math::
            <0|\mu|n>

        Args:
            state_number: Which excited state, counting from zero.
            dipole_integrals: Dipole integrals ordered as (x,y,z).

        Returns:
            Transition dipole moment.
        """

        if len(dipole_integrals) != 3:
            raise ValueError(f"Expected 3 dipole integrals got {len(dipole_integrals)}")
        number_excitations = len(self.excitation_energies)

        if not do_working_equations:
            # SLow version without RDMs
            # Get transfer operator: O = \sum_k ( Z_k X_k^\dagger + Y_k X_k ) with X \elementof {G,q}
            # Using normed response vectors
            for i, op in enumerate(self.q_ops + self.G_ops):
                G = op.operator
                if i == 0:
                    transfer_op = (
                        self.normed_response_vectors[i, state_number] * G.dagger
                        + self.normed_response_vectors[i + number_excitations, state_number] * G
                    )
                else:
                    transfer_op += (
                        self.normed_response_vectors[i, state_number] * G.dagger
                        + self.normed_response_vectors[i + number_excitations, state_number] * G
                    )
        else:
            # Get sub-transfer operators:
            # t_Zq_op = \sum_i Z_i q_i^\dagger
            # t_Yq_op = \sum_i Y_i q_i
            # t_ZG_op = \sum_i Z_i G_i^\dagger
            # t_YG_op = \sum_i Y_i G_i
            idx_shift = len(self.q_ops)
            """ OLD
            t_Zq_op = OperatorHybrid({})
            t_Yq_op = OperatorHybrid({})
            for i, op in enumerate(self.q_ops):
                if i == 0:
                    t_Zq_op = self.normed_response_vectors[i, state_number] * op.operator.dagger
                    t_Yq_op = self.normed_response_vectors[i + number_excitations, state_number] * op.operator
                else:
                    t_Zq_op += self.normed_response_vectors[i, state_number] * op.operator.dagger
                    t_Yq_op += (
                        self.normed_response_vectors[i + number_excitations, state_number] * op.operator
                    )
            """
            t_ZG_op = OperatorHybrid({})
            t_YG_op = OperatorHybrid({})
            for i, op in enumerate(self.G_ops):
                if i == 0:
                    t_ZG_op = self.normed_response_vectors[i + idx_shift, state_number] * op.operator.dagger
                    t_YG_op = (
                        self.normed_response_vectors[i + idx_shift + number_excitations, state_number]
                        * op.operator
                    )
                else:
                    t_ZG_op += self.normed_response_vectors[i + idx_shift, state_number] * op.operator.dagger
                    t_YG_op += (
                        self.normed_response_vectors[i + idx_shift + number_excitations, state_number]
                        * op.operator
                    )

        # Transform Integrals from AO to MO basis
        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])

        counter = 0
        # mux_op \sum_pq ( x_pq E_pq )
        # muy_op \sum_pq ( y_pq E_pq )
        # muz_op \sum_pq ( z_pq E_pq )
        for p in range(self.wf.num_spin_orbs // 2):
            for q in range(self.wf.num_spin_orbs // 2):
                Epq_op = epq_pauli(p, q, self.wf.num_spin_orbs, self.wf.num_elec)
                if counter == 0:
                    mux_op = mux[p, q] * Epq_op
                    muy_op = muy[p, q] * Epq_op
                    muz_op = muz[p, q] * Epq_op
                    counter += 1
                else:
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

        if not do_working_equations:
            # Slow version without RDMs
            # <0|\mu_x|n> = <0|[(\sum_pq x_pq E_pq), O]|0>
            if mux_op.operators != {}:
                transition_dipole_x = expectation_value_contracted(
                    self.wf.state_vector, commutator_contract(mux_op, transfer_op), self.wf.state_vector
                )
            # <0|\mu_y|n> = <0|[(\sum_pq y_pq E_pq), O]|0>
            if muy_op.operators != {}:
                transition_dipole_y = expectation_value_contracted(
                    self.wf.state_vector, commutator_contract(muy_op, transfer_op), self.wf.state_vector
                )
            # <0|\mu_z|n> = <0|[(\sum_pq z_pq E_pq), O]|0>
            if muz_op.operators != {}:
                transition_dipole_z = expectation_value_contracted(
                    self.wf.state_vector, commutator_contract(muz_op, transfer_op), self.wf.state_vector
                )
        else:
            q_part_x = get_orbital_response_property_gradient(
                self.wf.rdms,
                mux,
                self.wf.kappa_idx,
                self.wf.num_inactive_spin_orbs // 2,
                self.wf.num_active_spin_orbs // 2,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_y = get_orbital_response_property_gradient(
                self.wf.rdms,
                muy,
                self.wf.kappa_idx,
                self.wf.num_inactive_spin_orbs // 2,
                self.wf.num_active_spin_orbs // 2,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_z = get_orbital_response_property_gradient(
                self.wf.rdms,
                muz,
                self.wf.kappa_idx,
                self.wf.num_inactive_spin_orbs // 2,
                self.wf.num_active_spin_orbs // 2,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            if self.do_projected_operators or self.do_all_projected_operators:
                if mux_op.operators != {}:
                    transition_dipole_x = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, mux_op], self.wf.state_vector
                        #    )
                        q_part_x
                        + (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [mux_op], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(
                                self.wf.state_vector, [t_ZG_op], self.wf.state_vector
                            )
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_ZG_op, mux_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [mux_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [mux_op, t_YG_op], self.wf.state_vector
                        )
                        - (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [t_YG_op], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(
                                self.wf.state_vector, [mux_op], self.wf.state_vector
                            )
                        )
                    )

                # <0|\mu_y|n> =
                # - <0|t_Zq (\sum_pq y_pq E_pq)|0> + <0|(\sum_pq y_pq E_pq) t_ZG |0> - <0|t_ZG (\sum_pq y_pq E_pq)|0>
                # + <0|(\sum_pq y_pq E_pq) t_Yq|0> + <0|(\sum_pq y_pq E_pq) t_YG |0> - <0|t_YG (\sum_pq y_pq E_pq)|0>
                if muy_op.operators != {}:
                    transition_dipole_y = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, muy_op], self.wf.state_vector
                        #    )
                        q_part_y
                        + (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [muy_op], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(
                                self.wf.state_vector, [t_ZG_op], self.wf.state_vector
                            )
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_ZG_op, muy_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [muy_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [muy_op, t_YG_op], self.wf.state_vector
                        )
                        - (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [t_YG_op], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(
                                self.wf.state_vector, [muy_op], self.wf.state_vector
                            )
                        )
                    )

                # <0|\mu_z|n> =
                # - <0|t_Zq (\sum_pq z_pq E_pq)|0> + <0|(\sum_pq z_pq E_pq) t_ZG |0> - <0|t_ZG (\sum_pq z_pq E_pq)|0>
                # + <0|(\sum_pq z_pq E_pq) t_Yq|0> + <0|(\sum_pq z_pq E_pq) t_YG |0> - <0|t_YG (\sum_pq z_pq E_pq)|0>
                if muz_op.operators != {}:
                    transition_dipole_z = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, muz_op], self.wf.state_vector
                        #    )
                        q_part_z
                        + (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [muz_op], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(
                                self.wf.state_vector, [t_ZG_op], self.wf.state_vector
                            )
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_ZG_op, muz_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [muz_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [muz_op, t_YG_op], self.wf.state_vector
                        )
                        - (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [t_YG_op], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(
                                self.wf.state_vector, [muz_op], self.wf.state_vector
                            )
                        )
                    )
            elif (
                self.do_statetransfer_operators
                or self.do_ST_projected_operators
                or self.do_hermitian_statetransfer_operators
                or self.do_selfconsistent_operators
            ):
                if mux_op.operators != {}:
                    transition_dipole_x = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, mux_op], self.wf.state_vector
                        #    )
                        q_part_x
                        - expectation_value_hybrid_flow(
                            self.csf, [t_ZG_op, self.U.dagger, mux_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [mux_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [mux_op, self.U, t_YG_op], self.csf
                        )
                    )

                # <0|\mu_y|n> =
                # - <0|t_Zq (\sum_pq y_pq E_pq)|0> + <0|(\sum_pq y_pq E_pq) t_ZG |0> - <0|t_ZG (\sum_pq y_pq E_pq)|0>
                # + <0|(\sum_pq y_pq E_pq) t_Yq|0> + <0|(\sum_pq y_pq E_pq) t_YG |0> - <0|t_YG (\sum_pq y_pq E_pq)|0>
                if muy_op.operators != {}:
                    transition_dipole_y = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, muy_op], self.wf.state_vector
                        #    )
                        q_part_y
                        - expectation_value_hybrid_flow(
                            self.csf, [t_ZG_op, self.U.dagger, muy_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [muy_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [muy_op, self.U, t_YG_op], self.csf
                        )
                    )

                # <0|\mu_z|n> =
                # - <0|t_Zq (\sum_pq z_pq E_pq)|0> + <0|(\sum_pq z_pq E_pq) t_ZG |0> - <0|t_ZG (\sum_pq z_pq E_pq)|0>
                # + <0|(\sum_pq z_pq E_pq) t_Yq|0> + <0|(\sum_pq z_pq E_pq) t_YG |0> - <0|t_YG (\sum_pq z_pq E_pq)|0>
                if muz_op.operators != {}:
                    transition_dipole_z = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, muz_op], self.wf.state_vector
                        #    )
                        q_part_z
                        - expectation_value_hybrid_flow(
                            self.csf, [t_ZG_op, self.U.dagger, muz_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [muz_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [muz_op, self.U, t_YG_op], self.csf
                        )
                    )
            # naive
            else:
                # <0|\mu_x|n> =
                # - <0|t_Zq (\sum_pq x_pq E_pq)|0> + <0|(\sum_pq x_pq E_pq) t_ZG |0> - <0|t_ZG (\sum_pq x_pq E_pq)|0>
                # + <0|(\sum_pq x_pq E_pq) t_Yq|0> + <0|(\sum_pq x_pq E_pq) t_YG |0> - <0|t_YG (\sum_pq x_pq E_pq)|0>
                if mux_op.operators != {}:
                    transition_dipole_x = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, mux_op], self.wf.state_vector
                        #    )
                        q_part_x
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [mux_op, t_ZG_op], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_ZG_op, mux_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [mux_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [mux_op, t_YG_op], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_YG_op, mux_op], self.wf.state_vector
                        )
                    )

                # <0|\mu_y|n> =
                # - <0|t_Zq (\sum_pq y_pq E_pq)|0> + <0|(\sum_pq y_pq E_pq) t_ZG |0> - <0|t_ZG (\sum_pq y_pq E_pq)|0>
                # + <0|(\sum_pq y_pq E_pq) t_Yq|0> + <0|(\sum_pq y_pq E_pq) t_YG |0> - <0|t_YG (\sum_pq y_pq E_pq)|0>
                if muy_op.operators != {}:
                    transition_dipole_y = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, muy_op], self.wf.state_vector
                        #    )
                        q_part_y
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [muy_op, t_ZG_op], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_ZG_op, muy_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [muy_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [muy_op, t_YG_op], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_YG_op, muy_op], self.wf.state_vector
                        )
                    )

                # <0|\mu_z|n> =
                # - <0|t_Zq (\sum_pq z_pq E_pq)|0> + <0|(\sum_pq z_pq E_pq) t_ZG |0> - <0|t_ZG (\sum_pq z_pq E_pq)|0>
                # + <0|(\sum_pq z_pq E_pq) t_Yq|0> + <0|(\sum_pq z_pq E_pq) t_YG |0> - <0|t_YG (\sum_pq z_pq E_pq)|0>
                if muz_op.operators != {}:
                    transition_dipole_z = (
                        #    -expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [t_Zq_op, muz_op], self.wf.state_vector
                        #    )
                        q_part_z
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [muz_op, t_ZG_op], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_ZG_op, muz_op], self.wf.state_vector
                        )
                        #    + expectation_value_hybrid_flow(
                        #        self.wf.state_vector, [muz_op, t_Yq_op], self.wf.state_vector
                        #    )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [muz_op, t_YG_op], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [t_YG_op, muz_op], self.wf.state_vector
                        )
                    )

        return transition_dipole_x, transition_dipole_y, transition_dipole_z

    def get_oscillator_strength(
        self, state_number: int, dipole_integrals: Sequence[np.ndarray], do_working_equations: bool = False
    ) -> float:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\left>\right|^2

        Args:
            state_number: Target excited state (zero being the first excited state).
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Oscillator Strength.
        """

        # Get <0|<mu_{x,y,z}|n>
        transition_dipole_x, transition_dipole_y, transition_dipole_z = self.get_transition_dipole(
            state_number, dipole_integrals, do_working_equations=do_working_equations
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
            "Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n"
        )
        for i, exc_energy in enumerate(self.excitation_energies):
            osc_strength = self.get_oscillator_strength(i, dipole_integrals)
            exc_str = f"{exc_energy:2.6f}"
            exc_str_ev = f"{exc_energy*27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i+1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output
