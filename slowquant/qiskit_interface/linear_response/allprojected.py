from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.qiskit_interface.linear_response.lr_baseclass import (
    get_num_CBS_elements,
    get_num_nonCBS,
    quantumLRBaseClass,
)
from slowquant.qiskit_interface.operators import (
    hamiltonian_pauli_2i_2a,
    one_elec_op_0i_0a,
)
from slowquant.qiskit_interface.util import Clique
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_response_property_gradient,
)


class quantumLR(quantumLRBaseClass):

    def run(
        self,
        do_gradients: bool = True,
    ) -> None:
        """Run simulation of all projected LR matrix elements.

        Args:
            do_gradients: Calculate gradients w.r.t. orbital rotations and active space excitations.
        """
        idx_shift = self.num_q
        print("Gs", self.num_G)
        print("qs", self.num_q)

        if self.num_q != 0:
            self.H_2i_2a = hamiltonian_pauli_2i_2a(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.num_virtual_orbs,
            )

        # pre-calculate <0|G|0> and <0|HG|0>
        self._G_exp = []
        self._HG_exp = []
        for GJ in self.G_ops:
            self._G_exp.append(self.wf.QI.quantum_expectation_value(GJ.get_folded_operator(*self.orbs)))
            self._HG_exp.append(
                self.wf.QI.quantum_expectation_value((self.H_0i_0a * GJ).get_folded_operator(*self.orbs))
            )

        # Check gradients
        if do_gradients:
            grad = np.zeros(2 * self.num_q)
            for i, op in enumerate(self.q_ops):
                grad[i] = self.wf.QI.quantum_expectation_value(
                    (self.H_1i_1a * op).get_folded_operator(*self.orbs)
                )
                grad[i + self.num_q] = self.wf.QI.quantum_expectation_value(
                    (op.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
                )
            if len(grad) != 0:
                print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
                if np.max(np.abs(grad)) > 10**-3:
                    print("WARNING: Large Gradient detected in q of ", np.max(np.abs(grad)))

            grad = np.zeros(self.num_G)  # G^\dagger is the same
            for i in range(self.num_G):
                grad[i] = self._HG_exp[i] - (self.wf.energy_elec * self._G_exp[i])
            if len(grad) != 0:
                print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
                if np.max(np.abs(grad)) > 10**-3:
                    print("WARNING: Large Gradient detected in G of ", np.max(np.abs(grad)))

        # qq
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                val = self.wf.QI.quantum_expectation_value(
                    (qI.dagger * self.H_2i_2a * qJ).get_folded_operator(*self.orbs)
                )
                qq_exp = self.wf.QI.quantum_expectation_value(
                    (qI.dagger * qJ).get_folded_operator(*self.orbs)
                )
                val -= qq_exp * self.wf.energy_elec
                self.A[i, j] = self.A[j, i] = val
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = qq_exp

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs)
                )

        # Calculate Matrices
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * self.H_0i_0a * GJ).get_folded_operator(*self.orbs)
                )
                GG_exp = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * GJ).get_folded_operator(*self.orbs)
                )
                val -= GG_exp * self.wf.energy_elec
                val -= self._G_exp[i] * self._HG_exp[j]
                val += self._G_exp[i] * self._G_exp[j] * self.wf.energy_elec
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                val = self._HG_exp[i] * self._G_exp[j]
                val -= self._G_exp[i] * self._G_exp[j] * self.wf.energy_elec
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = (
                    GG_exp - (self._G_exp[i] * self._G_exp[j])
                )

    def _get_qbitmap(
        self,
        cliques: bool = False,
    ) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
        """Get qubit map of operators.

        Args:
            cliques: If using cliques.

        Returns:
            Qubit map of operators.
        """
        idx_shift = self.num_q
        print("Gs", self.num_G)
        print("qs", self.num_q)

        A = [[""] * self.num_params for _ in range(self.num_params)]
        B = [[""] * self.num_params for _ in range(self.num_params)]
        Sigma = [[""] * self.num_params for _ in range(self.num_params)]

        # pre-calculate <0|G|0> and <0|HG|0>
        G_exp = []  # save and use for properties
        HG_exp = []
        for GJ in self.G_ops:
            G_exp.append(self.wf.QI.op_to_qbit(GJ.get_folded_operator(*self.orbs)).paulis.to_labels())
            HG_exp.append(
                self.wf.QI.op_to_qbit((self.H_0i_0a * GJ).get_folded_operator(*self.orbs)).paulis.to_labels()
            )
        energy = self.wf.QI.op_to_qbit((self.H_0i_0a).get_folded_operator(*self.orbs)).paulis.to_labels()

        # qq
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                val = self.wf.QI.op_to_qbit(
                    (qI.dagger * self.H_2i_2a * qJ).get_folded_operator(*self.orbs)
                ).paulis.to_labels()
                qq_exp = self.wf.QI.op_to_qbit(
                    (qI.dagger * qJ).get_folded_operator(*self.orbs)
                ).paulis.to_labels()
                val += qq_exp + energy
                A[i][j] = A[j][i] = val
                # Make Sigma
                Sigma[i][j] = Sigma[j][i] = qq_exp

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                A[j][i + idx_shift] = A[i + idx_shift][j] = self.wf.QI.op_to_qbit(
                    (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs)
                ).paulis.to_labels()

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = self.wf.QI.op_to_qbit(
                    (GI.dagger * self.H_0i_0a * GJ).get_folded_operator(*self.orbs)
                ).paulis.to_labels()
                GG_exp = self.wf.QI.op_to_qbit(
                    (GI.dagger * GJ).get_folded_operator(*self.orbs)
                ).paulis.to_labels()
                val += GG_exp + energy
                val += G_exp[i] + HG_exp[j]
                val += G_exp[i] + G_exp[j] + energy
                A[i + idx_shift][j + idx_shift] = A[j + idx_shift][i + idx_shift] = val
                # Make B
                val = HG_exp[i] + G_exp[j]
                val += G_exp[i] + G_exp[j] + energy
                B[i + idx_shift][j + idx_shift] = B[j + idx_shift][i + idx_shift] = val
                # Make Sigma
                Sigma[i + idx_shift][j + idx_shift] = Sigma[j + idx_shift][i + idx_shift] = (
                    GG_exp + G_exp[i] + G_exp[j]
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

    def _run_std(
        self,
        no_coeffs: bool = False,
        verbose: bool = True,
        cv: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get standard deviation in matrix elements of LR equation.

        Args:
            no_coeffs: Set all coefficients of Pauli strings to 1
            verbose: Print analysis per operator row
            cv: Perform coefficient of varation analysis
        """
        idx_shift = self.num_q
        print("Gs", self.num_G)
        print("qs", self.num_q)

        A = np.zeros((self.num_params, self.num_params))
        B = np.zeros((self.num_params, self.num_params))
        Sigma = np.zeros((self.num_params, self.num_params))

        if not hasattr(self, "_G_exp") or len(self._G_exp) == 0:
            # pre-calculate <0|G|0> and <0|HG|0>
            self._G_exp = []  # save and use for properties
            self._HG_exp = []
            for GJ in self.G_ops:
                self._G_exp.append(self.wf.QI.quantum_expectation_value(GJ.get_folded_operator(*self.orbs)))
                self._HG_exp.append(
                    self.wf.QI.quantum_expectation_value((self.H_0i_0a * GJ).get_folded_operator(*self.orbs))
                )
        # pre-calculate std of <0|G|0> and <0|HG|0>
        var_G_exp = []  # save and use for properties
        var_HG_exp = []
        for GJ in self.G_ops:
            var_G_exp.append(
                self.wf.QI.quantum_variance(GJ.get_folded_operator(*self.orbs), no_coeffs=no_coeffs)
            )
            var_HG_exp.append(
                self.wf.QI.quantum_variance(
                    (self.H_0i_0a * GJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                )
            )
        var_energy = self.wf.QI.quantum_variance(
            (self.H_0i_0a).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
        )

        # qq
        self.H_2i_2a = hamiltonian_pauli_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                qq_exp = self.wf.QI.quantum_expectation_value(
                    (qI.dagger * qJ).get_folded_operator(*self.orbs)
                )
                var_qq_exp = self.wf.QI.quantum_variance(
                    (qI.dagger * qJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                )
                # Make A
                val = self.wf.QI.quantum_variance(
                    (qI.dagger * self.H_2i_2a * qJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                )
                val += (qq_exp**2 + var_qq_exp) * (self.wf.energy_elec**2 + var_energy) - (
                    qq_exp**2 * self.wf.energy_elec**2
                )
                A[i, j] = A[j, i] = np.sqrt(val)
                # Make Sigma
                Sigma[i, j] = Sigma[j, i] = np.sqrt(var_qq_exp)

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                A[j, i + idx_shift] = A[i + idx_shift, j] = np.sqrt(
                    self.wf.QI.quantum_variance(
                        (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                    )
                )

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = self.wf.QI.quantum_variance(
                    (GI.dagger * self.H_0i_0a * GJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                )
                var_GG_exp = self.wf.QI.quantum_variance(
                    (GI.dagger * GJ).get_folded_operator(*self.orbs), no_coeffs=no_coeffs
                )
                GG_exp = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * GJ).get_folded_operator(*self.orbs)
                )
                # Var(A*B) = (\mu(A)^2 + var(A)) * (\mu(B)^2 + var(B)) - \mu(A)^2 \mu(B)^2
                val += (GG_exp**2 + var_GG_exp) * (self.wf.energy_elec**2 + var_energy) - (
                    GG_exp**2 * self.wf.energy_elec**2
                )
                val += (self._G_exp[i] ** 2 + var_G_exp[i]) * (self._HG_exp[j] ** 2 + var_HG_exp[j]) - (
                    self._G_exp[i] ** 2 * self._HG_exp[j] ** 2
                )
                val += (self._G_exp[i] ** 2 + var_G_exp[i]) * (self._G_exp[j] ** 2 + var_G_exp[j]) * (
                    self.wf.energy_elec**2 + var_energy
                ) - (self._G_exp[i] ** 2 * self._G_exp[j] ** 2 * self.wf.energy_elec**2)
                A[i + idx_shift, j + idx_shift] = A[j + idx_shift, i + idx_shift] = np.sqrt(val)
                # Make B
                val = (self._G_exp[j] ** 2 + var_G_exp[j]) * (self._HG_exp[i] ** 2 + var_HG_exp[i]) - (
                    self._G_exp[j] ** 2 * self._HG_exp[i] ** 2
                )
                val += (self._G_exp[i] ** 2 + var_G_exp[i]) * (self._G_exp[j] ** 2 + var_G_exp[j]) * (
                    self.wf.energy_elec**2 + var_energy
                ) - (self._G_exp[i] ** 2 * self._G_exp[j] ** 2 * self.wf.energy_elec**2)
                B[i + idx_shift, j + idx_shift] = B[j + idx_shift, i + idx_shift] = np.sqrt(val)
                # Make Sigma
                val = (self._G_exp[i] ** 2 + var_G_exp[i]) * (self._G_exp[j] ** 2 + var_G_exp[j]) - (
                    self._G_exp[i] ** 2 * self._G_exp[j] ** 2
                )
                Sigma[i + idx_shift, j + idx_shift] = Sigma[j + idx_shift, i + idx_shift] = np.sqrt(
                    var_GG_exp + val
                )

        if no_coeffs:
            cv = False
        self._analyze_std(A, B, Sigma, verbose=verbose, cv=cv)
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
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
        )

        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])
        mux_op = one_elec_op_0i_0a(mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        muy_op = one_elec_op_0i_0a(muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        muz_op = one_elec_op_0i_0a(muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs)

        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            q_part_x = get_orbital_response_property_gradient(
                rdms,
                mux,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_excitation_vectors,
                state_number,
                number_excitations,
            )
            q_part_y = get_orbital_response_property_gradient(
                rdms,
                muy,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_excitation_vectors,
                state_number,
                number_excitations,
            )
            q_part_z = get_orbital_response_property_gradient(
                rdms,
                muz,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_excitation_vectors,
                state_number,
                number_excitations,
            )
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            exp_mux = self.wf.QI.quantum_expectation_value(mux_op.get_folded_operator(*self.orbs))
            exp_muy = self.wf.QI.quantum_expectation_value(muy_op.get_folded_operator(*self.orbs))
            exp_muz = self.wf.QI.quantum_expectation_value(muz_op.get_folded_operator(*self.orbs))
            for i, G in enumerate(self.G_ops):
                exp_G = self._G_exp[i]
                exp_Gmux = self.wf.QI.quantum_expectation_value(
                    (G.dagger * mux_op).get_folded_operator(*self.orbs)
                )
                exp_Gmuy = self.wf.QI.quantum_expectation_value(
                    (G.dagger * muy_op).get_folded_operator(*self.orbs)
                )
                exp_Gmuz = self.wf.QI.quantum_expectation_value(
                    (G.dagger * muz_op).get_folded_operator(*self.orbs)
                )

                g_part_x += self._Z_G_normed[i, state_number] * exp_G * exp_mux
                g_part_x -= self._Z_G_normed[i, state_number] * exp_Gmux
                g_part_x -= self._Y_G_normed[i, state_number] * exp_G * exp_mux
                g_part_x += self._Y_G_normed[i, state_number] * exp_Gmux
                g_part_y += self._Z_G_normed[i, state_number] * exp_G * exp_muy
                g_part_y -= self._Z_G_normed[i, state_number] * exp_Gmuy
                g_part_y -= self._Y_G_normed[i, state_number] * exp_G * exp_muy
                g_part_y += self._Y_G_normed[i, state_number] * exp_Gmuy
                g_part_z += self._Z_G_normed[i, state_number] * exp_G * exp_muz
                g_part_z -= self._Z_G_normed[i, state_number] * exp_Gmuz
                g_part_z -= self._Y_G_normed[i, state_number] * exp_G * exp_muz
                g_part_z += self._Y_G_normed[i, state_number] * exp_Gmuz

            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
