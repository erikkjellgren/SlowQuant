import numpy as np

from slowquant.qiskit_interface.linear_response.lr_baseclass import (
    get_num_CBS_elements,
    get_num_nonCBS,
    quantumLRBaseClass,
)
from slowquant.qiskit_interface.util import Clique
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_response_property_gradient_1e,
    get_orbital_response_property_gradient_2e,
)
from slowquant.unitary_coupled_cluster.operators import (
    hamiltonian_2i_2a,
    one_elec_op_0i_0a,
    hamiltonian_0i_0a,
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
            self.H_2i_2a = hamiltonian_2i_2a(
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
                val += self._G_exp[i] * self._G_exp[j] * self.wf.energy_elec
                val -= 1 / 2 * self._G_exp[i] * self._HG_exp[j]
                val -= 1 / 2 * self._G_exp[j] * self._HG_exp[i]
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                val = 1 / 2 * self._HG_exp[i] * self._G_exp[j]
                val += 1 / 2 * self._HG_exp[j] * self._G_exp[i]
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
                val += G_exp[j] + HG_exp[i]
                val += G_exp[i] + G_exp[j] + energy
                A[i + idx_shift][j + idx_shift] = A[j + idx_shift][i + idx_shift] = val
                # Make B
                val = HG_exp[i] + G_exp[j]
                val = HG_exp[j] + G_exp[i]
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
        self.H_2i_2a = hamiltonian_2i_2a(
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
                val += (
                    1
                    / 2
                    * (
                        (self._G_exp[i] ** 2 + var_G_exp[i]) * (self._HG_exp[j] ** 2 + var_HG_exp[j])
                        - (self._G_exp[i] ** 2 * self._HG_exp[j] ** 2)
                    )
                )
                val += (
                    1
                    / 2
                    * (
                        (self._G_exp[j] ** 2 + var_G_exp[j]) * (self._HG_exp[i] ** 2 + var_HG_exp[i])
                        - (self._G_exp[j] ** 2 * self._HG_exp[i] ** 2)
                    )
                )
                val += (self._G_exp[i] ** 2 + var_G_exp[i]) * (self._G_exp[j] ** 2 + var_G_exp[j]) * (
                    self.wf.energy_elec**2 + var_energy
                ) - (self._G_exp[i] ** 2 * self._G_exp[j] ** 2 * self.wf.energy_elec**2)
                A[i + idx_shift, j + idx_shift] = A[j + idx_shift, i + idx_shift] = np.sqrt(val)
                # Make B
                val = (
                    1
                    / 2
                    * (
                        (self._G_exp[j] ** 2 + var_G_exp[j]) * (self._HG_exp[i] ** 2 + var_HG_exp[i])
                        - (self._G_exp[j] ** 2 * self._HG_exp[i] ** 2)
                    )
                )
                val += (
                    1
                    / 2
                    * (
                        (self._G_exp[i] ** 2 + var_G_exp[i]) * (self._HG_exp[j] ** 2 + var_HG_exp[j])
                        - (self._G_exp[i] ** 2 * self._HG_exp[j] ** 2)
                    )
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
                V[idx + idx_shift_q, mu] = self.wf.QI.quantum_expectation_value((op).get_folded_operator(*self.orbs)) * self._G_exp[idx]
                V[idx + idx_shift_q, mu] -= self.wf.QI.quantum_expectation_value((op * G).get_folded_operator(*self.orbs))
        
        return np.vstack((V, fac * V))
