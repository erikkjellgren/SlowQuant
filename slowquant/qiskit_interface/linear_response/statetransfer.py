import numpy as np

from slowquant.qiskit_interface.linear_response.lr_baseclass import quantumLRBaseClass
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    get_determinant_expansion_from_operator_on_HF,
)
from slowquant.unitary_coupled_cluster.operators import one_elec_op_0i_0a, hamiltonian_0i_0a

class quantumLR(quantumLRBaseClass):
    def run(
        self,
        skip_orbital_rotations: bool = False,
        do_gradients: bool = True,
    ) -> None:
        """Run simulation of naive LR matrix elements.

        Args:
            skip_orbital_rotations: Skip orbital rotations in the linear response equations.
            do_gradients: Calculate gradients w.r.t. orbital rotations and active space excitations.
        """
        print("Gs", self.num_G)
        self.A = np.zeros((self.num_G, self.num_G))
        self.B = np.zeros((self.num_G, self.num_G))
        self.Sigma = np.zeros((self.num_G, self.num_G))
        self.states = {}
        hf_det = ""
        for i in range(2 * self.wf.num_active_orbs):
            if i % 2 == 0 and i // 2 < self.wf.num_active_elec_alpha:
                hf_det += "1"
                continue
            if i % 2 == 1 and i // 2 < self.wf.num_active_elec_beta:
                hf_det += "1"
                continue
            hf_det += "0"
        self.states = {"HF": ([1.0], [hf_det])}
        for i, G in enumerate(self.G_ops):
            coeffs, dets = get_determinant_expansion_from_operator_on_HF(
                G.get_folded_operator(*self.orbs),
                self.wf.num_active_orbs,
                self.wf.num_active_elec_alpha,
                self.wf.num_active_elec_beta,
            )
            self.states[f"G{i}"] = (coeffs, dets)

        if self.num_q != 0 and not skip_orbital_rotations:
            raise NotImplementedError(
                "Found orbital rotations and skip_orbital_rotations is set to False. Self-consistent is only implemented for active space parameters."
            )
        H_active = self.H_0i_0a.get_folded_operator(*self.orbs)
        if do_gradients:
            grad = np.zeros(2 * self.num_G)
            for i in range(self.num_G):
                # <CSF| Ud H U G |CSF>
                grad[i] = self.wf.QI.quantum_expectation_value_csfs(
                    self.states["HF"], H_active, self.states[f"G{i}"]
                )
                # <CSF| Gd Ud H U |HF>
                grad[i + self.num_G] = self.wf.QI.quantum_expectation_value_csfs(
                    self.states[f"G{i}"], H_active, self.states["HF"]
                )
            if len(grad) != 0:
                print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
                if np.max(np.abs(grad)) > 10**-3:
                    print("WARNING: Large Gradient detected in G of ", np.max(np.abs(grad)))

        # GG
        for j in range(self.num_G):
            for i in range(j, self.num_G):
                # Make A
                # <CSF| GId Ud H U GJ |CSF>
                val = self.wf.QI.quantum_expectation_value_csfs(
                    self.states[f"G{i}"], H_active, self.states[f"G{j}"]
                )
                # - delta_IJ E0
                if i == j:
                    val -= self.wf.energy_elec
                self.A[i, j] = self.A[j, i] = val
                # Make Sigma
                if i == j:
                    self.Sigma[i, j] = 1

    def get_property_gradient(self, int1e: np.ndarray, int2e: np.ndarray | None = None) -> np.ndarray:
        """Calculate property gradient.

        Args:
            int1e: one-electron property integrals in MO basis.
            int2e: two-electron property integrals in MO basis.

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

        V = np.zeros((len(self.G_ops), len(int1e)))

        for mu, int1e_mu in enumerate(int1e):
            if int2e is None:
                op = one_elec_op_0i_0a(int1e_mu, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.triplet)
            else:
                op = hamiltonian_0i_0a(int1e_mu, int2e[mu], self.wf.num_inactive_orbs, self.wf.num_active_orbs)
            for idx in range(self.G_ops):
                V[idx, mu] = self.wf.QI.quantum_expectation_value_csfs(
                    self.states["HF"], 
                    op.get_folded_operator(*self.orbs),
                    self.states[f"G{idx}"])
        
        return np.vstack((V, fac * V))