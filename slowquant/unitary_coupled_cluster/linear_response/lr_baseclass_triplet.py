import numpy as np
import scipy

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators_triplet import (
    G3,
    G4,
    G5,
    G6,
    Tpq,
    G1_sa_t,
    G2_1_sa_t,
    G2_2_sa_t,
    G2_3_sa_t,
    hamiltonian_0i_0a,
    hamiltonian_1i_1a,
    hamiltonian_2i_2a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction_triplet import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.util_triplet import (
    UccStructure,
    UpsStructure,
    iterate_t1_sa,
    iterate_t2_sa_t,
    iterate_t3,
    iterate_t4,
    iterate_t5,
    iterate_t6,
)

from slowquant.unitary_coupled_cluster.operator_matrix import (
    expectation_value,
)

class LinearResponseBaseClass:
    index_info: (
        tuple[list[int], dict[int, int], int, int, int, int, int, list[float], UpsStructure]
        | tuple[list[int], dict[int, int], int, int, int, int, int, list[float], UccStructure]
    )

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
        self.wf = wave_function
        if isinstance(self.wf, WaveFunctionUCC):
            self.index_info = (
                self.wf.idx2det,
                self.wf.det2idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.num_virtual_orbs,
                self.wf.num_active_elec_alpha,
                self.wf.num_active_elec_beta,
                self.wf.thetas,
                self.wf.ucc_layout,
            )
        elif isinstance(self.wf, WaveFunctionUPS):
            self.index_info = (
                self.wf.idx2det,
                self.wf.det2idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.num_virtual_orbs,
                self.wf.num_active_elec_alpha,
                self.wf.num_active_elec_beta,
                self.wf.thetas,
                self.wf.ups_layout,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")

        self.G_ops: list[FermionicOperator] = []
        self.q_ops: list[FermionicOperator] = []
        excitations = excitations.lower()

        if "s" in excitations:
            for a, i, _ in iterate_t1_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G1_sa_t(i, a))
        if "d" in excitations:
            for a, i, b, j, _, op_type in iterate_t2_sa_t(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                if op_type == 1:
                    self.G_ops.append(G2_1_sa_t(i, j, a, b))
                elif op_type == 2:
                    self.G_ops.append(G2_2_sa_t(i, j, a, b))
                elif op_type == 3:
                    self.G_ops.append(G2_3_sa_t(i, j, a, b))
        if "t" in excitations:
            for a, i, b, j, c, k in iterate_t3(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G3(i, j, k, a, b, c))
        if "q" in excitations:
            for a, i, b, j, c, k, d, l in iterate_t4(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G4(i, j, k, l, a, b, c, d))
        if "5" in excitations:
            for a, i, b, j, c, k, d, l, e, m in iterate_t5(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G5(i, j, k, l, m, a, b, c, d, e))
        if "6" in excitations:
            for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G6(i, j, k, l, m, n, a, b, c, d, e, f))
        for i, a in self.wf.kappa_no_activeactive_idx:
            op = 2 ** (-1 / 2) * Tpq(a, i)
            self.q_ops.append(op)

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))
        self.H_2i_2a = hamiltonian_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )
        self.H_1i_1a = hamiltonian_1i_1a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )
        self.H_0i_0a = hamiltonian_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
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
        if np.abs(np.min(hess_eigval)) < 10**-8:
            print("WARNING: Small eigenvalue in Hessian")
        elif np.min(hess_eigval) < 0:
            raise ValueError("Negative eigenvalue in Hessian.")

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.Sigma
        S[:size, size:] = self.Delta
        S[size:, :size] = -self.Delta
        S[size:, size:] = -self.Sigma
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")

        self.hessian = E2
        self.metric = S

        eigval, eigvec = scipy.linalg.eig(self.hessian, self.metric)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
        self.normed_response_vectors = np.zeros_like(self.response_vectors)
        self.num_q = len(self.q_ops)
        self.num_G = size - self.num_q
        self.Z_q = self.response_vectors[: self.num_q, :]
        self.Z_G = self.response_vectors[self.num_q : self.num_q + self.num_G, :]
        self.Y_q = self.response_vectors[self.num_q + self.num_G : 2 * self.num_q + self.num_G]
        self.Y_G = self.response_vectors[2 * self.num_q + self.num_G :]
        self.Z_q_normed = np.zeros_like(self.Z_q)
        self.Z_G_normed = np.zeros_like(self.Z_G)
        self.Y_q_normed = np.zeros_like(self.Y_q)
        self.Y_G_normed = np.zeros_like(self.Y_G)
        norms = self.get_excited_state_norm()
        for state_number, norm in enumerate(norms):
            if norm < 10**-10:
                print(f"WARNING: State number {state_number} could not be normalized. Norm of {norm}.")
                continue
            self.Z_q_normed[:, state_number] = self.Z_q[:, state_number] * (1 / norm) ** 0.5
            self.Z_G_normed[:, state_number] = self.Z_G[:, state_number] * (1 / norm) ** 0.5
            self.Y_q_normed[:, state_number] = self.Y_q[:, state_number] * (1 / norm) ** 0.5
            self.Y_G_normed[:, state_number] = self.Y_G[:, state_number] * (1 / norm) ** 0.5
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def get_excited_state_norm(self) -> np.ndarray:
        """Calculate the norm of excited states.

        Returns:
            Norm of excited states.
        """
        norms = np.zeros(len(self.response_vectors[0]))
        for state_number in range(len(self.response_vectors[0])):
            # Get Z_q Z_G Y_q and Y_G matrices
            ZZq = np.outer(self.Z_q[:, state_number], self.Z_q[:, state_number].transpose())
            YYq = np.outer(self.Y_q[:, state_number], self.Y_q[:, state_number].transpose())
            ZZG = np.outer(self.Z_G[:, state_number], self.Z_G[:, state_number].transpose())
            YYG = np.outer(self.Y_G[:, state_number], self.Y_G[:, state_number].transpose())

            norms[state_number] = np.sum(self.metric[: self.num_q, : self.num_q] * (ZZq - YYq)) + np.sum(
                self.metric[self.num_q : self.num_q + self.num_G, self.num_q : self.num_q + self.num_G]
                * (ZZG - YYG)
            )

        return norms
    
    def get_property_gradient(self, property_integrals):
        """Calculate property gradient.

        Args:
            property_integrals: Integrals (x,y,z) in AO basis.

        Returns:
            Property gradient.
        """
        mo_coeffs = self.wf.c_trans
        #out_shape = property_integrals[:-2]
        #property_integrals = property_integrals.reshape(-1, len(mo_coeffs), len(mo_coeffs))
        mo_integrals = np.einsum('uj,xuv,vi->xij', mo_coeffs, property_integrals, mo_coeffs)

        ops = self.q_ops + self.G_ops
        pg = []

        for G in (ops):
            u = np.zeros(len(mo_integrals), dtype=np.complex128)
            for i in range(len(mo_integrals[0])):
                for j in range(len(mo_integrals[0])):
                    P = Tpq(i, j)
                    ExpectationValue = expectation_value(
                        self.wf.ci_coeffs,
                        [G*P - P*G],
                        self.wf.ci_coeffs,
                        *self.index_info)

                    u[:] += mo_integrals[:,i,j]*ExpectationValue
            pg.extend(u)
        
        for op in (ops):
            G = op.dagger
            u = np.zeros(len(mo_integrals), dtype=np.complex128)
            for i in range(len(mo_integrals[0])):
                for j in range(len(mo_integrals[0])):
                    P = Tpq(i, j)
                    ExpectationValue = expectation_value(
                        self.wf.ci_coeffs,
                        [G*P - P*G],
                        self.wf.ci_coeffs,
                        *self.index_info)

                    u[:] += mo_integrals[:,i,j]*ExpectationValue
            pg.extend(u)

        #return np.reshape(pg, (len(mo_integrals),-1),  order='F').reshape(*out_shape, -1)
        return np.reshape(pg, (len(mo_integrals),-1),  order='F')
