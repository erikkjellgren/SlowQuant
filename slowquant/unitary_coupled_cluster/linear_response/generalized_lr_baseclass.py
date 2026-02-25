from collections.abc import Sequence

import numpy as np
import scipy

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G3,
    G4,
    G5,
    G6,
    G1,
    G2,
    G1_generalized, #AE added
    G2_generalized,
)
from slowquant.unitary_coupled_cluster.generalized_operators import (
    generalized_hamiltonian_0i_0a,
    generalized_hamiltonian_1i_1a,
    generalized_hamiltonian_full_space,
)
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.util import (
    UpsStructure,
    iterate_t1,
    iterate_t2,
    iterate_t3,
    iterate_t4,
    iterate_t5,
    iterate_t6,
    iterate_t6,
)


class LinearResponseBaseClass:
    index_info: tuple[CI_Info, list[float | complex], UpsStructure]

    def __init__(
        self,
        wave_function: GeneralizedWaveFunctionUPS, #Anna har fjernet WaveFunctionUCC + ændret til at alt løber over spin orbitaler
        excitations: str,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        self.wf = wave_function
        # if isinstance(self.wf, WaveFunctionUCC): udkommenteret for at køre uden UCC.
        #     self.index_info = (
        #         self.wf.ci_info,
        #         self.wf.thetas,
        #         self.wf.ucc_layout,
        #     )
        if isinstance(self.wf, GeneralizedWaveFunctionUPS):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ups_layout,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")

        self.G_ops: list[FermionicOperator] = []
        self.q_ops: list[FermionicOperator] = []
        excitations = excitations.lower()
        self.operator_labels_q = [] #AE
        self.operator_labels_G = [] #AE
        self.G_ops_finite: int = 0 #PERNILLE
        self.q_ops_finite: int = 0 #PERNILLE

        if "s" in excitations:
<<<<<<< HEAD
            # print("Active occupied spin idx") #AWE
            # print(self.wf.active_occ_spin_idx)
            # print("Active unooccupied spin idx")
            # print(self.wf.active_unocc_spin_idx)
            # print("Excitation idx")
            for a, i in iterate_t1(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx, is_spin_conserving=False): ## -diagonal jf HJ. Cross?
=======
            print("Active occupied spin idx") # AWE
            print(self.wf.active_occ_spin_idx) # AWE
            print("Active unooccupied spin idx") # AWE
            print(self.wf.active_unocc_spin_idx) # AWE
            print("Excitation idx")
            for a, i in iterate_t1(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx, is_spin_conserving=self.wf.ansatz_options["is_spin_conserving"]): ## -diagonal jf HJ. Cross?, # is_spin_conserving  AWE
>>>>>>> 32ecd3e9c871c322e41832b6fc7ea19ecd50836c
                self.G_ops.append(G1(i, a)) #AE from G1
                print('G1', i,a)
        if "d" in excitations:
            for a, i, b, j in iterate_t2(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx): 
                self.G_ops.append(G2(i, j, a, b)) #AE from G2
                print('G2',i, j, a, b)
                self.operator_labels_G.append(('G2',i,j,a,b))
        if "t" in excitations:
            for a, i, b, j, c, k in iterate_t3(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G3(i, j, k, a, b, c))
                print(i,k,k,a,b,c)
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

        for p, q in self.wf.kappa_no_activeactive_spin_idx:
            self.q_ops.append(G1(p, q)) #AE from G1 skal det være generalized??
            self.operator_labels_q.append(('q',p,q))
            print('qs:',p,q)
        # print('no active active', self.wf.kappa_no_activeactive_spin_idx)
        # print(operator_labels)
        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters), dtype=complex) #AE complex
        self.B = np.zeros((num_parameters, num_parameters), dtype=complex) #AE complex
        self.Sigma = np.zeros((num_parameters, num_parameters), dtype=complex) #AE complex
        self.Delta = np.zeros((num_parameters, num_parameters), dtype=complex) #AE complex
        self.H_1i_1a = generalized_hamiltonian_1i_1a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        # self.H_0i_0a = generalized_hamiltonian_0i_0a( #AE: virker ikke!
        #     self.wf.h_mo,
        #     self.wf.g_mo,
        #     self.wf.num_inactive_spin_orbs,
        #     self.wf.num_active_spin_orbs,
        # )
        

    def calc_excitation_energies(self) -> None:
        """Calculate excitation energies."""
        size = len(self.A)
        E2 = np.zeros((size * 2, size * 2), dtype=complex) #AE complex
        E2[:size, :size] = self.A
        E2[:size, size:] = self.B
        E2[size:, :size] = self.B.conjugate() #AE added conjugtate 
        E2[size:, size:] = self.A.conjugate() #AE added conjugtate 
        
        # print(E2)
        (
            hess_eigval,
            _,
        ) = np.linalg.eig(E2)
        print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")
        if np.abs(np.min(hess_eigval)) < 10**-8:
            print("WARNING: Small eigenvalue in Hessian")
        elif np.min(hess_eigval) < 0:
            print("Negative eigenvalue in Hessian.")
            #raise ValueError("Negative eigenvalue in Hessian.")

        S = np.zeros((size * 2, size * 2), dtype=complex) #AE complex
        S[:size, :size] = self.Sigma
        S[:size, size:] = self.Delta
        S[size:, :size] = -self.Delta.conjugate()
        S[size:, size:] = -self.Sigma.conjugate()

        #print(np.linalg.eig(S)) AWE
        
        # print(S)
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")
        self.hessian = E2
        self.metric = S

        # for i in range(len(self.hessian)):
        #     print(self.hessian[i][i], i)
                
        eigval, eigvec = scipy.linalg.eig(self.hessian, self.metric)      
        # # print('eigenval',eigval)
        # for i in range(len(eigval)):
        #     print('eigenval',eigval[i])
        #     print('eigvec ',eigvec[i])
        #     print('eigenvectors', max(abs(eigvec[i])))
        #     print('eigenvector index', np.argmax(abs(eigvec[i])))
        #     print('NEW INDEX')
        # print('eigvec ',eigvec)
        
      
        # for i in range(len(eigval)):
        #     print('eigenval',eigval[i])
                # print('eigvec ',eigvec[i])
                # print('eigenvectors', max(abs(eigvec[i])))
                # print('eigenvector index', np.argmax(abs(eigvec[i])))
                # print('NEW INDEX')
                
            # num=(eigvec.conj().T@(self.hessian)@eigvec)
            # den=(eigvec.conj().T@(self.metric)@eigvec)
            # print('eigenvalues on the diagonal?',num/den)
            
            # vec=eigvec[:,i]
            # for j in range(len(vec)):
            #     print(vec[j], self.operator_labels[j])
                
                
            # print('Max value eigvec', max(abs(vec)), 'Max value eigvec index', np.argmax(abs(vec)))
            # k = np.argmax(np.abs(vec))
            # print("dominant operator:", self.operator_labels[k])

            # print(self.operator_labels[k])

            
   
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
        self.normed_response_vectors = np.zeros_like(self.response_vectors, dtype=complex) #AE
        self.num_q = len(self.q_ops)
        
        
        #PERNILLES
        # self.num_q = self.q_ops_finite
        # self.num_qG = size
        
        
        self.num_G = size - self.num_q
        self.Z_q = self.response_vectors[: self.num_q, :]
        self.Z_G = self.response_vectors[self.num_q : self.num_q + self.num_G, :]
        self.Y_q = self.response_vectors[self.num_q + self.num_G : 2 * self.num_q + self.num_G]
        self.Y_G = self.response_vectors[2 * self.num_q + self.num_G :]
        
        
        #Pernille
        # self.Z_qG = self.response_vectors[: self.num_qG, :]
        # self.Y_qG = self.response_vectors[self.num_qG :]
        # self.Z_qG_normed = np.zeros_like(self.Z_qG, dtype=complex)
        # self.Y_qG_normed = np.zeros_like(self.Y_qG, dtype=complex)
        
        self.Z_q_normed = np.zeros_like(self.Z_q, dtype=complex) #AE
        self.Z_G_normed = np.zeros_like(self.Z_G, dtype=complex) #AE
        self.Y_q_normed = np.zeros_like(self.Y_q, dtype=complex) #AE
        self.Y_G_normed = np.zeros_like(self.Y_G, dtype=complex) #AE
        
        norms = self.get_excited_state_norm()
        for state_number, norm in enumerate(norms):
            if norm < 10**-10:
                print(f"WARNING: State number {state_number} could not be normalized. Norm of {norm}.")
                continue
            self.Z_q_normed[:, state_number] = self.Z_q[:, state_number] * (1 / norm) ** 0.5
            self.Z_G_normed[:, state_number] = self.Z_G[:, state_number] * (1 / norm) ** 0.5
            self.Y_q_normed[:, state_number] = self.Y_q[:, state_number] * (1 / norm) ** 0.5
            self.Y_G_normed[:, state_number] = self.Y_G[:, state_number] * (1 / norm) ** 0.5
            
            #Pernille
            # self.Z_qG_normed[:, state_number] = self.Z_qG[:, state_number] * (1 / norm) ** 0.5
            # self.Y_qG_normed[:, state_number] = self.Y_qG[:, state_number] * (1 / norm) ** 0.5
        
            
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def get_excited_state_norm(self) -> np.ndarray:
        """Calculate the norm of excited states.

        Returns:
            Norm of excited states.
        """
        norms = np.zeros(len(self.response_vectors[0]),dtype=complex) #AE complex
        for state_number in range(len(self.response_vectors[0])):
            # Get Z_q Z_G Y_q and Y_G matrices
            ZZq = np.outer(self.Z_q[:, state_number], self.Z_q[:, state_number].transpose())
            YYq = np.outer(self.Y_q[:, state_number], self.Y_q[:, state_number].transpose())
            ZZG = np.outer(self.Z_G[:, state_number], self.Z_G[:, state_number].transpose())
            YYG = np.outer(self.Y_G[:, state_number], self.Y_G[:, state_number].transpose())

            #Pernille
            # ZZqG = np.outer(self.Z_qG[:, state_number], self.Z_qG[:, state_number].transpose())
            # YYqG = np.outer(self.Y_qG[:, state_number], self.Y_qG[:, state_number].transpose())
            # norms[state_number] = np.sum(self.metric[: self.num_qG, : self.num_qG] * (ZZqG - YYqG))
            
            
            norms[state_number] = np.sum(self.metric[: self.num_q, : self.num_q] * (ZZq - YYq)) + np.sum(
                self.metric[self.num_q : self.num_q + self.num_G, self.num_q : self.num_q + self.num_G]
                * (ZZG - YYG)
            )
        return norms

    def get_transition_dipole(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        """Calculate transition dipole moment.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Transition dipole moment.
        """
        raise NotImplementedError

    def get_oscillator_strength(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\right>\right|^2

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Oscillator Strength.
        """
        transition_dipoles = self.get_transition_dipole(dipole_integrals)
        osc_strs = np.zeros(len(transition_dipoles))
        for idx, (excitation_energy, transition_dipole) in enumerate(
            zip(self.excitation_energies, transition_dipoles)
        ):
            osc_strs[idx] = (
                2
                / 3
                * excitation_energy
                * (transition_dipole[0] ** 2 + transition_dipole[1] ** 2 + transition_dipole[2] ** 2)
            )
        self.oscillator_strengths = osc_strs
        return osc_strs

    def get_formatted_oscillator_strength(self) -> str:
        """Create table of excitation energies and oscillator strengths.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Nicely formatted table.
        """
        if not hasattr(self, "oscillator_strengths"):
            raise ValueError(
                "Oscillator strengths have not been calculated. Run get_oscillator_strength() first."
            )

        output = (
            "Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n"
        )

        for i, (exc_energy, osc_strength) in enumerate(
            zip(self.excitation_energies, self.oscillator_strengths)
        ):
            exc_str = f"{exc_energy:2.6f}"
            exc_str_ev = f"{exc_energy * 27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i + 1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output