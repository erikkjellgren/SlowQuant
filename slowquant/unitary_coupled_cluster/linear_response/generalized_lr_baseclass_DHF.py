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
    DHF_hamiltonian_0i_0a,
    DHF_hamiltonian_1i_1a,
    DHF_hamiltonian_full_space,
)
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.util import (
    UpsStructure,
    iterate_t1,
    iterate_t2,
    iterate_t3,
    iterate_t4,
    iterate_t5,
    iterate_t6,
    iterate_t6,
    iterate_t2_generalized
)


class LinearResponseBaseClass:
    index_info: tuple[CI_Info, list[float | complex], UpsStructure]

    def __init__(
        self,
        wave_function: GeneralizedWaveFunctionUPS, #Anna har fjernet WaveFunctionUCC + ændret til at alt løber over spin orbitaler
        excitations: str,
        screen: bool,
        thresh_A: float,
        thresh_m: float,
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
        self.q_ops_old: list[FermionicOperator] = []
        excitations = excitations.lower()
        self.operator_labels_q = [] #AE
        self.operator_labels_q_old = [] #AE
        self.operator_labels_G = [] #AE
        self.screen = screen
        self.thresh_A = thresh_A
        self.thresh_m = thresh_m


        if "s" in excitations:
            for a, i in iterate_t1(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx, is_spin_conserving=False): ## -diagonal jf HJ. Cross?, # is_spin_conserving  AWE
                self.G_ops.append(G1(i, a)) #AE from G1
                #print('G1', i,a)
                self.operator_labels_G.append(('G1',i,a))
        if "d" in excitations:
            for a, i, b, j in iterate_t2(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx, is_spin_conserving=False): 
                self.G_ops.append(G2(i, j, a, b)) #AE from G2
                #print('G2',i, j, a, b)
                self.operator_labels_G.append(('G2',i,j,a,b))
        if "t" in excitations:
            for a, i, b, j, c, k in iterate_t3(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G3(i, j, k, a, b, c))
                # print('G3',a, i, b, j, c, k)
                # print(i,k,k,a,b,c)
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
            self.q_ops_old.append(G1(p, q))
            self.operator_labels_q_old.append(('q',p,q))
        for p, q in self.wf.kappa_no_activeactive_spin_idx_resp:
            self.q_ops.append(G1(p, q))
            self.operator_labels_q.append(('q',p,q))

        self.finite_excitations_idx: list[bool] = [True] * (len(self.q_ops)+len(self.G_ops))
        self.q_ops_finite: list[bool] = [True] * len(self.q_ops)
        self.G_ops_finite: list[bool] = [True] * len(self.G_ops)
        self.num_q_ops_finite = len(self.q_ops)
        self.num_G_ops_finite = len(self.G_ops)


        # Hessian and metric:
        num_parameters = len(self.G_ops) + len(self.q_ops)

        self.A = np.zeros((num_parameters, num_parameters), dtype=complex) #AE complex
        self.B = np.zeros((num_parameters, num_parameters), dtype=complex) #AE complex
        self.Sigma = np.zeros((num_parameters, num_parameters), dtype=complex) #AE complex
        self.Delta = np.zeros((num_parameters, num_parameters), dtype=complex) #AE complex

        self.hessian = None
        self.metric = None

        # Hamiltonians:
        self.H_1i_1a = DHF_hamiltonian_1i_1a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
            self.wf.num_spin_orbs_NES,
        )
        self.H_0i_0a = DHF_hamiltonian_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_spin_orbs_NES,
        )

        self.H = DHF_hamiltonian_full_space(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_spin_orbs_NES,
        )
        

        # Shieldings and coupling constants:
        self.shieldings = None
        self.sscc = None
        

    def calc_excitation_energies(self) -> None:
        """Calculate excitation energies."""
        size = len(self.G_ops) + len(self.q_ops)

        (
            hess_eigval,
            _,
        ) = np.linalg.eig(self.hessian)
        print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")
        if np.abs(np.min(hess_eigval)) < 10**-8:
            print("WARNING: Small eigenvalue in Hessian")
        elif np.min(hess_eigval) < 0:
            print("Negative eigenvalue in Hessian.")
            #raise ValueError("Negative eigenvalue in Hessian.")
        # for i in hess_eigval:
        #     if i < 0:
        #         print("Negative eigenvalue in Hessian:",i)
        #     #raise ValueError("Negative eigenvalue in Hessian.")
        
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")

        eigval, eigvec = la.eig(self.hessian, self.metric) 

        # Extra screening in the metric:        
        #eigval, eigvec, sigma_eigs, keep = solve_lr_drop_sigma_null(self.hessian, self.metric, cut=self.thresh_m)
   
        sorting = np.argsort(np.real(eigval.real)) #AE added np.real
        tmp = eigval[sorting][size:]
        self.excitation_energies = np.real(tmp[tmp < 1e4]) 
        self.response_vectors = (eigvec[:, sorting][:, size:]) #Removed np.real
        self.normed_response_vectors = np.zeros_like(self.response_vectors, dtype=complex) #AE

        self.num_q = len(self.q_ops)       
        self.num_G = len(self.G_ops)


        self.Z_q = self.response_vectors[: self.num_q, :]
        self.Z_G = self.response_vectors[self.num_q : self.num_q + self.num_G, :]
        self.Y_q = self.response_vectors[self.num_q + self.num_G : 2 * self.num_q + self.num_G]
        self.Y_G = self.response_vectors[2 * self.num_q + self.num_G :]
        
        self.Z_q_normed = np.zeros_like(self.Z_q, dtype=complex) #AE
        self.Z_G_normed = np.zeros_like(self.Z_G, dtype=complex) #AE
        self.Y_q_normed = np.zeros_like(self.Y_q, dtype=complex) #AE
        self.Y_G_normed = np.zeros_like(self.Y_G, dtype=complex) #AE


        norms = self.get_excited_state_norm()
        for state_number, norm in enumerate(norms):
            if abs(norm) < 10**-10: #AE change to abs
                print(f"WARNING: State number {state_number} could not be normalized. Norm of {norm}.")
                continue
            self.Z_q_normed[:, state_number] = self.Z_q[:, state_number] * (1/abs(norm))**0.5 * np.sign(norm.real) # AE added abs and np.sign
            self.Z_G_normed[:, state_number] = self.Z_G[:, state_number] * (1/abs(norm))**0.5 * np.sign(norm.real)
            self.Y_q_normed[:, state_number] = self.Y_q[:, state_number] * (1/abs(norm))**0.5 * np.sign(norm.real)
            self.Y_G_normed[:, state_number] = self.Y_G[:, state_number] * (1/abs(norm))**0.5 * np.sign(norm.real)
            
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1/abs(norm))**0.5 * np.sign(norm.real)  #AE added abs
            )

        with np.printoptions(precision=5, suppress=True, formatter={'float_kind': lambda x: f'{x:.5f}'}):
            print("Excitation energies:", self.excitation_energies)



    def get_excited_state_norm(self) -> np.ndarray:
        """Calculate the norm of excited states.

        Returns:
            Norm of excited states.
        """
        norms = np.zeros(len(self.response_vectors[0]),dtype=complex) #AE complex
        for state_number in range(len(self.response_vectors[0])):
            # Get Z_q Z_G Y_q and Y_G matrices
            ZZq = np.outer(self.Z_q[:, state_number], self.Z_q[:, state_number].conj().T) #AE transpose to conj
            YYq = np.outer(self.Y_q[:, state_number], self.Y_q[:, state_number].conj().T)
            ZZG = np.outer(self.Z_G[:, state_number], self.Z_G[:, state_number].conj().T)
            YYG = np.outer(self.Y_G[:, state_number], self.Y_G[:, state_number].conj().T)
            
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
    
    
    
   
   
import scipy.linalg as la
def solve_lr_drop_sigma_null(H, sigma, cut=1e-10):
    # Hermitize (important for numerical stability)
    Hh = 0.5*(H + H.conj().T)
    Sh = 0.5*(sigma + sigma.conj().T)
    # 1) eigen-decompose metric
    s, U = la.eigh(Sh)
    # 2) keep only non-null directions
    keep = np.abs(s) > cut * np.max(np.abs(s))
    Uk = U[:, keep]
    # 3) project both matrices consistently
    Hk = Uk.conj().T @ Hh @ Uk
    Sk = Uk.conj().T @ Sh @ Uk
    # 4) solve reduced generalized eigenproblem
    w, y = la.eig(Hk, Sk)
    # 5) backtransform eigenvectors
    v = Uk @ y
    return w, v, s, keep
 