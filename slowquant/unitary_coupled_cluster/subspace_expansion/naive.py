import numpy as np
import scipy

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import (
    G3,
    G4,
    G5,
    G6,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
    hamiltonian_0i_0a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.util import (
    UccStructure,
    UpsStructure,
    iterate_t1_sa,
    iterate_t2_sa,
    iterate_t3,
    iterate_t4,
    iterate_t5,
    iterate_t6,
)


class SubspaceExpansion:
    index_info: tuple[CI_Info, list[float], UpsStructure] | tuple[CI_Info, list[float], UccStructure]

    def __init__(
        self, wave_function: WaveFunctionUCC | WaveFunctionUPS, excitations: str, do_TDA: bool = False
    ) -> None:
        """Initialize subspace expansion by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            do_TDA: Apply Tamm-Dancoff approximation (default: False).
        """
        self.wf = wave_function
        if isinstance(self.wf, WaveFunctionUCC):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ucc_layout,
            )
        elif isinstance(self.wf, WaveFunctionUPS):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ups_layout,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")
        self.G_ops: list[FermionicOperator] = [FermionicOperator({"": []}, {"": 1.0})]
        excitations = excitations.lower()
        if "s" in excitations:
            for a, i, _ in iterate_t1_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G1_sa(i, a))
                if not do_TDA:
                    self.G_ops.append(G1_sa(i, a).dagger)
        if "d" in excitations:
            for a, i, b, j, _, op_type in iterate_t2_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                if op_type == 1:
                    self.G_ops.append(G2_1_sa(i, j, a, b))
                    if not do_TDA:
                        self.G_ops.append(G2_1_sa(i, j, a, b).dagger)
                elif op_type == 2:
                    self.G_ops.append(G2_2_sa(i, j, a, b))
                    if not do_TDA:
                        self.G_ops.append(G2_2_sa(i, j, a, b).dagger)
        if "t" in excitations:
            for a, i, b, j, c, k in iterate_t3(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G3(i, j, k, a, b, c))
                if not do_TDA:
                    self.G_ops.append(G3(i, j, k, a, b, c).dagger)
        if "q" in excitations:
            for a, i, b, j, c, k, d, l in iterate_t4(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G4(i, j, k, l, a, b, c, d))
                if not do_TDA:
                    self.G_ops.append(G4(i, j, k, l, a, b, c, d).dagger)
        if "5" in excitations:
            for a, i, b, j, c, k, d, l, e, m in iterate_t5(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G5(i, j, k, l, m, a, b, c, d, e))
                if not do_TDA:
                    self.G_ops.append(G5(i, j, k, l, m, a, b, c, d, e).dagger)
        if "6" in excitations:
            for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G6(i, j, k, l, m, n, a, b, c, d, e, f))
                if not do_TDA:
                    self.G_ops.append(G6(i, j, k, l, m, n, a, b, c, d, e, f).dagger)
        num_parameters = len(self.G_ops)

        H = np.zeros((num_parameters, num_parameters))
        S = np.zeros((num_parameters, num_parameters))

        H_0i_0a = hamiltonian_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        for j, GJ in enumerate(self.G_ops):
            GJ_ket = propagate_state([GJ], self.wf.ci_coeffs, *self.index_info)
            HGJ_ket = propagate_state([H_0i_0a], GJ_ket, *self.index_info)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                # Make H
                H[i, j] = H[j, i] = expectation_value(
                    GI_ket,
                    [],
                    HGJ_ket,
                    *self.index_info,
                )
                # Make S
                S[i, j] = S[j, i] = expectation_value(
                    GI_ket,
                    [],
                    GJ_ket,
                    *self.index_info,
                )
        self.H = H
        self.S = S
        eigval, _ = scipy.linalg.eig(H, S)
        sorting = np.argsort(eigval)
        eigval = eigval[sorting]
        self.E0 = np.real(eigval[0])
        self.excitation_energies = np.real(eigval[1:]) - self.E0
