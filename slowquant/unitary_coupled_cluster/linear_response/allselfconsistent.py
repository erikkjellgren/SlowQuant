import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.ci_spaces import (
    CI_Info,
    get_indexing_extended,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import (
    Epq,
    Tpq,
    hamiltonian_2i_2a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.util import UccStructure, UpsStructure


class LinearResponse(LinearResponseBaseClass):
    index_info_extended: tuple[CI_Info, list[float], UpsStructure] | tuple[CI_Info, list[float], UccStructure]

    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: str,
        triplet: bool = False,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            triplet: If the linear response should be triplet spin-adapted.
        """
        super().__init__(wave_function, excitations, triplet)
        # Overwrite Superclass
        ci_info = get_indexing_extended(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.num_active_elec_alpha,
            self.wf.num_active_elec_beta,
            2,
        )
        if isinstance(self.wf, WaveFunctionUCC):
            self.index_info_extended = (
                ci_info,
                self.wf.thetas,
                self.wf.ucc_layout,
            )
        elif isinstance(self.wf, WaveFunctionUPS):
            self.index_info_extended = (
                ci_info,
                self.wf.thetas,
                self.wf.ups_layout,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")
        num_det = len(ci_info.idx2det)
        self.csf_coeffs = np.zeros(num_det)
        hf_det = int(
            "1" * self.wf.int_gen.num_elec + "0" * (self.wf.num_spin_orbs - self.wf.int_gen.num_elec), 2
        )
        self.csf_coeffs[ci_info.det2idx[hf_det]] = 1
        self.ci_coeffs = propagate_state(["U"], self.csf_coeffs, *self.index_info_extended)
        self.q_ops: list[FermionicOperator] = []
        for i, a in self.wf.kappa_hf_like_idx:
            if not self.triplet:
                op = 2 ** (-1 / 2) * Epq(a, i)
            else:
                op = 2 ** (-1 / 2) * Tpq(a, i)
            self.q_ops.append(op)

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))

        H_2i_2a = hamiltonian_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        grad = np.zeros(2 * len(self.q_ops))
        print("WARNING!")
        print("Gradient working equations not implemented for self consistent q operators")
        if len(grad) != 0:
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        UdH00_ket = propagate_state(["Ud", self.H_0i_0a], self.ci_coeffs, *self.index_info_extended)
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state([op], self.csf_coeffs, *self.index_info_extended)
            # - <0| H U G |CSF>
            grad[i] = -expectation_value(
                UdH00_ket,
                [],
                G_ket,
                *self.index_info_extended,
            )
            # <CSF| Gd Ud H |0>
            grad[i + len(self.G_ops)] = expectation_value(
                G_ket,
                [],
                UdH00_ket,
                *self.index_info_extended,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        H_ket = propagate_state([H_2i_2a], self.ci_coeffs, *self.index_info_extended, do_unsafe=True)  # type: ignore
        UdH_ket = propagate_state(["Ud"], H_ket, *self.index_info_extended, do_unsafe=True)  # type: ignore
        for j, qJ in enumerate(self.q_ops):
            UdHUqJ_ket = propagate_state(
                ["Ud", H_2i_2a, "U", qJ],
                self.csf_coeffs,
                *self.index_info_extended,
                do_unsafe=True,  # type: ignore
            )
            qJUdH_ket = propagate_state([qJ], UdH_ket, *self.index_info_extended, do_unsafe=True)  # type: ignore
            qJdUdH_ket = propagate_state([qJ.dagger], UdH_ket, *self.index_info_extended, do_unsafe=True)  # type: ignore
            for i, qI in enumerate(self.q_ops[j:], j):
                qI_ket = propagate_state([qI], self.csf_coeffs, *self.index_info_extended, do_unsafe=True)  # type: ignore
                # Make A
                # <CSF| qId Ud H U qJ |CSF>
                val = expectation_value(
                    qI_ket,
                    [],
                    UdHUqJ_ket,
                    *self.index_info_extended,
                )
                # - 1/2<CSF| qId qJ Ud H |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        qI_ket,
                        [],
                        qJUdH_ket,
                        *self.index_info_extended,
                    )
                )
                # - 1/2<0| H U qId qJ |CSF>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        UdH_ket,
                        [qI.dagger, qJ],
                        self.csf_coeffs,
                        *self.index_info_extended,
                    )
                )
                self.A[i, j] = self.A[j, i] = val
                # Make B
                # -<CSF| qId qJd Ud H |0>
                val = -expectation_value(
                    qI_ket,
                    [],
                    qJdUdH_ket,
                    *self.index_info_extended,
                )
                self.B[i, j] = self.B[j, i] = val
                # Make Sigma
                if i == j:
                    self.Sigma[i, j] = self.Sigma[j, i] = 1
        for j, qJ in enumerate(self.q_ops):
            UdHUq_ket = propagate_state(
                ["Ud", self.H_1i_1a, "U", qJ],
                self.csf_coeffs,
                *self.index_info_extended,
                do_unsafe=True,  # type: ignore
            )
            qdUdH_ket = propagate_state(
                [qJ.dagger, "Ud", self.H_1i_1a],
                self.ci_coeffs,
                *self.index_info_extended,
                do_unsafe=True,  # type: ignore
            )
            for i, GI in enumerate(self.G_ops):
                G_ket = propagate_state([GI], self.csf_coeffs, *self.index_info_extended)
                # Make A
                # <CSF| Gd Ud H U q |CSF>
                val = expectation_value(
                    G_ket,
                    [],
                    UdHUq_ket,
                    *self.index_info_extended,
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                # - 1/2*<CSF| Gd qd Ud H |0>
                val = (
                    -1
                    / 2
                    * expectation_value(
                        G_ket,
                        [],
                        qdUdH_ket,
                        *self.index_info_extended,
                    )
                )
                # - 1/2<CSF| qd Gd Ud H |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        self.csf_coeffs,
                        [qJ.dagger, GI.dagger],
                        UdH_ket,
                        *self.index_info_extended,
                    )
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val
        for j, GJ in enumerate(self.G_ops):
            UdHUGJ_ket = propagate_state(
                ["Ud", self.H_0i_0a, "U", GJ], self.csf_coeffs, *self.index_info_extended
            )
            GJUdH_ket = propagate_state([GJ], UdH00_ket, *self.index_info_extended)
            GJdUdH_ket = propagate_state([GJ.dagger], UdH00_ket, *self.index_info_extended)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = propagate_state([GI], self.csf_coeffs, *self.index_info_extended)
                # Make A
                # <CSF| GId Ud H U GJ |CSF>
                val = expectation_value(
                    GI_ket,
                    [],
                    UdHUGJ_ket,
                    *self.index_info_extended,
                )
                # - 1/2<CSF| GId GJ Ud H |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        GI_ket,
                        [],
                        GJUdH_ket,
                        *self.index_info_extended,
                    )
                )
                # - 1/2<0| H U GId GJ |CSF>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        UdH00_ket,
                        [GI.dagger, GJ],
                        self.csf_coeffs,
                        *self.index_info_extended,
                    )
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                # - <CSF| GId GJd Ud H |0>
                val = -expectation_value(
                    GI_ket,
                    [],
                    GJdUdH_ket,
                    *self.index_info_extended,
                )
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                if i == j:
                    self.Sigma[i + idx_shift, j + idx_shift] = 1

    def get_property_gradient(self, property_integrals: np.ndarray | tuple[np.ndarray]) -> np.ndarray:
        """Calculate property gradient.

        Args:
            property_integrals: Integrals in AO basis.

        Returns:
            Property gradient.
        """
        size_mo = self.wf.num_inactive_orbs + self.wf.num_active_orbs + self.wf.num_virtual_orbs
        num_mo = len(property_integrals)
        mo = np.zeros((num_mo, size_mo, size_mo))
        for i, ao in enumerate(property_integrals):
            mo[i, :, :] += one_electron_integral_transform(self.wf.c_mo, ao)

        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), num_mo))
        
        if not self.triplet:
            E = Epq
        else:
            E = Tpq

        for idx, q in enumerate(self.q_ops):
            Uq_ket = propagate_state(["U", q], self.csf_coeffs, *self.index_info_extended)
            Uqd_ket = propagate_state(["U", q.dagger], self.csf_coeffs, *self.index_info_extended)
            for m in range(self.wf.num_inactive_orbs + self.wf.num_active_orbs + self.wf.num_virtual_orbs):
                for n in range(self.wf.num_inactive_orbs + self.wf.num_active_orbs + self.wf.num_virtual_orbs):
                    E_ket = propagate_state([E(m, n)], self.ci_coeffs, *self.index_info_extended)
                    Ed_ket = propagate_state([E(n, m)], self.ci_coeffs, *self.index_info_extended)
                    # < CSF | q Ud E | 0 >
                    val = expectation_value(
                        Uqd_ket,
                        [],
                        E_ket,
                        *self.index_info_extended,
                    )
                    # - < 0 | E U q | CSF >
                    val -= expectation_value(
                        Ed_ket,
                        [],
                        Uq_ket,
                        *self.index_info_extended,
                    )
                    V[idx, :] += mo[:, m, n] * val

        for idx, G in enumerate(self.G_ops):
            UG_ket = propagate_state(["U", G], self.csf_coeffs, *self.index_info_extended)
            UGd_ket = propagate_state(["U", G.dagger], self.csf_coeffs, *self.index_info_extended)
            # Inactive part
            for i in range(self.wf.num_inactive_orbs):
                E_ket = propagate_state([E(i, i)], self.ci_coeffs, *self.index_info_extended) 
                # < CSF | G Ud E | 0 >
                val = expectation_value(
                    UGd_ket, 
                    [], 
                    E_ket, 
                    *self.index_info_extended
                )
                # - < 0 | E U G | CSF >
                val -= expectation_value(
                    E_ket, # E_ket = Ed_ket for E(i,i)
                    [], 
                    UG_ket, 
                    *self.index_info_extended
                ) 
                V[idx + idx_shift_q, :] += mo[:, i, i] * val
            # Active part
            for v in range(self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                for w in range(self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                    E_ket = propagate_state([E(v, w)], self.ci_coeffs, *self.index_info_extended)
                    Ed_ket = propagate_state([E(w, v)], self.ci_coeffs, *self.index_info_extended)
                    # < CSF | G Ud E | 0 >
                    val = expectation_value(
                        UGd_ket, 
                        [], 
                        E_ket, 
                        *self.index_info_extended
                    )
                    # - < 0 | E U G | CSF >
                    val -= expectation_value(
                        Ed_ket, 
                        [], 
                        UG_ket, 
                        *self.index_info_extended
                    )
                    V[idx + idx_shift_q, :] += mo[:, v, w] * val
        if np.allclose(mo, mo.transpose(0, -1, -2)):
            return np.vstack((V, -1 * V))
        return np.vstack((V, V))