from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
import numpy as np
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import LinearResponseBaseClass


class Properties:
    def __init__(self, WF: WaveFunctionUCC | WaveFunctionUPS, LR: LinearResponseBaseClass | None = None) -> None:
        self.wf = WF
        self.lr = LR

    @property
    def excitation_energies(self) -> np.ndarray:
        """Calculate excitation energies.

        Returns:
            Excitation energies.
        """
        if self.lr is None:
            raise ValueError("LR must be defined to get excitation energies.")
        return self.lr.excitation_energies

    @property
    def oscillator_strengths(self) -> np.ndarray:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\right>\right|^2

        Returns:
            Oscillator Strength.
        """
        if self.lr is None:
            raise ValueError("LR must be defined to get oscillator strengths.")
        dipole_integrals = self.wf.int_gen.electric_dipole
        transition_dipoles = np.zeros((len(self.excitation_energies), 3), dtype=float) 
        transition_dipoles[:,0] = self.lr.get_transition_property(dipole_integrals[0])
        transition_dipoles[:,1] = self.lr.get_transition_property(dipole_integrals[1])
        transition_dipoles[:,2] = self.lr.get_transition_property(dipole_integrals[2])
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
        return osc_strs

    @property
    def get_formatted_oscillator_strength(self) -> str:
        """Create table of excitation energies and oscillator strengths.

        Returns:
            Nicely formatted table.
        """
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


