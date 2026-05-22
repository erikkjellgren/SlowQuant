from typing import Any

from qiskit.primitives import (
    BaseSamplerV1,
    BaseSamplerV2,
)
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.sa_circuit_wavefunction import WaveFunctionSACircuit
from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


def circuit_wavefunction_from_ups(
    ups_wf: WaveFunctionUPS | WaveFunctionSAUPS,
    primitive: BaseSamplerV1 | BaseSamplerV2,
    mapper: FermionicMapper,
    ISA: bool = False,
    pass_manager_options: dict[str, Any] | None = None,
    shots: None | int = None,
    max_shots_per_run: int = 100000,
    do_M_mitigation: bool = False,
    do_M_ansatz0: bool = False,
    do_M_ansatz0_plus: bool = False,
    do_postselection: bool = False,
) -> WaveFunctionCircuit | WaveFunctionSACircuit:
    """Convert UPS wavefunction to circuit wavefunction.

    Args:
        ups_wf: Unitary product state wavefunction object.
        primitive: Qiskit Sampler object.
        mapper: Qiskit mapper object, e.g. JW or Parity.
        ISA: Use ISA for submitting to IBM quantum. Locally transpiling is performed.
        pass_manager_options: Dictionary to define custom pass manager.
        shots: Number of shots. None means ideal simulator.
        max_shots_per_run: Maximum number of shots allowed in a single run. Set to 100000 per IBM machines.
        do_M_mitigation: Do error mitigation via read-out correlation matrix.
        do_M_ansatz0: Use the ansatz with theta=0 when constructing the read-out correlation matrix.
        do_M_ansatz0_plus: Creates M0 for each initial superposition state. Only used for SA-VQE.
        do_postselection: Use postselection to preserve number of particles in the computational basis.

    Returns:
        Circuit wavefunction.
    """
    QI = QuantumInterface(
        primitive=primitive,
        ansatz=ups_wf.ups_layout,
        mapper=mapper,
        ISA=ISA,
        pass_manager_options=pass_manager_options,
        ansatz_options=None,
        shots=shots,
        max_shots_per_run=max_shots_per_run,
        do_M_mitigation=do_M_mitigation,
        do_M_ansatz0=do_M_ansatz0,
        do_M_ansatz0_plus=do_M_ansatz0_plus,
        do_postselection=do_postselection,
    )
    if isinstance(ups_wf, WaveFunctionUPS):
        wf = WaveFunctionCircuit(
            (ups_wf.num_active_elec, ups_wf.num_active_orbs),
            ups_wf.c_mo,
            ups_wf.int_gen.int_obj,
            QI,
            include_active_kappa=ups_wf._include_active_kappa,
        )
        wf.thetas = ups_wf.thetas
        return wf
    elif isinstance(ups_wf, WaveFunctionSAUPS):
        sawf = WaveFunctionSACircuit(
            (ups_wf.num_active_elec, ups_wf.num_active_orbs),
            ups_wf.c_mo,
            ups_wf.int_gen.int_obj,
            ups_wf._states,
            QI,
            include_active_kappa=ups_wf._include_active_kappa,
        )
        sawf.thetas = ups_wf.thetas
        return sawf
    else:
        raise TypeError(f"Got unknown wavefunction type: {type(ups_wf)}")
