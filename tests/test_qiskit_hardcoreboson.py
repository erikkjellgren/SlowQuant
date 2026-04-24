from qiskit_aer.primitives import Sampler

import slowquant.SlowQuant as sq
from slowquant.qiskit_interface.hcb_circuit_wavefunction import WaveFunctionHCBCircuit
from slowquant.qiskit_interface.hcb_interface import HCBQuantumInterface
from slowquant.unitary_coupled_cluster.hcb_ups_wavefunction import WaveFunctionHCBUPS


def test_h2o_fullspace() -> None:
    """Test LiH fUCCpD(4,6)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
           H   1.6  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    WFref = WaveFunctionHCBUPS((4, 6), SQobj.hartree_fock.mo_coeff, SQobj, "fuccpd")
    WFref.run_wf_optimization_1step("BFGS", False)

    QI = HCBQuantumInterface(
        Sampler(),
        "fuccpd",
        shots=None,
    )
    qWF = WaveFunctionHCBCircuit(
        (4, 6),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        QI,
    )
    qWF.run_wf_optimization_1step("BFGS", False, tol=10**-6)

    assert abs(qWF.energy_elec - WFref.energy_elec) < 10**-8
    assert abs(qWF.energy_elec - -8.8701086132016069) < 10**-8


def test_h2o_4_4() -> None:
    """Test H2O fUCCpD(4,4)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """O   0.0  0.0           0.1035174918;
    H   0.0  0.7955612117 -0.4640237459;
    H   0.0 -0.7955612117 -0.4640237459;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    WFref = WaveFunctionHCBUPS((4, 4), SQobj.hartree_fock.mo_coeff, SQobj, "fuccpd")
    WFref.run_wf_optimization_1step("BFGS", False)

    QI = HCBQuantumInterface(
        Sampler(),
        "fuccpd",
        shots=None,
    )
    qWF = WaveFunctionHCBCircuit(
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        QI,
    )
    qWF.run_wf_optimization_1step("BFGS", False, tol=10**-6)

    assert abs(qWF.energy_elec - WFref.energy_elec) < 10**-8
    assert abs(qWF.energy_elec - -83.96638862128532) < 10**-8
