# type: ignore
import numpy as np
import pyscf
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper

import slowquant.SlowQuant as sq
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.wavefunction import WaveFunction as WaveFunctionQC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


def test_tups() -> None:
    """Test tUPS is conventional <--> qiskit compatibile."""
    # Setup initial stuff
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0.0           0.0  0.0;
           H  1.6717072740  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    WF.run_ups(True)

    assert abs(WF.energy_elec - -8.82891657651419) < 10**-8

    # Circuit based UPS wave function
    mapper = JordanWignerMapper()
    primitive = Sampler(run_options={"shots": None})
    QI = QuantumInterface(
        primitive, "tUPS", mapper, ansatz_options={"n_layers": 1, "skip_last_singles": True}
    )
    qWF = WaveFunctionQC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_trans,
        h_core,
        g_eri,
        QI,
    )
    qWF.ansatz_parameters = WF.thetas

    assert abs(qWF.energy_elec - -8.82891657651419) < 10**-8


def test_fucc() -> None:
    """Test fUCC is conventional <--> qiskit compatibile."""
    # Setup initial stuff
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0.0           0.0  0.0;
           H  1.6717072740  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "fUCC",
        ansatz_options={},
        include_active_kappa=True,
    )
    WF.run_ups(True)

    assert abs(WF.energy_elec - -8.828916576513892) < 10**-8

    # Circuit based UPS wave function
    mapper = JordanWignerMapper()
    primitive = Sampler(run_options={"shots": None})
    QI = QuantumInterface(primitive, "fUCCSD", mapper, ansatz_options={})
    qWF = WaveFunctionQC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_trans,
        h_core,
        g_eri,
        QI,
    )
    qWF.ansatz_parameters = WF.thetas

    assert abs(qWF.energy_elec - -8.828916576513892) < 10**-8


def test_ksafupccgsd() -> None:
    """Test k-SAfUpCCGSD is conventional <--> qiskit compatibile."""
    # Setup initial stuff
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0.0           0.0  0.0;
           H  1.6717072740  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "kSAfUpCCGSD",
        ansatz_options={"n_layers": 1},
        include_active_kappa=True,
    )
    WF.run_ups(True)

    assert abs(WF.energy_elec - -8.828916576543133) < 10**-8

    # Circuit based UPS wave function
    mapper = JordanWignerMapper()
    primitive = Sampler(run_options={"shots": None})
    QI = QuantumInterface(primitive, "kSAfUpCCGSD", mapper, ansatz_options={"n_layers": 1})
    qWF = WaveFunctionQC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_trans,
        h_core,
        g_eri,
        QI,
    )
    qWF.ansatz_parameters = WF.thetas

    assert abs(qWF.energy_elec - -8.828916576543133) < 10**-8


def test_duccsd() -> None:
    """Test dUCCSD is conventional <--> qiskit compatibile."""
    # Setup initial stuff
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0.0           0.0  0.0;
           H  1.6717072740  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "dUCCSD",
        ansatz_options={},
        include_active_kappa=True,
    )
    WF.run_ups(True)

    assert abs(WF.energy_elec - -8.82891657653415) < 10**-8

    # Circuit based UPS wave function
    mapper = JordanWignerMapper()
    primitive = Sampler(run_options={"shots": None})
    QI = QuantumInterface(primitive, "dUCCSD", mapper, ansatz_options={})
    qWF = WaveFunctionQC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_trans,
        h_core,
        g_eri,
        QI,
    )
    qWF.ansatz_parameters = WF.thetas

    assert abs(qWF.energy_elec - -8.82891657653415) < 10**-8


def test_ksadupccgsd() -> None:
    """Test k-SAdUpCCGSD is conventional <--> qiskit compatibile."""
    # Setup initial stuff
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0.0           0.0  0.0;
           H  1.6717072740  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    # Conventional UPS wave function
    WF = WaveFunctionUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "kSAdUpCCGSD",
        ansatz_options={"n_layers": 1},
        include_active_kappa=True,
    )
    WF.run_ups(True)

    assert abs(WF.energy_elec - -8.828916576542285) < 10**-8

    # Circuit based UPS wave function
    mapper = JordanWignerMapper()
    primitive = Sampler(run_options={"shots": None})
    QI = QuantumInterface(primitive, "kSAdUpCCGSD", mapper, ansatz_options={"n_layers": 1})
    qWF = WaveFunctionQC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_trans,
        h_core,
        g_eri,
        QI,
    )
    qWF.ansatz_parameters = WF.thetas

    assert abs(qWF.energy_elec - -8.828916576542285) < 10**-8


def test_lih_fucc_allparameters() -> None:
    """Tests that a change in all parameters in fucc works.
    This example was used to test a bug in post-selection.
    """
    atom = "Li .0 .0 .0; H .0 .0 1.672"
    basis = "sto3g"

    # PySCF
    mol = pyscf.M(atom=atom, basis=basis, unit="angstrom")
    rhf = pyscf.scf.RHF(mol).run()

    # SlowQuant
    WF = WaveFunctionUPS(
        mol.nao * 2,
        mol.nelectron,
        (2, 3),
        rhf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        "fUCC",
    )
    sampler = Sampler()
    mapper = JordanWignerMapper()

    QI = QuantumInterface(sampler, "fUCCSD", mapper)

    qWF = WaveFunctionQC(
        mol.nao * 2,
        mol.nelectron,
        (2, 3),
        WF.c_trans,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        QI,
    )
    for n in range(len(WF.thetas)):
        thetas = np.zeros(len(WF.thetas))
        thetas[n] = 1
        WF.thetas = thetas
        qWF.ansatz_parameters = thetas
        assert abs(qWF.energy_elec - WF.energy_elec) < 10**-12
