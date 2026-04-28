import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.hcb_ups_wavefunction import WaveFunctionHCBUPS
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a, hamiltonian_hcb_0i_0a
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


def test_h2o_fullspace() -> None:
    """Test H2O fUCCpD(10,7)."""
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
    WFref = WaveFunctionUPS((10, 7), SQobj.hartree_fock.mo_coeff, SQobj, "fucc", ansatz_options={"pD": True})
    WFref.run_wf_optimization_1step("BFGS", False)

    WF = WaveFunctionHCBUPS(
        (10, 7),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "fuccpd",
    )
    WF.run_wf_optimization_1step("BFGS", False)

    assert abs(WF.energy_elec - WFref.energy_elec) < 10**-8
    assert abs(WF.energy_elec - -83.98446121828943) < 10**-8


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
    WFref = WaveFunctionUPS((4, 4), SQobj.hartree_fock.mo_coeff, SQobj, "fucc", ansatz_options={"pD": True})
    WFref.run_wf_optimization_1step("BFGS", False)

    WF = WaveFunctionHCBUPS(
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "fuccpd",
    )
    WF.run_wf_optimization_1step("BFGS", False)

    assert abs(WF.energy_elec - WFref.energy_elec) < 10**-8
    assert abs(WF.energy_elec - -83.96638862128532) < 10**-8


def test_convert_fermionic2hcb() -> None:
    """Test conversion of fermionic operator to hard-core boson operator."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0  0.0 0.0;
    H   1.6 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    WF = WaveFunctionUPS((4, 6), SQobj.hartree_fock.mo_coeff, SQobj, "fucc", ansatz_options={"pD": True})
    WF.run_wf_optimization_1step("BFGS", False)

    WF2 = WaveFunctionHCBUPS(
        (4, 6),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "fuccpd",
    )
    WF2.run_wf_optimization_1step("BFGS", False)

    Hf = hamiltonian_0i_0a(WF.h_mo, WF.g_mo, WF.num_inactive_orbs, WF.num_active_orbs)
    Hb = hamiltonian_hcb_0i_0a(WF2.hr1, WF2.hr2, WF2.num_inactive_orbs, WF2.num_active_orbs)
    Hfb = Hf.get_hardcoreboson_form()

    assert (Hb - Hfb).operators == {}
