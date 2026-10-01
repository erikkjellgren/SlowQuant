import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.properties import Properties
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_H2_631g_naive_q():
    """Test of excitation energies and oscialltor strengths for naive, projected, selfconsistent and statetransfer LR with H2(2,2)/6-31G."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # OO-UCCSD
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # Linear Response
    print("\nNaive")
    prop = Properties(WF, "naive")
    excita_naive, osc_strs_naive = prop.get_excitation_energies(osc_strs=True)

    print("\nProjected")
    prop = Properties(WF, "proj")
    excita_proj, osc_strs_proj = prop.get_excitation_energies(osc_strs=True)    

    print("\nSelfconsistent")
    prop = Properties(WF, "sc")
    excita_sc, osc_strs_sc = prop.get_excitation_energies(osc_strs=True)

    print("\nStatetransfer")
    prop = Properties(WF, "st")
    excita_st, osc_strs_st = prop.get_excitation_energies(osc_strs=True)

    excita = np.array([excita_naive, excita_proj, excita_sc, excita_st])
    osc_strs = np.array([osc_strs_naive, osc_strs_proj, osc_strs_sc, osc_strs_st])

    thresh = 10**-4

    # Check excitation energies
    assert np.all(abs(excita[:,0] - 0.574413) < thresh)
    assert np.all(abs(excita[:,1] - 1.043177) < thresh)
    assert np.all(abs(excita[:,2] - 1.139481) < thresh)
    assert np.all(abs(excita[:,3] - 1.365960) < thresh)
    assert np.all(abs(excita[:,4] - 1.831196) < thresh)
    assert np.all(abs(excita[:,5] - 2.581273) < thresh)

    # Check oscillator strengths
    assert np.all(abs(osc_strs[:,0] - 0.6338) < thresh)
    assert np.all(abs(osc_strs[:,1] - 0.0) < thresh)
    assert np.all(abs(osc_strs[:,2] - 0.0) < thresh)
    assert np.all(abs(osc_strs[:,3] - 0.0311) < thresh)
    assert np.all(abs(osc_strs[:,4] - 0.0421) < thresh)
    assert np.all(abs(osc_strs[:,5] - 0.0) < thresh)


def test_LiH_sto3g_naive_q():
    """Test of excitation energies and oscialltor strength for naive, projected, selfconsistent and statetransfer LR with LiH(2,2)/STO-3G."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.671707274 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # oo-UCCSD
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # Linear Response
    print("\nNaive")
    prop = Properties(WF, "naive")
    excita_naive, osc_strs_naive = prop.get_excitation_energies(osc_strs=True)

    print("\nProjected")
    prop = Properties(WF, "proj")
    excita_proj, osc_strs_proj = prop.get_excitation_energies(osc_strs=True)    

    print("\nSelfconsistent")
    prop = Properties(WF, "sc")
    excita_sc, osc_strs_sc = prop.get_excitation_energies(osc_strs=True)

    print("\nStatetransfer")
    prop = Properties(WF, "st")
    excita_st, osc_strs_st = prop.get_excitation_energies(osc_strs=True)

    excita = np.array([excita_naive, excita_proj, excita_sc, excita_st])
    osc_strs = np.array([osc_strs_naive, osc_strs_proj, osc_strs_sc, osc_strs_st])

    thresh = 10**-4

    # Check excitation energies
    assert np.all(abs(excita[:,0] - 0.129471) < thresh)
    assert np.all(abs(excita[:,1] - 0.178744) < thresh)
    assert np.all(abs(excita[:,2] - 0.178744) < thresh)
    assert np.all(abs(excita[:,3] - 0.604674) < thresh)
    assert np.all(abs(excita[:,4] - 0.646694) < thresh)
    assert np.all(abs(excita[:,5] - 0.740616) < thresh)
    assert np.all(abs(excita[:,6] - 0.740616) < thresh)
    assert np.all(abs(excita[:,7] - 1.002882) < thresh)
    assert np.all(abs(excita[:,8] - 2.074820) < thresh)
    assert np.all(abs(excita[:,9] - 2.137192) < thresh)
    assert np.all(abs(excita[:,10] - 2.137192) < thresh)
    assert np.all(abs(excita[:,11] - 2.455124) < thresh)
    assert np.all(abs(excita[:,12] - 2.9543838) < thresh)

    # Check oscillator strengths
    assert np.all(abs(osc_strs[:,0] - 0.049952) < thresh)
    assert np.all(abs(osc_strs[:,1] - 0.241200) < thresh)
    assert np.all(abs(osc_strs[:,2] - 0.241200) < thresh)
    assert np.all(abs(osc_strs[:,3] - 0.1580497) < thresh)
    assert np.all(abs(osc_strs[:,4] - 0.166598) < thresh)
    assert np.all(abs(osc_strs[:,5] - 0.010376) < thresh)
    assert np.all(abs(osc_strs[:,6] - 0.010376) < thresh)
    assert np.all(abs(osc_strs[:,7] - 0.006250) < thresh)
    assert np.all(abs(osc_strs[:,8] - 0.062374) < thresh)
    assert np.all(abs(osc_strs[:,9] - 0.128854) < thresh)
    assert np.all(abs(osc_strs[:,10] - 0.128854) < thresh)
    assert np.all(abs(osc_strs[:,11] - 0.046008) < thresh)
    assert np.all(abs(osc_strs[:,12] - 0.003907) < thresh)


def test_H2_631g_proj_q():
    """Test of excitation energies and oscialltor strength for all-projected and projected-statetransfer LR with H2(2,2)/6-31G."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # OO-UCCSD
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # Linear Response
    print("\nAll-projected")
    prop = Properties(WF, "allproj")
    excita_allproj, osc_strs_allproj = prop.get_excitation_energies(osc_strs=True)

    print("\nProjected-statetransfer")
    prop = Properties(WF, "projst")
    excita_projst, osc_strs_projst = prop.get_excitation_energies(osc_strs=True)

    excita = np.array([excita_allproj, excita_projst])
    osc_strs = np.array([osc_strs_allproj, osc_strs_projst])

    thresh = 10**-4

    # Check excitation energies
    assert np.all(abs(excita[:,0] - 0.57549309) < thresh)
    assert np.all(abs(excita[:,1] - 1.04824448) < thresh)
    assert np.all(abs(excita[:,2] - 1.14842879) < thresh)
    assert np.all(abs(excita[:,3] - 1.48434251) < thresh)
    assert np.all(abs(excita[:,4] - 1.96225079) < thresh)
    assert np.all(abs(excita[:,5] - 2.59296189) < thresh)

    # Check oscillator strengths
    assert np.all(abs(osc_strs[:,0] - 0.646005715) < thresh)
    assert np.all(abs(osc_strs[:,1] - 0.0) < thresh)
    assert np.all(abs(osc_strs[:,2] - 0.0) < thresh)
    assert np.all(abs(osc_strs[:,3] - 4.68927085e-02) < thresh)
    assert np.all(abs(osc_strs[:,4] - 2.07917839e-02) < thresh)
    assert np.all(abs(osc_strs[:,5] - 0.0) < thresh)


def test_LiH_sto3g_proj_q():
    """Test of excitation energies and oscialltor strength for all-projected and projected-statetransfer LR with LiH(2,2)/STO-3G."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.67 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # OO-UCCSD
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # Linear Response
    print("\nAll-projected")
    prop = Properties(WF, "allproj")
    excita_allproj, osc_strs_allproj = prop.get_excitation_energies(osc_strs=True)

    print("\nProjected-statetransfer")
    prop = Properties(WF, "projst")
    excita_projst, osc_strs_projst = prop.get_excitation_energies(osc_strs=True)

    excita = np.array([excita_allproj, excita_projst])
    osc_strs = np.array([osc_strs_allproj, osc_strs_projst])

    thresh = 10**-4

    # Check excitation energies
    solutions = np.array(
        [
            0.12973325,
            0.18092772,
            0.18092772,
            0.60537673,
            0.64747507,
            0.74982736,
            0.74982736,
            1.00424791,
            2.07489682,
            2.13720681,
            2.13720681,
            2.45601762,
            2.95607806,]
    )

    assert np.allclose(excita[0], solutions, atol=thresh)
    assert np.allclose(excita[1], solutions, atol=thresh)

    # Check oscillator strengths
    assert np.all(abs(osc_strs[:,0] - 0.04994788) < thresh)
    assert np.all(abs(osc_strs[:,1] - 0.25097391) < thresh)
    assert np.all(abs(osc_strs[:,2] - 0.25097391) < thresh)
    assert np.all(abs(osc_strs[:,3] - 0.16147543) < thresh)
    assert np.all(abs(osc_strs[:,4] - 0.16109274) < thresh)
    assert np.all(abs(osc_strs[:,5] - 0.01834264) < thresh)
    assert np.all(abs(osc_strs[:,6] - 0.01834264) < thresh)
    assert np.all(abs(osc_strs[:,7] - 0.00672061) < thresh)
    assert np.all(abs(osc_strs[:,8] - 0.06322828) < thresh)
    assert np.all(abs(osc_strs[:,9] - 0.13384300) < thresh)
    assert np.all(abs(osc_strs[:,10] - 0.13384300) < thresh)
    assert np.all(abs(osc_strs[:,11] - 0.04662360) < thresh)
    assert np.all(abs(osc_strs[:,12] - 0.00381938) < thresh)


def test_H2_631g_allST():
    """Test of excitation energies and oscialltor strength for all-statetransfer LR with H2(2,2)/6-31G."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # OO-UCCSD
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # Linear Response
    prop = Properties(WF, "allst")
    excita, osc_strs = prop.get_excitation_energies(osc_strs=True)

    thresh = 10**-4

    # Check excitation energies
    assert np.all(abs(excita[0] - 0.57773553) < thresh)
    assert np.all(abs(excita[1] - 1.05253656) < thresh)
    assert np.all(abs(excita[2] - 1.63445659) < thresh)
    assert np.all(abs(excita[3] - 1.64921366) < thresh)

    # Check oscillator strengths
    assert np.all(abs(osc_strs[0] - 0.650294311) < thresh)
    assert np.all(abs(osc_strs[1] - 0.0) < thresh)
    assert np.all(abs(osc_strs[2] - 6.23019972e-02) < thresh)
    assert np.all(abs(osc_strs[3] - 0.0) < thresh)


def test_LiH_sto3g_allST():
    """Test of excitation energies and oscialltor strength for all-statetransfer LR with LiH(2,2)/STO-3G."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.67 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # OO-UCCSD
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # Linear Response
    prop = Properties(WF, "allst")
    excita, osc_strs = prop.get_excitation_energies(osc_strs=True)

    thresh = 10**-3

    # Check excitation energies
    solutions = np.array(
        [
            0.1851181,
            0.24715136,
            0.24715136,
            0.6230648,
            0.85960395,
            2.07752209,
            2.13720198,
            2.13720198,
            2.55113802,
        ]
    )

    assert np.allclose(excita, solutions, atol=thresh)

    # Check oscillator strengths
    assert np.all(abs(osc_strs[0] - 0.06668878) < thresh)
    assert np.all(abs(osc_strs[1] - 0.33360367) < thresh)
    assert np.all(abs(osc_strs[2] - 0.33360367) < thresh)
    assert np.all(abs(osc_strs[3] - 0.30588158) < thresh)
    assert np.all(abs(osc_strs[4] - 0.02569977) < thresh)
    assert np.all(abs(osc_strs[5] - 0.06690658) < thresh)
    assert np.all(abs(osc_strs[6] - 0.13411942) < thresh)
    assert np.all(abs(osc_strs[7] - 0.13411942) < thresh)
    assert np.all(abs(osc_strs[8] - 0.04689274) < thresh)


def test_H2_631g_allSC():
    """Test of excitation energies and oscialltor strength for all-selfconsistent LR with H2(2,2)/6-31G."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # OO-UCCSD
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # Linear Response
    prop = Properties(WF, "allsc")
    excita, osc_strs = prop.get_excitation_energies(osc_strs=True)

    thresh = 10**-4

    # Check excitation energies
    assert np.all(abs(excita[0] - 0.57751618) < thresh)
    assert np.all(abs(excita[1] - 1.04796405) < thresh)
    assert np.all(abs(excita[2] - 1.63423404) < thresh)
    assert np.all(abs(excita[3] - 1.64907314) < thresh)

    # Check oscillator strengths
    assert np.all(abs(osc_strs[0] - 0.638731917) < thresh)
    assert np.all(abs(osc_strs[1] - 0.0) < thresh)
    assert np.all(abs(osc_strs[2] - 6.63456989e-02) < thresh)
    assert np.all(abs(osc_strs[3] - 0.0) < thresh)


def test_LiH_sto3g_allSC():
    """Test of excitation energies and oscialltor strength for all-selfconsistent LR with LiH(2,2)/STO-3G."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.67 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # OO-UCCSD
    WF = WaveFunctionUCC(
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )
    WF.run_wf_optimization_1step("BFGS", True)

    # Linear Response
    prop = Properties(WF, "allsc")
    excita, osc_strs = prop.get_excitation_energies(osc_strs=True)

    thresh = 10**-4

    # Check excitation energies
    solutions = np.array(
        [
            0.18563041, 
            0.24713336, 
            0.24713336, 
            0.62310207, 
            0.85953354, 
            2.07735631,
            2.13715369, 
            2.13715369, 
            2.55046675,]
    )

    assert np.allclose(excita, solutions, atol=thresh)

    # Check oscillator strengths
    assert np.all(abs(osc_strs[0] - 0.06548333) < thresh)
    assert np.all(abs(osc_strs[1] - 0.30867389) < thresh)
    assert np.all(abs(osc_strs[2] - 0.30867389) < thresh)
    assert np.all(abs(osc_strs[3] - 0.30674692) < thresh)
    assert np.all(abs(osc_strs[4] - 0.02573047) < thresh)
    assert np.all(abs(osc_strs[5] - 0.06604172) < thresh)
    assert np.all(abs(osc_strs[6] - 0.12944289) < thresh)
    assert np.all(abs(osc_strs[7] - 0.12944289) < thresh)
    assert np.all(abs(osc_strs[8] - 0.04646674) < thresh)
