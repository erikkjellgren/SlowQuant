import slowquant.SlowQuant as sq


def test_unrestricted_hartree_fock_h2o_sto3g() -> None:
    """Test unrestricted Hartree-Fock for H2O with STO-3G.

    Should give the exact same as restricted Hartree-Fock
    when orbitals are not forced to break symmetry.
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """O 0.000000000000  -0.143225816552   0.000000000000;
           H 1.638036840407   1.136548822547  -0.000000000000;
           H -1.638036840407   1.136548822547  -0.000000000000;""",
        distance_unit="bohr",
    )
    A.set_basis_set("sto-3g")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    A.hartree_fock.uhf_lumo_homo_mix_coeff = 0.0
    A.hartree_fock.run_unrestricted_hartree_fock()
    assert abs(A.hartree_fock.E_hf - A.hartree_fock.E_uhf) < 10**-6
