import slowquant.SlowQuant as sq


def test_properties_ch3_sto3g() -> None:
    """Test unrestricted Hartree-Fock for CH3 with STO-3G.

    Reference data: Szabo and Ostlund
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """C   0.000000000000  0.000000000000   0.000000000000;
           H   2.039000000000  0.000000000000   0.000000000000;
           H  -1.019500000000  1.765825798316   0.000000000000;
           H  -1.019500000000 -1.765825798316   0.000000000000""",
        distance_unit="bohr",
    )
    A.set_basis_set("sto-3g")
    A.init_hartree_fock()
    A.hartree_fock.de_threshold = 10**-6
    A.hartree_fock.rmsd_threshold = 10**-6
    A.hartree_fock.run_unrestricted_hartree_fock()
    assert abs(A.hartree_fock.spin_contamination - 0.0152) < 10**-4


def test_properties_ch3_431g() -> None:
    """Test unrestricted Hartree-Fock for CH3 with 4-31G.

    Reference data: Szabo and Ostlund
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """C   0.000000000000  0.000000000000   0.000000000000;
           H   2.039000000000  0.000000000000   0.000000000000;
           H  -1.019500000000  1.765825798316   0.000000000000;
           H  -1.019500000000 -1.765825798316   0.000000000000""",
        distance_unit="bohr",
    )
    A.set_basis_set("4-31g")
    A.init_hartree_fock()
    A.hartree_fock.de_threshold = 10**-6
    A.hartree_fock.rmsd_threshold = 10**-6
    A.hartree_fock.run_unrestricted_hartree_fock()
    assert abs(A.hartree_fock.spin_contamination - 0.0122) < 10**-4


def test_properties_ch3_631gs() -> None:
    """Test unrestricted Hartree-Fock for CH3 with 6-31G*.

    Reference data: Szabo and Ostlund
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """C   0.000000000000  0.000000000000   0.000000000000;
           H   2.039000000000  0.000000000000   0.000000000000;
           H  -1.019500000000  1.765825798316   0.000000000000;
           H  -1.019500000000 -1.765825798316   0.000000000000""",
        distance_unit="bohr",
    )
    A.set_basis_set("6-31g*")
    A.init_hartree_fock()
    A.hartree_fock.de_threshold = 10**-7
    A.hartree_fock.rmsd_threshold = 10**-7
    A.hartree_fock.max_scf_iterations = 200
    A.hartree_fock.run_unrestricted_hartree_fock()
    assert abs(A.hartree_fock.spin_contamination - 0.0118) < 10**-4


def test_properties_ch3_631gss() -> None:
    """Test unrestricted Hartree-Fock for CH3 with 6-31G**.

    Reference data: Szabo and Ostlund
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """C   0.000000000000  0.000000000000   0.000000000000;
           H   2.039000000000  0.000000000000   0.000000000000;
           H  -1.019500000000  1.765825798316   0.000000000000;
           H  -1.019500000000 -1.765825798316   0.000000000000""",
        distance_unit="bohr",
    )
    A.set_basis_set("6-31g**")
    A.init_hartree_fock()
    A.hartree_fock.de_threshold = 10**-6
    A.hartree_fock.rmsd_threshold = 10**-6
    A.hartree_fock.run_unrestricted_hartree_fock()
    assert abs(A.hartree_fock.spin_contamination - 0.0114) < 10**-4
