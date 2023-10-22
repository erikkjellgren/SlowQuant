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


def test_unrestricted_hartree_fock_n2_631gs() -> None:
    """Test unresticted Hartree-Fock N2 with 6-31G*.

    Does NOT give the same as the reference. (Reference for N2: -108.94235)
    Reference: Szabo and Ostlund
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """N   0.000000000000  0.000000000000   0.000000000000;
                      N  2.074000000000   0.000000000000  -0.000000000000""",
        distance_unit="bohr",
    )
    A.set_basis_set("6-31G*")
    A.init_hartree_fock()
    A.hartree_fock.run_unrestricted_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    assert abs(A.hartree_fock.E_uhf + A.molecule.nuclear_repulsion - (-108.942686389118)) < 10**-6
    # Check RHF and UHF gives the same for closed shell case.
    assert abs(A.hartree_fock.E_hf - A.hartree_fock.E_uhf) < 10**-6

    A = sq.SlowQuant()
    A.set_molecule(
        """N   0.000000000000  0.000000000000   0.000000000000;
                      N  2.074000000000   0.000000000000  -0.000000000000""",
        distance_unit="bohr",
    )
    A.set_basis_set("6-31G*")
    A.molecule.molecular_charge = 1
    A.init_hartree_fock()
    A.hartree_fock.run_unrestricted_hartree_fock()
    assert abs(A.hartree_fock.E_uhf + A.molecule.nuclear_repulsion - (-108.36597)) < 10**-5


def test_unrestricted_hartree_fock_li_631gss() -> None:
    """Test unresticted Hartree-Fock Li with 6-31G**.

    Reference: https://github.com/rpmuller/pyquante2/blob/master/pyquante2/scf/hamiltonians.py
    """
    A = sq.SlowQuant()
    A.set_molecule("""Li   0.000000000000  0.000000000000   0.000000000000;""", distance_unit="bohr")
    A.set_basis_set("6-31G**")
    A.init_hartree_fock()
    A.hartree_fock.run_unrestricted_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    assert abs(A.hartree_fock.E_uhf - (-7.4313707537)) < 10**-5
