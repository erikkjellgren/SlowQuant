from numpy.testing import assert_allclose

import slowquant.SlowQuant as sq


def test_properties_h2o_sto3g() -> None:
    """Test restricted Hartree-Fock for H2O with STO-3G.

    Reference data: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303
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
    A.init_properties()
    assert_allclose(
        A.properties.get_dipole_moment(A.hartree_fock.rdm1), [0, 0.603521296525, 0], atol=10**-6
    )


def test_properties_h2o_dz() -> None:
    """Test restricted Hartree-Fock for H2O with DZ.

    Reference data: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """O 0.000000000000  -0.143225816552   0.000000000000;
           H 1.638036840407   1.136548822547  -0.000000000000;
           H -1.638036840407   1.136548822547  -0.000000000000;""",
        distance_unit="bohr",
    )
    A.set_basis_set("dz")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    A.init_properties()
    assert_allclose(
        A.properties.get_dipole_moment(A.hartree_fock.rdm1), [0, 1.070995737060, 0], atol=10**-6
    )


def test_properties_h2o_dzp() -> None:
    """Test restricted Hartree-Fock for H2O with DZP.

    Does NOT give the same as in the reference.
    Reference data: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """O 0.000000000000  -0.143225816552   0.000000000000;
           H 1.638036840407   1.136548822547  -0.000000000000;
           H -1.638036840407   1.136548822547  -0.000000000000;""",
        distance_unit="bohr",
    )
    A.set_basis_set("dzp")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    A.init_properties()
    assert_allclose(A.properties.get_dipole_moment(A.hartree_fock.rdm1), [0, 9.227290e-01, 0], atol=10**-6)


def test_properties_ch4_sto3g() -> None:
    """Test restricted Hartree-Fock for CH4 with STO-3G.

    Reference data: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """C  -0.000000000000   0.000000000000   0.000000000000;
           H   1.183771681898  -1.183771681898  -1.183771681898;
           H   1.183771681898   1.183771681898   1.183771681898;
           H  -1.183771681898   1.183771681898  -1.183771681898;
           H  -1.183771681898  -1.183771681898   1.183771681898""",
        distance_unit="bohr",
    )
    A.set_basis_set("sto-3g")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    A.init_properties()
    assert_allclose(A.properties.get_dipole_moment(A.hartree_fock.rdm1), [0, 0, 0], atol=10**-6)
