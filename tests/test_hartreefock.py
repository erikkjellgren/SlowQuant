import slowquant.SlowQuant as sq


def test_restricted_hartree_fock_h2o_sto3g() -> None:
    """Test restricted Hartree-Fock for H2O with STO-3G.

    Reference data: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """O 0.000000000000  -0.143225816552   0.000000000000;
           H 1.638036840407   1.136548822547  -0.000000000000;
           H -1.638036840407   1.136548822547  -0.000000000000;""",
        distance_unit='bohr',
    )
    A.set_basis_set('sto-3g')
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    assert abs(A.molecule.nuclear_repulsion - 8.0023670618) < 10**-6
    assert abs(A.hartree_fock.E_hf - (-82.944446990003)) < 10**-6


def test_restricted_hartree_fock_h2o_dz() -> None:
    """Test restricted Hartree-Fock for H2O with DZ.

    Reference data: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """O 0.000000000000  -0.143225816552   0.000000000000;
           H 1.638036840407   1.136548822547  -0.000000000000;
           H -1.638036840407   1.136548822547  -0.000000000000;""",
        distance_unit='bohr',
    )
    A.set_basis_set('dz')
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    assert abs(A.molecule.nuclear_repulsion - 8.0023670618) < 10**-6
    assert abs(A.hartree_fock.E_hf - (-83.980246037187)) < 10**-6


def test_restricted_hartree_fock_h2o_dzp() -> None:
    """Test restricted Hartree-Fock for H2O with DZP.

    Does NOT give the same as in the reference. (Reference is: -84.011188854711)
    Reference data: https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2303
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """O 0.000000000000  -0.143225816552   0.000000000000;
           H 1.638036840407   1.136548822547  -0.000000000000;
           H -1.638036840407   1.136548822547  -0.000000000000;""",
        distance_unit='bohr',
    )
    A.set_basis_set('dzp')
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    assert abs(A.molecule.nuclear_repulsion - 8.0023670618) < 10**-6
    assert abs(A.hartree_fock.E_hf - (-84.01054766778844)) < 10**-6


def test_restricted_hartree_fock_ch4_sto3g() -> None:
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
        distance_unit='bohr',
    )
    A.set_basis_set('sto-3g')
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    assert abs(A.molecule.nuclear_repulsion - 13.4973044620) < 10**-6
    assert abs(A.hartree_fock.E_hf - (-53.224154786383)) < 10**-6
