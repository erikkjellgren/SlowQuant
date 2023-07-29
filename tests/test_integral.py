import numpy as np
from numpy.testing import assert_allclose

import slowquant.SlowQuant as sq


def test_cauchy_schwarz_inequality() -> None:
    """Test the use of Cauchy Schwarz inequaility.

    Test that integrals are identical within screening threshold, compared to a reference.

    Test that all integrals are zero if the sceening threshold is huge.
    """
    ref = sq.SlowQuant()
    ref.set_molecule(
        """H   0.000000000000   0.000000000000   0.000000000000;
                        H   1.000000000000   0.000000000000   0.000000000000;
                        H   2.000000000000   0.000000000000   0.000000000000;
                        H   3.000000000000   0.000000000000   0.000000000000;
                        H   4.000000000000   0.000000000000   0.000000000000;
                        H   5.000000000000   0.000000000000   0.000000000000""",
        distance_unit='bohr',
    )
    ref.set_basis_set('dzp')
    ref.integral.cauchy_schwarz_threshold = 10**-20
    ref_eri = ref.integral.electron_repulsion_tensor

    A = sq.SlowQuant()
    A.set_molecule(
        """H   0.000000000000   0.000000000000   0.000000000000;
                      H   1.000000000000   0.000000000000   0.000000000000;
                      H   2.000000000000   0.000000000000   0.000000000000;
                      H   3.000000000000   0.000000000000   0.000000000000;
                      H   4.000000000000   0.000000000000   0.000000000000;
                      H   5.000000000000   0.000000000000   0.000000000000""",
        distance_unit='bohr',
    )
    A.set_basis_set('dzp')
    A.integral.cauchy_schwarz_threshold = 10**-8
    assert_allclose(A.integral.electron_repulsion_tensor, ref_eri, atol=10**-8)

    A = sq.SlowQuant()
    A.set_molecule(
        """H   0.000000000000   0.000000000000   0.000000000000;
                      H   1.000000000000   0.000000000000   0.000000000000;
                      H   2.000000000000   0.000000000000   0.000000000000;
                      H   3.000000000000   0.000000000000   0.000000000000;
                      H   4.000000000000   0.000000000000   0.000000000000;
                      H   5.000000000000   0.000000000000   0.000000000000""",
        distance_unit='bohr',
    )
    A.set_basis_set('dzp')
    A.integral.cauchy_schwarz_threshold = 10**-3
    assert_allclose(A.integral.electron_repulsion_tensor, ref_eri, atol=10**-3)

    A = sq.SlowQuant()
    A.set_molecule(
        """H   0.000000000000   0.000000000000   0.000000000000;
                      H   1.000000000000   0.000000000000   0.000000000000;
                      H   2.000000000000   0.000000000000   0.000000000000;
                      H   3.000000000000   0.000000000000   0.000000000000;
                      H   4.000000000000   0.000000000000   0.000000000000;
                      H   5.000000000000   0.000000000000   0.000000000000""",
        distance_unit='bohr',
    )
    A.set_basis_set('dzp')
    A.integral.cauchy_schwarz_threshold = 10**10
    assert_allclose(A.integral.electron_repulsion_tensor, np.zeros(np.shape(ref_eri)), atol=10**-8)
