import pyscf
import slowquant.unitary_coupled_cluster.linear_response.naive as naivelr
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def get_polarisability(geometry, basis, active_space, charge=0, unit='bohr'):
    """
    Calculate the polarisability
    """
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, charge=charge, unit=unit)
    rhf = mol.RHF().run()
    mo_coeff = rhf.mo_coeff

    # SlowQuant
    WF = WaveFunctionUCC(
        active_space,
        mo_coeff,
        mol,
        "SD",
    )

    # Optimize WF
    if active_space[1] == mol.nao:
        WF.run_wf_optimization_1step('SLSQP', False)
    else:
        WF.run_wf_optimization_1step('SLSQP', True)
    print("Energy elec", WF.energy_elec)

    # Singlet Linear Response
    LR = naivelr.LinearResponse(WF, excitations="SD")
    alpha = LR.get_polarisability()

    print(f'Polarisabilities:\n \t xx: {alpha[0,0]:.4f} \t yy: {alpha[1,1]:.4f} \t zz: {alpha[2,2]:.4f}')

    return alpha


def test_H2_sto3g_naive():
    """
    Test of polarisability for naive LR with H2(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;"""
    alpha = get_polarisability(geometry, basis='sto-3g', active_space=(2,2), unit='angstrom')

    thresh = 10**-4

    # Check excitation energies - reference dalton mcscf
    assert abs(alpha[0,0] - 2.775271948863) < thresh
    assert abs(alpha[1,1] - 0.0) < thresh
    assert abs(alpha[2,2] - 0.0) < thresh


def test_LiH_sto3g_naive():
    """
    Test of polarisability for naive LR with LiH(2,2)/STO-3G
    """
    geometry = """H  0.0   0.0  0.0;
            Li  0.8  0.0  0.0;"""
    alpha = get_polarisability(geometry, basis='sto-3g', active_space=(2,2), unit='angstrom')

    thresh = 10**-2

    # Check excitation energies - reference dalton mcscf
    assert abs(alpha[0,0] - 0.5238650005008) < thresh
    assert abs(alpha[1,1] - 20.01552907544) < thresh
    assert abs(alpha[2,2] - 20.01552907544) < thresh


def test_H10_sto3g_naive():
    """
    Test of polarisability for naive LR with H10(2,2)/STO-3G
    """
    geometry = """H  0.0  0.0  0.0;
           H  1.4  0.0  0.0;
           H  2.8  0.0  0.0;
           H  4.2  0.0  0.0;
           H  5.6  0.0  0.0;
           H  7.0  0.0  0.0;
           H  8.4  0.0  0.0;
           H  9.8  0.0  0.0;
           H 11.2  0.0  0.0;
           H 12.6  0.0  0.0;"""

    alpha = get_polarisability(geometry, basis='sto-3g', active_space=(2,2), unit='bohr')

    thresh = 10**-3

    # Check excitation energies - reference dalton mcscf
    assert abs(alpha[0,0] - 72.343635) < thresh
    assert abs(alpha[1,1] - 0.0) < thresh
    assert abs(alpha[2,2] - 0.0) < thresh
