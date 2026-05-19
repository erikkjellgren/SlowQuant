import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.linear_response import naive
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def get_triplet_excita(geometry, basis, active_space, unit='bohr'):
    """
    Calculate the triplet spin-adapted excitation energies
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(geometry, distance_unit=unit)
    SQobj.set_basis_set(basis)

    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()

    # OO-UCCSD
    WF = WaveFunctionUCC(
        active_space,
        SQobj.hartree_fock.mo_coeff,
        SQobj,
        "SD",
    )

    # Optimize wavefunction
    if active_space[1] == SQobj.molecule.number_bf:
        WF.run_wf_optimization_1step('SLSQP', False)
    else:
        WF.run_wf_optimization_1step('SLSQP', True)

    # Linear Response
    LR = naive.LinearResponse(WF, excitations="SD", triplet=True)
    LR.calc_excitation_energies()

    print('Triplet excitation energies (hartree):')
    for i, en in enumerate(LR.excitation_energies):
        print(f'\t{i+1}: {en:.4f}')

    return LR.excitation_energies


def test_H2_sto3g_naive_triplet():
    """
    Test of triplet excitation energies for naive LR with H2(2,2)/STO-3G
    """
    geometry = """H  0.0  0.0  0.0;
            H  0.74  0.0  0.0;"""
    excita = get_triplet_excita(geometry=geometry, basis='sto-3g', active_space=(2,2), unit='angstrom')
    
    thresh = 10**-4

    # Check excitation energies
    assert abs(excita[0] - 0.606510) < thresh


def test_LiH_sto3g_naive_triplet():
    """
    Test of triplet excitation energies for naive LR with LiH(2,2)/STO-3G
    """
    # Slowquant Object with parameters and setup
    geometry = """Li  0.0  0.0  0.0;
            H  0.8  0.0  0.0;"""

    excita = get_triplet_excita(geometry=geometry, basis='sto-3g', active_space=(2,2), unit='angstrom')

    thresh = 10**-4

    # Check excitation energies
    assert abs(excita[0] - 0.102103) < thresh
    assert abs(excita[1] - 0.138627) < thresh
    assert abs(excita[2] - 0.138627) < thresh
    assert abs(excita[3] - 0.459503) < thresh
    assert abs(excita[4] - 0.676406) < thresh
    assert abs(excita[5] - 0.676406) < thresh
    assert abs(excita[6] - 0.786396) < thresh
    assert abs(excita[7] - 2.120330) < thresh
    assert abs(excita[8] - 2.195168) < thresh
    assert abs(excita[9] - 2.195168) < thresh
    assert abs(excita[10] - 2.597856) < thresh
    assert abs(excita[11] - 3.060383) < thresh


def test_H10_sto3g_naive_triplet():
    """
    Test of triplet excitation energies for naive LR with H10(2,2)/STO-3G
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

    excita = get_triplet_excita(geometry=geometry, basis='sto-3g', active_space=(2,2), unit='bohr')

    thresh = 10**-4

    # Check excitation energies
    assert abs(excita[0] - 0.174084) < thresh
    assert abs(excita[1] - 0.283825) < thresh
    assert abs(excita[2] - 0.399352) < thresh
    assert abs(excita[3] - 0.595528) < thresh
    assert abs(excita[4] - 0.661147) < thresh
    assert abs(excita[5] - 0.729917) < thresh
    assert abs(excita[6] - 0.820365) < thresh
    assert abs(excita[7] - 0.858801) < thresh
    assert abs(excita[8] - 1.013200) < thresh
    assert abs(excita[9] - 1.030337) < thresh
    assert abs(excita[10] - 1.096057) < thresh
    assert abs(excita[11] - 1.130836) < thresh
    assert abs(excita[12] - 1.186165) < thresh
    assert abs(excita[13] - 1.230153) < thresh
    assert abs(excita[14] - 1.337101) < thresh
    assert abs(excita[15] - 1.356619) < thresh
    assert abs(excita[16] - 1.368000) < thresh
    assert abs(excita[17] - 1.472716) < thresh
    assert abs(excita[18] - 1.523427) < thresh
    assert abs(excita[19] - 1.598596) < thresh
    assert abs(excita[20] - 1.685950) < thresh
    assert abs(excita[21] - 1.712598) < thresh
    assert abs(excita[22] - 1.856746) < thresh
    assert abs(excita[23] - 2.032029) < thresh
    assert abs(excita[24] - 2.079022) < thresh
    assert abs(excita[25] - 2.111754) < thresh
    assert abs(excita[26] - 2.210035) < thresh
    assert abs(excita[27] - 2.242963) < thresh
    assert abs(excita[28] - 2.408660) < thresh
    assert abs(excita[29] - 2.504838) < thresh
    assert abs(excita[30] - 2.564356) < thresh
    assert abs(excita[31] - 2.751666) < thresh

test_H2_sto3g_naive_triplet()
test_LiH_sto3g_naive_triplet()
test_H10_sto3g_naive_triplet()