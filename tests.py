import BasisSet as BS
import DIIS as DIIS
import HartreeFock as HF
import MolecularIntegrals as MI
import MPn as MP
import Properties as PROP
import Qfit as QFIT
import Utilityfunc as UF
import numpy as np
import HFrun as HFrun

def test_N():
    check = 5.701643762839922
    calc = BS.N(1.0,1.0,1.0,1.0)
    assert abs(check-calc) < 10**-12
    calc = MI.N(1.0,1.0,1.0,1.0)
    assert abs(check-calc) < 10**-12

def test_gaussian_product_center():
    check = 3.5
    calc = MI.gaussian_product_center(1.0,2.0,3.0,4.0)
    assert abs(check - calc) < 10**-12

def test_magvec():
    check = 5.196152422706632
    calc = QFIT.magvec([1,2,3],[4,5,6])
    assert abs(check - calc) < 10**-12

def test_centerofcharge():
    check1 = 0.1020452034
    check2 = 0.162516435
    check3 = 0.0
    input1 = np.array([[10,0,0,0],[8.0, 0.0, 0.0, 0],[1.0, 1.70075339, 0.0, 0],[1.0, -0.68030136, 1.62516435, 0.0]])
    calc1, calc2, calc3 = QFIT.centerofcharge(input1)
    assert abs(check1 - calc1) < 10**-8
    assert abs(check2 - calc2) < 10**-8
    assert abs(check3 - calc3) < 10**-8

def test_solveFit():
    check = np.array([-4.0, 4.5])
    A = np.array([[1.0,2.0],[3.0,4.0]])
    B = np.array([5.0,6.0])
    calc = QFIT.solveFit(A,B)
    assert abs(check[0] - calc[0]) < 10**-12
    assert abs(check[1] - calc[1]) < 10**-12

def test_makepoints():
    check = np.array([[  4.92471737e-16,   0.00000000e+00,  -4.02133690e+00,
          0.00000000e+00,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   4.02133690e+00,
          0.00000000e+00,   0.00000000e+00]])
    settting = {'Griddensity':0.07, 'vdW scaling':1.4}
    input1 = np.array([[10,0,0,0],[8.0, 0.0, 0.0, 0],[1.0, 1.70075339, 0.0, 0],[1.0, -0.68030136, 1.62516435, 0.0]])
    calc = QFIT.makepoints(settting, input1)
    assert np.sum(np.abs(check-calc)) < 10**-8

def test_basissets():
    Basis = ['STO3G','DZ','DZP']
    Atoms = {'STO3G':[1,6,7,8],'DZ':[1,8],'DZP':[1,8]}
    for i in Basis:
        Atom = Atoms[i]
        for j in Atom:
            A = BS.bassiset([[0,0,0,0],[j,0,0,0]], set={'basisset':i})
            sum1 = 0
            sum2 = 0
            sum3 = 0
            assert len(A[0][5]) == A[0][4]
            for k in range(0, len(A[0][5])):
                sum1 = sum1 + A[0][5][k][-1]
                sum2 = sum2 + A[0][5][k][-2]
                sum3 = sum3 + A[0][5][k][-3]
            assert sum1 == sum2
            assert sum1 == sum3

def test_boys():
    m = [0.5, 13.0, 20.6, 25.0, 64.0, 75.5, 80.3, 4.0, 8.5, 15.3, 1.8, 30, 46.8, 100.0]
    x = [6.8, 14.1, 32.4, 6.4, 50.0, 40.8, 78.2, 7.0, 3.6, 20.7, 25.3, 26.0, 37.6, 125.1]
    scale = [2,7,14,5,24,20,36,4,3,10,4,13,18,55]
    check = [7.34475165333247E-02,
            1.56775160456192E-07,
            2.17602798734846E-14,
            4.28028518677348E-05,
            5.67024356263279E-24,
            2.63173492081630E-20,
            6.35062774057122E-36,
            8.03538503977806E-04,
            2.31681539108704E-03,
            5.40914879973724E-10,
            3.45745419193244E-04,
            3.57321060811178E-13,
            1.91851951160577E-18,
            7.75391047694625E-55]
    for i in range(0, len(x)):
        assert abs(MI.boys(m[i], x[i])-check[i])*10**scale[i] < 10**-8
    
def test_HartreeFock1():
    settings = np.genfromtxt('Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    input    = np.genfromtxt('testfiles/inputH2O.csv', delimiter=';')
    set['DIIS'] = 'No'
    set['basisset'] = 'STO3G'
    VNN      = np.load('testfiles/enucH2O_STO3G.npy')
    Te       = np.load('testfiles/EkinH2O_STO3G.npy')
    S        = np.load('testfiles/overlapH2O_STO3G.npy')
    VeN      = np.load('testfiles/nucattH2O_STO3G.npy')
    Vee      = np.load('testfiles/twointH2O_STO3G.npy')
    Dcheck   = np.genfromtxt('testfiles/dH2O_STO3G.csv',delimiter=';')
    basis    = BS.bassiset(input, set)
    results  = {}
    CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN, Te, S, VeN, Vee, results)
    for i in range(0, len(D)):
        for j in range(0, len(D)):
            assert abs(Dcheck[i,j] - D[i,j]) < 10**-7
    
def test_HartreeFock2():
    settings = np.genfromtxt('Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    input    = np.genfromtxt('testfiles/inputCH4.csv', delimiter=';')
    set['DIIS'] = 'No'
    set['basisset'] = 'STO3G'
    VNN      = np.load('testfiles/enucCH4_STO3G.npy')
    Te       = np.load('testfiles/EkinCH4_STO3G.npy')
    S        = np.load('testfiles/overlapCH4_STO3G.npy')
    VeN      = np.load('testfiles/nucattCH4_STO3G.npy')
    Vee      = np.load('testfiles/twointCH4_STO3G.npy')
    Dcheck   = np.genfromtxt('testfiles/dCH4_STO3G.csv',delimiter=';')
    basis    = BS.bassiset(input, set)
    results  = {}
    CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN, Te, S, VeN, Vee, results=results)
    for i in range(0, len(D)):
        for j in range(0, len(D)):
            assert abs(Dcheck[i,j] - D[i,j]) < 10**-7

def test_HartreeFock3():
    settings = np.genfromtxt('Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    input    = np.genfromtxt('testfiles/inputH2O.csv', delimiter=';')
    set['DIIS'] = 'No'
    set['basisset'] = 'DZ'
    VNN      = np.load('testfiles/enucH2O_DZ.npy')
    Te       = np.load('testfiles/EkinH2O_DZ.npy')
    S        = np.load('testfiles/overlapH2O_DZ.npy')
    VeN      = np.load('testfiles/nucattH2O_DZ.npy')
    Vee      = np.load('testfiles/twointH2O_DZ.npy')
    Dcheck   = np.genfromtxt('testfiles/dH2O_DZ.csv',delimiter=';')
    basis    = BS.bassiset(input, set)
    results  = {}
    CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN, Te, S, VeN, Vee, results=results)
    for i in range(0, len(D)):
        for j in range(0, len(D)):
            assert abs(Dcheck[i,j] - D[i,j]) < 10**-7

def test_MP2_1():
    settings = np.genfromtxt('Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    set['basisset'] = 'STO3G'
    set['MPn'] = 'MP2'
    input    = np.genfromtxt('testfiles/inputCH4.csv', delimiter=';')
    basis    = BS.bassiset(input, set)
    F        = np.load('testfiles/faoCH4_STO3G.npy')
    C        = np.load('testfiles/cmoCH4_STO3G.npy')
    UF.TransformMO(C, basis, set, Vee=np.load('testfiles/twointCH4_STO3G.npy'))
    calc = MP.MP2(basis, input, F, C)
    check = -0.056046676165
    assert abs(calc - check) < 10**-7


def test_MP2_2():
    settings = np.genfromtxt('Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    set['basisset'] = 'DZ'
    set['MPn'] = 'MP2'
    input    = np.genfromtxt('testfiles/inputH2O.csv', delimiter=';')
    basis    = BS.bassiset(input, set)
    F        = np.load('testfiles/faoH2O_DZ.npy')
    C        = np.load('testfiles/cmoH2O_DZ.npy')
    UF.TransformMO(C, basis, set, Vee=np.load('testfiles/twointH2O_DZ.npy'))
    calc = MP.MP2(basis, input, F, C)
    check = -0.152709879075
    assert abs(calc - check) < 10**-7


def test_derivative():
    settings = np.genfromtxt('Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    input = np.array([[8, 0, 0, 0],[8, 0, 0, 0]])
    basis = BS.bassiset(input, set)
    MI.rungeometric_derivatives(input, basis)
    VNe = np.load('1dynucatt.npy')
    S   = np.load('1dyoverlap.npy')
    Te  = np.load('1dyEkin.npy')
    VNN = np.load('1dyenuc.npy')
    ERI = np.load('1dytwoint.npy')
    
    assert np.max(np.abs(ERI)) < 10**-12
    assert np.max(np.abs(VNN)) < 10**-12
    assert np.max(np.abs(Te)) < 10**-12
    assert np.max(np.abs(S)) < 10**-12
    assert np.max(np.abs(VNe)) < 10**-12

## REGRESSION TESTS
def test_prop():
    HFrun.run('testfiles/inputH2O.csv','testfiles/settingsPROP.csv')
    check = open('testfiles/outPROP.txt','r')
    calc = open('out.txt')
    for line in check:
        if line[0:3] == 'MP2':
            checkMP2 = float(line[12:])
        if line[0:5] == 'Total':
            checkMolDip = float(line[7:])
        if line[0:5] == 'Atom1':
            checkMulChr = float(line[7:])
    
    for line in calc:
        if line[0:3] == 'MP2':
            calcMP2 = float(line[12:])
        if line[0:5] == 'Total':
            calcMolDip = float(line[7:])
        if line[0:5] == 'Atom1':
            calcMulChr = float(line[7:])
    
    assert checkMP2 == calcMP2
    assert checkMolDip == calcMolDip
    assert checkMulChr == calcMulChr

def test_qfit():
    HFrun.run('testfiles/input2_H2O.csv','testfiles/settingsQFIT.csv')
    check = open('testfiles/outQFIT.txt','r')
    calc = open('out.txt')
    for line in check:
        if line[0:4] == 'RMSD':
            checkRMSD = float(line[6:])
    
    for line in calc:
        if line[0:4] == 'RMSD':
            calcRMSD = float(line[6:])
    
    assert checkRMSD == calcRMSD
        
    
def test_geoopt():
    HFrun.run('testfiles/inputH2.csv','testfiles/settingsGEO.csv')
    check = open('testfiles/outGEO.txt','r')
    calc = open('out.txt')
    for line in check:
        if line[0:3] == 'MP2':
            checkMP2 = float(line[12:])

    for line in calc:
        if line[0:3] == 'MP2':
            calcMP2 = float(line[12:])
    
    assert checkMP2 == calcMP2
