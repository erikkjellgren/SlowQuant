import numpy as np
import SlowQuant as HFrun
import slowquant.basissets.BasisSet as BS
import slowquant.hartreefock.DIIS as DIIS
import slowquant.hartreefock.runHartreeFock as HF   
from slowquant.hartreefock.HartreeFock import HartreeFock
import slowquant.molecularintegrals.runMolecularIntegrals as MI
from slowquant.molecularintegrals.runMIcython import boysPrun
import slowquant.mollerplesset.runMPn as MP
import slowquant.properties.runProperties as prop
import slowquant.qfit.Qfit as QFIT
import slowquant.integraltransformation.IntegralTransform as UF

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
        assert abs(boysPrun(m[i], x[i])-check[i])*10**scale[i] < 10**-8
    
def test_HartreeFock1():
    settings = np.genfromtxt('slowquant/Standardsettings.csv', delimiter = ';', dtype='str')
    results = {}
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    input    = np.genfromtxt('data/testfiles/inputH2O.csv', delimiter=';')
    set['DIIS'] = 'No'
    set['basisset'] = 'STO3G'
    results['VNN']      = np.load('data/testfiles/enucH2O_STO3G.npy')
    results['Te']       = np.load('data/testfiles/EkinH2O_STO3G.npy')
    results['S']        = np.load('data/testfiles/overlapH2O_STO3G.npy')
    results['VNe']      = np.load('data/testfiles/nucattH2O_STO3G.npy')
    results['Vee']      = np.load('data/testfiles/twointH2O_STO3G.npy')
    Dcheck   = np.genfromtxt('data/testfiles/dH2O_STO3G.csv',delimiter=';')
    basis    = BS.bassiset(input, set)
    results = HF.runHartreeFock(input, set, results, print_SCF='No')
    D = results['D']
    for i in range(0, len(D)):
        for j in range(0, len(D)):
            assert abs(Dcheck[i,j] - D[i,j]) < 10**-7
    
def test_HartreeFock2():
    settings = np.genfromtxt('slowquant/Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    input    = np.genfromtxt('data/testfiles/inputCH4.csv', delimiter=';')
    set['DIIS'] = 'No'
    set['basisset'] = 'STO3G'
    results = {}
    results['VNN']      = np.load('data/testfiles/enucCH4_STO3G.npy')
    results['Te']       = np.load('data/testfiles/EkinCH4_STO3G.npy')
    results['S']        = np.load('data/testfiles/overlapCH4_STO3G.npy')
    results['VNe']      = np.load('data/testfiles/nucattCH4_STO3G.npy')
    results['Vee']      = np.load('data/testfiles/twointCH4_STO3G.npy')
    Dcheck   = np.genfromtxt('data/testfiles/dCH4_STO3G.csv',delimiter=';')
    basis    = BS.bassiset(input, set)
    results = HF.runHartreeFock(input, set, results, print_SCF='No')
    D = results['D']
    for i in range(0, len(D)):
        for j in range(0, len(D)):
            assert abs(Dcheck[i,j] - D[i,j]) < 10**-7

def test_HartreeFock3():
    settings = np.genfromtxt('slowquant/Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    input    = np.genfromtxt('data/testfiles/inputH2O.csv', delimiter=';')
    set['DIIS'] = 'No'
    set['basisset'] = 'DZ'
    results = {}
    results['VNN']      = np.load('data/testfiles/enucH2O_DZ.npy')
    results['Te']       = np.load('data/testfiles/EkinH2O_DZ.npy')
    results['S']        = np.load('data/testfiles/overlapH2O_DZ.npy')
    results['VNe']      = np.load('data/testfiles/nucattH2O_DZ.npy')
    results['Vee']      = np.load('data/testfiles/twointH2O_DZ.npy')
    Dcheck   = np.genfromtxt('data/testfiles/dH2O_DZ.csv',delimiter=';')
    basis    = BS.bassiset(input, set)
    results = HF.runHartreeFock(input, set, results, print_SCF='No')
    D = results['D']
    for i in range(0, len(D)):
        for j in range(0, len(D)):
            assert abs(Dcheck[i,j] - D[i,j]) < 10**-7

def test_MP2_1():
    settings = np.genfromtxt('slowquant/Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    set['basisset'] = 'STO3G'
    set['MPn'] = 'MP2'
    results  = {}
    input    = np.genfromtxt('data/testfiles/inputCH4.csv', delimiter=';')
    basis    = BS.bassiset(input, set)
    results['F']        = np.load('data/testfiles/faoCH4_STO3G.npy')
    results['C_MO']     = np.load('data/testfiles/cmoCH4_STO3G.npy')
    results['Vee']      = np.load('data/testfiles/twointCH4_STO3G.npy')
    results = MP.runMPn(input, results, set)
    check = -0.056046676165
    assert abs(results['EMP2'] - check) < 10**-7


def test_MP2_2():
    settings = np.genfromtxt('slowquant/Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    set['basisset'] = 'DZ'
    set['MPn'] = 'MP2'
    input    = np.genfromtxt('data/testfiles/inputH2O.csv', delimiter=';')
    basis    = BS.bassiset(input, set)
    results  = {}
    results['F']         = np.load('data/testfiles/faoH2O_DZ.npy')
    results['C_MO']      = np.load('data/testfiles/cmoH2O_DZ.npy')
    results['Vee']       = np.load('data/testfiles/twointH2O_DZ.npy')
    results = MP.runMPn(input, results, set)
    check = -0.152709879075
    assert abs(results['EMP2'] - check) < 10**-7


def test_derivative():
    # Tests that a single atom have no geometric gradient
    settings = np.genfromtxt('slowquant/Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    results = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    input = np.array([[8, 0, 0, 0],[8, 0.0, 0.0, 0.0]])
    basis = BS.bassiset(input, set)
    results = MI.rungeometric_derivatives(input, basis, set, results)
    VNe = results['1dyVNe']
    S   = results['1dyS']
    Te  = results['1dyTe']
    VNN = results['1dyVNN']
    ERI = results['1dyVee']
    
    assert np.max(np.abs(ERI)) < 10**-12
    assert np.max(np.abs(VNN)) < 10**-12
    assert np.max(np.abs(Te)) < 10**-12
    assert np.max(np.abs(S)) < 10**-12
    assert np.max(np.abs(VNe)) < 10**-12

## REGRESSION TESTS
def test_prop():
    results = HFrun.run('data/testfiles/inputH2O.csv','data/testfiles/settingsPROP.csv')
    check = open('data/testfiles/outPROP.txt','r')
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
    HFrun.run('data/testfiles/input2_H2O.csv','data/testfiles/settingsQFIT.csv')
    check = open('data/testfiles/outQFIT.txt','r')
    calc = open('out.txt')
    for line in check:
        if line[0:4] == 'RMSD':
            checkRMSD = float(line[6:])
    
    for line in calc:
        if line[0:4] == 'RMSD':
            calcRMSD = float(line[6:])
    
    assert checkRMSD == calcRMSD
        
        
def test_geoopt():
    results = HFrun.run('data/testfiles/H2.csv','data/testfiles/settingsGEO.csv')
    e = 0.000001
    dp = np.load('data/testfiles/enucp.npy')
    dm = np.load('data/testfiles/enucm.npy')
    dVNN = results['1dxVNN']
    dnVNN = (dp-dm)/(2*e)
    assert np.max(np.abs(dVNN-dnVNN)) < 10**-9
    
    e = 0.000001
    dp = np.load('data/testfiles/overlapp.npy')
    dm = np.load('data/testfiles/overlapm.npy')
    dS = results['1dxS']
    dnS = (dp-dm)/(2*e)
    assert np.max(np.abs(dS-dnS)) < 10**-9
    
    e = 0.000001
    dp = np.load('data/testfiles/Ekinp.npy')
    dm = np.load('data/testfiles/Ekinm.npy')
    dTe = results['1dxTe']
    dnTe = (dp-dm)/(2*e)
    assert np.max(np.abs(dTe-dnTe)) < 10**-9
    
    e = 0.000001
    dp = np.load('data/testfiles/nucattp.npy')
    dm = np.load('data/testfiles/nucattm.npy')
    dVNe = results['1dxVNe']
    dnVNe = (dp-dm)/(2*e)
    assert np.max(np.abs(dVNe-dnVNe)) < 10**-9
    
    e = 0.000001
    dp = np.load('data/testfiles/twointp.npy')
    dm = np.load('data/testfiles/twointm.npy')
    dERI = results['1dxVee']
    dnERI = (dp-dm)/(2*e)
    assert np.max(np.abs(dERI-dnERI)) < 10**-9
    

def test_UHF():
    results = HFrun.run('data/testfiles/inputH2_UHF.csv','data/testfiles/settingsUHF.csv')
    check = open('data/testfiles/outUHF.txt','r')
    for line in check:
        if line[0:2] == '27':
            checkUHF = float(line[23:30])
    
    assert abs(checkUHF - results['UHFenergy']) < 10**-4


def test_Lowdin():
    HFrun.run('data/testfiles/inputH2O.csv','data/testfiles/settingsLowdin.csv')
    check = open('data/testfiles/outLowdin.txt','r')
    calc = open('out.txt')
    for line in check:
        if line[0:5] == 'Atom1':
            checkLow = float(line[7:])
    for line in calc:
        if line[0:5] == 'Atom1':
            calcLow = float(line[7:])
    
    assert calcLow == checkLow

def test_Ffunction():
    results = HFrun.run('data/testfiles/Hm.csv','data/testfiles/settingFfunctions.csv')
    assert results['HFenergy'] + 0.475129018306 < 10**-5

def test_CIS():
    results = HFrun.run('data/testfiles/inputH2O.csv','data/testfiles/settingsCIS.csv')
    check = [0.2872555,0.2872555,0.2872555,0.344424996,0.344424996,0.344424996,0.356461759,0.365988995,0.365988995,0.365988995,0.394513799,0.394513799,0.394513799,0.416071739,0.505628288,0.514289997,0.514289997,0.514289997,0.555191886,0.563055764,0.563055764,0.563055764,0.655318449,0.910121689,1.108770966,1.108770966,1.108770966,1.200096133,1.200096133,1.200096133,1.300785195,1.325762065,19.95852641,19.95852641,19.95852641,20.01097942,20.01134209,20.01134209,20.01134209,20.05053194]
    for i in range(len(results['CIS Exc'])):
        assert results['CIS Exc'][i] - check[i] < 10**-6

def test_RPA():
    results = HFrun.run('data/testfiles/inputH2O.csv','data/testfiles/settingRPA.csv')
    check = [0.285163717,0.285163717,0.285163717,0.299743447,0.299743447,0.299743447,0.352626661,0.352626661,0.352626661,0.354778253,0.365131311,0.365131311,0.365131311,0.415317495,0.50010114,0.510661051,0.510661051,0.510661051,0.546071909,0.546071909,0.546071909,0.551371885,0.650270712,0.873425371,1.103818796,1.103818796,1.103818796,1.195787071,1.195787071,1.195787071,1.283205318,1.323742189,19.95850406,19.95850406,19.95850406,20.01094716,20.01130746,20.01130746,20.01130746,20.05049194]
    for i in range(len(results['RPA Exc'])):
        assert results['RPA Exc'][i] - check[i] < 10**-6

def test_MP3():
    results = HFrun.run('data/testfiles/inputH2.csv','data/testfiles/settingMP3.csv')
    print(results)
    calc = results['EMP2'] + results['EMP3']
    check = -0.0180
    assert abs(calc - check) < 10**-5

def test_CCSD():
    results = HFrun.run('data/testfiles/inputH2O.csv','data/testfiles/settingCCSD.csv')
    check = -0.070680088376-0.000099877272
    calc = results['ECCSD']+results['E(T)']
    assert abs(calc-check) < 10**-10

def test_GeoOptimization():
    # New geometry optimization test, after bug was found
    results = HFrun.run('data/testfiles/inputH2O.csv','data/testfiles/settingFullGeoOpt.csv')
    check = -74.9658980993
    assert abs(check-results['HFenergy']) < 10**-10

def test_GeoOptimizationNUM():
    # New geometry optimization test, after bug was found
    results = HFrun.run('data/testfiles/inputH2O.csv','data/testfiles/settingsGEONUM.csv')
    check = -74.9658980993
    assert abs(check-results['HFenergy']) < 10**-10

def test_BOMD():
    results = HFrun.run('data/testfiles/inputH2O.csv','data/testfiles/settingBOMD.csv')
    check = -75.5667945588
    assert abs(check-results['HFenergy']) < 10**-10