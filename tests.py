import BasisSet as BS
import DIIS as DIIS
import HartreeFock as HF
import MolecularIntegrals as MI
import MPn as MP
import Properties as PROP
import Qfit as QFIT
import Utilityfunc as UF
import numpy as np


def test_N():
    check = 5.701643762839922
    calc = BS.N(1.0,1.0,1.0,1.0)
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