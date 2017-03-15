import BasisSet as BS
import MolecularIntegrals as MI
import HartreeFock as HF
import os
import numpy as np
import time
import Properties as prop
import MPn as MP

input = np.genfromtxt('inputH2O.csv', delimiter=';')
settings = np.genfromtxt('settings.csv', delimiter = ';', dtype='str')
set = {}
for i in range(len(settings)):
    set.update({settings[i][0]:settings[i][1]})

basis = BS.bassiset(input, set)
start = time.time()
MI.runIntegrals(input, basis)
print(time.time()-start, 'INTEGRALS')
start = time.time()
FAO, D = HF.HartreeFock(input, set, basis)
print(time.time()-start, 'HF')
start = time.time()
prop.runprop(basis, input, D, set)
print(time.time()-start, 'Properties')

