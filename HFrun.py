import BasisSet as BS
import MolecularIntegrals as MI
import HartreeFock as HF
import os
import numpy as np
import time

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
Cr, Fnew = HF.HartreeFock(input, set, basis)
print(time.time()-start, 'HF')

