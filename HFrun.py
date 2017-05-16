import BasisSet as BS
import MolecularIntegrals as MI
import HartreeFock as HF
import os   
import numpy as np
import time
import Properties as prop
import MPn as MP
import Qfit as QF
import Utilityfunc as utilF 

settings = np.genfromtxt('Standardsettings.csv', delimiter = ';', dtype='str')
set = {}
for i in range(len(settings)):
    set.update({settings[i][0]:settings[i][1]})

input = np.genfromtxt('inputH2O.csv', delimiter=';')
results = {}
settings = np.genfromtxt('settings.csv', delimiter = ';', dtype='str')
for i in range(len(settings)):
    set[settings[i][0]] = settings[i][1]


basis = BS.bassiset(input, set)
start = time.time()
MI.runIntegrals(input, basis)
print(time.time()-start, 'INTEGRALS')
start = time.time()
CMO, FAO, D = HF.HartreeFock(input, set, basis, VNN=np.load('enuc.npy'), Te=np.load('Ekin.npy'), S=np.load('overlap.npy'), VeN=np.load('nucatt.npy'), Vee=np.load('twoint.npy'))
print(time.time()-start, 'HF')
start = time.time()
utilF.TransformMO(CMO, basis, set, Vee=np.load('twoint.npy'))
print(time.time()-start, 'MO transform')
start = time.time()
results = prop.runprop(basis, input, D, set, results)
print(time.time()-start, 'PROPERTIES')
start = time.time()
MP.runMPn(basis, input, FAO, CMO, set)
print(time.time()-start, 'MP2')
start = time.time()
QF.runQfit(basis, input, D, set, results)
print(time.time()-start, 'QFIT')

