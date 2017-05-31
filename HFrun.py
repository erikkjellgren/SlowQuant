import BasisSet as BS
import MolecularIntegrals as MI
import HartreeFock as HF   
import numpy as np
import time
import Properties as prop
import MPn as MP
import Qfit as QF
import Utilityfunc as utilF 
import GeometryOptimization as GO

def run(inputname, settingsname):
    settings = np.genfromtxt('Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    
    input = np.genfromtxt(str(inputname), delimiter=';')
    results = {}
    settings = np.genfromtxt(str(settingsname), delimiter = ';', dtype='str')
    for i in range(len(settings)):
        set[settings[i][0]] = settings[i][1]
    
    output = open('out.txt', 'w')
    for key in set:
        output.write(str(key)+'    '+str(set[key])+'\n')
    output.write('\n \n')
    for i in range(0, len(input)):
        for j in range(0, 4):
            output.write("{: 12.8e}".format(input[i,j]))
            output.write("\t \t")
        output.write('\n')
    output.write('\n \n')
    output.close()
    
    if set['GeoOpt'] == 'Yes':
        input = GO.runGO(input, set)
    
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

if __name__ == "__main__":
    run('inputH2.csv', 'settings.csv')
