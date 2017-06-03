import numpy as np
import time
from slowquant import BasisSet as BS
from slowquant import MolecularIntegrals as MI
from slowquant import HartreeFock as HF   
from slowquant import Properties as prop
from slowquant import MPn as MP
from slowquant import Qfit as QF
from slowquant import IntegralTransform as utilF 
from slowquant import GeometryOptimization as GO
from slowquant import UHF

def run(inputname, settingsname):
    settings = np.genfromtxt('slowquant/Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    
    input = np.genfromtxt(str(inputname), delimiter=';')
    results = {}
    
    output = open('out.txt', 'w')
    output.write('User specified settings: \n')
    settings = np.genfromtxt(str(settingsname), delimiter = ';', dtype='str')
    for i in range(len(settings)):
        set[settings[i][0]] = settings[i][1]
        output.write('    '+str(settings[i][0])+'    '+str(settings[i][1])+'\n')
    output.write('\n \n')

    output.write('Inputfile: \n')
    for i in range(0, len(input)):
        for j in range(0, 4):
            output.write("   {: 12.8e}".format(input[i,j]))
            output.write("\t \t")
        output.write('\n')
    output.write('\n \n')
    output.close()
    
    if set['Initial method'] == 'UHF':
        basis = BS.bassiset(input, set)
        start = time.time()
        MI.runIntegrals(input, basis, set)
        print(time.time()-start, 'INTEGRALS')
        C_alpha, F_alpha, D_alpha, C_beta, F_beta, D_beta, results = UHF.HartreeFock(input, set, basis, VNN=np.load('slowquant/temp/enuc.npy'), Te=np.load('slowquant/temp/Ekin.npy'), S=np.load('slowquant/temp/overlap.npy'), VeN=np.load('slowquant/temp/nucatt.npy'), Vee=np.load('slowquant/temp/twoint.npy'), results=results)

    elif set['GeoOpt'] == 'Yes':
        input, results = GO.runGO(input, set, results)
    
    if set['Initial method'] == 'HF':
        basis = BS.bassiset(input, set)
        start = time.time()
        MI.runIntegrals(input, basis, set)
        print(time.time()-start, 'INTEGRALS')
        start = time.time()
        CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN=np.load('slowquant/temp/enuc.npy'), Te=np.load('slowquant/temp/Ekin.npy'), S=np.load('slowquant/temp/overlap.npy'), VeN=np.load('slowquant/temp/nucatt.npy'), Vee=np.load('slowquant/temp/twoint.npy'), results=results)
        print(time.time()-start, 'HF')
        start = time.time()
        utilF.TransformMO(CMO, basis, set, Vee=np.load('slowquant/temp/twoint.npy'))
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
    run('inputExampleH2O.csv', 'settingExample.csv')
