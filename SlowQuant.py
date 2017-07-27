import numpy as np
import time
import sys
from slowquant import BasisSet as BS
from slowquant import runMolecularIntegrals as MI
from slowquant import HartreeFock as HF   
from slowquant import Properties as prop
from slowquant import MPn as MP
from slowquant import Qfit as QF
from slowquant import IntegralTransform as utilF 
from slowquant import GeometryOptimization as GO
from slowquant import UHF
from slowquant import CI
from slowquant import CC

import slowquant.molecularintegrals.runMIcython as prof

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
        results = MI.runIntegrals(input, basis, set, results)
        print(time.time()-start, 'INTEGRALS')
        
        start = time.time()
        C_alpha, F_alpha, D_alpha, C_beta, F_beta, D_beta, results = UHF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results)
        print(time.time()-start, 'UHF')

    elif set['GeoOpt'] == 'Yes':
        input, results = GO.runGO(input, set, results)
    
    if set['Initial method'] == 'HF':
        basis = BS.bassiset(input, set)
        
        start = time.time()
        results = MI.runIntegrals(input, basis, set, results)
        print(time.time()-start, 'INTEGRALS')
        
        start = time.time()
        results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results)
        print(time.time()-start, 'HF')
        
        start = time.time()
        results = utilF.runTransform(set, results)
        print(time.time()-start, 'MO transform')
        
        start = time.time()
        results = prop.runprop(basis, input, set, results)
        print(time.time()-start, 'PROPERTIES')
        
        start = time.time()
        results = MP.runMPn(input, results, set)
        print(time.time()-start, 'Perturbation')
        
        start = time.time()
        results = QF.runQfit(basis, input, set, results)
        print(time.time()-start, 'QFIT')
        
        start = time.time()
        results = CI.runCI(set, results, input)
        print(time.time()-start, 'CI')
        
        start = time.time()
        results = CC.runCC(input, set, results)
        print(time.time()-start, 'CC')
        
    return results

    
if __name__ == "__main__":
    total = time.time()
    mol = str(sys.argv[1])
    set = str(sys.argv[2])
    results = run(mol, set)
    print(time.time() - total, 'Execution time')
    
