import slowquant.hartreefock.runHartreeFock as HF
import numpy as np
import slowquant.basissets.BasisSet as BS
import slowquant.molecularintegrals.runMolecularIntegrals as MI

def nForce(input, set, results, print_time='No', print_scf='Yes'):
    dX = np.zeros(len(input))
    dY = np.zeros(len(input))
    dZ = np.zeros(len(input))
    for j in range(1, len(input)):
        input[j,1] += 10**-6
        basis = BS.bassiset(input, set)
        results = MI.runIntegrals(input, basis, set, results)
        input[j,1] -= 10**-6
        results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
        xplus = results['HFenergy']
        input[j,1] -= 10**-6
        basis = BS.bassiset(input, set)
        results = MI.runIntegrals(input, basis, set, results)
        input[j,1] += 10**-6
        results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
        xminus = results['HFenergy']
        
        input[j,2] += 10**-6
        basis = BS.bassiset(input, set)
        results = MI.runIntegrals(input, basis, set, results)
        input[j,2] -= 10**-6
        results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
        yplus = results['HFenergy']
        input[j,2] -= 10**-6
        basis = BS.bassiset(input, set)
        results = MI.runIntegrals(input, basis, set, results)
        input[j,2] += 10**-6
        results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
        yminus = results['HFenergy']
        
        input[j,3] += 10**-6
        basis = BS.bassiset(input, set)
        results = MI.runIntegrals(input, basis, set, results)
        input[j,3] -= 10**-6
        results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
        zplus = results['HFenergy']
        input[j,3] -= 10**-6
        basis = BS.bassiset(input, set)
        results = MI.runIntegrals(input, basis, set, results)
        input[j,3] += 10**-6
        results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
        zminus = results['HFenergy']
        
        dX[j] = (xplus-xminus)/(2*10**-6)
        dY[j] = (yplus-yminus)/(2*10**-6)
        dZ[j] = (zplus-zminus)/(2*10**-6)
    
    return dX, dY, dZ