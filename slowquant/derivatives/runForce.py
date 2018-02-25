import numpy as np
import slowquant.molecularintegrals.runMolecularIntegrals as MI
import slowquant.hartreefock.runHartreeFock as HF
import slowquant.basissets.BasisSet as BS
from slowquant.derivatives.Force import Force

def runForce(input, set, results, print_time='No', print_scf='Yes'):
    basis = BS.bassiset(input, set['basisset'])
    results = MI.runIntegrals(input, basis, set, results)
    results = MI.rungeometric_derivatives(input, basis, set, results, print_time=print_time)
    results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
    
    CMO = results['C_MO'] 
    FAO = results['F'] 
    D = results['D'] 
    dX, dY, dZ, results = Force(input, D, CMO, FAO, results)
    
    return dX, dY, dZ, results