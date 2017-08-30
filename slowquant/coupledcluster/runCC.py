import numpy as np
from slowquant.coupledcluster.PythonCC import CCSD
from slowquant.integraltransformation.IntegralTransform import Transform2eMO, Transform2eSPIN

def runCC(input, set, results):
    if set['CC'] == 'CCSD' or set['CC'] == 'CCSD(T)':
        results['VeeMO'] = Transform2eMO(results['C_MO'],results['Vee'])
        results['VeeMOspin'] = Transform2eSPIN(results['VeeMO'])
        if set['CC'] == 'CCSD(T)':
            Elist = CCSD(occ=int(input[0,0]), F=results['F'], C=results['C_MO'], VeeMOspin=results['VeeMOspin'], maxiter=int(set['CC Max iterations'])+1, deTHR=float(set['CC Energy Threshold']), rmsTHR=float(set['CC RMSD Threshold']),runCCSDT=1)
            results['E(T)'] = Elist[2]
        else:
            Elist = CCSD(occ=int(input[0,0]), F=results['F'], C=results['C_MO'], VeeMOspin=results['VeeMOspin'], maxiter=int(set['CC Max iterations'])+1, deTHR=float(set['CC Energy Threshold']), rmsTHR=float(set['CC RMSD Threshold']))
        results['EMP2'] = Elist[0]
        results['ECCSD'] = Elist[1] 
    return results