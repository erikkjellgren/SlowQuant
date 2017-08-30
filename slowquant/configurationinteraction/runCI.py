import numpy as np
from slowquant.configurationinteraction.CI import CIS
from slowquant.integraltransformation.IntegralTransform import Transform2eMO, Transform2eSPIN

def runCI(set, results, input):
    if set['CI'] == 'CIS':
        results['VeeMO'] = Transform2eMO(results['C_MO'],results['Vee'])
        results['VeeMOspin'] = Transform2eSPIN(results['VeeMO'])
        Exc = CIS(occ=int(input[0,0]), F=results['F'], C=results['C_MO'], VeeMOspin=results['VeeMOspin'])
        output = open('out.txt', 'a')
        output.write('CIS Excitation Energies: \n')
        output.write(' # \t\t Hartree \n')
        output.write('-- \t\t -------------- \n')
        for i in range(len(Exc)):
            output.write(str(i+1)+'\t\t')
            output.write("{: 12.8e}".format(Exc[i]))
            output.write('\n')
        output.write('\n \n')
        output.close()
        results['CIS Exc'] = Exc
    return results