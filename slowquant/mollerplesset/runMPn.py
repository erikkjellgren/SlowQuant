from slowquant.mollerplesset.MPn import MP2, MP3, DCPT2
import slowquant.integraltransformation.IntegralTransform as IT

def runMPn(input, results, set):
    if set['MPn'] == 'MP2' or set['MPn'] == 'MP3':
        results['VeeMO'] = IT.Transform2eMO(C=results['C_MO'], Vee=results['Vee'])
        EMP2 = MP2(occ=int(input[0][0]/2), F=results['F'], C=results['C_MO'], VeeMO=results['VeeMO'])
        output = open('out.txt', 'a')
        output.write('\n \n')
        output.write('MP2 Energy \t')
        output.write("{: 10.8f}".format(EMP2))
        output.close()
        results['EMP2'] = EMP2
    if set['MPn'] == 'MP3':
        results['VeeMOspin'] = IT.Transform2eSPIN(VeeMO=results['VeeMO'])
        EMP3 = MP3(occ=int(input[0][0]), F=results['F'], C=results['C_MO'], VeeMOspin=results['VeeMOspin'])
        output = open('out.txt', 'a')
        output.write('\n \n')
        output.write('MP3 Energy \t')
        output.write("{: 10.8f}".format(EMP3))
        output.close()
        results['EMP3'] = EMP3
    elif set['MPn'] == 'DCPT2':
        results['VeeMO'] = IT.Transform2eMO(C=results['C_MO'], Vee=results['Vee'])
        EDCPT2 = DCPT2(occ=int(input[0][0]/2), F=results['F'], C=results['C_MO'], VeeMO=results['VeeMO'])
        output = open('out.txt', 'a')
        output.write('\n \n')
        output.write('DCPT2 Energy \t')
        output.write("{: 10.8f}".format(EDCPT2))
        output.close()
        results['EDCPT2'] = EDCPT2
    return results