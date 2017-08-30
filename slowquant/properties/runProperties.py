import slowquant.molecularintegrals.runMolecularIntegrals as MI
from slowquant.properties.Properties import dipolemoment, MulCharge, LowdinCharge, RPA
from slowquant.integraltransformation.IntegralTransform import Transform2eMO, Transform2eSPIN

def runprop(basis, input, set, results):
    if set['Charge'] == 'Mulliken':
        output = open('out.txt', 'a')
        output.write('\n \n')
        output.write('Mulliken Charges \n')
        qvec = MulCharge(basis, input, results['D'], results['S'])
        for i in range(len(qvec)):
            output.write('Atom'+str(i+1)+'\t')
            output.write("{: 10.8f}".format(qvec[i]))
            output.write('\n')
        output.close()
        results['MullikenCharge'] = qvec
    elif set['Charge'] == 'Lowdin':
        output = open('out.txt', 'a')
        output.write('\n \n')
        output.write('Lowdin Charges \n')
        qvec = LowdinCharge(basis, input, results['D'], results['S'])
        for i in range(len(qvec)):
            output.write('Atom'+str(i+1)+'\t')
            output.write("{: 10.8f}".format(qvec[i]))
            output.write('\n')
        output.close()
        results['LowdinCharge'] = qvec
    if set['Dipole'] == 'Yes' or set['Multipolefit'] == 'Dipole':
        results = MI.run_dipole_int(basis, input, results)
        ux, uy, uz, u = dipolemoment(input,results['D'], results['mu_x'], results['mu_y'], results['mu_z'])
        output = open('out.txt', 'a')
        output.write('\n \nMolecular dipole moment \n')
        output.write('X \t \t')
        output.write("{: 10.8f}".format(ux))
        output.write('\nY \t \t')
        output.write("{: 10.8f}".format(uy))
        output.write('\nZ \t \t')
        output.write("{: 10.8f}".format(uz))
        output.write('\nTotal \t')
        output.write("{: 10.8f}".format(u))
        output.close()
        results['dipolex'] = ux
        results['dipoley'] = uy
        results['dipolez'] = uz
        results['dipoletot'] = u
    if set['Excitation'] == 'RPA':
        results['VeeMO'] = Transform2eMO(results['C_MO'],results['Vee'])
        results['VeeMOspin'] = Transform2eSPIN(results['VeeMO'])
        Exc = RPA(occ=int(input[0,0]), F=results['F'], C=results['C_MO'], VeeMOspin=results['VeeMOspin'])
        output = open('out.txt', 'a')
        output.write('RPA Excitation Energies: \n')
        output.write(' # \t\t Hartree \n')
        output.write('-- \t\t -------------- \n')
        for i in range(len(Exc)):
            output.write(str(i+1)+'\t\t')
            output.write("{: 12.8e}".format(Exc[i]))
            output.write('\n')
        output.write('\n \n')
        output.close()
        results['RPA Exc'] = Exc
    return results