import numpy as np
import MolecularIntegrals as MI
import HartreeFock as HF
import BasisSet as BS

def run_analytic(input, set, results, settings):
    maxstep = int(set['Max iteration GeoOpt'])
    GeoOptol = float(set['Geometry Tolerance'])
    stepsize = float(set['Gradient Decent Step'])
    for i in range(1, maxstep):
        basis = BS.bassiset(input, set)
        MI.runIntegrals(input, basis, settings)
        MI.rungeometric_derivatives(input, basis)
        CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN=np.load('enuc.npy'), Te=np.load('Ekin.npy'), S=np.load('overlap.npy'), VeN=np.load('nucatt.npy'), Vee=np.load('twoint.npy'), results=results)
        CTMO = np.transpose(CMO)
        eps = np.dot(np.dot(CTMO, FAO),CMO)
        P = 2*D

        Q = np.zeros((len(CMO),len(CMO)))
        for v in range(0, len(CMO)):
            for u in range(0, len(CMO)):
                for a in range(0, int(input[0][0]/2)):
                    Q[v,u] += 2*eps[a,a]*CMO[u,a]*CMO[v,a]

        dX = np.zeros(len(input))
        dY = np.zeros(len(input))
        dZ = np.zeros(len(input))
        
        for j in range(1, len(input)):
            dxenuc = np.load(str(j)+'dxenuc.npy')
            dyenuc = np.load(str(j)+'dyenuc.npy')
            dzenuc = np.load(str(j)+'dzenuc.npy')
            
            dxEkin = np.load(str(j)+'dxEkin.npy')
            dyEkin = np.load(str(j)+'dyEkin.npy')
            dzEkin = np.load(str(j)+'dzEkin.npy')
            
            dxoverlap = np.load(str(j)+'dxoverlap.npy')
            dyoverlap = np.load(str(j)+'dyoverlap.npy')
            dzoverlap = np.load(str(j)+'dzoverlap.npy')
            
            dxnucatt = np.load(str(j)+'dxnucatt.npy')
            dynucatt = np.load(str(j)+'dynucatt.npy')
            dznucatt = np.load(str(j)+'dznucatt.npy')
            
            dxtwoint = np.load(str(j)+'dxtwoint.npy')
            dytwoint = np.load(str(j)+'dytwoint.npy')
            dztwoint = np.load(str(j)+'dztwoint.npy')
            
            dxHcore = 0
            dyHcore = 0
            dzHcore = 0
            
            dxERI = 0
            dyERI = 0
            dzERI = 0
            
            dxS = 0
            dyS = 0
            dzS = 0
            
            for u in range(0, len(P)):
                for v in range(0, len(P)):
                    dxHcore += P[v,u]*(dxEkin[u,v]+dxnucatt[u,v])
                    dyHcore += P[v,u]*(dyEkin[u,v]+dynucatt[u,v])
                    dzHcore += P[v,u]*(dzEkin[u,v]+dznucatt[u,v])
            
            for u in range(0, len(P)):
                for v in range(0, len(P)):
                    for l in range(0, len(P)):
                        for s in range(0, len(P)):
                            dxERI += 0.5*P[v,u]*P[l,s]*(dxtwoint[u,v,s,l]-0.5*dxtwoint[u,l,s,v])
                            dyERI += 0.5*P[v,u]*P[l,s]*(dytwoint[u,v,s,l]-0.5*dytwoint[u,l,s,v])
                            dzERI += 0.5*P[v,u]*P[l,s]*(dztwoint[u,v,s,l]-0.5*dztwoint[u,l,s,v])
            
            for u in range(0, len(dxoverlap)):
                for v in range(0, len(dxoverlap)):
                    dxS += Q[v,u]*dxoverlap[u,v]
                    dyS += Q[v,u]*dyoverlap[u,v]
                    dzS += Q[v,u]*dzoverlap[u,v]
            
            
            dX[j] = dxHcore + dxERI - dxS + dxenuc[0]
            dY[j] = dyHcore + dyERI - dyS + dyenuc[0]
            dZ[j] = dzHcore + dzERI - dzS + dzenuc[0]
        
        
        for j in range(1, len(dX)):
            input[j,1] = input[j,1] - stepsize*dX[j]
            input[j,2] = input[j,2] - stepsize*dY[j]
            input[j,3] = input[j,3] - stepsize*dZ[j]
        
        output = open('out.txt', 'a')
        for j in range(1, len(input)):
            for k in range(0, 4):
                output.write("{: 12.8e}".format(input[j,k]))
                output.write("\t \t")
            output.write('\n')
        output.write('\n \n')
        output.close()
        
        if np.max(np.abs(dX)) < 10**-GeoOptol and np.max(np.abs(dY)) < 10**-GeoOptol and np.max(np.abs(dZ)) < 10**-GeoOptol:
            break
            
    return input


def run_numeric(input, set, results):
    maxstep = int(set['Max iteration GeoOpt'])
    GeoOptol = float(set['Geometry Tolerance'])
    stepsize = float(set['Gradient Decent Step'])
    for i in range(1, maxstep):
        dX = np.zeros(len(input))
        dY = np.zeros(len(input))
        dZ = np.zeros(len(input))
    
        for j in range(1, len(input)):
            input[j,1] += 10**-6
            basis = BS.bassiset(input, set)
            MI.runIntegrals(input, basis, settings)
            input[j,1] -= 10**-6
            CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN=np.load('enuc.npy'), Te=np.load('Ekin.npy'), S=np.load('overlap.npy'), VeN=np.load('nucatt.npy'), Vee=np.load('twoint.npy'), results=results, print_SCF='No')
            xplus = results['HFenergy']
            input[j,1] -= 10**-6
            basis = BS.bassiset(input, set)
            MI.runIntegrals(input, basis, settings)
            input[j,1] += 10**-6
            CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN=np.load('enuc.npy'), Te=np.load('Ekin.npy'), S=np.load('overlap.npy'), VeN=np.load('nucatt.npy'), Vee=np.load('twoint.npy'), results=results, print_SCF='No')
            xminus = results['HFenergy']
            
            input[j,2] += 10**-6
            basis = BS.bassiset(input, set)
            MI.runIntegrals(input, basis, settings)
            input[j,2] -= 10**-6
            CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN=np.load('enuc.npy'), Te=np.load('Ekin.npy'), S=np.load('overlap.npy'), VeN=np.load('nucatt.npy'), Vee=np.load('twoint.npy'), results=results, print_SCF='No')
            yplus = results['HFenergy']
            input[j,2] -= 10**-6
            basis = BS.bassiset(input, set)
            MI.runIntegrals(input, basis, settings)
            input[j,2] += 10**-6
            CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN=np.load('enuc.npy'), Te=np.load('Ekin.npy'), S=np.load('overlap.npy'), VeN=np.load('nucatt.npy'), Vee=np.load('twoint.npy'), results=results, print_SCF='No')
            yminus = results['HFenergy']
            
            input[j,3] += 10**-6
            basis = BS.bassiset(input, set)
            MI.runIntegrals(input, basis, settings)
            input[j,3] -= 10**-6
            CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN=np.load('enuc.npy'), Te=np.load('Ekin.npy'), S=np.load('overlap.npy'), VeN=np.load('nucatt.npy'), Vee=np.load('twoint.npy'), results=results, print_SCF='No')
            zplus = results['HFenergy']
            input[j,3] -= 10**-6
            basis = BS.bassiset(input, set)
            MI.runIntegrals(input, basis, settings)
            input[j,3] += 10**-6
            CMO, FAO, D, results = HF.HartreeFock(input, set, basis, VNN=np.load('enuc.npy'), Te=np.load('Ekin.npy'), S=np.load('overlap.npy'), VeN=np.load('nucatt.npy'), Vee=np.load('twoint.npy'), results=results, print_SCF='No')
            zminus = results['HFenergy']
            
            dX[j] = (xplus-xminus)/(2*10**-6)
            dY[j] = (yplus-yminus)/(2*10**-6)
            dZ[j] = (zplus-zminus)/(2*10**-6)
        
        for j in range(1, len(dX)):
            input[j,1] = input[j,1] - stepsize*dX[j]
            input[j,2] = input[j,2] - stepsize*dY[j]
            input[j,3] = input[j,3] - stepsize*dZ[j]
        
        output = open('out.txt', 'a')
        for j in range(1, len(dX)):
                output.write("{: 12.8e}".format(dX[j]))
                output.write("\t \t")
                output.write("{: 12.8e}".format(dY[j]))
                output.write("\t \t")
                output.write("{: 12.8e}".format(dZ[j]))
                output.write('\n')
        output.write('\n \n')
                
        for j in range(1, len(input)):
            for k in range(0, 4):
                output.write("{: 12.8e}".format(input[j,k]))
                output.write("\t \t")
            output.write('\n')
        output.write('\n \n')
        output.close()
        
        if np.max(np.abs(dX)) < 10**-GeoOptol and np.max(np.abs(dY)) < 10**-GeoOptol and np.max(np.abs(dZ)) < 10**-GeoOptol:
            break
        
    return input
    
        
def runGO(input, set, results):
    if set['Force Numeric'] == 'Yes':
        input = run_numeric(input, set, results, settings)
    else:
        input = run_analytic(input, set, results, settings)
    
    return input, results
