import numpy as np
from slowquant import runMolecularIntegrals as MI
from slowquant import HartreeFock as HF
from slowquant import BasisSet as BS

def Force(input, set, results, print_time='No', print_scf='Yes'):
    basis = BS.bassiset(input, set)
    results = MI.runIntegrals(input, basis, set, results)
    results = MI.rungeometric_derivatives(input, basis, set, results, print_time=print_time)
    results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results, print_SCF=print_scf)
    D = results['D']
    CMO = results['C_MO']
    FAO = results['F']
    CTMO = np.transpose(CMO)
    eps = np.dot(np.dot(CTMO, FAO),CMO)
    P = 2*D
    occ = int(input[0][0]/2)

    Q = 2*np.einsum('aa,ua,va->vu',eps[0:occ,0:occ],CMO[:,0:occ],CMO[:,0:occ])

    dX = np.zeros(len(input))
    dY = np.zeros(len(input))
    dZ = np.zeros(len(input))
    
    for j in range(1, len(input)):
        dxenuc = results[str(j)+'dxVNN']
        dyenuc = results[str(j)+'dyVNN']
        dzenuc = results[str(j)+'dzVNN']
        
        dxEkin = results[str(j)+'dxTe']
        dyEkin = results[str(j)+'dyTe']
        dzEkin = results[str(j)+'dzTe']
        
        dxoverlap = results[str(j)+'dxS']
        dyoverlap = results[str(j)+'dyS']
        dzoverlap = results[str(j)+'dzS']
        
        dxnucatt = results[str(j)+'dxVNe']
        dynucatt = results[str(j)+'dyVNe']
        dznucatt = results[str(j)+'dzVNe']
        
        dxtwoint = results[str(j)+'dxVee']
        dytwoint = results[str(j)+'dyVee']
        dztwoint = results[str(j)+'dzVee']
        
        dxHcore = np.einsum('vu,uv->',P, dxEkin) + np.einsum('vu,uv->', P, dxnucatt)
        dyHcore = np.einsum('vu,uv->',P, dyEkin) + np.einsum('vu,uv->', P, dynucatt)
        dzHcore = np.einsum('vu,uv->',P, dzEkin) + np.einsum('vu,uv->', P, dznucatt)

        dxERI = 0.5*np.einsum('vu,ls,uvsl->',P,P,dxtwoint)
        dxERI += -0.25*np.einsum('vu,ls,ulsv->',P,P,dxtwoint)
        dyERI = 0.5*np.einsum('vu,ls,uvsl->',P,P,dytwoint)
        dyERI += -0.25*np.einsum('vu,ls,ulsv->',P,P,dytwoint)
        dzERI = 0.5*np.einsum('vu,ls,uvsl->',P,P,dztwoint)
        dzERI += -0.25*np.einsum('vu,ls,ulsv->',P,P,dztwoint)
        
        dxS = np.einsum('vu,uv->',Q,dxoverlap)
        dyS = np.einsum('vu,uv->',Q,dyoverlap)
        dzS = np.einsum('vu,uv->',Q,dzoverlap)
        
        dX[j] = dxHcore + dxERI - dxS + dxenuc[0]
        dY[j] = dyHcore + dyERI - dyS + dyenuc[0]
        dZ[j] = dzHcore + dzERI - dzS + dzenuc[0]
    
    return dX, dY, dZ, results