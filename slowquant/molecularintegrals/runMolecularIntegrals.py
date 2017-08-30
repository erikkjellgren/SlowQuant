import numpy as np
import copy
import time
from slowquant.molecularintegrals.runMIcython import runCythonIntegrals, runE, runCythonRunGeoDev, runQMESPcython
from slowquant.molecularintegrals.MIpython import nucrep, nucdiff, Nrun, Ndiff1, Ndiff2, u_ObaraSaika

##CALC OF INTEGRALS
def runIntegrals(input, basis, settings, results):
    # Nuclear-nuclear repulsion
    VNN = np.zeros(1)
    VNN[0] = nucrep(input)
    
    # Reform basisset information
    
    # basisidx [number of primitives, start index in basisfloat and basisint]
    # basisfloat array of float values for basis, N, zeta, c, x, y, z 
    # basisint array of integer values for basis, l, m, n
    basisidx   = np.zeros((len(basis),2))
    startidx = 0
    for i in range(0, len(basisidx)):
        basisidx[i,0] = basis[i][4]
        basisidx[i,1] = startidx
        startidx     += basisidx[i,0]
    basisidx = basisidx.astype(np.int32)
    
    basisfloat = np.zeros((np.sum(basisidx[:,0]),6))
    basisint   = np.zeros((np.sum(basisidx[:,0]),3))
    
    idxfi = 0
    for i in range(0, len(basisidx)):
        for j in range(0, basis[i][4]):
            basisfloat[idxfi,0] = basis[i][5][j][0]
            basisfloat[idxfi,1] = basis[i][5][j][1]
            basisfloat[idxfi,2] = basis[i][5][j][2]
            basisfloat[idxfi,3] = basis[i][1]
            basisfloat[idxfi,4] = basis[i][2]
            basisfloat[idxfi,5] = basis[i][3]
            
            basisint[idxfi,0]  = basis[i][5][j][3]
            basisint[idxfi,1]  = basis[i][5][j][4]
            basisint[idxfi,2]  = basis[i][5][j][5]
            
            idxfi += 1
    
    basisint = basisint.astype(np.int32)
    
    Na = np.zeros((len(basisidx),len(basisidx)))
    S = np.zeros((len(basisidx),len(basisidx)))
    T = np.zeros((len(basisidx),len(basisidx)))
    ERI = np.zeros((len(basisidx),len(basisidx),len(basisidx),len(basisidx)))
    # Making array to save E
    E1arr = np.zeros((len(basisidx),len(basisidx),np.max(basisidx[:,0]),np.max(basisidx[:,0]),np.max(basisint[:,0:3])*2+1))
    E2arr = np.zeros((len(basisidx),len(basisidx),np.max(basisidx[:,0]),np.max(basisidx[:,0]),np.max(basisint[:,0:3])*2+1))
    E3arr = np.zeros((len(basisidx),len(basisidx),np.max(basisidx[:,0]),np.max(basisidx[:,0]),np.max(basisint[:,0:3])*2+1))
    # Array to store R values, only created once if created here
    R1buffer = np.zeros((4*np.max(basisint)+1,4*np.max(basisint)+1,4*np.max(basisint)+1))
    Rbuffer = np.zeros((4*np.max(basisint)+1,4*np.max(basisint)+1,4*np.max(basisint)+1,3*4*np.max(basisint)+1))
    
    Na, S, T, ERI = runCythonIntegrals(basisidx, basisfloat, basisint, input, Na, S, T, ERI, E1arr, E2arr, E3arr, R1buffer, Rbuffer)

    results['VNN'] = VNN
    results['VNe'] = np.array(Na)
    results['S'] = np.array(S)
    results['Te'] = np.array(T)
    results['Vee'] = np.array(ERI)
    return results
    #END OF two electron integrals


def rungeometric_derivatives(input, basis, settings, results, print_time='Yes'):    
    # Reform basisset information
    
    # basisidx [number of primitives, start index in basisfloat and basisint]
    # basisfloat array of float values for basis, N, zeta, c, x, y, z, Nxp, Nxm, Nyp, Nym, Nzp, Nzm 
    # basisint array of integer values for basis, l, m, n, atomidx
    basisidx   = np.zeros((len(basis),2))
    startidx = 0
    for i in range(0, len(basisidx)):
        basisidx[i,0] = basis[i][4]
        basisidx[i,1] = startidx
        startidx     += basisidx[i,0]
    basisidx = basisidx.astype(np.int32)
    
    basisfloat = np.zeros((np.sum(basisidx[:,0]),12))
    basisint   = np.zeros((np.sum(basisidx[:,0]),4))
    
    # Bad solution but not time limiting, cannot be bothered to fix
    # Calculate derivative normalization
    Nxplus  = copy.deepcopy(basis)
    Nxminus = copy.deepcopy(basis)
    Nyplus  = copy.deepcopy(basis)
    Nyminus = copy.deepcopy(basis)
    Nzplus  = copy.deepcopy(basis)
    Nzminus = copy.deepcopy(basis)
    for i in range(len(basis)):
        for j in range(len(basis[i][5])):
            Nxplus[i][5][j][3] += 1
            if Nxminus[i][5][j][3] != 0:
                Nxminus[i][5][j][3] -= 1
            
            Nyplus[i][5][j][4] += 1
            if Nyminus[i][5][j][4] != 0:
                Nyminus[i][5][j][4] -= 1
            
            Nzplus[i][5][j][5] += 1    
            if Nzminus[i][5][j][5] != 0:
                Nzminus[i][5][j][5] -= 1
                
    Nxplus = Nrun(Nxplus)
    Nxminus = Nrun(Nxminus)
    Nyplus = Nrun(Nyplus)
    Nyminus = Nrun(Nyminus)
    Nzplus = Nrun(Nzplus)
    Nzminus = Nrun(Nzminus)
    
    # Fix basisset information to go into Cython run function
    # basisfloat array of float values for basis, N, zeta, c, x, y, z, Nxp, Nxm, Nyp, Nym, Nzp, Nzm 
    # basisint array of integer values for basis, l, m, n, atomidx
    idxfi = 0
    for i in range(0, len(basisidx)):
        for j in range(0, basis[i][4]):
            basisint[idxfi,0]  = basis[i][5][j][3]
            basisint[idxfi,1]  = basis[i][5][j][4]
            basisint[idxfi,2]  = basis[i][5][j][5]
            basisint[idxfi,3]  = basis[i][6]
            
            basisfloat[idxfi,0]  = basis[i][5][j][0]
            basisfloat[idxfi,1]  = basis[i][5][j][1]
            basisfloat[idxfi,2]  = basis[i][5][j][2]
            basisfloat[idxfi,3]  = basis[i][1]
            basisfloat[idxfi,4]  = basis[i][2]
            basisfloat[idxfi,5]  = basis[i][3]
            basisfloat[idxfi,6]  = Nxplus[i][5][j][0]*Ndiff1(basisint[idxfi,0], basisfloat[idxfi,1])
            if basisint[idxfi,0] != 0:
                basisfloat[idxfi,7]  = Nxminus[i][5][j][0]*Ndiff2(basisint[idxfi,0], basisfloat[idxfi,1])
            basisfloat[idxfi,8]  = Nyplus[i][5][j][0]*Ndiff1(basisint[idxfi,1], basisfloat[idxfi,1])
            if basisint[idxfi,1] != 0:
                basisfloat[idxfi,9]  = Nyminus[i][5][j][0]*Ndiff2(basisint[idxfi,1], basisfloat[idxfi,1])
            basisfloat[idxfi,10] = Nzplus[i][5][j][0]*Ndiff1(basisint[idxfi,2], basisfloat[idxfi,1])
            if basisint[idxfi,2] != 0:
                basisfloat[idxfi,11] = Nzminus[i][5][j][0]*Ndiff2(basisint[idxfi,2], basisfloat[idxfi,1])
            
            idxfi += 1
    
    basisint = basisint.astype(np.int32)
    
    # Precalculat all E, they are the same for all atom derivatives
    Earr = runE(basisidx, basisfloat, basisint, input)
    # Array to store R values, only created once if created here
    R1buffer = np.zeros((4*np.max(basisint)+2,4*np.max(basisint)+2,4*np.max(basisint)+2))
    Rbuffer = np.zeros((4*np.max(basisint)+2,4*np.max(basisint)+2,4*np.max(basisint)+2,3*4*(np.max(basisint)+1)+1))
    
    for atomidx in range(1, len(input)):
        start = time.time()
        # Nuclear-Nuclear repulsion
        VNN = np.zeros(1)
        VNN[0] = nucdiff(input, atomidx, 1)
        results[str(atomidx)+'dxVNN'] = VNN
        
        VNN = np.zeros(1)
        VNN[0] = nucdiff(input, atomidx, 2)
        results[str(atomidx)+'dyVNN'] = VNN
        
        VNN = np.zeros(1)
        VNN[0] = nucdiff(input, atomidx, 3)
        results[str(atomidx)+'dzVNN'] = VNN
        
        # Integrals
        Sxarr= np.zeros((len(basisidx),len(basisidx)))
        Syarr= np.zeros((len(basisidx),len(basisidx)))
        Szarr= np.zeros((len(basisidx),len(basisidx)))
        Txarr = np.zeros((len(basisidx),len(basisidx)))
        Tyarr = np.zeros((len(basisidx),len(basisidx)))
        Tzarr = np.zeros((len(basisidx),len(basisidx)))
        VNexarr = np.zeros((len(basisidx),len(basisidx)))
        VNeyarr = np.zeros((len(basisidx),len(basisidx)))
        VNezarr = np.zeros((len(basisidx),len(basisidx)))
        ERIx = np.zeros((len(basisidx),len(basisidx),len(basisidx),len(basisidx)))
        ERIy = np.zeros((len(basisidx),len(basisidx),len(basisidx),len(basisidx)))
        ERIz = np.zeros((len(basisidx),len(basisidx),len(basisidx),len(basisidx)))
        
        Sxarr, Syarr, Szarr, Txarr, Tyarr, Tzarr, VNexarr, VNeyarr, VNezarr, ERIx, ERIy, ERIz = runCythonRunGeoDev(basisidx, basisfloat, basisint, input, Earr, Sxarr, Syarr, Szarr, Txarr, Tyarr, Tzarr, VNexarr, VNeyarr, VNezarr, ERIx, ERIy, ERIz, atomidx, R1buffer, Rbuffer)
        
        results[str(atomidx)+'dxS']   = np.array(Sxarr)
        results[str(atomidx)+'dyS']   = np.array(Syarr)
        results[str(atomidx)+'dzS']   = np.array(Szarr)
        results[str(atomidx)+'dxTe']  = np.array(Txarr)
        results[str(atomidx)+'dyTe']  = np.array(Tyarr)
        results[str(atomidx)+'dzTe']  = np.array(Tzarr)
        results[str(atomidx)+'dxVNe'] = np.array(VNexarr)
        results[str(atomidx)+'dyVNe'] = np.array(VNeyarr)
        results[str(atomidx)+'dzVNe'] = np.array(VNezarr)
        results[str(atomidx)+'dxVee'] = np.array(ERIx)
        results[str(atomidx)+'dyVee'] = np.array(ERIy)
        results[str(atomidx)+'dzVee'] = np.array(ERIz)
        if print_time == 'Yes':
            print(time.time()-start, 'Integral derivative: atom'+str(atomidx))
    
    return results

def run_dipole_int(basis, input, results):
    X = np.zeros((len(basis),len(basis)))
    Y = np.zeros((len(basis),len(basis)))
    Z = np.zeros((len(basis),len(basis)))
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                idx = np.zeros(2)
                idx[0] = k
                idx[1] = l
                idx = idx.astype(int)
                calcx = 0
                calcy = 0
                calcz = 0
                for i in range(basis[idx[0]][4]):
                    for j in range(basis[idx[1]][4]):
                        x, y, z = u_ObaraSaika(basis[idx[0]][5][i][1], basis[idx[1]][5][j][1], basis[idx[0]][1], basis[idx[0]][2], basis[idx[0]][3], basis[idx[1]][1], basis[idx[1]][2], basis[idx[1]][3], basis[idx[0]][5][i][3], basis[idx[1]][5][j][3], basis[idx[0]][5][i][4], basis[idx[1]][5][j][4],basis[idx[0]][5][i][5], basis[idx[1]][5][j][5], basis[idx[0]][5][i][0], basis[idx[1]][5][j][0], basis[idx[0]][5][i][2], basis[idx[1]][5][j][2], input)
                        calcx += x
                        calcy += y
                        calcz += z
                X[k,l] = calcx
                X[l,k] = calcx
                Y[k,l] = calcy
                Y[l,k] = calcy
                Z[k,l] = calcz
                Z[l,k] = calcz
    
    results['mu_x'] = X
    results['mu_y'] = Y
    results['mu_z'] = Z
    return results

def runQMESP(basis, input, rcx, rcy ,rcz):
    # Reform basisset information
    # basisidx [number of primitives, start index in basisfloat and basisint]
    # basisfloat array of float values for basis, N, zeta, c, x, y, z 
    # basisint array of integer values for basis, l, m, n
    basisidx   = np.zeros((len(basis),2))
    startidx = 0
    for i in range(0, len(basisidx)):
        basisidx[i,0] = basis[i][4]
        basisidx[i,1] = startidx
        startidx     += basisidx[i,0]
    basisidx = basisidx.astype(np.int32)
    
    basisfloat = np.zeros((np.sum(basisidx[:,0]),6))
    basisint   = np.zeros((np.sum(basisidx[:,0]),3))
    
    idxfi = 0
    for i in range(0, len(basisidx)):
        for j in range(0, basis[i][4]):
            basisfloat[idxfi,0] = basis[i][5][j][0]
            basisfloat[idxfi,1] = basis[i][5][j][1]
            basisfloat[idxfi,2] = basis[i][5][j][2]
            basisfloat[idxfi,3] = basis[i][1]
            basisfloat[idxfi,4] = basis[i][2]
            basisfloat[idxfi,5] = basis[i][3]
            
            basisint[idxfi,0]  = basis[i][5][j][3]
            basisint[idxfi,1]  = basis[i][5][j][4]
            basisint[idxfi,2]  = basis[i][5][j][5]
            
            idxfi += 1
    
    basisint = basisint.astype(np.int32)
    # Array to store R values, only created once if created here
    R1buffer = np.zeros((2*np.max(basisint)+1,2*np.max(basisint)+1,2*np.max(basisint)+1))
    Rbuffer = np.zeros((2*np.max(basisint)+1,2*np.max(basisint)+1,2*np.max(basisint)+1,3*2*np.max(basisint)+1))
    
    Ve = np.zeros((len(basis),len(basis)))
    Zc = -1.0
    
    Ve = runQMESPcython(basisidx, basisfloat, basisint, Ve, Zc, rcx, rcy ,rcz, R1buffer, Rbuffer)
        
    return np.array(Ve)
