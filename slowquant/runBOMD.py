import random as rng
import numpy as np
import time
from slowquant import BasisSet as BS
from slowquant import runMolecularIntegrals as MI
from slowquant import HartreeFock as HF

def Forces(results, input, set):
    basis = BS.bassiset(input, set)
    results = MI.runIntegrals(input, basis, set, results)
    results = MI.rungeometric_derivatives(input, basis, set, results, print_time='No')
    results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results, print_SCF='No')
    D = results['D']
    CMO = results['C_MO']
    FAO = results['F']
    CTMO = np.transpose(CMO)
    eps = np.dot(np.dot(CTMO, FAO),CMO)
    P = 2*D

    Q = np.zeros((len(CMO),len(CMO)))
    for v in range(0, len(CMO)):
        for u in range(0, len(CMO)):
            for a in range(0, int(input[0][0]/2)):
                Q[v,u] += 2*eps[a,a]*CMO[u,a]*CMO[v,a]
    
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
        
        
        input[j,8]  = -1.0*(dxHcore + dxERI - dxS + dxenuc[0])
        input[j,9]  = -1.0*(dyHcore + dyERI - dyS + dyenuc[0])
        input[j,10] = -1.0*(dzHcore + dzERI - dzS + dzenuc[0])
    
    return input, results
    
def VelocityVerlet(input, dt, results, set):
    
    Forces_old = np.zeros((len(input),len(input[0])))
    # Update position
    for i in range(1, len(input)):
        input[i,1] = input[i,1] + input[i,5]*dt + 0.5*input[i,8]/input[i,4]*dt*dt
        input[i,2] = input[i,2] + input[i,6]*dt + 0.5*input[i,9]/input[i,4]*dt*dt
        input[i,3] = input[i,3] + input[i,7]*dt + 0.5*input[i,10]/input[i,4]*dt*dt
        
        # Save old forces
        Forces_old[i,8] = input[i,8]
        Forces_old[i,9] = input[i,9]
        Forces_old[i,10] = input[i,10]
        
    # Get forces
    input, results = Forces(results, input, set)
    
    # Update velocity
    for i in range(1, len(input)):
        input[i,5] = input[i,5] + 0.5*(Forces_old[i,8]+input[i,8])*dt/input[i,4]
        input[i,6] = input[i,6] + 0.5*(Forces_old[i,9]+input[i,9])*dt/input[i,4]
        input[i,7] = input[i,7] + 0.5*(Forces_old[i,10]+input[i,10])*dt/input[i,4]

    return input, results


def BOMD(input, set, results):
    
    Temperature = float(set['Temperature'])/(3.1577464*10**5)
    steps       = int(set['steps'])
    stepsize    = float(set['stepsize'])
    
    # Reform input for BOMD
    # inputBOMD
    # [atom, x, y, z, mass, vx, vy, vz, fx, fy, fz]
    inputBOMD = np.zeros((len(input),11))
    inputBOMD[:,0:4] = input
    for i in range(1,len(inputBOMD)):
        if inputBOMD[i,0] == 1:
            inputBOMD[i,4] = 1.00794
        elif inputBOMD[i,0] == 8:
            inputBOMD[i,4] = 15.9994
    
    # Run simulation
    output = open('out.txt', 'a')
    output.write('Iter')
    output.write("\t")
    output.write('EHF')
    output.write("\t \t \t \t \t")
    output.write('EKIN')
    output.write("\t \t \t \t")
    output.write('HF iterations')
    output.write("\t \t \t \t")
    output.write('calc time')
    output.write('\n')
    
    # Inital placement in trajectory
    coords = open('coords.xyz', 'w')
    coords.write(str(len(inputBOMD)-1))
    coords.write('\n')
    coords.write('\n')
    for i in range(1, len(inputBOMD)):
        if inputBOMD[i,0] == 1:
            atom = 'H'
        elif inputBOMD[i,0] == 8:
            atom = 'O'
        coords.write(atom+' '+str(inputBOMD[i,1])+' '+str(inputBOMD[i,2])+' '+str(inputBOMD[i,3]))
        coords.write('\n')
        
    for step in range(1, steps+1):
   
        steptime = time.time()
        inputBOMD, results = VelocityVerlet(inputBOMD, stepsize, results, set)
        
        # Write results from previous step
        Ekin = 0.0
        for i in range(1, len(inputBOMD)):
            Ekin += 0.5*inputBOMD[i,4]*(inputBOMD[i,5]**2+inputBOMD[i,6]**2+inputBOMD[i,7]**2)
        
        print(step, time.time()-steptime)
        output.write(str(step))
        output.write("\t \t")
        output.write("{:14.10f}".format(results['HFenergy']))
        output.write("\t \t")
        output.write("{:14.10f}".format(Ekin))
        output.write("\t \t")
        output.write(str(results['HF iterations']))
        output.write("\t \t \t \t")
        output.write("{: 12.8e}".format(time.time()-steptime))
        output.write('\n')
        
        # Write trajectory
        coords.write(str(len(inputBOMD)-1))
        coords.write('\n')
        coords.write('\n')
        for i in range(1, len(inputBOMD)):
            if inputBOMD[i,0] == 1:
                atom = 'H'
            elif inputBOMD[i,0] == 8:
                atom = 'O'
            coords.write(atom+' '+str(inputBOMD[i,1])+' '+str(inputBOMD[i,2])+' '+str(inputBOMD[i,3]))
            coords.write('\n')
        
    
    output.close()
    coords.close()
    return results

    
    