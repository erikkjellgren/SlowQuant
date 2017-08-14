import random as rng
import numpy as np
import time
from slowquant import BasisSet as BS
from slowquant import runMolecularIntegrals as MI
from slowquant import HartreeFock as HF
from slowquant import Force as F

def Get_Forces(results, input, set):
    dX, dY, dZ, results = F.Force(input, set, results, print_scf='No')
    input[:,8]  = -1.0*dX
    input[:,9]  = -1.0*dY
    input[:,10] = -1.0*dZ
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
    input, results = Get_Forces(results, input, set)
    
    # Update velocity
    for i in range(1, len(input)):
        input[i,5] = input[i,5] + 0.5*(Forces_old[i,8]+input[i,8])*dt/input[i,4]
        input[i,6] = input[i,6] + 0.5*(Forces_old[i,9]+input[i,9])*dt/input[i,4]
        input[i,7] = input[i,7] + 0.5*(Forces_old[i,10]+input[i,10])*dt/input[i,4]

    return input, results


def BOMD(input, set, results):
    
    Temperature = float(set['Temperature'])/(3.1577464*10**5) # Convert from K til a.u.
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

    
    