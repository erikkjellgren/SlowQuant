import numpy as np
import slowquant.molecularintegrals.runMolecularIntegrals as MI
import slowquant.hartreefock.HartreeFock as HF
import slowquant.derivatives.runForce as F

def Get_Forces(results, input, set):
    dX, dY, dZ, results = F.runForce(input, set, results, print_scf='No')
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