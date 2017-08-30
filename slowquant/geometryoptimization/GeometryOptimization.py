import numpy as np
import time
import slowquant.derivatives.runForce as F
from slowquant.numerical.numForce import nForce

def GeoOpt(input, set, results):
    maxstep = int(set['Max iteration GeoOpt'])+1
    GeoOptol = float(set['Geometry Tolerance'])
    stepsize = float(set['Gradient Descent Step'])
    
    for i in range(1, maxstep):
        if set['Force Numeric'] == 'Yes':
            dX, dY, dZ = nForce(input, set, results)
        else:
            dX, dY, dZ, results = F.runForce(input, set, results)
        
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
        
        if np.max(np.abs(dX)) < GeoOptol and np.max(np.abs(dY)) < GeoOptol and np.max(np.abs(dZ)) < GeoOptol:
            break
            
    return input, results

         
def runGO(input, set, results):
    input, results = GeoOpt(input, set, results)
    return input, results

