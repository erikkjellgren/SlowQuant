import numpy as np
import time
from slowquant import Force as F

def run_analytic(input, set, results):
    maxstep = int(set['Max iteration GeoOpt'])+1
    GeoOptol = float(set['Geometry Tolerance'])
    stepsize = float(set['Gradient Descent Step'])
    
    for i in range(1, maxstep):
        dX, dY, dZ, results = F.Force(input, set, results)
        
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


def run_numeric(input, set, results):
    maxstep = int(set['Max iteration GeoOpt'])+1
    GeoOptol = float(set['Geometry Tolerance'])
    stepsize = float(set['Gradient Descent Step'])
    for i in range(1, maxstep):
        start = time.time()
        dX = np.zeros(len(input))
        dY = np.zeros(len(input))
        dZ = np.zeros(len(input))
    
        for j in range(1, len(input)):
            input[j,1] += 10**-6
            basis = BS.bassiset(input, set)
            results = MI.runIntegrals(input, basis, set, results)
            input[j,1] -= 10**-6
            results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results, print_SCF='No')
            xplus = results['HFenergy']
            input[j,1] -= 10**-6
            basis = BS.bassiset(input, set)
            results = MI.runIntegrals(input, basis, set, results)
            input[j,1] += 10**-6
            results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results, print_SCF='No')
            xminus = results['HFenergy']
            
            input[j,2] += 10**-6
            basis = BS.bassiset(input, set)
            results = MI.runIntegrals(input, basis, set, results)
            input[j,2] -= 10**-6
            results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results, print_SCF='No')
            yplus = results['HFenergy']
            input[j,2] -= 10**-6
            basis = BS.bassiset(input, set)
            results = MI.runIntegrals(input, basis, set, results)
            input[j,2] += 10**-6
            results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results, print_SCF='No')
            yminus = results['HFenergy']
            
            input[j,3] += 10**-6
            basis = BS.bassiset(input, set)
            results = MI.runIntegrals(input, basis, set, results)
            input[j,3] -= 10**-6
            results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results, print_SCF='No')
            zplus = results['HFenergy']
            input[j,3] -= 10**-6
            basis = BS.bassiset(input, set)
            results = MI.runIntegrals(input, basis, set, results)
            input[j,3] += 10**-6
            results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results, print_SCF='No')
            zminus = results['HFenergy']
            
            dX[j] = (xplus-xminus)/(2*10**-6)
            dY[j] = (yplus-yminus)/(2*10**-6)
            dZ[j] = (zplus-zminus)/(2*10**-6)
        
        for j in range(1, len(dX)):
            input[j,1] = input[j,1] - stepsize*dX[j]
            input[j,2] = input[j,2] - stepsize*dY[j]
            input[j,3] = input[j,3] - stepsize*dZ[j]
            
        print(time.time()-start, 'GeoOpt step '+str(i))
        
        # Get energy of new structure
        basis = BS.bassiset(input, set)
        results = MI.runIntegrals(input, basis, set, results)
        results = HF.HartreeFock(input, set, basis, VNN=results['VNN'], Te=results['Te'], S=results['S'], VeN=results['VNe'], Vee=results['Vee'], results=results)
        
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
        input = run_numeric(input, set, results)
    else:
        input = run_analytic(input, set, results)
    
    return input, results
