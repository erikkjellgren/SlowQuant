import slowquant.hartreefock.runHartreeFock as HF

def nForce(input, set, results, print_time='No', print_scf='Yes'):
    input[j,1] += 10**-6
    basis = BS.bassiset(input, set)
    results = MI.runIntegrals(input, basis, set, results)
    input[j,1] -= 10**-6
    results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
    xplus = results['HFenergy']
    input[j,1] -= 10**-6
    basis = BS.bassiset(input, set)
    results = MI.runIntegrals(input, basis, set, results)
    input[j,1] += 10**-6
    results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
    xminus = results['HFenergy']
    
    input[j,2] += 10**-6
    basis = BS.bassiset(input, set)
    results = MI.runIntegrals(input, basis, set, results)
    input[j,2] -= 10**-6
    results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
    yplus = results['HFenergy']
    input[j,2] -= 10**-6
    basis = BS.bassiset(input, set)
    results = MI.runIntegrals(input, basis, set, results)
    input[j,2] += 10**-6
    results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
    yminus = results['HFenergy']
    
    input[j,3] += 10**-6
    basis = BS.bassiset(input, set)
    results = MI.runIntegrals(input, basis, set, results)
    input[j,3] -= 10**-6
    results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
    zplus = results['HFenergy']
    input[j,3] -= 10**-6
    basis = BS.bassiset(input, set)
    results = MI.runIntegrals(input, basis, set, results)
    input[j,3] += 10**-6
    results = HF.runHartreeFock(input, set, results, print_SCF=print_scf)
    zminus = results['HFenergy']
    
    dX[j] = (xplus-xminus)/(2*10**-6)
    dY[j] = (yplus-yminus)/(2*10**-6)
    dZ[j] = (zplus-zminus)/(2*10**-6)
    
    return dX, dY, dZ