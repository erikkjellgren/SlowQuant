import numpy as np
from slowquant.coupledcluster.CythonCC import runCythonCCSD, runCythonPerturbativeT

def doublebar(i,j,k,l, Vee):
    return Vee[i,j,k,l] - Vee[i,j,l,k]
    
def tautilde(i,j,a,b,tdouble,tsingle):
    return tdouble[i,j,a,b] + 0.5*(tsingle[i,a]*tsingle[j,b]-tsingle[i,b]*tsingle[j,a])

def tau(i,j,a,b,tdouble,tsingle):
    return tdouble[i,j,a,b] + tsingle[i,a]*tsingle[j,b]-tsingle[i,b]*tsingle[j,a]


def CCSD(occ, F, C, VeeMOspin, maxiter, deTHR, rmsTHR, runCCSDT=0):           
    # Make the spin MO fock matrix
    Fspin = np.zeros((len(F)*2,len(F)*2))
    Cspin = np.zeros((len(F)*2,len(F)*2))
    for p in range(1,len(F)*2+1):
        for q in range(1,len(F)*2+1):
            Fspin[p-1,q-1] = F[(p+1)//2-1,(q+1)//2-1] * (p%2 == q%2)
            Cspin[p-1,q-1] = C[(p+1)//2-1,(q+1)//2-1] * (p%2 == q%2)
    FMOspin = np.dot(np.transpose(Cspin),np.dot(Fspin,Cspin))

    # Make inital T1 and T2 guess
    dimension = len(VeeMOspin)
    tsingle = np.zeros((dimension,dimension))
    tdouble = np.zeros((dimension,dimension,dimension,dimension))
    EMP2 = 0
    for i in range(0, occ):
        for j in range(0, occ):
            for a in range(occ, dimension):
                for b in range(occ, dimension):
                    Dijab = FMOspin[i,i]+FMOspin[j,j]-FMOspin[a,a]-FMOspin[b,b]
                    tdouble[i,j,a,b] = doublebar(i,j,a,b,VeeMOspin)/Dijab
                    EMP2 += 0.25*doublebar(i,j,a,b,VeeMOspin)*tdouble[i,j,a,b]
    
    output = open('out.txt', 'a')
    output.write('\nMP2 Energy \t')
    output.write("{: 10.8f}".format(EMP2))
    
    output.write('\n \n')
    output.write('Iter')
    output.write("\t")
    output.write(' ECCSD')
    output.write("\t \t \t \t")
    output.write('dE')
    output.write("\t \t \t \t \t")
    output.write(' rmsDsingle')
    output.write("\t \t \t")
    output.write('rmsDdouble')
    output.write('\n')
    
    E0CCSD = 0.0
    for iter in range(1, maxiter):
        ECCSD, tsingle, tdouble, tsingle_old, tdouble_old = runCythonCCSD(FMOspin, VeeMOspin, tsingle, tdouble, dimension, occ)
        
        rmsDsingle = 0
        rmsDsingle = np.sum((np.array(tsingle) - np.array(tsingle_old))**2)
        rmsDsingle = (rmsDsingle)**0.5
        
        rmsDdouble = 0
        rmsDdouble = np.sum((np.array(tdouble) - np.array(tdouble_old))**2)
        rmsDdouble = (rmsDdouble)**0.5
        
        dE = ECCSD - E0CCSD
        E0CCSD = ECCSD
        
        output.write(str(iter))
        output.write("\t \t")
        output.write("{:14.10f}".format(ECCSD))
        output.write("\t \t")
        output.write("{:12.8e}".format(dE))
        output.write("\t \t")
        output.write("{: 12.8e}".format(rmsDsingle))
        output.write("\t \t")
        output.write("{: 12.8e}".format(rmsDdouble))
        output.write('\n')
        
        if np.abs(dE) < deTHR and rmsDsingle < rmsTHR and rmsDdouble -rmsTHR:
            break
    
    
    output.write('\n \n')
    if runCCSDT == 1:
        ET = runCythonPerturbativeT(FMOspin, VeeMOspin, tsingle, tdouble, dimension, occ)
        
        output.write('E(T) \t')
        output.write("{:14.10f}".format(ET))
        
        output.close()
        return EMP2, ECCSD, ET
        
    else:
        output.close()
        return EMP2, ECCSD