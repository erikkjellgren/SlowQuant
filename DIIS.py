import numpy as np
from time import sleep

def DIIS(F,D,S,Efock,Edens,basis,numbF):
    
    Edens[1:] = Edens[:len(Edens)-1]
    Efock[1:] = Efock[:len(Efock)-1]
    Edens[0] = D
    Efock[0] = F

    #Check for not used errorMatrixes
    check = 0
    for i in range(0, numbF):
        if np.sum(Edens[i]) == 0:
            check+=1
    
    Err = np.zeros((numbF-check,len(basis),len(basis)))
    for i in range(0, numbF-check):
        Err[i] =  np.dot(np.dot(Efock[i],Edens[i]),S)
        Err[i] -= np.dot(np.dot(S, Edens[i]),Efock[i])
    Emax = np.max(np.abs(Err))
    
    #Construct B and b0
    B = -1*np.ones((numbF+1-check,numbF+1-check))
    B[numbF-check,numbF-check] = 0
    b0 = np.zeros(numbF+1-check)
    b0[numbF-check] = -1
    
    for i in range(0, numbF-check):
        for j in range(0, numbF-check):
            B[i,j] = np.trace(np.dot(Err[i], Err[j]))
    
    c = np.linalg.solve(B, b0)
    
    Fprime = np.zeros((len(basis),len(basis)))
    
    for i in range(0, numbF-check):
        Fprime += c[i]*Efock[i]

    return Fprime, Efock, Edens, Emax
    

def runDIIS(Fnew,Dnew,Snew,iter,set,basis,errF,errD):
    Steps = int(set['Keep Steps'])
    
    if iter == 1:
        Efock = np.zeros((Steps, len(basis), len(basis)))
        Edens = np.zeros((Steps, len(basis), len(basis)))
        F = Fnew
        Emax = 'None'
    else:
        F, Efock, Edens, Emax = DIIS(Fnew,Dnew,Snew,errF,errD,basis,Steps)
    
    return F, Efock, Edens, Emax
    