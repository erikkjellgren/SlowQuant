import numpy as np
from time import sleep

def DIIS(F,D,S,Efock,Edens,numbF):
    
    Edens[1:] = Edens[:len(Edens)-1]
    Efock[1:] = Efock[:len(Efock)-1]
    Edens[0] = D
    Efock[0] = F
    
    Err = np.zeros((numbF,len(F),len(F)))
    for i in range(0, numbF):
        Err[i] =  np.dot(np.dot(Efock[i],Edens[i]),S)
        Err[i] -= np.dot(np.dot(S, Edens[i]),Efock[i])
    Emax = np.max(np.abs(Err))
    
    #Construct B and b0
    B = -1*np.ones((numbF+1,numbF+1))
    B[numbF,numbF] = 0
    b0 = np.zeros(numbF+1)
    b0[numbF] = -1
    
    for i in range(0, numbF):
        for j in range(0, numbF):
            B[i,j] = np.trace(np.dot(Err[i], Err[j]))
    
    c = np.linalg.solve(B, b0)

    Fprime = np.zeros((len(F),len(F)))
    
    for i in range(0, numbF):
        Fprime += c[i]*Efock[i]
    
    return Fprime, Efock, Edens, Emax
    

def runDIIS(Fnew,Dnew,Snew,iter,Steps,errF,errD):
    
    if iter == 1:
        Efock = np.zeros((Steps, len(Fnew), len(Fnew)))
        Edens = np.zeros((Steps, len(Fnew), len(Fnew)))
        Efock[0] = Fnew
        Edens[0] = Dnew
        F = Fnew
        Emax = 'None'
    else:
        if iter < Steps:
            F, Efock, Edens, Emax = DIIS(Fnew,Dnew,Snew,errF,errD,iter)
        else:
            F, Efock, Edens, Emax = DIIS(Fnew,Dnew,Snew,errF,errD,Steps)
    
    return F, Efock, Edens, Emax
    