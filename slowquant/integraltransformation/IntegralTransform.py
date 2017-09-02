import numpy as np

def Transform2eMO(C, Vee):
    VeeMO = np.zeros((len(Vee),len(Vee),len(Vee),len(Vee)))

    MO1 = np.zeros((len(Vee),len(Vee),len(Vee),len(Vee)))
    MO2 = np.zeros((len(Vee),len(Vee),len(Vee),len(Vee)))
    MO3 = np.zeros((len(Vee),len(Vee),len(Vee),len(Vee)))
    
    for s in range(0, len(C)):
        for sig in range(0, len(C)):
            MO1[:,:,:,s] += C[sig,s]*Vee[:,:,:,sig]
        
        for r in range(0, len(C)):
            for lam in range(0, len(C)):
                MO2[:,:,r,s] += C[lam,r]*MO1[:,:,lam,s]
        
            for q in range(0, len(C)):
                for nu in range(0, len(C)):
                    MO3[:,q,r,s] += C[nu,q]*MO2[:,nu,r,s]
                
                for p in range(0, len(C)):
                    for mu in range(0, len(C)):
                        VeeMO[p,q,r,s] += C[mu,p]*MO3[mu,q,r,s]
    return VeeMO


def Transform2eSPIN(VeeMO):
    VeeMOspin = np.zeros((len(VeeMO)*2,len(VeeMO)*2,len(VeeMO)*2,len(VeeMO)*2))
    for p in range(1,len(VeeMO)*2+1):
        for r in range(1,len(VeeMO)*2+1):
            if (p%2 == r%2):
                for q in range(1,len(VeeMO)*2+1):
                    for s in range(1,len(VeeMO)*2+1):
                        if (q%2 == s%2):
                            VeeMOspin[p-1,q-1,r-1,s-1] = VeeMO[(p+1)//2-1,(r+1)//2-1,(q+1)//2-1,(s+1)//2-1]
    return VeeMOspin


def Transform1eMO(S, C):
    SMO = np.zeros((len(C),len(C)))
    for i in range(len(C)):
        for j in range(len(C)):
            for p in range(len(C)):
                for q in range(len(C)):
                    SMO[i,j] += C[p,i]*C[q,j]*S[p,q]
    return SMO


def Transform1eSPIN(S):
    Sspin = np.zeros((len(S)*2,len(S)*2))
    for p in range(1,len(S)*2+1):
        for q in range(1,len(S)*2+1):
            if (p%2 == q%2):
                Sspin[p-1,q-1] = S[(p+1)//2-1,(q+1)//2-1]
    return Sspin
