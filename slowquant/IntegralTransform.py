import numpy as np

def TransformMO(C, basis, set):
    Vee=np.load('slowquant/temp/twoint.npy')
    VeeMO = np.zeros((len(basis),len(basis),len(basis),len(basis)))

    MO1 = np.zeros((len(basis),len(basis),len(basis),len(basis)))
    MO2 = np.zeros((len(basis),len(basis),len(basis),len(basis)))
    MO3 = np.zeros((len(basis),len(basis),len(basis),len(basis)))
    
    for s in range(0, len(basis)):
        for sig in range(0, len(basis)):
            MO1[:,:,:,s] += C[sig,s]*Vee[:,:,:,sig]
        
        for r in range(0, len(basis)):
            for lam in range(0, len(basis)):
                MO2[:,:,r,s] += C[lam,r]*MO1[:,:,lam,s]
        
            for q in range(0, len(basis)):
                for nu in range(0, len(basis)):
                    MO3[:,q,r,s] += C[nu,q]*MO2[:,nu,r,s]
                
                for p in range(0, len(basis)):
                    for mu in range(0, len(basis)):
                        VeeMO[p,q,r,s] += C[mu,p]*MO3[mu,q,r,s]

    np.save('slowquant/temp/twointMO.npy', VeeMO)

def Transform2eSPIN():
    VeeMO=np.load('slowquant/temp/twointMO.npy')
    VeeMOspin = np.zeros((len(VeeMO)*2,len(VeeMO)*2,len(VeeMO)*2,len(VeeMO)*2))
    for p in range(1,len(VeeMO)*2+1):
        for r in range(1,len(VeeMO)*2+1):
            if (p%2 == r%2):
                for q in range(1,len(VeeMO)*2+1):
                    for s in range(1,len(VeeMO)*2+1):
                        if (q%2 == s%2):
                            VeeMOspin[p-1,q-1,r-1,s-1] = VeeMO[(p+1)//2-1,(r+1)//2-1,(q+1)//2-1,(s+1)//2-1]
        
    np.save('slowquant/temp/twointMOspin.npy', VeeMOspin)

def runTransform(CMO, basis, set, FAO):
    if set['MPn'] == 'MP2' or set['CI'] == 'CIS' or set['Excitation'] == 'RPA' or set['MPn'] == 'MP3' or set['MPn'] == 'DCPT2':
        TransformMO(CMO, basis, set)
    if set['CI'] == 'CIS' or set['Excitation'] == 'RPA' or set['MPn'] == 'MP3':
        Transform2eSPIN()
    