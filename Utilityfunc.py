import numpy as np

def TransformMO(C, basis, set, Vee):
    #Check for key that require MO integrals
    MOcheck = 0
    if set['MPn'] == 'MP2':
        MOcheck = 1
    if MOcheck == 1:
        #Loading two electron integrals
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

        np.save('twointMO.npy', VeeMO)

