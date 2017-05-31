import numpy as np
import math
import scipy.misc as scm

def N(a, l, m, n):
    part1 = (2.0/math.pi)**(3.0/4.0)
    part2 = 2.0**(l+m+n) * a**((2.0*l+2.0*m+2.0*n+3.0)/(4.0))
    part3 = math.sqrt(scm.factorial2(int(2*l-1))*scm.factorial2(int(2*m-1))*scm.factorial2(int(2*n-1)))
    N = part1 * ((part2)/(part3))
    return N


def bassiset(input, set):
    basisname = set['basisset']
    basisload = np.genfromtxt('basissets/'+str(basisname)+'.csv', dtype=str, delimiter=';')

    basis_out = []
    idx = 1
    for i in range(1, len(input)):
        writecheck = 0
        firstcheck = 0
        typecheck = 0
        for j in range(len(basisload)):
            if writecheck == 1:
                if basisload[j,0] == 'S' or basisload[j,0] == 'P' or basisload[j,0] == 'D':
                    if firstcheck != 0:
                        if typecheck == 'S':
                            basis_func = np.array(basis_func, dtype=float)
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            idx += 1
                        elif typecheck == 'P':
                            basis_func = np.array(basis_func, dtype=float)
                            basis_func[:,3] = 1 
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,3] = 0
                            idx += 1
                            
                            basis_func[:,4] = 1 
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,4] = 0
                            idx += 1
                            
                            basis_func[:,5] = 1 
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,5] = 0
                            idx += 1
                        elif typecheck == 'D':
                            basis_func = np.array(basis_func, dtype=float)
                            basis_func[:,3] = 2 
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,3] = 0
                            idx += 1
                            
                            basis_func[:,4] = 2 
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,4] = 0
                            idx += 1
                            
                            basis_func[:,5] = 2
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,5] = 0
                            idx += 1
                            
                            basis_func[:,3] = 1
                            basis_func[:,4] = 1
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,3] = 0
                            basis_func[:,4] = 0
                            idx += 1
                            
                            basis_func[:,3] = 1
                            basis_func[:,5] = 1
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,3] = 0
                            basis_func[:,4] = 0
                            idx += 1
                            
                            basis_func[:,4] = 1
                            basis_func[:,5] = 1
                            basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                            basis_func[:,4] = 0
                            basis_func[:,5] = 0
                            idx += 1
                    basis_func = []
                    typecheck = basisload[j,0]
                    firstcheck = 1
                    basis_func.append([0,basisload[j,1],basisload[j,2],0,0,0])
                else:
                    basis_func.append([0,basisload[j,1],basisload[j,2],0,0,0])
    
            
            if basisload[j+1,0] == 'FOR' and writecheck == 1:
                writecheck = 0
                if typecheck == 'S':
                    basis_func = np.array(basis_func, dtype=float)
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    idx += 1
                elif typecheck == 'P':
                    basis_func = np.array(basis_func, dtype=float)
                    basis_func[:,3] = 1 
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,3] = 0
                    idx += 1
                    
                    basis_func[:,4] = 1 
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,4] = 0
                    idx += 1
                    
                    basis_func[:,5] = 1 
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,5] = 0
                    idx += 1
                elif typecheck == 'D':
                    basis_func = np.array(basis_func, dtype=float)
                    basis_func[:,3] = 2 
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,3] = 0
                    idx += 1
                    
                    basis_func[:,4] = 2 
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,4] = 0
                    idx += 1
                    
                    basis_func[:,5] = 2
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,5] = 0
                    idx += 1
                    
                    basis_func[:,3] = 1
                    basis_func[:,4] = 1
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,3] = 0
                    basis_func[:,4] = 0
                    idx += 1
                    
                    basis_func[:,3] = 1
                    basis_func[:,5] = 1
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,3] = 0
                    basis_func[:,4] = 0
                    idx += 1
                    
                    basis_func[:,4] = 1
                    basis_func[:,5] = 1
                    basis_out.append([idx,input[i,1],input[i,2],input[i,3],len(basis_func),np.ndarray.tolist(np.copy(basis_func)), i])
                    basis_func[:,4] = 0
                    basis_func[:,5] = 0
                    idx += 1
                break
                
            if basisload[j,1] == 'H' and input[i,0] == 1:
                writecheck = 1
            elif basisload[j,1] == 'Be' and input[i,0] == 2:
                writecheck = 1
            elif basisload[j,1] == 'Li' and input[i,0] == 3:
                writecheck = 1
            elif basisload[j,1] == 'B' and input[i,0] == 5:
                writecheck = 1
            elif basisload[j,1] == 'C' and input[i,0] == 6:
                writecheck = 1
            elif basisload[j,1] == 'N' and input[i,0] == 7:
                writecheck = 1
            elif basisload[j,1] == 'O' and input[i,0] == 8:
                writecheck = 1
            elif basisload[j,1] == 'F' and input[i,0] == 9:
                writecheck = 1
            elif basisload[j,1] == 'Ne' and input[i,0] == 10:
                writecheck = 1
            elif basisload[j,1] == 'Na' and input[i,0] == 11:
                writecheck = 1
            elif basisload[j,1] == 'Mg' and input[i,0] == 12:
                writecheck = 1
            elif basisload[j,1] == 'Al' and input[i,0] == 13:
                writecheck = 1
            elif basisload[j,1] == 'Si' and input[i,0] == 14:
                writecheck = 1
            elif basisload[j,1] == 'P' and input[i,0] == 15:
                writecheck = 1
            elif basisload[j,1] == 'S' and input[i,0] == 16:
                writecheck = 1
            elif basisload[j,1] == 'Cl' and input[i,0] == 17:
                writecheck = 1
            elif basisload[j,1] == 'Ar' and input[i,0] == 18:
                writecheck = 1
            elif basisload[j,1] == 'K' and input[i,0] == 19:
                writecheck = 1
            elif basisload[j,1] == 'Ca' and input[i,0] == 20:
                writecheck = 1

    for i in range(len(basis_out)):
        for j in range(len(basis_out[i][5])):
            basis_out[i][5][j][3] = int(basis_out[i][5][j][3])
            basis_out[i][5][j][4] = int(basis_out[i][5][j][4])
            basis_out[i][5][j][5] = int(basis_out[i][5][j][5])
            basis_out[i][5][j][0] = N(basis_out[i][5][j][1], basis_out[i][5][j][3], basis_out[i][5][j][4], basis_out[i][5][j][5])
    return basis_out
