import numpy as np
import time
from slowquant.bomd.BOMD import VelocityVerlet

def runBOMD(input, set, results):
    Temperature = float(set['Temperature'])/(3.1577464*10**5) # Convert from K til a.u.
    steps       = int(set['steps'])
    stepsize    = float(set['stepsize'])
    
    # Converters
    AtomNrtoName =  {1: 'H',2: 'He',3: 'Li',4: 'Be',5: 'B',6: 'C',7: 'N',8: 'O',9: 'F',10: 'Ne',11: 'Na',12: 'Mg',13: 'Al',14: 'Si',15: 'P',16: 'S',17: 'Cl',18: 'Ar',19: 'K',20: 'Ca',21: 'Sc',22: 'Ti',23: 'V',24: 'Cr',25: 'Mn',26: 'Fe',27: 'Co',28: 'Ni',29: 'Cu',30: 'Zn',31: 'Ga',32: 'Ge',33: 'As',34: 'Se',35: 'Br',36: 'Kr',37: 'Rb',38: 'Sr',39: 'Y',40: 'Zr',41: 'Nb',42: 'Mo',43: 'Tc',44: 'Ru',45: 'Rh',46: 'Pd',47: 'Ag',48: 'Cd',49: 'In',50: 'Sn',51: 'Sb',52: 'Te',53: 'I',54: 'Xe',55: 'Cs',56: 'Ba',57: 'La',58: 'Ce',59: 'Pr',60: 'Nd',61: 'Pm',62: 'Sm',63: 'Eu',64: 'Gd',65: 'Tb',66: 'Dy',67: 'Ho',68: 'Er',69: 'Tm',70: 'Yb',71: 'Lu',72: 'Hf',73: 'Ta',74: 'W',75: 'Re',76: 'Os',77: 'Ir',78: 'Pt',79: 'Au',80: 'Hg',81: 'Tl',82: 'Pb',83: 'Bi',84: 'Po',85: 'At',86: 'Rn',87: 'Fr',88: 'Ra',89: 'Ac',90: 'Th',91: 'Pa',92: 'U',93: 'Np',94: 'Pu',95: 'Am',96: 'Cm',97: 'Bk',98: 'Cf',99: 'Es',100: 'Fm',101: 'Md',102: 'No',103: 'Lr',104: 'Rf',105: 'Db',106: 'Sg',107: 'Bh',108: 'Hs',109: 'Mt'}
    
    # Reform input for BOMD
    # inputBOMD
    # [atom, x, y, z, mass, vx, vy, vz, fx, fy, fz]
    inputBOMD = np.zeros((len(input),11))
    inputBOMD[:,0:4] = input
    for i in range(1,len(inputBOMD)):
        if inputBOMD[i,0] == 1:
            inputBOMD[i,4] = 1.00794
        elif inputBOMD[i,0] == 8:
            inputBOMD[i,4] = 15.9994
    
    # Run simulation
    output = open('out.txt', 'a')
    output.write('Iter')
    output.write("\t")
    output.write('EHF')
    output.write("\t \t \t \t \t")
    output.write('EKIN')
    output.write("\t \t \t \t")
    output.write('calc time')
    output.write('\n')
    
    # Inital placement in trajectory
    traj = open('traj.xyz', 'w')
    traj.write(str(len(inputBOMD)-1))
    traj.write('\n')
    traj.write('\n')
    for i in range(1, len(inputBOMD)):
        atom = AtomNrtoName[inputBOMD[i,0]]
        traj.write(atom+' '+str(inputBOMD[i,1])+' '+str(inputBOMD[i,2])+' '+str(inputBOMD[i,3]))
        traj.write('\n')
        
    for step in range(1, steps+1):
   
        steptime = time.time()
        inputBOMD, results = VelocityVerlet(inputBOMD, stepsize, results, set)
        
        # Write results from previous step
        Ekin = 0.0
        for i in range(1, len(inputBOMD)):
            Ekin += 0.5*inputBOMD[i,4]*(inputBOMD[i,5]**2+inputBOMD[i,6]**2+inputBOMD[i,7]**2)
        
        print(step, time.time()-steptime)
        output.write(str(step))
        output.write("\t \t")
        output.write("{:14.10f}".format(results['HFenergy']))
        output.write("\t \t")
        output.write("{:14.10f}".format(Ekin))
        output.write("\t \t \t \t")
        output.write("{: 12.8e}".format(time.time()-steptime))
        output.write('\n')
        
        # Write trajectory
        traj.write(str(len(inputBOMD)-1))
        traj.write('\n')
        traj.write('\n')
        for i in range(1, len(inputBOMD)):
            atom = AtomNrtoName[inputBOMD[i,0]]
            traj.write(atom+' '+str(inputBOMD[i,1])+' '+str(inputBOMD[i,2])+' '+str(inputBOMD[i,3]))
            traj.write('\n')
        
    
    output.close()
    traj.close()
    return results
    
    