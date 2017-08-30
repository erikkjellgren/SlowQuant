import numpy as np
from slowquant.hartreefock.HartreeFock import HartreeFock
from slowquant.hartreefock.UHF import UnrestrictedHartreeFock

def runHartreeFock(input, set, results, print_SCF='Yes'):
    VNN=results['VNN']
    Te=results['Te']
    S=results['S']
    VeN=results['VNe']
    Vee=results['Vee']
    if set['Initial method'] == 'HF' or set['Initial method'] == 'BOMD':
        dethr   = float(set['SCF Energy Threshold'])
        rmsthr  = float(set['SCF RMSD Threshold'])
        maxiter = int(set['SCF Max iterations'])
        diis_steps = int(set['Keep Steps'])
        do_diis = set['DIIS'] 
        EHF, C, F, D, iter = HartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=dethr, rmsTHR=dethr, Maxiter=maxiter, DO_DIIS=do_diis, DIIS_steps=diis_steps, print_SCF=print_SCF)
        results['HFenergy'] = EHF
        results['C_MO']     = C
        results['F']        = F
        results['D']        = D
        results['HF iterations'] = iter
    elif set['Initial method'] == 'UHF':
        dethr   = float(set['SCF Energy Threshold'])
        rmsthr  = float(set['SCF RMSD Threshold'])
        maxiter = int(set['SCF Max iterations'])
        uhf_mix = float(set['UHF mix guess'])
        EUHF, C_alpha, F_alpha, D_alpha, C_beta, F_beta, D_beta, iter = UnrestrictedHartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=dethr, rmsTHR=dethr, Maxiter=maxiter, UHF_mix=uhf_mix, print_SCF=print_SCF)
        results['UHFenergy'] = EUHF
        results['C_MO_alpha']    = C_alpha
        results['F_alpha']       = F_alpha
        results['D_alpha']       = D_alpha
        results['C_MO_beta']     = C_beta
        results['F_beta']        = F_beta
        results['D_beta']        = D_beta
        results['HF iterations'] = iter
        
    return results