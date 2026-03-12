import numpy as np
import pyscf
from pyscf import scf, mcscf, fci
from pyscf.data import nist
import sys
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS

# Read geometry file
def my_read_xyz_file(inp):
    inp = inp[0]
    try:
        with open(inp, "r") as file:
            input_names = ["basis", "spin", "charge", "active_space"]
            input = ["", int(0), int(0), ()]
            geometry_list = []
            geometry = ""
            active_space_list = []
            lines = file.readlines()
            l = iter(lines)
            for line in l:
                new_line = line.strip().split()
                if len(new_line) >=5:
                    for a in range(len(input_names)):
                        if type(input[a]) == str:
                            input[a] = new_line[a+1]
                        elif type(input[a]) == int:
                            input[a] = int(new_line[a+1])
                        elif type(input[a]) == tuple:
                            for char in new_line[a+1]:
                                try:
                                    active_space_list.append(int(char))
                                except:
                                    continue
                elif len(new_line) == 4:
                    for a in range(len(new_line)):
                        geometry_list.append(new_line[a])
                    geometry_list.append(";")
                else:
                    continue
            if len(active_space_list) >= 3:
                active_space_list_str = []
                for num in range(len(active_space_list)):
                    active_space_list_str.append(str(active_space_list[num]))
                active_orb = "".join(active_space_list_str[2:])
            
            active_space = ((active_space_list[0], active_space_list[1]), int(active_orb))
            input[-1] = active_space
            geometry = " ".join(geometry_list)
    except FileNotFoundError: 
        print(f"file: {inp} not found")
    
    return input, geometry

def get_wf_optimized_unrestricted(geometry, basis, active_space, unit='bohr', charge=0, spin=0):
    """
    Optimize unrestricted wavefunction for HFC's with pp-utUPS
    """
    print(f"geometry: {geometry}, basis: {basis}, active space:, {active_space}, charge: {charge}, spin (2s+1): {spin+1}")
    #PySCF UHF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # mc = mcscf.UCASSCF(mf, active_space[1], active_space[0])
    res = mc.kernel(mf.mo_coeff)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
    
    #Slowquant
    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "utups",
        {"n_layers":2},
        include_active_kappa=True,
    )
    
    WF.run_wf_optimization_1step("bfgs", orbital_optimization=True, tol=1e-6, maxiter=5000)
    # WF.run_wf_optimization_1step("slsqp", True)
    print("Electronic energy")
    print(WF.energy_elec_RDM)

    print("thetas")
    print(WF.thetas)
    print("alpha mo_coeff")
    print(WF.c_a_mo)
    print("beta mo_coeff")
    print(WF.c_b_mo)
    print("# active orbs")
    print(WF.num_active_orbs)
    print("# inactive orbs")
    print(WF.num_inactive_orbs)
    print("one-electron alpha rdm")
    print(WF.rdm1aa)
    print("one-electron beta rdm")
    print(WF.rdm1bb)



def molecule():
    info = my_read_xyz_file(inp=sys.argv[1:])
    geometry = info[1]
    basis = info[0][0]
    spin = info[0][1]
    charge = info[0][2]
    active_space = info[0][3]

    get_wf_optimized_unrestricted(geometry=geometry, basis=basis, spin=spin, charge=charge, active_space=active_space, unit="angstrom")

molecule()