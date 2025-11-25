import pyscf
from pyscf import scf, mcscf
import slowquant.SlowQuant as sq
import io
import sys

import slowquant.unitary_coupled_cluster.linear_response.unrestricted_naive as unaive
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
import slowquant.unitary_coupled_cluster.linear_response.naive as naive

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
            active_space = ((active_space_list[0], active_space_list[1]), active_space_list[2])
            input[-1] = active_space
            geometry = " ".join(geometry_list)
    except FileNotFoundError: 
        print(f"file: {inp} not found")
    
    return input, geometry



def get_unrestricted_excitation_energy(geometry, basis, active_space, charge=0, spin=0, unit="bohr"):
    """
    Calculate unrestricted excitation energies
    """
    # Info for output file
    print(f'geometry: {geometry}, basis: {basis}, active space: {active_space}, charge: {charge}, spin (2S+1): {spin+1}')
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, charge=charge, spin=spin, unit=unit)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0]) 
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
        "fuccsd",
        {"n_layers":2},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_1step("bfgs", True)

    print("Electronic energy", WF.energy_elec_RDM)

    ULR = unaive.LinearResponseUPS(WF, excitations="SDTQ")
    ULR.calc_excitation_energies()
    print(f'excitation energies: {ULR.excitation_energies}')


def molecule():
    info = my_read_xyz_file(inp=sys.argv[1:])
    geometry = info[1]
    basis = info[0][0]
    spin = info[0][1]
    charge = info[0][2]
    active_space = info[0][3]

    get_unrestricted_excitation_energy(geometry=geometry, basis=basis, spin=spin, charge=charge, active_space=active_space, unit="angstrom")



molecule()

