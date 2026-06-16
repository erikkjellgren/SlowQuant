import numpy as np
import sys
import pyscf
from pyscf import mp, scf, mp, tools
from pyscf.mp import ump2
# from pyscf.mp import ump2
import scipy
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



def get_ump2_nat_orb(geometry, basis, active_space, unit='bohr', charge=0, spin=0):

    print(f"geometry: {geometry}, basis: {basis}, active space:, {active_space}, charge: {charge}, spin (2s+1): {spin+1}")

    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()
    mo = mf.mo_coeff
    pt = mp.UMP2(mf)
    ump2_E, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
    rdm1 = pt.make_rdm1()

    # print(rdm1.shape)
    occ_a, no_a = scipy.linalg.eigh(rdm1[0])
    occ_b, no_b = scipy.linalg.eigh(rdm1[1])

    print("mos and nos should be of size nao x nao and they are")
    print(f"no.shape alpha", no_a.shape)
    print(f"no.shape beta", no_b.shape)
    print("mo.shape alpha", )
    print(mo.shape)
    print("mol.nao")
    print(mol.nao)

    # print("trace of rdm1, sum of occupation numbers, and number of electrons")
    # print(f"numpy.trace(rdm1_alpha)", np.trace(rdm1[0]))
    # print(f"numpy.trace(rdm1_beta", np.trace(rdm1[0]))
    # print(f"numpy.sum(occ_a)", np.sum(occ_a))
    # print(f"numpy.sum(occ_a)", np.sum(occ_b))
    # print("mol.nelectron")
    # print(mol.nelectron)

    # # eigenvalues are sorted in ascending order so reorder
    # print("BEFORE REORDER")
    # print("occ alpha")
    # print(occ_a)
    # print("occ beta")
    # print(occ_b)


    # occ_a = occ_a[::-1]
    # no_a = no_a[:, ::-1]
    # occ_b = occ_b[::-1]
    # no_b = no_b[:, ::-1]
    # print("AFTER REORDER")
    # print("occ alpha")
    # print(occ_a)
    # print("occ beta")
    # print(occ_b)


    # mp = pyscf.mp.ump2.UMP2(mf, frozen=None, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
    # # mf = mp.ump2.UMP2(mol)
    # # mf.kernel()
    # print("only pyscf functions")
    # print(pyscf.mp.ump2.get_nmo(mp))
    # # print(na_orb)
    # print(pyscf.mp.ump2.get_nocc(mp))
    # mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
    # res = mc.kernel(mf.mo_coeff)
    
    # h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    # g_eri = mol.intor("int2e")


def molecule():
    info = my_read_xyz_file(inp=sys.argv[1:])
    geometry = info[1]
    basis = info[0][0]
    spin = info[0][1]
    charge = info[0][2]
    active_space = info[0][3]

    get_ump2_nat_orb(geometry=geometry, basis=basis, spin=spin, charge=charge, active_space=active_space, unit="angstrom")

molecule()