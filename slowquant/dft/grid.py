import numpy as np

from slowquant.dft.constants import BRAGG, LEBEDEV, POPLE_RADII


def SG1_grid(coordinate, charge, n_rad, n_ang=0):
    # https://www.sciencedirect.com/science/article/pii/0009261493801259?via%3Dihub
    grid_parameters = []
    grid = []
    pople_radii = POPLE_RADII[Z]
    for i in range(1, n_rad + 1):
        # includes a factor of 4pi
        w = 8.0 * np.pi * pople_radii**3 * (n_rad + 1.0) * i**5 * (n_rad + 1 - i) ** (-7)
        r = pople_radii * i**2 * (n_rad + 1 - i) ** (-2)
        if n_ang == 0:
            if charge in range(1, 3):  # H-He
                alphas = [0.25, 0.5, 1.0, 4.5]
            elif charge in range(3, 11):  # Li-Ne
                alphas = [0.1667, 0.500, 0.900, 3.5]
            else:  # Na-Ar
                alphas = [0.1, 0.4, 0.8, 2.5]
            if r < alphas[0] * pople_radii:
                n_lebedev = 6
            elif r < alphas[1] * pople_radii:
                n_lebedev = 38
            elif r < alphas[2] * pople_radii:
                n_lebedev = 86
            elif r < alphas[3] * pople_radii:
                n_lebedev = 194
            else:
                n_lebedev = 86
        else:
            n_lebedev = n_ang
        grid_parameters.append([r, w, n_lebedev])
    for r_rad, w_rad, n_lebedev in grid_parameters:
        for x_ang, y_ang, z_ang, w_ang in LEBEDEV[n_lebedev]:
            w = w_rad * w_ang
            grid.append(
                [
                    r_rad * x_ang + coordinate[0],
                    r_rad * y_ang + coordinate[1],
                    r_rad * z_ang + coordinate[2],
                    w,
                ]
            )
    return np.array(grid)


def becke_reweight(coordinates, charges, atom_grids):
    # https://aip-scitation-org.proxy1-bib.sdu.dk/doi/abs/10.1063/1.454033
    for grid_idx in range(0, len(atom_grids)):
        for point_idx in range(0, len(atom_grids[grid_idx])):
            xyz_point = atom_grids[grid_idx][point_idx, :3]
            Ps = []
            for atom_idx in range(0, len(atom_xyzZ)):
                s_product = 1
                r_ip = np.linalg.norm(atom_xyzZ[atom_idx, :3] - xyz_point)
                for atom_idx2 in range(0, len(atom_xyzZ)):
                    if atom_idx == atom_idx2:
                        continue
                    r_ij = np.linalg.norm(coordinates[atom_idx] - coordinates[atom_idx2])
                    r_jp = np.linalg.norm(coordinates[atom_idx2] - xyz_point)
                    mu = (r_ip - r_jp) / r_ij
                    if charges[atom_idx] != charges[atom_idx2]:
                        chi = BRAGG[int(charges[atom_idx])] / BRAGG[int(charges[atom_idx2])]
                        u = (chi - 1) / (chi + 1)
                        a = u / (u * u - 1)
                        a = min(a, 0.5)
                        a = max(a, -0.5)
                        mu += a * (1 - mu * mu)
                    p1 = 1.5 * mu - 0.5 * mu**3
                    p2 = 1.5 * p1 - 0.5 * p1**3
                    p3 = 1.5 * p2 - 0.5 * p2**3
                    s_product *= 0.5 * (1 - p3)
                Ps.append(s_product)
            P = Ps[grid_idx] / np.sum(Ps)
            atom_grids[grid_idx][point_idx, 3] = P * atom_grids[grid_idx][point_idx, 3]
    return atom_grids


def generate_grid(mol_obj):
    atomic_grids = []
    for atom in mol_obj.atoms:
        atomic_grids.append(atom.coordinate, atom.charge, 64)
    reweighted_atomic_grids = becke_reweight(atom_xyzZ, grids)
    grid = reweighted_atomic_grids[0]
    for i in range(1, len(reweighted_atomic_grids)):
        grid = np.vstack((grid, reweighted_atomic_grids[i]))
    return grid
