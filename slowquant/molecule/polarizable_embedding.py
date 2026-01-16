#!/usr/bin/env python

import numpy as np
from collections import namedtuple
from pyscf.data.nist import BOHR

def multipole_length(n:int)->int:
    return (n+1)*(n+2)//2

def read_potfile(filename:str) -> namedtuple:
    with open(filename, 'r') as f:
        section = None
        multipoles = {}
        polarizabilities = None
        exclusion_lists = None
        while (line := f.readline()):
            if '@COORDINATES' in line:
                num_sites = int(f.readline())
                unit = f.readline().strip()
                coordinates = np.zeros((num_sites, 3))
                for i in range(num_sites):
                    element, x, y, z, *_ = f.readline().split()
                    coordinates[i,:] = [float(x), float(y), float(z)]
            # section headers
            if '@MULTIPOLES' in line:
                section = 'multipoles'
                line = f.readline()
            if '@POLARIZABILITIES' in line:
                section = 'polarizabilities'
            if 'EXCLISTS' in line:
                section = 'exclusion_lists'
            # read the data
            if section == 'multipoles':
                order = int(line.split()[-1])
                if order > 2:
                    raise ValueError('Only support for multipoles up to quadrupoles')
                # num_multipoles may or may not be less than num_sites
                num_multipoles = int(f.readline())
                multipoles[order] = np.zeros((num_sites, multipole_length(order)))
                for i in range(num_multipoles):
                    index, *values = f.readline().split()
                    index = int(index) - 1
                    values = list(map(float, values))
                    multipoles[order][index, :] = values
                if order == 2:
                    # unpack quadrupoles from packed index (xx, xy, xz, yy, yz, zz) -> (xx, xy, xz, yx, yy, yz, zx, zy, zz)
                    # and remove trace
                    unpacked = np.zeros((num_multipoles, 9))
                    quadrupoles = multipoles[2]
                    trace = quadrupoles[:, 0] + quadrupoles[:, 3] + quadrupoles[:, 5]
                    unpacked[:, 0] = quadrupoles[:, 0] - trace / 3 # xx
                    unpacked[:, 1] = quadrupoles[:, 1]             # xy
                    unpacked[:, 2] = quadrupoles[:, 2]             # xz
                    unpacked[:, 3] = quadrupoles[:, 1]             # yx=xy
                    unpacked[:, 4] = quadrupoles[:, 3] - trace / 3 # yy 
                    unpacked[:, 5] = quadrupoles[:, 4]             # yz
                    unpacked[:, 6] = quadrupoles[:, 2]             # zx=xz
                    unpacked[:, 7] = quadrupoles[:, 4]             # zy=yz
                    unpacked[:, 8] = quadrupoles[:, 5] - trace / 3 # zz
                    multipoles[order] = unpacked
            if section == 'polarizabilities':
                order = f.readline().strip()
                num_polarizabilities = int(f.readline())
                if order != 'ORDER 1 1':
                    raise ValueError(f'Cannot handle polarizability order: {order}')
                # num_polarizabilities may or may not be less than num_site s
                polarizabilities = np.zeros((num_sites, 6))
                for i in range(num_polarizabilities):
                    index, *values = f.readline().split()
                    index = int(index) - 1
                    values = list(map(float, values))
                    polarizabilities[index, :] = values
            if section == 'exclusion_lists':
                num_exclusions, max_length = list(map(int, f.readline().split()))
                exclusion_lists = np.zeros((num_exclusions, max_length), dtype=np.int64)
                for i in range(num_exclusions):
                    values = list(map(lambda val: int(val) - 1, f.readline().split()))
                    index = values[0]
                    exclusion_lists[index, :] = values
    if unit == 'AU':
        unit_factor = 1.0
    elif unit == 'AA':
        unit_factor = 1/BOHR
    else:
        raise ValueError(f'Invalid coordinate unit ({unit}) in potfile)')
    coordinates *= unit_factor
    return namedtuple('potfile', ('coordinates', 'multipoles', 'polarizabilities', 'exclusion_lists'))(coordinates, multipoles, polarizabilities, exclusion_lists)

def T0(Rab):
    x = Rab[..., 0]
    y = Rab[..., 1]
    z = Rab[..., 2]
    shape = Rab.shape[:-1] + (3, ) * 0
    result = np.zeros(shape, dtype=np.float64)
    result[..., ] = 1 / np.sqrt(x**2 + y**2 + z**2)
    return result

def T1(Rab):
    x = Rab[..., 0]
    y = Rab[..., 1]
    z = Rab[..., 2]
    shape = Rab.shape[:-1] + (3, ) * 1
    result = np.zeros(shape, dtype=np.float64)
    x0 = (x**2 + y**2 + z**2)**(-3 / 2)
    result[..., 0] = -x * x0
    result[..., 1] = -x0 * y
    result[..., 2] = -x0 * z
    return result

def T2(Rab):
    x = Rab[..., 0]
    y = Rab[..., 1]
    z = Rab[..., 2]
    shape = Rab.shape[:-1] + (3, ) * 2
    result = np.zeros(shape, dtype=np.float64)
    x0 = x**2
    x1 = y**2
    x2 = z**2
    x3 = x1 + x2
    x4 = (x0 + x3)**(-5 / 2)
    x5 = 3 * x * x4
    x6 = x5 * y
    x7 = x5 * z
    x8 = 3 * x4 * y * z
    result[..., 0, 0] = -x4 * (-2 * x0 + x3)
    result[..., 0, 1] = x6
    result[..., 0, 2] = x7
    result[..., 1, 0] = x6
    result[..., 1, 1] = -x4 * (x0 - 2 * x1 + x2)
    result[..., 1, 2] = x8
    result[..., 2, 0] = x7
    result[..., 2, 1] = x8
    result[..., 2, 2] = -x4 * (x0 + x1 - 2 * x2)
    return result


def T3(Rab):
    x = Rab[..., 0]
    y = Rab[..., 1]
    z = Rab[..., 2]
    shape = Rab.shape[:-1] + (3, ) * 3
    result = np.zeros(shape, dtype=np.float64)
    x0 = x**2
    x1 = y**2
    x2 = 3 * x1
    x3 = z**2
    x4 = 3 * x3
    x5 = x1 + x3
    x6 = (x0 + x5)**(-7 / 2)
    x7 = 3 * x6
    x8 = x * x7
    x9 = x7 * (-4 * x0 + x5)
    x10 = x9 * y
    x11 = x9 * z
    x12 = x0 - 4 * x1 + x3
    x13 = x12 * x8
    x14 = -15 * x * x6 * y * z
    x15 = x0 + x1 - 4 * x3
    x16 = x15 * x8
    x17 = 3 * x0
    x18 = x7 * y
    x19 = x7 * z
    x20 = x12 * x19
    x21 = x15 * x18
    result[..., 0, 0, 0] = x8 * (-2 * x0 + x2 + x4)
    result[..., 0, 0, 1] = x10
    result[..., 0, 0, 2] = x11
    result[..., 0, 1, 0] = x10
    result[..., 0, 1, 1] = x13
    result[..., 0, 1, 2] = x14
    result[..., 0, 2, 0] = x11
    result[..., 0, 2, 1] = x14
    result[..., 0, 2, 2] = x16
    result[..., 1, 0, 0] = x10
    result[..., 1, 0, 1] = x13
    result[..., 1, 0, 2] = x14
    result[..., 1, 1, 0] = x13
    result[..., 1, 1, 1] = x18 * (-2 * x1 + x17 + x4)
    result[..., 1, 1, 2] = x20
    result[..., 1, 2, 0] = x14
    result[..., 1, 2, 1] = x20
    result[..., 1, 2, 2] = x21
    result[..., 2, 0, 0] = x11
    result[..., 2, 0, 1] = x14
    result[..., 2, 0, 2] = x16
    result[..., 2, 1, 0] = x14
    result[..., 2, 1, 1] = x20
    result[..., 2, 1, 2] = x21
    result[..., 2, 2, 0] = x16
    result[..., 2, 2, 1] = x21
    result[..., 2, 2, 2] = x19 * (x17 + x2 - 2 * x3)
    return result

def compute_potential(multipoles, multipole_coordinates, target_coordinates):
    potential = np.zeros(target_coordinates.shape[0])
    Rab = multipole_coordinates[:,None] - target_coordinates
    if 0 in multipoles:
        potential += np.einsum('pi,p->i', T0(Rab), multipoles[0].ravel())
    if 1 in multipoles:
        potential += np.einsum('pij,pj->i', T1(Rab), multipoles[1])
    if 2 in multipoles:
        potential += np.einsum('pijk,pjk->i', T2(Rab), multipoles[2].reshape(-1,3,3)) / 2
    return potential

def compute_nuclear_multipole_energy(nuclear_charges):
    potential = compute_potential(multipoles, multipole_coordinates, nuclear_coordinates)
    energy = np.dot(potential, nuclear_charges)
    return energy

def compute_nuclear_field(nuclear_charges, nuclear_positions, target_coordinates):
    Rab = nuclear_positions[:,None] - target_coordinates
    field = np.einsum('pix,p->ix', T1(Rab), nuclear_charges)
    return field

def compute_multipole_field(multipoles, multipole_coordinates, target_coordinates, exclusion_lists):
    field = np.zeros((target_coordinates.shape[0], 3))
    mask = np.ones(target_coordinates.shape[0], dtype=bool)
    for i in range(target_coordinates.shape[0]):
        # set mask
        for exclusion in exclusion_lists[i]:
            mask[exclusion] = False
        Rab = multipole_coordinates[mask, :] - target_coordinates[i, :]
        if 0 in multipoles:
            field[i, :] += np.einsum('px,p->x', T1(Rab), multipoles[0][mask, :].ravel())
        if 1 in multipoles:
            field[i, :] += np.einsum('pjx,pj->x', T2(Rab), multipoles[1][mask, :])
        if 2 in multipoles:
            field[i, :] += np.einsum('pjkx,pjk->x', T3(Rab), multipoles[2][mask, :].reshape(-1,3,3)) / 2
        # release mask
        for exclusion in exclusion_lists[i]:
            mask[exclusion] = True
    return field
