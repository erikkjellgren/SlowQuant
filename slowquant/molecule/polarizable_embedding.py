#!/usr/bin/env python

import numpy as np
from collections import namedtuple

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
            if '@POLARIZABILITIES' in line:
                section = 'polarizabilities'
            if 'EXCLISTS' in line:
                section = 'exclusion_lists'
            # read the data
            if section == 'multipoles':
                order = int(f.readline().split()[-1])
                # num_multipoles may or may not be less than num_sites
                num_multipoles = int(f.readline())
                multipoles[order] = np.zeros((num_sites, multipole_length(order)))
                for i in range(num_multipoles):
                    index, *values = f.readline().split()
                    index = int(index) - 1
                    values = list(map(float, values))
                    multipoles[order][index, :] = values
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


                    
    return namedtuple('potfile', ('coordinates', 'multipoles', 'polarizabilities', 'exclusion_lists'))(coordinates, multipoles, polarizabilities, exclusion_lists)
