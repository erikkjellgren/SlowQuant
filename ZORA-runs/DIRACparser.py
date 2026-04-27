#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from DIRACparser_functions import *


def unpack_upper_triangular(vec):
    """
    Convert flattened upper-triangular array into full square symmetric matrix.
    """
    # Solve n*(n+1)/2 = vec.size
    n = int((np.sqrt(1 + 8*vec.size) - 1)//2)
    full = np.zeros((n,n), dtype=vec.dtype)
    iu = np.triu_indices(n)
    full[iu] = vec
    # Fill lower triangle
    full = full + np.triu(full,1).T
    return full


def build_4c_AO_tensor(nef_dict):
    """
    Build 3x3 spin-spin AO tensor from NEF integrals.
    Handles 1D, flattened, or upper-triangular NEF arrays.
    Returns:
        SS_AO: (3, 3, n4c, n4c)
        nAO: number of physical AOs (4 spinors per AO)
    """
    # Pick first NEF to detect dimension
    first_matrix = next(iter(nef_dict.values()))

    # Handle 1D arrays
    if first_matrix.ndim == 1:
        first_matrix = unpack_upper_triangular(first_matrix)

    n4c = first_matrix.shape[0]
    if n4c % 4 != 0:
        print("Warning: 4c spinor dimension not divisible by 4")
    nAO = n4c // 4
    print(f"Detected 4c AO dimension: {n4c}, number of physical AOs: {nAO}")

    # Initialize 3x3 AO spin-spin tensor
    SS_AO = np.zeros((3, 3, n4c, n4c), dtype=first_matrix.dtype)

    # NEF → tensor mapping (typical)
    mapping = {
        "NEF 001 FTTF": (0, 0),
        "NEF 002 FTTF": (1, 1),
        "NEF 003 FTTF": (2, 2),
        "NEF 004 FTTF": (0, 1),
        "NEF 005 FTTF": (0, 2),
        "NEF 006 FTTF": (1, 0),
        "NEF 007 FTTF": (1, 2),
        "NEF 008 FTTF": (2, 0),
        "NEF 009 FTTF": (2, 1)
    }

    # Fill tensor
    for key, (i, j) in mapping.items():
        if key in nef_dict:
            mat = nef_dict[key]
            # If 1D, reconstruct full matrix
            if mat.ndim == 1:
                mat = unpack_upper_triangular(mat)
            SS_AO[i, j] = mat
        else:
            print(f"Warning: {key} not found, leaving zeros")

    return SS_AO, nAO




def main():

    if len(sys.argv) != 2:  # 1 for script name + 1 required
            raise ValueError("ERROR: This script requires exactly 1 argument: An AOPROPER file placed in the current directory")

    # Get name of input-file:
    input = sys.argv[1]

    # Opening Input files:
    try:
        contents = read_dirac_file(input)

    except FileNotFoundError:
        print("File not found. Please make sure the file exists.")
    except IOError:
        print("An error occurred while reading the file.")

    #print(contents.keys())
    #print(contents["ao_matrices"].keys())
    print(contents.keys())

    # # Extract NEF blocks
    # nef_dict = {k: v for k, v in contents["ao_matrices"].items() if k.startswith("NEF")}
    # if not nef_dict:
    #     print("No NEF integrals found in the file.")
    #     sys.exit(1)

    # # Build 4c AO spin-spin tensor
    # SS_AO, nAO = build_4c_AO_tensor(nef_dict)
    # print("AO spin-spin tensor built successfully.")
    # print("Tensor shape:", SS_AO.shape)




    
if __name__ == '__main__':
    main()