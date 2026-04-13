#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import FortranFile
import h5py


def detect_file_type(filename):
    """
    Detect whether file is HDF5 or old Fortran unformatted.
    """
    with open(filename, "rb") as f:
        signature = f.read(8)

    # HDF5 files start with this signature
    if signature.startswith(b"\x89HDF"):
        return "hdf5"

    return "fortran"

def read_label_old(f):
    record = f.read_record(np.dtype("4a8"))
    stars = record[0].decode("utf-8")

    if stars != "********":
        raise IOError("Error reading label, file may be corrupted")

    date_run = record[1].decode("utf-8")
    time_run = record[2].decode("utf-8")
    label = record[3].decode("utf-8")

    return label.strip(), date_run.strip(), time_run.strip()

def read_label(f):
    record = f.read_record(np.dtype("S8"))  # array of 8-byte strings
    
    # numpy.bytes_ needs decode differently
    stars    = record[0].tobytes().decode("utf-8").strip()
    date_run = record[1].tobytes().decode("utf-8").strip()
    time_run = record[2].tobytes().decode("utf-8").strip()
    label    = record[3].tobytes().decode("utf-8").strip()

    if stars != "********":
        raise IOError(f"Expected '********', got '{stars}' — file may be corrupted or wrong format")

    return label, date_run, time_run

def clean_operator_name(raw_name):
    """
    Remove storage flags like TFFT, FFFT, FTTF
    and return clean operator label.
    """
    parts = raw_name.split()

    # Remove storage pattern if present
    if parts[-1] in ("TFFT", "FFFT", "FTTF"):
        parts = parts[:-1]

    name = parts[0]

    # Optional: normalize some names
    if name == "MOLFIELDTFFT":
        name = "MOLFIELD"

    if name == "KINENERGTFFF":
        name = "KINENERG"

    return name

def read_fortran_dirac(file_name):
    contents = {}

    with FortranFile(file_name, "r") as f:
        while True:
            label, date_run, time_run = read_label(f)

            if label in ("EOF", "EOFLABEL"):
                break

            data = f.read_reals()

            # AOPROPER hack
            if "AOPROPER" in file_name:
                contents[label + time_run] = data
            else:
                contents[label] = data

    return contents

def read_hdf5_dirac(file_name):
    contents = {
        "ao_matrices": {},
        "other": {}
    }

    with h5py.File(file_name, "r") as f:

        def recurse(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return

            # Check if this is an AO matrix
            if "ao_matrices" in name:
                raw = name.split("/")[-1]
                clean = clean_operator_name(raw)
                contents["ao_matrices"][raw] = obj[:]
            else:
                contents["other"][name] = obj[:]

        f.visititems(recurse)

    return contents

def read_fortran_aomomat(file_name):
    """
    Read DIRAC AOMOMAT file — contains MO coefficients.
    Record structure:
        Record 1: label (4x 8-char strings)
        Record 2: integer header [n_basis, n_mo, ...]
        Record 3+: complex MO coefficients, column by column
    """
    contents = {"mo_coefficients": None, "header": None}

    with FortranFile(file_name, "r") as f:
        # Record 1: label
        try:
            label, date_run, time_run = read_label(f)
            print(f"Label: {label}, Date: {date_run}, Time: {time_run}")
        except Exception as e:
            print(f"Warning: could not read label: {e}")

        # Record 2: try reading as integers (header info)
        try:
            header = f.read_ints()
            print(f"Header integers: {header}")
            contents["header"] = header
        except Exception as e:
            print(f"Warning: could not read header: {e}")

        # Record 3+: read all remaining records as reals or complex
        all_data = []
        while True:
            try:
                # Try complex first
                data = f.read_reals(dtype=np.float64)
                all_data.append(data)
                print(f"  Read record: {len(data)} float64 values")
            except Exception:
                break

        if all_data:
            flat = np.concatenate(all_data)
            print(f"Total float64 values: {len(flat)}")
            # Try to interpret as complex (pairs of real+imag)
            if len(flat) % 2 == 0:
                cplx = flat[::2] + 1j * flat[1::2]
                print(f"As complex: {len(cplx)} values")
                # Try to reshape to square matrix
                n = int(np.sqrt(len(cplx)))
                if n * n == len(cplx):
                    contents["mo_coefficients"] = cplx.reshape(n, n)
                    print(f"MO coefficient matrix: ({n} x {n})")
                else:
                    contents["mo_coefficients"] = cplx
            else:
                contents["mo_coefficients"] = flat

    return contents

def read_dirac_file(file_name):
    file_type = detect_file_type(file_name)

    if file_type == "hdf5":
        print("Detected HDF5 file")
        return read_hdf5_dirac(file_name)

    elif file_type == "fortran":
        print("Detected Fortran unformatted file")
        # Dispatch based on filename
        if "AOMOMAT" in file_name:
            return read_fortran_aomomat(file_name)
        else:
            return read_fortran_dirac(file_name)

    else:
        raise ValueError("Unknown DIRAC file format")
