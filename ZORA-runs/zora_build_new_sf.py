import struct
import numpy as np

# ---------------------------------------------------------------
# File reading utilities
# ---------------------------------------------------------------

def read_fortran_record(f):
    marker_bytes = f.read(4)
    if len(marker_bytes) < 4:
        return None
    nbytes = struct.unpack("<i", marker_bytes)[0]
    data = f.read(nbytes)
    f.read(4)  # trailing marker
    return nbytes, data

def read_int(f):
    nbytes, data = read_fortran_record(f)
    return struct.unpack("<i", data[:4])[0]

def read_column(f, nbf):
    nbytes, data = read_fortran_record(f)
    return np.array(struct.unpack(f"<{nbf}d", data))

def read_matrix(f, nbf):
    """Read a full nbf x nbf matrix stored column by column."""
    mat = np.zeros((nbf, nbf))
    for col in range(nbf):
        mat[:, col] = read_column(f, nbf)
    return mat

# ---------------------------------------------------------------
# Main function: read .zora_so file and return Hamiltonian
# and scaling matrices
# ---------------------------------------------------------------

def read_zora_sf(filename):
    """
    Read NWChem .zora_so binary file and return the full ZORA
    Hamiltonian contribution and scaling matrices.

    Returns
    -------
    nbf          : number of AO basis functions
    H_zora       : 4x4 complex ZORA Hamiltonian contribution
                   (add T + V_ne to diagonal blocks to complete)
    zora_scale_sf : list of 2 matrices (alpha, beta) for scalar scaling
    zora_scale_so : list of 3 matrices (z, y, x) for SO scaling
    """
    with open(filename, 'rb') as f:
        nsets = read_int(f)
        nbf   = read_int(f)
        mult  = read_int(f)

        zora_sf       = [read_matrix(f, nbf) for _ in range(2)]
        zora_scale_sf = [read_matrix(f, nbf) for _ in range(2)]

    H_zora = np.zeros((2*nbf, 2*nbf), dtype=complex)
    H_zora[:nbf,  :nbf]  =   zora_sf[0].astype(complex)
    H_zora[nbf:,  nbf:]  =   zora_sf[1].astype(complex) 
    H_zora[:nbf,  nbf:]  =   np.zeros_like(zora_sf[0])
    H_zora[nbf:,  :nbf]  =   np.zeros_like(zora_sf[0])

    return H_zora


if __name__ == "__main__":
    import sys
    zora_filename = sys.argv[1] if len(sys.argv) > 1 else "H2_zora.zora_so"

    H_zora = read_zora_sf(zora_filename)

    #print(f"nbf = {nbf}")
    print(f"H_zora shape: {H_zora.shape}")
    print(f"Hermiticity check: max|H_zora - H_zora†| = "
          f"{np.max(np.abs(H_zora - H_zora.conj().T)):.2e}")
