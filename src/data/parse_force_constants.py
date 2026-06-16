"""
Phonopy FORCE_CONSTANTS (.fc) parser.

File format
-----------
Line 0 : "<n_atoms>  <n_pairs>"
Line 1 : "<i>  <j>"          (1-indexed atom pair)
Lines 2-4 : 3×3 force-constant matrix  [eV/Å²]
...
"""
import numpy as np


def parse_force_constants(fc_path: str):
    """
    Parse a Phonopy FORCE_CONSTANTS file.

    Returns
    -------
    n_atoms : int
    ifcs    : np.ndarray  shape (n_atoms, n_atoms, 3, 3)  [eV/Å²]
    """
    with open(fc_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    n_atoms = int(lines[0].split()[0])
    ifcs    = np.zeros((n_atoms, n_atoms, 3, 3), dtype=np.float64)

    idx = 1
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 2:
            try:
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                idx += 1
                for row in range(3):
                    vals = list(map(float, lines[idx].split()))
                    ifcs[i, j, row, :] = vals[:3]
                    idx += 1
            except (ValueError, IndexError):
                idx += 1
        else:
            idx += 1

    return n_atoms, ifcs
