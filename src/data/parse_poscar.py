"""
POSCAR / .psc file parser for MAX phase structures.
"""
import numpy as np


def parse_poscar(path: str) -> dict:
    """
    Parse a POSCAR or .psc file.

    Returns
    -------
    dict with keys:
        lattice       : np.ndarray (3,3)  Cartesian lattice vectors [Å]
        positions     : np.ndarray (N,3)  Cartesian atomic positions [Å]
        elements      : list[str]         unique element symbols
        counts        : list[int]         atoms per element
        atom_elements : list[str]         element symbol for each atom
        n_atoms       : int
        volume        : float             unit-cell volume [Å³]
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    scaling = float(lines[1].strip())
    lattice = np.array(
        [list(map(float, lines[i].split())) for i in range(2, 5)]
    ) * scaling

    line5 = lines[5].strip().split()
    line6 = lines[6].strip().split()

    if line5[0].isalpha():
        elements   = line5
        counts     = list(map(int, line6))
        coord_line = 7
    else:
        counts     = list(map(int, line5))
        elements   = [f'X{i}' for i in range(len(counts))]
        coord_line = 6

    coord_type = lines[coord_line].strip()[0].upper()
    n_atoms    = sum(counts)

    positions = np.array([
        list(map(float, lines[i].split()[:3]))
        for i in range(coord_line + 1, coord_line + 1 + n_atoms)
    ])

    if coord_type == 'D':          # Direct → Cartesian
        positions = positions @ lattice

    atom_elements = []
    for elem, cnt in zip(elements, counts):
        atom_elements.extend([elem] * cnt)

    return {
        'lattice':       lattice.astype(np.float32),
        'positions':     positions.astype(np.float32),
        'elements':      elements,
        'counts':        counts,
        'atom_elements': atom_elements,
        'n_atoms':       n_atoms,
        'volume':        float(np.abs(np.linalg.det(lattice))),
    }
