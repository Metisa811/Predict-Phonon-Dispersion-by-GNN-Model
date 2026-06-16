"""
Phonon calculator: Force Constants → Phonon frequencies.

IFC (8,8,3,3)  →  Dynamical matrix D(q)  →  ω(q) via eigendecomposition
"""
import numpy as np

AMU_TO_EV_FS2 = 1.0364e-4          # conversion factor
THz_FACTOR    = 15.633              # eV/Å²·amu  →  THz

ATOMIC_MASSES = {
    'H':1.008,'He':4.003,'Li':6.941,'Be':9.012,'B':10.81,'C':12.011,
    'N':14.007,'O':15.999,'F':18.998,'Ne':20.180,'Na':22.990,'Mg':24.305,
    'Al':26.982,'Si':28.086,'P':30.974,'S':32.06,'Cl':35.453,'Ar':39.948,
    'K':39.098,'Ca':40.078,'Sc':44.956,'Ti':47.867,'V':50.942,'Cr':51.996,
    'Mn':54.938,'Fe':55.845,'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38,
    'Ga':69.723,'Ge':72.630,'As':74.922,'Se':78.971,'Br':79.904,'Kr':83.798,
    'Rb':85.468,'Sr':87.62,'Y':88.906,'Zr':91.224,'Nb':92.906,'Mo':95.95,
    'Tc':98.0,'Ru':101.07,'Rh':102.906,'Pd':106.42,'Ag':107.868,'Cd':112.414,
    'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.904,'Xe':131.293,
    'Cs':132.905,'Ba':137.327,'La':138.905,'Hf':178.49,'Ta':180.948,'W':183.84,
    'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.084,'Au':196.967,'Hg':200.592,
    'Tl':204.38,'Pb':207.2,'Bi':208.980,
}
DEFAULT_MASS = 100.0


def ifc_to_phonon_frequencies(
    ifcs:     np.ndarray,      # (n_atoms, n_atoms, 3, 3)  eV/Å²
    positions: np.ndarray,     # (n_atoms, 3)  Å  Cartesian
    elements:  list,           # len = n_atoms
    q_points:  np.ndarray,     # (n_q, 3)  fractional or Cartesian
    lattice:   np.ndarray,     # (3, 3)  Å
) -> np.ndarray:               # (n_q, 3*n_atoms)  THz
    """
    Compute phonon frequencies from real-space force constants.
    """
    n_atoms = len(elements)
    masses  = np.array(
        [ATOMIC_MASSES.get(e, DEFAULT_MASS) for e in elements],
        dtype=np.float64
    )

    n_q   = len(q_points)
    freqs = np.zeros((n_q, 3 * n_atoms), dtype=np.float64)

    # Lattice vectors (simple supercell: just R=0 here)
    R = np.zeros(3)

    for qi, q in enumerate(q_points):
        q_cart = q @ np.linalg.inv(lattice).T  # to Cartesian reciprocal
        D = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=complex)

        for i in range(n_atoms):
            for j in range(n_atoms):
                phase = np.exp(1j * np.dot(q_cart, positions[j] - positions[i]))
                prefactor = 1.0 / np.sqrt(masses[i] * masses[j])
                for a in range(3):
                    for b in range(3):
                        D[3*i+a, 3*j+b] += prefactor * ifcs[i, j, a, b] * phase

        D = (D + D.conj().T) / 2.0    # Hermitian symmetry
        eigvals = np.linalg.eigvalsh(D)
        # ω = sqrt(|λ|) × sign(λ)  [keep imaginary as negative]
        freqs[qi] = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * THz_FACTOR

    return freqs
