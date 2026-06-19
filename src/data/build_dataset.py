"""
Build the unified MAX-phase dataset:
    POSCAR + Force Constants + Atomic Features + Elastic Constants
    → list of sample dicts  →  saved as .npy splits

Kaggle paths (adjust as needed):
    FC_DIR     = /kaggle/input/pgcnndata/extracted_force_constant_All/extracted_force_constant_C
    POSCAR_DIR = /kaggle/input/pgcnndata/all_poscar/all_poscar
    BANDS_DIR  = /kaggle/input/pgcnndata/extracted_bands_C
    CSV_FILE   = /kaggle/input/features-dataset/ptable2.csv
    ELASTIC_FILE = /kaggle/input/features-dataset/mechanical_data_fixed.json
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.data.parse_poscar import parse_poscar
from src.data.parse_force_constants import parse_force_constants


def build_dataset(
    fc_dir: str,
    poscar_dir: str,
    bands_dir: str,
    csv_file: str = '/kaggle/input/d/metisa81/feature-dataset-split/5_geometry_radius_volume.csv',
    elastic_file: str = None,
    output_dir: str = './dataset',
    cutoff_radius: float = 6.0,
    random_seed: int = 42,
):
    """
    Notes
    -----
    csv_file default: '5_geometry_radius_volume.csv' (7 atomic features).
    Chosen after a real train/test comparison across 4 feature sets on the
    actual 358-material MAX Phase dataset (Dual Graph GNN, June 2026):

        feature_set                 n_features  test_mae
        5_geometry_radius_volume     7          0.429   <- winner (full 1000-epoch run)
        9_CLEAN_no_missing         100          0.576
        2_phonon_PROVEN_best10       12          1.110
        0_RECOMMENDED_2025_BEST      14          1.162

    See notebooks/11_Feature_Comparison_MAX_Phase.ipynb for the full experiment.
    Simple geometric features (atomic radius + volume) outperformed larger/more
    "sophisticated" feature sets that were originally tuned for Matbench Phonons
    (1265 samples) — likely overfitting on this smaller 358-material dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Atomic features ---
    df = pd.read_csv(csv_file)
    elem_col = next(
        c for c in df.columns if c.lower() in ['symbol', 'element', 'atom']
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    feat_matrix = scaler.fit_transform(df[numeric_cols].fillna(0))
    element_features = {
        str(row[elem_col]).strip(): feat_matrix[i].astype(np.float32)
        for i, (_, row) in enumerate(df.iterrows())
    }
    feat_dim = feat_matrix.shape[1]

    # --- Elastic constants ---
    with open(elastic_file) as f:
        elastic_json = json.load(f)
    elastic_keys = ['C11','C12','C13','C33','C44','C66',
                    'B_Hill','G_Hill','E_Hill','Poisson_Hill',
                    'Pugh_ratio','Debye_temperature']
    elastic_data = {}
    for entry in elastic_json:
        mat = entry.get('material', '')
        vals = [float(entry.get(k, 0) or 0) for k in elastic_keys]
        elastic_data[mat] = np.array(vals, dtype=np.float32)
    elastic_dim = len(elastic_keys)

    # --- File matching ---
    fc_dict     = {Path(f).stem: f for f in Path(fc_dir).glob('*.fc')}
    poscar_dict = {Path(f).stem: f for f in Path(poscar_dir).glob('*.psc')}
    band_dict   = {Path(f).stem: np.load(f, allow_pickle=True)
                   for f in Path(bands_dir).glob('*.npy')}

    common = sorted(set(fc_dict) & set(poscar_dict) & set(band_dict))
    print(f'Matched materials: {len(common)}')

    # target shape
    shapes = [np.array(band_dict[f]).shape for f in common]
    from collections import Counter
    target_shape = Counter(shapes).most_common(1)[0][0]

    dataset = []
    failed  = []
    for formula in tqdm(common, desc='Building dataset'):
        try:
            band_arr = np.array(band_dict[formula])
            if band_arr.shape != target_shape:
                continue
            poscar = parse_poscar(str(poscar_dict[formula]))
            n_atoms, ifcs = parse_force_constants(str(fc_dict[formula]))
            if n_atoms != poscar['n_atoms']:
                continue

            node_feats = np.array([
                element_features.get(e, np.zeros(feat_dim, np.float32))
                for e in poscar['atom_elements']
            ], dtype=np.float32)

            dataset.append({
                'formula':          formula,
                'n_atoms':          n_atoms,
                'lattice':          poscar['lattice'],
                'positions':        poscar['positions'],
                'elements':         poscar['atom_elements'],
                'node_features':    node_feats,
                'force_constants':  ifcs.astype(np.float32),
                'elastic_constants': elastic_data.get(
                    formula, np.zeros(elastic_dim, np.float32)),
                'has_elastic':      formula in elastic_data,
                'y_phonon':         band_arr.astype(np.float32),
            })
        except Exception as e:
            failed.append((formula, str(e)))

    print(f'Success: {len(dataset)}, Failed: {len(failed)}')

    # train/val/test split
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(len(dataset))
    n_tr = int(0.8 * len(dataset))
    n_va = int(0.1 * len(dataset))
    splits = {
        'train': [dataset[i] for i in idx[:n_tr]],
        'val':   [dataset[i] for i in idx[n_tr:n_tr+n_va]],
        'test':  [dataset[i] for i in idx[n_tr+n_va:]],
    }
    for name, data in splits.items():
        np.save(os.path.join(output_dir, f'{name}_data.npy'), data, allow_pickle=True)
    print(f"Saved to {output_dir}")
    return splits
