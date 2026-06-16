"""
Main training script for the phonon GNN.

Usage (Kaggle / local):
    python -m src.training.train \
        --dataset_dir ./dataset \
        --output_dir  ./results/run1
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.gnn import build_model, rbf_encode, CUTOFF
from src.training.losses import smoothness_loss


# ── hyperparameters ────────────────────────────────────────────────────────
BATCH_SIZE     = 16
EPOCHS         = 600
LR             = 3e-4
PATIENCE       = 100
HIDDEN_DIM     = 256
N_LAYERS       = 5
LAMBDA_SMOOTH  = 1.5
LAMBDA_ELASTIC = 0.2


def build_graph(sample, cutoff=CUTOFF):
    pos    = sample['positions']
    n      = len(pos)
    diff   = pos[:, None] - pos[None]
    dist   = np.linalg.norm(diff, axis=-1)

    src, dst, vecs, lens = [], [], [], []
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] < cutoff:
                src.append(i); dst.append(j)
                vecs.append(diff[i, j]); lens.append(dist[i, j])

    if not src:          # fallback: 4 nearest neighbours
        for i in range(n):
            for j in np.argsort(dist[i])[1:5]:
                src.append(i); dst.append(j)
                vecs.append(diff[i, j]); lens.append(dist[i, j])

    lens   = np.array(lens, np.float32)
    dirs   = np.array(vecs, np.float32) / (lens[:, None] + 1e-8)
    edges  = np.column_stack([src, dst]).astype(np.int32)
    return edges, rbf_encode(lens), dirs


def prepare_batch(samples):
    nodes_l, edges_l, rbf_l, dir_l, gidx_l = [], [], [], [], []
    y_ph_l, y_el_l, mask_l = [], [], []
    off = 0
    for gid, s in enumerate(samples):
        nodes_l.append(s['node_features'])
        e, r, d = build_graph(s)
        edges_l.append(e + off)
        rbf_l.append(r); dir_l.append(d)
        y_ph_l.append(s['y_phonon'].reshape(-1))
        y_el_l.append(s['elastic_constants'])
        mask_l.append(float(s['has_elastic']))
        gidx_l.append(np.full(s['n_atoms'], gid, np.int32))
        off += s['n_atoms']

    inp = (
        np.vstack(nodes_l).astype(np.float32),
        np.vstack(edges_l).astype(np.int32),
        np.vstack(rbf_l).astype(np.float32),
        np.vstack(dir_l).astype(np.float32),
        np.concatenate(gidx_l).astype(np.int32),
    )
    tgt = (
        np.array(y_ph_l, np.float32),
        np.array(y_el_l, np.float32),
        np.array(mask_l, np.float32),
    )
    return inp, tgt


def main(dataset_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_data = list(np.load(f'{dataset_dir}/train_data.npy', allow_pickle=True))
    val_data   = list(np.load(f'{dataset_dir}/val_data.npy',   allow_pickle=True))

    # Normalise phonon targets
    all_f = np.concatenate([s['y_phonon'].flatten() for s in train_data])
    ph_mean, ph_std = float(all_f.mean()), float(all_f.std())
    for s in train_data + val_data:
        s['y_phonon'] = (s['y_phonon'] - ph_mean) / (ph_std + 1e-8)

    meta = train_data[0]
    atomic_dim = meta['node_features'].shape[1]
    elastic_dim = meta['elastic_constants'].shape[0]
    output_dim  = meta['y_phonon'].size

    model = build_model(atomic_dim, output_dim, elastic_dim, HIDDEN_DIM, N_LAYERS)
    model.summary()

    n_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    lr_sched  = keras.optimizers.schedules.CosineDecay(LR, n_steps, alpha=0.05)
    optimizer = keras.optimizers.AdamW(lr_sched, weight_decay=1e-5)
    mse       = keras.losses.MeanSquaredError()

    def step(inp, tgt, training):
        y_ph, y_el, mask = tgt
        if training:
            with tf.GradientTape() as tape:
                out    = model(inp, training=True)
                l_ph   = mse(y_ph, out['phonon'])
                l_sm   = smoothness_loss(out['phonon'],
                                         meta['y_phonon'].shape[0],
                                         meta['y_phonon'].shape[1])
                l_el   = mse(y_el, out['elastic']) * tf.reduce_mean(mask)
                loss   = l_ph + LAMBDA_SMOOTH * l_sm + LAMBDA_ELASTIC * l_el
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                (tf.clip_by_norm(g, 1.0), v)
                for g, v in zip(grads, model.trainable_variables) if g is not None
            )
        else:
            out  = model(inp, training=False)
            loss = mse(y_ph, out['phonon'])
        return float(loss)

    best_val, pat = np.inf, 0
    for epoch in range(1, EPOCHS + 1):
        np.random.shuffle(train_data)
        tl = np.mean([step(prepare_batch(train_data[i:i+BATCH_SIZE]), True)
                      for i in range(0, len(train_data), BATCH_SIZE)])
        vl = np.mean([step(prepare_batch(val_data[i:i+BATCH_SIZE]), False)
                      for i in range(0, len(val_data), BATCH_SIZE)])
        if epoch % 20 == 0:
            print(f'Epoch {epoch:04d} | Train {tl:.4f} | Val {vl:.4f}')
        if vl < best_val:
            best_val, pat = vl, 0
            model.save_weights(f'{output_dir}/best.weights.h5')
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f'Early stop at epoch {epoch}')
                break

    np.save(f'{output_dir}/phonon_norm.npy',
            {'mean': ph_mean, 'std': ph_std}, allow_pickle=True)
    print(f'Done. Best val loss: {best_val:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', default='./dataset')
    p.add_argument('--output_dir',  default='./results/run1')
    args = p.parse_args()
    main(args.dataset_dir, args.output_dir)
