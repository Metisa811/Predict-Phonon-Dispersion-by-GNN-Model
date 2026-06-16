"""
E(3)-Equivariant GNN with RBF edge encoding for phonon band prediction.

Architecture
------------
  Node features (D)  →  Dense(H)  →  RBFMessagePassing × L
                                    →  AttentionPooling
                                    →  Phonon head  →  ω(q)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


N_RBF       = 64
CUTOFF      = 6.0   # Å
RBF_CENTERS = np.linspace(0.5, CUTOFF, N_RBF).astype(np.float32)
RBF_GAMMA   = 2.0 / (RBF_CENTERS[1] - RBF_CENTERS[0]) ** 2


def rbf_encode(distances: np.ndarray) -> np.ndarray:
    d = distances.reshape(-1, 1)
    return np.exp(-RBF_GAMMA * (d - RBF_CENTERS) ** 2).astype(np.float32)


class RBFMessageLayer(keras.layers.Layer):
    """Direction-aware message passing with RBF distance encoding."""

    def __init__(self, hidden_dim: int, **kw):
        super().__init__(**kw)
        self.msg_net = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='silu'),
            keras.layers.LayerNormalization(),
            keras.layers.Dense(hidden_dim),
        ])
        self.update_net = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='silu'),
            keras.layers.LayerNormalization(),
            keras.layers.Dense(hidden_dim),
        ])
        self.gate = keras.layers.Dense(hidden_dim, activation='sigmoid')

    def call(self, node_feat, edges, edge_rbf, edge_dirs):
        src, dst = edges[:, 0], edges[:, 1]
        h_src    = tf.gather(node_feat, src)
        h_dst    = tf.gather(node_feat, dst)

        dir_proj = tf.reduce_sum(h_src * edge_dirs, axis=-1, keepdims=True)
        msg      = self.msg_net(tf.concat([h_src, h_dst, edge_rbf, dir_proj], -1))
        msg      = msg * self.gate(tf.concat([h_src, edge_rbf], -1))

        n_nodes = tf.shape(node_feat)[0]
        agg     = tf.math.unsorted_segment_sum(msg, dst, n_nodes)
        h_new   = self.update_net(tf.concat([node_feat, agg], -1))
        return node_feat + h_new   # residual


class AttentionPooling(keras.layers.Layer):
    def __init__(self, hidden_dim: int, **kw):
        super().__init__(**kw)
        self.attn = keras.layers.Dense(1)
        self.proj = keras.layers.Dense(hidden_dim, activation='silu')

    def call(self, h, graph_idx):
        n_g    = tf.reduce_max(graph_idx) + 1
        scores = self.attn(h)
        max_s  = tf.gather(tf.math.unsorted_segment_max(scores, graph_idx, n_g), graph_idx)
        exp_s  = tf.exp(scores - max_s)
        sum_s  = tf.gather(tf.math.unsorted_segment_sum(exp_s, graph_idx, n_g), graph_idx)
        w      = exp_s / (sum_s + 1e-10)
        pooled = tf.math.unsorted_segment_sum(h * w, graph_idx, n_g)
        return self.proj(pooled)


def build_model(
    atomic_feat_dim: int,
    output_dim: int,
    elastic_dim: int,
    hidden_dim: int = 256,
    n_layers:   int = 5,
) -> keras.Model:
    """
    Build and return the GNN model.

    Parameters
    ----------
    atomic_feat_dim : int   number of per-atom input features
    output_dim      : int   n_qpoints × n_bands  (flattened)
    elastic_dim     : int   number of elastic constant features
    """
    nodes    = keras.Input(shape=(atomic_feat_dim,),  name='nodes')
    edges    = keras.Input(shape=(2,), dtype=tf.int32, name='edges')
    e_rbf    = keras.Input(shape=(N_RBF,),            name='edge_rbf')
    e_dir    = keras.Input(shape=(3,),                name='edge_dir')
    gidx     = keras.Input(shape=(), dtype=tf.int32,  name='gidx')

    x = keras.layers.Dense(hidden_dim, activation='silu')(nodes)
    x = keras.layers.LayerNormalization()(x)

    for i in range(n_layers):
        x = RBFMessageLayer(hidden_dim, name=f'mp{i}')(x, edges, e_rbf, e_dir)
        if i < n_layers - 1:
            x = keras.layers.Dropout(0.1)(x)

    g = AttentionPooling(hidden_dim)(x, gidx)

    # Phonon head
    h = keras.layers.Dense(512, activation='silu')(g)
    h = keras.layers.LayerNormalization()(h)
    h = keras.layers.Dropout(0.2)(h)
    h = keras.layers.Dense(512, activation='silu')(h)
    h = keras.layers.Dropout(0.15)(h)
    h = keras.layers.Dense(256, activation='silu')(h)
    out_phonon  = keras.layers.Dense(output_dim,  name='phonon')(h)

    # Elastic head
    e = keras.layers.Dense(128, activation='silu')(g)
    e = keras.layers.Dropout(0.2)(e)
    out_elastic = keras.layers.Dense(elastic_dim, name='elastic')(e)

    return keras.Model(
        inputs=[nodes, edges, e_rbf, e_dir, gidx],
        outputs={'phonon': out_phonon, 'elastic': out_elastic},
    )
