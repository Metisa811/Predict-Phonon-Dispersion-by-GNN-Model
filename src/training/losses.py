"""
Physics-informed loss functions.
"""
import tensorflow as tf


def smoothness_loss(y_pred, n_qpoints: int, n_bands: int) -> tf.Tensor:
    """Second-order finite-difference smoothness along q-path."""
    y = tf.reshape(y_pred, [-1, n_qpoints, n_bands])
    d2 = y[:, 2:, :] - 2 * y[:, 1:-1, :] + y[:, :-2, :]
    return tf.reduce_mean(tf.square(d2))


def symmetry_loss(ifcs_pred) -> tf.Tensor:
    """
    Enforce IFC_ij = IFC_ji^T
    ifcs_pred shape: (batch, n_atoms, n_atoms, 3, 3)
    """
    ifc_T = tf.transpose(ifcs_pred, [0, 2, 1, 4, 3])
    return tf.reduce_mean(tf.square(ifcs_pred - ifc_T))


def acoustic_sum_rule_loss(ifcs_pred) -> tf.Tensor:
    """
    Enforce: sum_j IFC_ij = 0  for all i
    ifcs_pred shape: (batch, n_atoms, n_atoms, 3, 3)
    """
    asr = tf.reduce_sum(ifcs_pred, axis=2)   # (batch, n_atoms, 3, 3)
    return tf.reduce_mean(tf.square(asr))
