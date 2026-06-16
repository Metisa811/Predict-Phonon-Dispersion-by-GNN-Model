"""
Evaluation script: load best model, compute MAE/RMSE on test set,
plot predicted vs true phonon bands.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def evaluate(model, test_data, ph_mean, ph_std, output_dir,
             prepare_batch_fn, batch_size=16):

    all_pred, all_true = [], []
    for i in range(0, len(test_data), batch_size):
        inp, tgt = prepare_batch_fn(test_data[i:i+batch_size])
        out  = model(inp, training=False)
        pred = out['phonon'].numpy() * ph_std + ph_mean
        true = tgt[0] * ph_std + ph_mean
        all_pred.append(pred)
        all_true.append(true)

    pred = np.vstack(all_pred)
    true = np.vstack(all_true)

    mae  = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    print(f'Test MAE : {mae:.4f} THz')
    print(f'Test RMSE: {rmse:.4f} THz')

    # scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    flat_t = true.flatten()[::20]
    flat_p = pred.flatten()[::20]
    ax.scatter(flat_t, flat_p, s=3, alpha=0.3)
    mn, mx = flat_t.min(), flat_t.max()
    ax.plot([mn, mx], [mn, mx], 'r--', label='Perfect')
    ax.set_xlabel('True frequency (THz)')
    ax.set_ylabel('Predicted frequency (THz)')
    ax.set_title(f'Phonon prediction  MAE={mae:.4f} THz')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter.png'), dpi=150)
    plt.close()

    np.save(os.path.join(output_dir, 'predictions.npy'), pred)
    np.save(os.path.join(output_dir, 'ground_truth.npy'), true)
    return {'mae': mae, 'rmse': rmse}
