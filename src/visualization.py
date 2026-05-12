"""Visualization helpers for Sparse CLIP experiments.

All plot_* functions accept an optional `ax` for embedding in larger layouts,
OR omit it to get a standalone figure returned.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


# ── Per-axes primitives ───────────────────────────────────────────────────────

def _modality_on_ax(ax: plt.Axes, z_img: torch.Tensor, z_txt: torch.Tensor,
                    title: str = '') -> dict:
    """Draw modality-score histogram (Figure 3b style) on `ax`."""
    img_sum = z_img.sum(0).cpu().float()
    txt_sum = z_txt.sum(0).cpu().float()
    total   = img_sum + txt_sum
    active  = total > 0
    ratio   = (img_sum[active] / total[active]).numpy()

    ax.hist(ratio, bins=40, range=(0, 1), color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2, label='ideal (0.5)')
    ax.axvline(float(ratio.mean()), color='orange', linestyle='--', linewidth=1.2,
               label=f'mean={ratio.mean():.3f}')
    ax.set_xlabel('Image activation ratio')
    ax.set_ylabel('# features')
    ax.set_title(title or 'Modality Score Distribution')
    ax.legend(fontsize=8)

    n_total = int(active.sum())
    n_multi = int(((ratio > 0.2) & (ratio < 0.8)).sum())
    ax.text(0.02, 0.95,
            f'Active: {n_total:,}\nMultimodal: {n_multi} ({n_multi / n_total * 100:.1f}%)',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return {'n_active': n_total, 'n_multimodal': n_multi, 'mean_ratio': float(ratio.mean())}


def _heatmap_on_ax(ax: plt.Axes, z: torch.Tensor,
                   n_samples: int = 64, n_features: int = 48,
                   title: str = '') -> None:
    """Draw feature-activation heatmap (samples × top-features) on `ax`."""
    feat_act = z.sum(0).cpu()
    top_feat = feat_act.topk(min(n_features, z.shape[1])).indices
    sub = z[:n_samples, top_feat].cpu().float().numpy()   # (S, F)

    im = ax.imshow(sub.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Activation', pad=0.02)
    ax.set_xlabel('Sample index')
    ax.set_ylabel(f'Top-{n_features} features')
    ax.set_title(title or 'Feature Activation Heatmap')


def _simmat_on_ax(ax: plt.Axes, img_z: torch.Tensor, txt_z: torch.Tensor,
                  n: int = 48, title: str = '') -> None:
    """Draw image–text cosine similarity matrix on `ax`.

    The diagonal should be bright (positive pairs).
    """
    img  = img_z[:n].cpu().float()
    txt  = txt_z[:n].cpu().float()
    sims = (img @ txt.T).numpy()

    im = ax.imshow(sims, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Cosine sim', pad=0.02)
    ax.set_xlabel('Text index')
    ax.set_ylabel('Image index')
    ax.set_title(title or 'Image–Text Similarity')

    diag = float(np.diag(sims).mean())
    off  = float((sims.sum() - np.trace(sims)) / (n * n - n))
    ax.text(0.02, 0.02,
            f'diag mean={diag:.3f}\noff-diag={off:.3f}',
            transform=ax.transAxes, va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# ── Standalone figure wrappers ────────────────────────────────────────────────

def plot_modality_distribution(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    title: str = '',
    out_path=None,
) -> plt.Figure:
    """Standalone modality-score histogram."""
    fig, ax = plt.subplots(figsize=(6, 4))
    _modality_on_ax(ax, z_img, z_txt, title)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_modality_grid(
    models: dict,                           # {name: (z_img, z_txt)}
    ncols: int = 3,
    out_path=None,
) -> plt.Figure:
    """Side-by-side modality histograms for multiple models."""
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, (name, (z_img, z_txt)) in zip(axes, models.items()):
        _modality_on_ax(ax, z_img, z_txt, name)
    for ax in axes[len(models):]:
        ax.set_visible(False)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_eval_grid(
    models: dict,                           # {name: {'z_img', 'z_txt', 'img_z', 'txt_z'}}
    out_path=None,
) -> plt.Figure:
    """3-row grid per model: modality hist / feature heatmap / retrieval matrix.

    models dict values must have keys:
        z_img  — raw sparse image activations (N, D)
        z_txt  — raw sparse text activations (N, D)
        img_z  — L2-normalised image sparse reps (N, D)
        txt_z  — L2-normalised text sparse reps (N, D)
    """
    names = list(models.keys())
    ncols = len(names)
    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 14))
    if ncols == 1:
        axes = axes.reshape(3, 1)

    for col, name in enumerate(names):
        m = models[name]
        _modality_on_ax(axes[0, col], m['z_img'], m['z_txt'], f'{name}\nModality Distribution')
        _heatmap_on_ax(axes[1, col], m['z_img'],               title=f'{name}\nFeature Heatmap (img)')
        _simmat_on_ax( axes[2, col], m['img_z'], m['txt_z'],   title=f'{name}\nRetrieval Sim Matrix')

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig
