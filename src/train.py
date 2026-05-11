import json
import pathlib
from typing import Callable

import torch
from torch.utils.data import DataLoader

from .metrics import l0_sparsity, retrieval_at_k, clarity_score, cross_modal_score


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    head: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str,
) -> float:
    """Run one epoch; return mean loss."""
    head.train()
    total_loss = 0.0
    for img_emb, txt_emb in loader:
        img_emb = img_emb.to(device)
        txt_emb = txt_emb.to(device)
        _, img_z = head(img_emb)
        _, txt_z = head(txt_emb)
        loss = loss_fn(img_z, txt_z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    head: torch.nn.Module,
    val_img_emb: torch.Tensor,
    val_txt_emb: torch.Tensor,
    CONFIG: dict,
) -> dict:
    """Compute all metrics on the validation set.

    val_img_emb / val_txt_emb are the frozen CLIP embeddings, which serve as
    both the input to SparseHead and as e(x) for the Clarity metric.
    """
    head.eval()
    device = CONFIG["device"]
    k = CONFIG["retrieval_k"]
    tau = CONFIG["clarity_tau"]
    n_min = CONFIG["clarity_n_min"]

    v_img = val_img_emb.to(device)
    v_txt = val_txt_emb.to(device)

    z_img, img_z = head(v_img)
    z_txt, txt_z = head(v_txt)

    ret = retrieval_at_k(img_z, txt_z, k=k)
    # Clarity: co-activation coherence using original CLIP embeddings as e(x)
    cl = clarity_score(z_img, v_img, tau=tau, n_min=n_min)
    # Cross-modal: image-activation ratio per feature (0.5 = perfectly multimodal)
    cm_mean, _ = cross_modal_score(z_img, z_txt)

    return {
        "l0_img": l0_sparsity(z_img),
        "l0_txt": l0_sparsity(z_txt),
        "clarity": cl,
        "cross_modal": cm_mean,
        **ret,
    }


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    head: torch.nn.Module,
    metrics: dict,
    name: str,
    variation: str,
    CONFIG: dict,
) -> None:
    """Save model weights and metrics JSON under results/variation/."""
    out_dir: pathlib.Path = CONFIG["results_dir"] / variation
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{name}_model.pt"
    metrics_path = out_dir / f"{name}_metrics.json"

    torch.save(head.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] Saved {model_path} and {metrics_path}")
