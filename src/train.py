import json
import pathlib
import re
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from .metrics import l0_sparsity, retrieval_at_k, clarity_score, cross_modal_score, active_feature_pct


def make_run_tag(epochs: int, eff_batch: int) -> str:
    """Build a short run identifier from training settings, e.g. 'e45_b8192'."""
    return f"e{epochs}_b{eff_batch}"


def _parse_run_tag(run_tag: str):
    """Parse 'e45_b8192' → (45, 8192). Returns (None, None) on failure."""
    m = re.match(r'e(\d+)_b(\d+)', run_tag or '')
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


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

    ret_k1 = retrieval_at_k(img_z, txt_z, k=k)
    ret_k5 = retrieval_at_k(img_z, txt_z, k=5)
    # Clarity: co-activation coherence using original CLIP embeddings as e(x)
    cl = clarity_score(z_img, v_img, tau=tau, n_min=n_min)
    # Cross-modal: image-activation ratio per feature (0.5 = perfectly multimodal)
    cm_mean, _ = cross_modal_score(z_img, z_txt)
    # Active feature %: fraction of features activating for ≥ n_min images
    act_pct = active_feature_pct(z_img, tau=tau, n_min=n_min)

    return {
        "l0_img": l0_sparsity(z_img),
        "l0_txt": l0_sparsity(z_txt),
        "active_pct": act_pct,
        "clarity": cl,
        "cross_modal": cm_mean,
        **ret_k1,
        **ret_k5,
    }


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    head: torch.nn.Module,
    metrics: dict,
    name: str,
    variation: str,
    CONFIG: dict,
    run_tag: str = "",
) -> None:
    """Save model weights and metrics JSON under results/variation/.

    run_tag (e.g. 'e45_b8192') is appended to filenames so different
    epoch/batch-size runs don't overwrite each other.
    """
    out_dir: pathlib.Path = CONFIG["results_dir"] / variation
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{run_tag}" if run_tag else ""
    model_path = out_dir / f"{name}{suffix}_model.pt"
    metrics_path = out_dir / f"{name}{suffix}_metrics.json"

    torch.save(head.state_dict(), model_path)

    epochs, eff_batch = _parse_run_tag(run_tag)
    payload = {
        "meta": {
            "name": name,
            "variation": variation,
            "run_tag": run_tag,
            "epochs": epochs,
            "eff_batch": eff_batch,
        },
        **metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[train] Saved {model_path} and {metrics_path}")


# ── Results loader ────────────────────────────────────────────────────────────

def load_results(
    results_dir: pathlib.Path,
    variation: Optional[str] = None,
    name: Optional[str] = None,
    epochs: Optional[int] = None,
    eff_batch: Optional[int] = None,
) -> list[dict]:
    """Load all *_metrics.json files, optionally filtered by metadata fields.

    Each returned dict contains the full metrics plus a 'meta' sub-dict:
        {
          "meta": {"name": "infonce", "variation": "variation1",
                   "run_tag": "e45_b8192", "epochs": 45, "eff_batch": 8192},
          "IR@1": 0.42, "clarity": 0.51, ...
        }

    Args:
        results_dir: top-level results directory (CONFIG['results_dir'])
        variation:   filter to one variation subfolder, e.g. 'variation1'
        name:        filter by model name, e.g. 'infonce' or 'gaussian'
        epochs:      filter by number of epochs
        eff_batch:   filter by effective batch size

    Example::
        rows = load_results(CONFIG['results_dir'], variation='variation1', epochs=45)
        for r in rows:
            print(r['meta']['name'], r['IR@1'], r['clarity'])
    """
    root = results_dir / variation if variation else results_dir
    rows = []
    for p in sorted(root.glob("**/*_metrics.json")):
        with open(p) as f:
            data = json.load(f)

        # Back-fill meta for JSON files written before this change
        meta = data.setdefault("meta", {})
        if not meta.get("run_tag"):
            # Try to infer from filename: e.g. infonce_e45_b8192_metrics.json
            stem  = p.stem  # 'infonce_e45_b8192_metrics'
            match = re.search(r'_(e\d+_b\d+)_metrics$', stem)
            if match:
                tag = match.group(1)
                ep, eb = _parse_run_tag(tag)
                meta.setdefault("run_tag",   tag)
                meta.setdefault("epochs",    ep)
                meta.setdefault("eff_batch", eb)
            # Infer name: everything before the first _e\d
            nm = re.sub(r'_e\d+.*$', '', stem)
            meta.setdefault("name", nm)
            # Infer variation from parent directory name
            meta.setdefault("variation", p.parent.name)

        if name      is not None and meta.get("name")      != name:      continue
        if epochs    is not None and meta.get("epochs")    != epochs:    continue
        if eff_batch is not None and meta.get("eff_batch") != eff_batch: continue

        rows.append(data)
    return rows
