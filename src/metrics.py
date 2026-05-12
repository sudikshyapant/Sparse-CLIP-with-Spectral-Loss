import torch
import torch.nn.functional as F


def active_feature_pct(z: torch.Tensor, tau: float = 0.001, n_min: int = 2) -> float:
    """Percentage of features that activate for ≥ n_min samples (paper Table 3a 'Active F%')."""
    n_active = (z > tau).sum(0)                      # (D,) count per feature
    return (n_active >= n_min).float().mean().item() * 100


@torch.no_grad()
def zero_shot_accuracy(
    head,
    cls_embs: torch.Tensor,
    img_embs: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    batch_size: int = 512,
) -> float:
    """Top-1 zero-shot accuracy using sparse representations.

    Args:
        head:      SparseHead (or any module returning (z_sparse, z_norm))
        cls_embs:  (C, E) L2-normalised CLIP text embeddings for C class names
        img_embs:  (N, E) L2-normalised CLIP image embeddings
        labels:    (N,) integer ground-truth class indices
    """
    head.eval()
    _, cls_z = head(cls_embs.to(device))        # (C, D) L2-normalised sparse
    preds = []
    for i in range(0, len(img_embs), batch_size):
        _, z = head(img_embs[i:i + batch_size].to(device))
        preds.append((z @ cls_z.T).argmax(1).cpu())
    return (torch.cat(preds) == labels).float().mean().item()


def l0_sparsity(z: torch.Tensor) -> float:
    """Mean number of active (non-zero) features per sample."""
    return (z > 0).float().sum(dim=-1).mean().item()


def clarity_score(
    z: torch.Tensor,
    clip_emb: torch.Tensor,
    tau: float = 0.001,
    n_min: int = 2,
) -> float:
    """Clarity metric (Dreyer et al. 2025 / paper Eq. 2).

    For each feature i active on ≥ n_min images, compute the average pairwise
    cosine similarity among those images (using the original CLIP embeddings),
    then average over all qualifying features.

    Uses the identity:
        Σ_{j≠k, active} sim(e_j, e_k) = ‖Σ_j e_j‖² - n_i
    (unit-norm embeddings, so ‖e_j‖²=1 for every j).

    Args:
        z:        (N, D) sparse activations from SparseHead
        clip_emb: (N, E) original CLIP image embeddings (the reference encoder e(x))
        tau:      activation threshold (paper: 0.001)
        n_min:    minimum activating images per feature (paper: 2)

    Returns:
        scalar Clarity value
    """
    N, D = z.shape
    emb = F.normalize(clip_emb.float(), dim=-1)     # (N, E) unit-norm

    A = (z > tau).float()                           # (N, D) binary activation mask
    n_active = A.sum(0)                             # (D,) number of activating images
    valid = n_active >= n_min                       # (D,) bool

    if not valid.any():
        return 0.0

    # S[:,i] = sum of CLIP embeddings for images activating feature i
    # Computed in chunks of 1024 features to stay within GPU memory.
    chunk = 1024
    s_sq = torch.zeros(D, device=z.device, dtype=torch.float32)

    for start in range(0, D, chunk):
        A_c = A[:, start:start + chunk]            # (N, chunk)
        S_c = A_c.T @ emb                          # (chunk, E)
        s_sq[start:start + chunk] = S_c.pow(2).sum(1)

    # Sum of pairwise similarities (diagonal removed via the identity above)
    sum_pair = s_sq - n_active                     # (D,)

    n = n_active[valid]
    mean_pair = sum_pair[valid] / (n * (n - 1))    # normalise per feature
    return mean_pair.mean().item()


def cross_modal_score(z_img: torch.Tensor, z_txt: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Per-feature image-activation ratio (Figure 3b in paper).

    A ratio near 0.5 means the feature fires equally for images and text
    (truly multimodal).  Returns (mean ratio, per-feature ratio tensor).
    """
    img_sum = z_img.sum(0)                          # (D,) total image activation
    txt_sum = z_txt.sum(0)                          # (D,) total text activation
    total = img_sum + txt_sum
    active = total > 0
    ratio = img_sum[active] / total[active]         # (D',) ∈ [0,1]
    return ratio.mean().item(), ratio


def retrieval_at_k(
    img_z: torch.Tensor,
    txt_z: torch.Tensor,
    k: int = 1,
) -> dict:
    """Image→Text and Text→Image Recall@K using cosine similarity.

    Args:
        img_z: (N, D) L2-normalised image embeddings
        txt_z: (N, D) L2-normalised text embeddings (i-th row matches i-th image)
    """
    sims = img_z @ txt_z.T                          # (N, N)
    N = sims.size(0)
    targets = torch.arange(N, device=sims.device)

    def recall(sim_matrix):
        topk = sim_matrix.topk(k, dim=1).indices
        hits = (topk == targets.unsqueeze(1)).any(1)
        return hits.float().mean().item()

    return {f"IR@{k}": recall(sims), f"TR@{k}": recall(sims.T)}
