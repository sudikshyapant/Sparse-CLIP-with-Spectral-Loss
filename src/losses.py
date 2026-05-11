import torch
import torch.nn as nn
import torch.nn.functional as F


# ── InfoNCE ───────────────────────────────────────────────────────────────────

def infonce_loss(img: torch.Tensor, txt: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style).

    Args:
        img:       (B, D) L2-normalised image embeddings
        txt:       (B, D) L2-normalised text embeddings
        log_scale: scalar learnable logit scale = log(1/τ); init at log(1/0.07)

    Returns:
        scalar mean loss
    """
    logits = img @ txt.T * log_scale.exp()          # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


# ── Spectral loss ─────────────────────────────────────────────────────────────

def spectral_loss(img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    """Spectral contrastive loss (HaoChen et al. 2022).

    L = -(1/B) Σ_{pos pairs (i,j)} ⟨z_i, z_j⟩  +  (1/B²) Σ_{i,k} ⟨z_i, z_k⟩²

    Positive pairs: (img_i, txt_i).  No temperature needed.
    The quadratic repulsion term acts over all cross-modal pairs in the batch.
    """
    B = img.size(0)
    sim = img @ txt.T                           # (B, B)  all cross-modal inner products
    pos_term = sim.diagonal().mean()            # mean of matched-pair similarities
    repulsion = sim.pow(2).sum() / (B * B)      # mean squared inner product (all pairs)
    return -pos_term + repulsion


# ── Kernel functions ──────────────────────────────────────────────────────────

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma_sq: float = 0.07) -> torch.Tensor:
    """RBF kernel: exp(-‖x-y‖² / (2σ²)).

    With σ²=0.07 and unit-norm vectors this is equivalent to InfoNCE with τ=0.07
    (the constant exp(-1/σ²) cancels in the softmax).
    """
    dist_sq = torch.cdist(x, y).pow(2)
    return torch.exp(-dist_sq / (2 * sigma_sq))


def poly_kernel(x: torch.Tensor, y: torch.Tensor, degree: int = 2, c: float = 1.0) -> torch.Tensor:
    """Polynomial kernel: (x·y + c)^degree.

    For unit-norm vectors x·y ∈ [-1,1], so (x·y + 1) ∈ [0,2] — always non-negative.
    """
    return (x @ y.T + c).pow(degree)


class MixtureKernel(nn.Module):
    """Learnable convex combination of Gaussian and Polynomial kernels.

    k_M(u,v) = α·k_G(u,v) + (1-α)·k_P(u,v),  α ∈ [0,1] learnable.
    """

    def __init__(self, alpha_init: float = 0.5):
        super().__init__()
        # Store unconstrained; clamp to [0,1] in forward so α stays valid.
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.clamp(0.0, 1.0)
        return alpha * gaussian_kernel(x, y) + (1 - alpha) * poly_kernel(x, y)


# ── Kernel InfoNCE ────────────────────────────────────────────────────────────

def kernel_infonce_loss(img: torch.Tensor, txt: torch.Tensor, kernel_fn) -> torch.Tensor:
    """InfoNCE where similarity is replaced by an arbitrary kernel.

    The kernel encodes the scale, so no separate temperature is needed.
    Diagonal entries of K are positive pairs.
    """
    K = kernel_fn(img, txt)                     # (B, B)
    targets = torch.arange(K.size(0), device=K.device)
    return (F.cross_entropy(K, targets) + F.cross_entropy(K.T, targets)) / 2
