import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseHead(nn.Module):
    """Linear → ReLU projection from CLIP embed_dim to sparse_dim.

    Returns (z_sparse, z_norm) where z_norm is L2-normalised for loss computation.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor):
        z = F.relu(self.proj(x))            # (B, out_dim) — sparse activations
        z_norm = F.normalize(z, dim=-1)     # (B, out_dim) — unit-norm for cosine losses
        return z, z_norm
