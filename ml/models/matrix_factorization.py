from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MatrixFactorization(nn.Module):
    """Minimal bias-free user/item dot-product Matrix Factorization."""

    def __init__(self, user_count: int, item_count: int, latent_dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(user_count, latent_dim)
        self.item_embeddings = nn.Embedding(item_count, latent_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def forward(self, user_indices: Tensor, item_indices: Tensor) -> Tensor:
        return (self.user_embeddings(user_indices) * self.item_embeddings(item_indices)).sum(dim=-1)

    def score_all_items(self, user_indices: Tensor) -> Tensor:
        return self.user_embeddings(user_indices) @ self.item_embeddings.weight.T


def bce_objective(logits: Tensor, targets: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def bpr_objective(positive_scores: Tensor, unknown_scores: Tensor) -> Tensor:
    return -F.logsigmoid(positive_scores - unknown_scores).mean()
