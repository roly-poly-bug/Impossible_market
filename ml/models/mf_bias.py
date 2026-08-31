from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


BiasType = Literal["item_bias", "user_item_bias"]


class BiasedMatrixFactorization(nn.Module):
    def __init__(self, user_count: int, item_count: int, latent_dim: int, bias_type: BiasType) -> None:
        super().__init__()
        self.bias_type = bias_type
        self.user_embeddings = nn.Embedding(user_count, latent_dim)
        self.item_embeddings = nn.Embedding(item_count, latent_dim)
        self.item_bias = nn.Embedding(item_count, 1)
        self.user_bias = nn.Embedding(user_count, 1) if bias_type == "user_item_bias" else None
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)
        nn.init.zeros_(self.item_bias.weight)
        if self.user_bias is not None:
            nn.init.zeros_(self.user_bias.weight)

    def personal_component(self, users: Tensor, items: Tensor) -> Tensor:
        return (self.user_embeddings(users) * self.item_embeddings(items)).sum(dim=-1)

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        score = self.personal_component(users, items) + self.item_bias(items).squeeze(-1)
        if self.user_bias is not None:
            score = score + self.user_bias(users).squeeze(-1)
        return score

    def score_all_items(self, users: Tensor) -> Tensor:
        score = self.user_embeddings(users) @ self.item_embeddings.weight.T
        score = score + self.item_bias.weight.squeeze(-1)
        if self.user_bias is not None:
            score = score + self.user_bias(users)
        return score
