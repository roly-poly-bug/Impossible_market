from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.matrix_factorization import EvaluationData, evaluate_model
from ml.models.matrix_factorization import MatrixFactorization
from ml.representations.mf_signal import RepresentationData
from ml.training.mf_trainer import MFTrainingConfig, TrainingResult, seed_training


def confidence_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * confidence).mean()


def _loader(data: RepresentationData, config: MFTrainingConfig) -> DataLoader:
    rows = data.weighted_bce_examples
    dataset = TensorDataset(
        torch.tensor([row[0] for row in rows], dtype=torch.long),
        torch.tensor([row[1] for row in rows], dtype=torch.long),
        torch.tensor([row[2] for row in rows], dtype=torch.float32),
        torch.tensor([row[3] for row in rows], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)


def _selection(metrics: list[dict[str, object]], task: str, metric: str) -> float:
    row = next(row for row in metrics if row["task"] == task and row["k"] == 10)
    return float(row[metric])


@dataclass
class SignalTrainingResult:
    representation: str
    result: TrainingResult


def train_signal_representation(
    data: RepresentationData,
    evaluation: EvaluationData,
    config: MFTrainingConfig,
) -> SignalTrainingResult:
    seed_training(config)
    model = MatrixFactorization(
        len(data.indexed.user_ids), len(data.indexed.item_ids), config.latent_dim
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = _loader(data, config)
    best_metric = float("-inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        for users, items, targets, confidence in loader:
            optimizer.zero_grad()
            loss = confidence_weighted_bce(model(users, items), targets, confidence)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss for {data.spec.name}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _ = evaluate_model(
            model, data.indexed, evaluation, split="validation", k_values=(10,)
        )
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "validation_purchase_recall_at_10": _selection(validation, "purchase", "recall"),
            "validation_purchase_ndcg_at_10": _selection(validation, "purchase", "ndcg"),
            "validation_viewplus_ndcg_at_10": _selection(validation, "viewplus", "ndcg"),
            "validation_favoriteplus_ndcg_at_10": _selection(validation, "favoriteplus", "ndcg"),
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError(f"Non-finite training history for {data.spec.name}")
        history.append(row)
        selection = float(row["validation_purchase_ndcg_at_10"])
        if selection > best_metric + 1e-12:
            best_metric = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return SignalTrainingResult(
        data.spec.name,
        TrainingResult("bce", model, history, best_epoch, best_metric),
    )
