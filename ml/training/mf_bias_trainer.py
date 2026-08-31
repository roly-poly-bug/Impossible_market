from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from ml.evaluation.matrix_factorization import EvaluationData, evaluate_model
from ml.models.mf_bias import BiasType, BiasedMatrixFactorization
from ml.representations.mf_signal import RepresentationData
from ml.training.mf_signal_trainer import _loader, _selection, confidence_weighted_bce
from ml.training.mf_trainer import MFTrainingConfig, seed_training


@dataclass
class BiasTrainingResult:
    model_type: str
    model: BiasedMatrixFactorization
    history: list[dict[str, object]]
    best_epoch: int
    best_validation_purchase_ndcg_at_10: float


def train_bias_model(
    bias_type: BiasType,
    data: RepresentationData,
    evaluation: EvaluationData,
    config: MFTrainingConfig,
) -> BiasTrainingResult:
    seed_training(config)
    model = BiasedMatrixFactorization(len(data.indexed.user_ids), len(data.indexed.item_ids), config.latent_dim, bias_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = _loader(data, config)
    best_metric, best_epoch, best_state, stale = float("-inf"), 0, None, 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for users, items, targets, confidence in loader:
            optimizer.zero_grad()
            loss = confidence_weighted_bce(model(users, items), targets, confidence)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss for {bias_type}")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        validation, _ = evaluate_model(model, data.indexed, evaluation, split="validation", k_values=(10,))
        row = {
            "epoch": epoch, "train_loss": sum(losses) / len(losses),
            "validation_purchase_recall_at_10": _selection(validation, "purchase", "recall"),
            "validation_purchase_ndcg_at_10": _selection(validation, "purchase", "ndcg"),
            "validation_viewplus_ndcg_at_10": _selection(validation, "viewplus", "ndcg"),
            "validation_favoriteplus_ndcg_at_10": _selection(validation, "favoriteplus", "ndcg"),
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("Non-finite training history")
        history.append(row)
        metric = float(row["validation_purchase_ndcg_at_10"])
        if metric > best_metric + 1e-12:
            best_metric, best_epoch, best_state, stale = metric, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience: break
    if best_state is None: raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return BiasTrainingResult(bias_type, model, history, best_epoch, best_metric)
