from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.matrix_factorization import EvaluationData, evaluate_model
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.representations.mf_signal import RepresentationData
from ml.training.mf_signal_trainer import _selection
from ml.training.mf_trainer import MFTrainingConfig, seed_training


def weighted_bpr_loss(positive_scores, comparison_scores, confidence):
    return (-F.logsigmoid(positive_scores - comparison_scores) * confidence).mean()


def _loader(data: RepresentationData, config: MFTrainingConfig):
    rows = data.sampled.triples
    confidence = data.positive_confidence
    dataset = TensorDataset(
        torch.tensor([row[0] for row in rows], dtype=torch.long),
        torch.tensor([row[1] for row in rows], dtype=torch.long),
        torch.tensor([row[2] for row in rows], dtype=torch.long),
        torch.tensor([confidence[(row[0], row[1])] for row in rows], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)


@dataclass
class ObjectiveTrainingResult:
    objective: str
    model: BiasedMatrixFactorization
    history: list[dict[str, object]]
    best_epoch: int
    best_validation_purchase_ndcg_at_10: float


def train_weighted_bpr(data: RepresentationData, evaluation: EvaluationData, config: MFTrainingConfig):
    seed_training(config)
    model = BiasedMatrixFactorization(len(data.indexed.user_ids), len(data.indexed.item_ids), config.latent_dim, "item_bias")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = _loader(data, config)
    best_metric, best_epoch, best_state, stale = float("-inf"), 0, None, 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train(); losses = []
        for users, positives, comparisons, confidence in loader:
            optimizer.zero_grad()
            loss = weighted_bpr_loss(model(users, positives), model(users, comparisons), confidence)
            if not torch.isfinite(loss): raise ValueError("Non-finite weighted BPR loss")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        validation, _ = evaluate_model(model, data.indexed, evaluation, split="validation", k_values=(10,))
        row = {"epoch": epoch, "train_loss": sum(losses)/len(losses), "validation_purchase_recall_at_10": _selection(validation,"purchase","recall"), "validation_purchase_ndcg_at_10": _selection(validation,"purchase","ndcg"), "validation_viewplus_ndcg_at_10": _selection(validation,"viewplus","ndcg"), "validation_favoriteplus_ndcg_at_10": _selection(validation,"favoriteplus","ndcg")}
        if not all(math.isfinite(float(value)) for value in row.values()): raise ValueError("Non-finite BPR history")
        history.append(row); metric = float(row["validation_purchase_ndcg_at_10"])
        if metric > best_metric + 1e-12:
            best_metric, best_epoch, best_state, stale = metric, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience: break
    if best_state is None: raise RuntimeError("BPR training produced no checkpoint")
    model.load_state_dict(best_state)
    return ObjectiveTrainingResult("bpr", model, history, best_epoch, best_metric)
