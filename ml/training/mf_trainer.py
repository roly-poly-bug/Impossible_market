from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.matrix_factorization import EvaluationData, evaluate_model
from ml.models.matrix_factorization import MatrixFactorization, bce_objective, bpr_objective
from ml.training.mf_data import IndexedInteractions, SampledTrainingData


ModelType = Literal["bce", "bpr"]


@dataclass(frozen=True)
class MFTrainingConfig:
    latent_dim: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 100
    patience: int = 5
    negative_ratio: int = 4
    seed: int = 42
    deterministic_algorithms: bool = True
    torch_num_threads: int = 1

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrainingResult:
    model_type: ModelType
    model: MatrixFactorization
    history: list[dict[str, object]]
    best_epoch: int
    best_validation_purchase_ndcg_at_10: float


def seed_training(config: MFTrainingConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_num_threads)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)


def _loader(
    model_type: ModelType,
    sampled: SampledTrainingData,
    config: MFTrainingConfig,
) -> DataLoader:
    if model_type == "bce":
        rows = sampled.bce_examples
        dataset = TensorDataset(
            torch.tensor([row[0] for row in rows], dtype=torch.long),
            torch.tensor([row[1] for row in rows], dtype=torch.long),
            torch.tensor([row[2] for row in rows], dtype=torch.float32),
        )
    else:
        rows = sampled.triples
        dataset = TensorDataset(
            torch.tensor([row[0] for row in rows], dtype=torch.long),
            torch.tensor([row[1] for row in rows], dtype=torch.long),
            torch.tensor([row[2] for row in rows], dtype=torch.long),
        )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)


def _selection_values(metrics: list[dict[str, object]]) -> dict[str, float]:
    def value(task: str, metric: str) -> float:
        row = next(row for row in metrics if row["task"] == task and row["k"] == 10)
        return float(row[metric])

    return {
        "validation_purchase_recall_at_10": value("purchase", "recall"),
        "validation_purchase_ndcg_at_10": value("purchase", "ndcg"),
        "validation_viewplus_ndcg_at_10": value("viewplus", "ndcg"),
        "validation_favoriteplus_ndcg_at_10": value("favoriteplus", "ndcg"),
    }


def train_matrix_factorization(
    model_type: ModelType,
    indexed: IndexedInteractions,
    sampled: SampledTrainingData,
    evaluation: EvaluationData,
    config: MFTrainingConfig,
) -> TrainingResult:
    seed_training(config)
    model = MatrixFactorization(
        len(indexed.user_ids), len(indexed.item_ids), config.latent_dim
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = _loader(model_type, sampled, config)
    best_metric = float("-inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            if model_type == "bce":
                users, items, targets = batch
                loss = bce_objective(model(users, items), targets)
            else:
                users, positive_items, unknown_items = batch
                loss = bpr_objective(
                    model(users, positive_items), model(users, unknown_items)
                )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_metrics, _ = evaluate_model(
            model,
            indexed,
            evaluation,
            split="validation",
            k_values=(10,),
        )
        values = _selection_values(validation_metrics)
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                **values,
            }
        )
        selection = values["validation_purchase_ndcg_at_10"]
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
    return TrainingResult(model_type, model, history, best_epoch, best_metric)


def save_checkpoint(result: TrainingResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": result.model_type,
            "best_epoch": result.best_epoch,
            "state_dict": result.model.state_dict(),
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    *,
    user_count: int,
    item_count: int,
    latent_dim: int,
) -> MatrixFactorization:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = MatrixFactorization(user_count, item_count, latent_dim)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
