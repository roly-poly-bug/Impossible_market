from __future__ import annotations

import math
from pathlib import Path

import torch

from ml.evaluation.mf_latent_dim import normalized_norm, parameter_counts
from ml.experiments.mf_latent_dim_experiment import DIMENSIONS, _config_for
from ml.experiments.mf_latent_dim_export import _csv
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.training.mf_trainer import MFTrainingConfig


def test_configurable_latent_dimension_and_score_shapes():
    for dimension in DIMENSIONS:
        model = BiasedMatrixFactorization(3, 5, dimension, "item_bias")
        assert model.user_embeddings.weight.shape == (3, dimension)
        assert model.item_embeddings.weight.shape == (5, dimension)
        assert model.score_all_items(torch.tensor([0, 1])).shape == (2, 5)


def test_parameter_count_correctness():
    for dimension in DIMENSIONS:
        counts = parameter_counts(BiasedMatrixFactorization(1000, 200, dimension, "item_bias"))
        assert counts["user_embedding_parameters"] == 1000 * dimension
        assert counts["item_embedding_parameters"] == 200 * dimension
        assert counts["item_bias_parameters"] == 200
        assert counts["total_trainable_parameters"] == 1200 * dimension + 200


def test_same_training_config_except_latent_dim():
    base = MFTrainingConfig()
    configs = [_config_for(base, dimension).as_dict() for dimension in DIMENSIONS]
    for key in configs[0]:
        values = {config[key] for config in configs}
        assert values == set(DIMENSIONS) if key == "latent_dim" else len(values) == 1


def test_same_fixed_samples_are_wired_once():
    source = Path("ml/experiments/mf_latent_dim_experiment.py").read_text(encoding="utf-8")
    assert source.count("load_cart_signal(") == 1
    assert "data, evaluation" in source or "data.indexed, evaluation" in source


def test_no_future_or_hidden_dimension_leakage():
    source = Path("ml/experiments/mf_latent_dim_experiment.py").read_text(encoding="utf-8")
    assert "hidden" not in source.lower()
    representation = Path("ml/representations/mf_cart_signal.py").read_text(encoding="utf-8")
    assert "validation_relevance" not in representation and "test_relevance" not in representation


def test_validation_only_selection_and_test_guard():
    trainer = Path("ml/training/mf_bias_trainer.py").read_text(encoding="utf-8")
    experiment = Path("ml/experiments/mf_latent_dim_experiment.py").read_text(encoding="utf-8")
    assert "validation_purchase_ndcg_at_10" in trainer
    assert "SignalFinalTestEvaluator" in experiment
    assert "DIMENSIONS = (8, 16, 32, 64)" in experiment


def test_embedding_diagnostics_are_finite():
    model = BiasedMatrixFactorization(3, 5, 8, "item_bias")
    values = [float(model.user_embeddings.weight.detach().norm()), float(model.item_embeddings.weight.detach().norm())]
    assert all(math.isfinite(value) for value in values)
    assert math.isfinite(normalized_norm(values[0], 8))


def test_score_decomposition_correctness():
    model = BiasedMatrixFactorization(1, 2, 2, "item_bias")
    with torch.no_grad():
        model.user_embeddings.weight[:] = torch.tensor([[2.0, 3.0]])
        model.item_embeddings.weight[:] = torch.tensor([[4.0, 5.0], [1.0, 1.0]])
        model.item_bias.weight[:] = torch.tensor([[0.5], [-0.5]])
    personal = model.personal_component(torch.tensor([0]), torch.tensor([0])).item()
    assert personal == 23.0
    assert model(torch.tensor([0]), torch.tensor([0])).item() == personal + 0.5


def test_personalization_metrics_are_exported():
    source = Path("ml/experiments/mf_latent_dim_experiment.py").read_text(encoding="utf-8")
    for name in ("unique_purchase_top10_lists", "average_pairwise_top10_overlap", "average_cart_popularity_top10_overlap"):
        assert name in source


def test_artifact_reproducibility(tmp_path: Path):
    rows = [{"latent_dim": 8, "value": 1.0}]
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    _csv(first, rows, ("latent_dim", "value")); _csv(second, rows, ("latent_dim", "value"))
    assert first.read_bytes() == second.read_bytes()
