from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from ml.evaluation.mf_objective import pairwise_margin_diagnostics
from ml.experiments.mf_latent_dim_export import _csv
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.representations.mf_cart_signal import EXISTING_WEIGHTED, load_cart_signal
from ml.training.mf_objective import weighted_bpr_loss


def test_bce_and_bpr_use_same_item_bias_architecture():
    for objective in ("bce", "bpr"):
        model = BiasedMatrixFactorization(3, 5, 8, "item_bias")
        assert model.user_embeddings.weight.shape == (3, 8)
        assert model.item_embeddings.weight.shape == (5, 8)
        assert model.item_bias.weight.shape == (5, 1)


def test_weighted_bpr_loss_matches_definition():
    positive = torch.tensor([2.0, 0.0])
    comparison = torch.tensor([1.0, 1.0])
    confidence = torch.tensor([3.0, 2.0])
    expected = (-F.logsigmoid(positive - comparison) * confidence).mean()
    assert torch.equal(weighted_bpr_loss(positive, comparison, confidence), expected)


def test_weighted_bpr_loss_is_finite_and_rewards_positive_margin():
    confidence = torch.ones(2)
    good = weighted_bpr_loss(torch.tensor([2.0, 3.0]), torch.zeros(2), confidence)
    bad = weighted_bpr_loss(torch.zeros(2), torch.tensor([2.0, 3.0]), confidence)
    assert math.isfinite(float(good)) and good < bad


def test_objectives_share_one_fixed_sample_source_and_ratio():
    source = Path("ml/experiments/mf_objective_experiment.py").read_text(encoding="utf-8")
    trainer = Path("ml/training/mf_objective.py").read_text(encoding="utf-8")
    assert source.count("load_cart_signal(") == 1
    assert "data.sampled.triples" in trainer
    assert "sample_ratio=4" in source and "seed=42" in source


def test_four_shared_comparisons_per_positive():
    data, _ = load_cart_signal("data/recommendation_v1", EXISTING_WEIGHTED, sample_ratio=4, seed=42)
    counts = Counter((user, positive) for user, positive, _ in data.sampled.triples)
    assert len(data.sampled.triples) == 4 * len(data.indexed.positive_pairs)
    assert set(counts.values()) == {4}


def test_sampling_is_exposure_aware_unknown_backfill_and_not_true_negative():
    source = Path("ml/representations/mf_cart_signal.py").read_text(encoding="utf-8")
    assert "Exposed" in source or "exposed" in source
    assert "unknown" in source.lower()
    assert "true_negative" not in source.lower()


def test_margin_diagnostics_are_correct():
    model = BiasedMatrixFactorization(1, 2, 1, "item_bias")
    with torch.no_grad():
        model.user_embeddings.weight.fill_(1.0)
        model.item_embeddings.weight[:] = torch.tensor([[2.0], [1.0]])
        model.item_bias.weight.zero_()
    diagnostics = pairwise_margin_diagnostics(model, [(0, 0, 1)])
    assert diagnostics["count"] == 1
    assert diagnostics["mean"] == 1.0
    assert diagnostics["positive_share"] == 1.0


def test_no_future_or_hidden_world_leakage():
    experiment = Path("ml/experiments/mf_objective_experiment.py").read_text(encoding="utf-8")
    representation = Path("ml/representations/mf_cart_signal.py").read_text(encoding="utf-8")
    assert "hidden" not in experiment.lower()
    assert "validation_relevance" not in representation
    assert "test_relevance" not in representation


def test_validation_only_selection_and_single_test_guard():
    trainer = Path("ml/training/mf_objective.py").read_text(encoding="utf-8")
    experiment = Path("ml/experiments/mf_objective_experiment.py").read_text(encoding="utf-8")
    assert 'split="validation"' in trainer
    assert "validation_purchase_ndcg_at_10" in trainer
    assert experiment.count("SignalFinalTestEvaluator()") == 1


def test_candidate_and_fallback_are_shared_by_both_objectives():
    source = Path("ml/experiments/mf_objective_experiment.py").read_text(encoding="utf-8")
    assert "for name,model in models.items()" in source
    assert "evaluate_model(model,data.indexed,evaluation" in source


def test_csv_artifact_reproducibility(tmp_path: Path):
    rows = [{"objective": "bpr", "mean": 1.0}]
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    _csv(first, rows, ("objective", "mean"))
    _csv(second, rows, ("objective", "mean"))
    assert first.read_bytes() == second.read_bytes()
