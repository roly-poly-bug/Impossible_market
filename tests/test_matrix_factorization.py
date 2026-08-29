from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from ml.evaluation.matrix_factorization import EvaluationData, FinalTestEvaluator, evaluate_model
from ml.experiments.mf_experiment import MFExperimentResult
from ml.experiments.mf_export import export_mf_results
from ml.models.matrix_factorization import MatrixFactorization, bce_objective, bpr_objective
from ml.training.mf_data import IndexedInteractions, load_binary_viewplus_training, sample_random_unknowns
from ml.training.mf_trainer import MFTrainingConfig, TrainingResult, load_checkpoint, save_checkpoint, train_matrix_factorization


def _indexed() -> IndexedInteractions:
    return IndexedInteractions(
        user_ids=("u1", "u2"),
        item_ids=("a", "b", "c", "d", "e", "f"),
        user_to_index={"u1": 0, "u2": 1},
        item_to_index={item: index for index, item in enumerate(("a", "b", "c", "d", "e", "f"))},
        positive_pairs=((0, 0), (1, 1)),
        unknown_items_by_user={0: (2, 3, 4, 5), 1: (0, 2, 4, 5)},
        observed_non_conversion_pairs=frozenset({(0, 1), (1, 3)}),
        cold_user_indices=frozenset(),
        cold_item_indices=frozenset({2, 3, 4, 5}),
    )


def _evaluation() -> EvaluationData:
    relevance = {
        split: {
            task: {"u1": ["c"], "u2": ["a"]}
            for task in ("purchase", "viewplus", "favoriteplus")
        }
        for split in ("validation", "test")
    }
    seen = {task: {"u1": ["a"], "u2": ["b"]} for task in ("purchase", "viewplus", "favoriteplus")}
    candidates = {task: ("a", "b", "c", "d", "e", "f") for task in ("purchase", "viewplus", "favoriteplus")}
    cart_rankings = {task: ("a", "b", "c", "d", "e", "f") for task in candidates}
    return EvaluationData(relevance, seen, candidates, cart_rankings, {item: 0.0 for item in candidates["purchase"]})


def test_mf_score_shapes_and_finite_losses() -> None:
    model = MatrixFactorization(3, 5, 16)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([1, 2, 3])
    scores = model(users, items)
    assert scores.shape == (3,)
    assert model.score_all_items(users).shape == (3, 5)
    assert torch.isfinite(bce_objective(scores, torch.tensor([1.0, 0.0, 1.0])))
    assert torch.isfinite(bpr_objective(scores, scores - 0.5))


def test_seeded_sampling_is_reproducible_and_unknown_only() -> None:
    data = _indexed()
    first = sample_random_unknowns(data, negative_ratio=4, seed=42)
    second = sample_random_unknowns(data, negative_ratio=4, seed=42)
    assert first == second
    assert len(first.triples) == len(data.positive_pairs) * 4
    positives = set(data.positive_pairs)
    for user, _, unknown in first.triples:
        assert (user, unknown) not in positives
        assert (user, unknown) not in data.observed_non_conversion_pairs
    for positive in data.positive_pairs:
        sampled = [unknown for user, item, unknown in first.triples if (user, item) == positive]
        assert len(sampled) == len(set(sampled)) == 4


def test_loader_uses_only_viewplus_state_and_ignores_ground_truth(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("user_id", "product_id", "state", "preference_match", "future_purchase"))
        writer.writeheader()
        writer.writerows(
            (
                {"user_id": "u", "product_id": "a", "state": "positive", "preference_match": 1, "future_purchase": 1},
                {"user_id": "u", "product_id": "b", "state": "observed_non_conversion", "preference_match": 0, "future_purchase": 0},
                {"user_id": "u", "product_id": "c", "state": "unknown", "preference_match": 1, "future_purchase": 1},
            )
        )
    data = load_binary_viewplus_training(path)
    assert data.positive_pairs == ((0, 0),)
    assert data.observed_non_conversion_pairs == {(0, 1)}
    assert data.unknown_items_by_user[0] == (2,)


def test_full_ranking_candidate_and_seen_exclusion() -> None:
    indexed = _indexed()
    model = MatrixFactorization(2, 6, 2)
    with torch.no_grad():
        model.user_embeddings.weight[:] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        model.item_embeddings.weight[:] = torch.tensor([[9.0, 0.0], [8.0, 0.0], [7.0, 0.0], [6.0, 0.0], [5.0, 0.0], [4.0, 0.0]])
    rows, top10 = evaluate_model(model, indexed, _evaluation(), split="validation", k_values=(1,))
    assert top10["u1"][0] == "b"
    assert "a" not in top10["u1"]
    assert {row["eligible_users"] for row in rows} == {2}


def test_final_test_evaluator_can_only_be_used_once() -> None:
    evaluator = FinalTestEvaluator()
    model = MatrixFactorization(2, 6, 2)
    evaluator.evaluate({"bce": model}, _indexed(), _evaluation())
    with pytest.raises(RuntimeError, match="only once"):
        evaluator.evaluate({"bce": model}, _indexed(), _evaluation())


def test_checkpoint_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = iter((0.1, 0.2, 0.15))

    def fake_evaluate(*args, **kwargs):
        ndcg = next(sequence)
        rows = [
            {"task": task, "k": 10, "recall": ndcg, "ndcg": ndcg}
            for task in ("purchase", "viewplus", "favoriteplus")
        ]
        return rows, {}

    monkeypatch.setattr("ml.training.mf_trainer.evaluate_model", fake_evaluate)
    data = _indexed()
    sampled = sample_random_unknowns(data, negative_ratio=4, seed=42)
    result = train_matrix_factorization(
        "bpr",
        data,
        sampled,
        _evaluation(),
        MFTrainingConfig(latent_dim=2, batch_size=8, max_epochs=3, patience=2),
    )
    assert result.best_epoch == 2
    path = tmp_path / "model.pt"
    save_checkpoint(result, path)
    restored = load_checkpoint(path, user_count=2, item_count=6, latent_dim=2)
    users = torch.tensor([0, 1])
    assert torch.equal(result.model.score_all_items(users), restored.score_all_items(users))


def test_default_v1_config_is_fixed() -> None:
    config = MFTrainingConfig()
    assert config.latent_dim == 16
    assert config.negative_ratio == 4
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.batch_size == 1024
    assert config.max_epochs == 100
    assert config.patience == 5
    assert config.seed == 42


def test_experiment_artifacts_are_reproducible(tmp_path: Path) -> None:
    indexed = _indexed()
    sampled = sample_random_unknowns(indexed, negative_ratio=4, seed=42)
    model = MatrixFactorization(2, 6, 2)
    history = [{
        "epoch": 1,
        "train_loss": 0.5,
        "validation_purchase_recall_at_10": 0.1,
        "validation_purchase_ndcg_at_10": 0.1,
        "validation_viewplus_ndcg_at_10": 0.1,
        "validation_favoriteplus_ndcg_at_10": 0.1,
    }]
    training = {
        name: TrainingResult(name, model, history, 1, 0.1)
        for name in ("bce", "bpr")
    }
    metric_rows = [
        {
            "task": task,
            "split": split,
            "k": k,
            "eligible_users": 1,
            "recall": 0.1,
            "ndcg": 0.1,
            "hit_rate": 0.1,
            "precision": 0.01,
        }
        for split in ("validation", "test")
        for task in ("purchase", "viewplus", "favoriteplus")
        for k in (5, 10, 20)
    ]
    diagnostics = {
        name: {"all_values_finite": True} for name in ("bce", "bpr")
    }
    comparison = [
        {
            "model": model_name,
            "k": 10,
            "eligible_users": 1,
            "recall": 0.1,
            "ndcg": 0.1,
            "hit_rate": 0.1,
            "precision": 0.01,
        }
        for model_name in ("cart_popularity", "bce_mf", "bpr_mf")
    ]
    result = MFExperimentResult(
        MFTrainingConfig(latent_dim=2),
        indexed,
        sampled,
        training,
        {name: metric_rows for name in ("bce", "bpr")},
        diagnostics,
        comparison,
    )
    dataset = tmp_path / "dataset"
    popularity = tmp_path / "popularity"
    dataset.mkdir()
    popularity.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    (popularity / "manifest.json").write_text("{}\n", encoding="utf-8")
    first = export_mf_results(result, tmp_path / "first", dataset_dir=dataset, popularity_dir=popularity)
    second = export_mf_results(result, tmp_path / "second", dataset_dir=dataset, popularity_dir=popularity)
    assert first == second
    for path in ("comparison.csv", "bce/metrics.csv", "bpr/training_history.csv"):
        assert (tmp_path / "first" / path).read_bytes() == (tmp_path / "second" / path).read_bytes()
