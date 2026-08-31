from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from ml.evaluation.matrix_factorization import EvaluationData
from ml.evaluation.mf_signal import SignalFinalTestEvaluator, purchase_cold_start_diagnostic
from ml.representations.mf_signal import (
    BINARY_VIEW,
    FAVORITEPLUS,
    LOG_VIEW,
    PURCHASE_ONLY,
    WEIGHTED,
    load_representation,
    positive_confidence,
    weighted_strength,
)
from ml.training.mf_signal_trainer import confidence_weighted_bce, train_signal_representation
from ml.training.mf_trainer import MFTrainingConfig


FIELDS = (
    "user_id", "product_id", "task", "state", "is_positive",
    "is_observed_non_conversion", "is_unknown", "impression_count", "view_count",
    "favorite_count", "cart_count", "purchase_count", "first_interaction_at",
    "last_interaction_at", "future_purchase", "hidden_preference",
)


def _write_task(path: Path, task: str, positive_product: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for user in ("u1", "u2"):
            for index, product in enumerate(("a", "b", "c", "d", "e", "f")):
                if product == positive_product:
                    state = "positive"
                    view, favorite, cart, purchase = 3, int(task == "favoriteplus"), 0, int(task == "purchase")
                elif product == "b":
                    state = "observed_non_conversion"
                    view, favorite, cart, purchase = 0, 0, 0, 0
                else:
                    state = "unknown"
                    view, favorite, cart, purchase = 0, 0, 0, 0
                writer.writerow(
                    {
                        "user_id": user, "product_id": product, "task": task, "state": state,
                        "is_positive": int(state == "positive"),
                        "is_observed_non_conversion": int(state == "observed_non_conversion"),
                        "is_unknown": int(state == "unknown"), "impression_count": int(state != "unknown"),
                        "view_count": view, "favorite_count": favorite, "cart_count": cart,
                        "purchase_count": purchase, "first_interaction_at": "", "last_interaction_at": "",
                        "future_purchase": 1, "hidden_preference": 999,
                    }
                )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _write_task(tmp_path / "train_viewplus.csv", "viewplus", "a")
    _write_task(tmp_path / "train_favoriteplus.csv", "favoriteplus", "c")
    _write_task(tmp_path / "train_purchase.csv", "purchase", "d")
    return tmp_path


def test_binary_view_representation(dataset: Path) -> None:
    data = load_representation(dataset, BINARY_VIEW)
    assert len(data.indexed.positive_pairs) == 2
    assert set(data.positive_confidence.values()) == {1.0}


def test_log_view_confidence_correctness() -> None:
    row = {"view_count": "3", "favorite_count": "0", "cart_count": "0", "purchase_count": "0"}
    assert positive_confidence(LOG_VIEW, row) == pytest.approx(1 + math.log1p(3))


def test_favoriteplus_and_purchase_positive_definitions(dataset: Path) -> None:
    favorite = load_representation(dataset, FAVORITEPLUS)
    purchase = load_representation(dataset, PURCHASE_ONLY)
    assert {favorite.indexed.item_ids[item] for _, item in favorite.indexed.positive_pairs} == {"c"}
    assert {purchase.indexed.item_ids[item] for _, item in purchase.indexed.positive_pairs} == {"d"}


def test_weighted_strength_and_confidence() -> None:
    row = {"view_count": "2", "favorite_count": "1", "cart_count": "1", "purchase_count": "1"}
    strength = math.log1p(2) + 3 + 5 + 8
    assert weighted_strength(row) == pytest.approx(strength)
    assert positive_confidence(WEIGHTED, row) == pytest.approx(1 + math.log1p(strength))


def test_confidence_weighted_bce_is_finite() -> None:
    loss = confidence_weighted_bce(
        torch.tensor([0.1, -0.2]), torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0])
    )
    assert torch.isfinite(loss)


def test_confidence_is_not_normalized(dataset: Path) -> None:
    data = load_representation(dataset, LOG_VIEW)
    assert data.confidence_stats["normalized"] is False
    assert data.confidence_stats["mean"] == pytest.approx(1 + math.log1p(3))


@pytest.mark.parametrize("name", [BINARY_VIEW, LOG_VIEW, FAVORITEPLUS, WEIGHTED, PURCHASE_ONLY])
def test_representation_specific_unknown_sampling_excludes_observed(dataset: Path, name: str) -> None:
    data = load_representation(dataset, name)
    positives = set(data.indexed.positive_pairs)
    assert len(data.sampled.triples) == 4 * len(positives)
    for user, _, unknown in data.sampled.triples:
        assert (user, unknown) not in positives
        assert (user, unknown) not in data.indexed.observed_non_conversion_pairs
        assert unknown in data.indexed.unknown_items_by_user[user]


def test_loader_ignores_future_and_hidden_columns(dataset: Path) -> None:
    first = load_representation(dataset, LOG_VIEW, seed=42)
    second = load_representation(dataset, LOG_VIEW, seed=42)
    assert first.sampled.triples == second.sampled.triples
    assert first.positive_confidence == second.positive_confidence


def test_coverage_calculation(dataset: Path) -> None:
    data = load_representation(dataset, FAVORITEPLUS)
    assert data.coverage["positive_pair_count"] == 2
    assert data.coverage["training_user_count"] == 2
    assert data.coverage["training_item_count"] == 1
    assert data.coverage["density"] == pytest.approx(2 / 12)


def test_training_selection_never_calls_test(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_evaluate(*args, **kwargs):
        calls.append(kwargs["split"])
        rows = [
            {"task": task, "k": 10, "recall": 0.1, "ndcg": 0.1}
            for task in ("purchase", "viewplus", "favoriteplus")
        ]
        return rows, {}

    monkeypatch.setattr("ml.training.mf_signal_trainer.evaluate_model", fake_evaluate)
    data = load_representation(dataset, LOG_VIEW)
    empty = EvaluationData({}, {}, {}, {}, {})
    result = train_signal_representation(
        data, empty, MFTrainingConfig(latent_dim=2, batch_size=8, max_epochs=2, patience=1)
    )
    assert result.result.best_epoch == 1
    assert calls == ["validation", "validation"]


def test_purchase_cold_start_fallback_groups() -> None:
    from ml.training.mf_data import IndexedInteractions

    indexed = IndexedInteractions(
        ("u1", "u2"), ("a", "b"), {"u1": 0, "u2": 1}, {"a": 0, "b": 1},
        ((0, 0),), {0: (1,), 1: (0, 1)}, frozenset(), frozenset({1}), frozenset({1}),
    )
    evaluation = EvaluationData(
        {"test": {"purchase": {"u1": ["a"], "u2": ["b"]}}}, {}, {}, {}, {}
    )
    result = purchase_cold_start_diagnostic(
        indexed, evaluation, {"u1": ["a"], "u2": ["a"]}
    )
    assert result["fallback_usage_count"] == 1
    assert result["fallback_users"]["recall_at_10"] == 0.0
    assert result["learned_users"]["recall_at_10"] == 1.0


def test_signal_final_test_evaluator_is_single_use() -> None:
    evaluator = SignalFinalTestEvaluator()
    assert evaluator.evaluate({}) == {}
    with pytest.raises(RuntimeError, match="only once"):
        evaluator.evaluate({})


def test_result_csv_artifact_is_reproducible(tmp_path: Path) -> None:
    from ml.experiments.mf_signal_export import _write_csv

    rows = [{"representation": "log_view", "count": 2}]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_csv(first, rows, ("representation", "count"))
    _write_csv(second, rows, ("representation", "count"))
    assert first.read_bytes() == second.read_bytes()
