from __future__ import annotations

import math
from pathlib import Path

import pytest

from ml.baselines.popularity import (
    SIGNAL_CART,
    SIGNAL_FAVORITEPLUS,
    SIGNAL_LOG_VIEW,
    SIGNAL_PURCHASE,
    SIGNAL_TOTAL_VIEW,
    SIGNAL_UNIQUE_VIEW_USERS,
    SIGNAL_WEIGHTED,
    TrainInteraction,
    WeightedSignalConfig,
    build_popularity_scores,
    deterministic_ranking,
    recommend_from_global_ranking,
)
from ml.evaluation.popularity import evaluate_popularity_ranking
from ml.experiments.popularity_experiment import PopularityExperimentResult
from ml.experiments.popularity_export import export_popularity_results
from ml.representations.interaction import (
    REPRESENTATION_BINARY_VIEWPLUS,
    REPRESENTATION_FAVORITEPLUS,
    REPRESENTATION_LOG_VIEW,
    REPRESENTATION_PURCHASE,
    REPRESENTATION_VIEW_COUNT,
    REPRESENTATION_WEIGHTED,
    event_signal_overlap,
    interaction_representation_stats,
    representation_value,
)


def _row(
    user_id: str,
    product_id: str,
    *,
    views: int = 0,
    favorites: int = 0,
    carts: int = 0,
    purchases: int = 0,
) -> TrainInteraction:
    return TrainInteraction(
        user_id=user_id,
        product_id=product_id,
        impression_count=int(views > 0),
        view_count=views,
        favorite_count=favorites,
        cart_count=carts,
        purchase_count=purchases,
    )


@pytest.fixture
def weighted_config() -> WeightedSignalConfig:
    return WeightedSignalConfig()


def test_popularity_score_definitions_are_correct(
    weighted_config: WeightedSignalConfig,
) -> None:
    rows = (
        _row("u1", "a", views=3, favorites=1, carts=1, purchases=1),
        _row("u2", "a", views=1),
        _row("u1", "b", views=2, carts=1),
    )
    scores = build_popularity_scores(rows, ("a", "b", "cold"), weighted_config=weighted_config)

    assert scores[SIGNAL_TOTAL_VIEW] == {"a": 4.0, "b": 2.0, "cold": 0.0}
    assert scores[SIGNAL_UNIQUE_VIEW_USERS] == {"a": 2.0, "b": 1.0, "cold": 0.0}
    assert scores[SIGNAL_LOG_VIEW]["a"] == pytest.approx(math.log1p(3) + math.log1p(1))
    assert scores[SIGNAL_FAVORITEPLUS]["a"] == 1.0
    assert scores[SIGNAL_FAVORITEPLUS]["b"] == 1.0
    assert scores[SIGNAL_CART]["a"] == 1.0
    assert scores[SIGNAL_PURCHASE]["a"] == 1.0
    assert scores[SIGNAL_WEIGHTED]["a"] == pytest.approx(
        math.log1p(3) + math.log1p(1) + 3 + 5 + 8
    )


def test_score_builder_uses_only_rows_explicitly_provided_as_train(
    weighted_config: WeightedSignalConfig,
) -> None:
    train_rows = (_row("u1", "a", views=1),)
    future_test_purchase = _row("u1", "b", purchases=1)

    scores = build_popularity_scores(
        train_rows,
        ("a", "b"),
        weighted_config=weighted_config,
    )

    assert future_test_purchase.product_id == "b"
    assert scores[SIGNAL_PURCHASE]["b"] == 0
    assert scores[SIGNAL_WEIGHTED]["b"] == 0


def test_deterministic_tie_break_candidate_filter_and_cold_item() -> None:
    scores = {"b": 2.0, "a": 2.0, "cold": 0.0, "excluded": 99.0}
    assert deterministic_ranking(scores, ("cold", "b", "a")) == (
        "a",
        "b",
        "cold",
    )


def test_exclude_seen_is_optional() -> None:
    ranking = ("a", "b", "c")
    assert recommend_from_global_ranking(
        ranking, seen_items=("a",), exclude_seen=True, k=2
    ) == ["b", "c"]
    assert recommend_from_global_ranking(
        ranking, seen_items=("a",), exclude_seen=False, k=2
    ) == ["a", "b"]


def test_metric_integration_uses_same_eligible_users_and_seen_policy() -> None:
    values = evaluate_popularity_ranking(
        ("a", "b", "c"),
        {"u1": ["b"], "u2": [], "u3": ["a"]},
        {"u1": ["a"], "u3": []},
        exclude_seen=True,
        k_values=(1,),
    )[0]
    assert values["eligible_users"] == 2
    assert values["recall"] == 1.0
    assert values["ndcg"] == 1.0
    assert values["hit_rate"] == 1.0
    assert values["precision"] == 1.0


@pytest.mark.parametrize(
    ("representation", "expected"),
    (
        (REPRESENTATION_BINARY_VIEWPLUS, 1.0),
        (REPRESENTATION_VIEW_COUNT, 3.0),
        (REPRESENTATION_LOG_VIEW, math.log1p(3)),
        (REPRESENTATION_FAVORITEPLUS, 1.0),
        (REPRESENTATION_PURCHASE, 1.0),
        (REPRESENTATION_WEIGHTED, math.log1p(3) + 3 + 5 + 8),
    ),
)
def test_interaction_representation_values(
    representation: str,
    expected: float,
    weighted_config: WeightedSignalConfig,
) -> None:
    row = _row("u1", "a", views=3, favorites=1, carts=1, purchases=1)
    assert representation_value(
        representation,
        row,
        weighted_config=weighted_config,
    ) == pytest.approx(expected)


def test_weighted_config_is_external_and_validated() -> None:
    config = WeightedSignalConfig(1, 2, 4, 8)
    assert config.weights == {"view": 1, "favorite": 2, "cart": 4, "purchase": 8}
    with pytest.raises(ValueError, match="non-negative"):
        WeightedSignalConfig(purchase_weight=-1)
    with pytest.raises(ValueError, match="log1p"):
        WeightedSignalConfig(view_transform="raw")


def test_representation_stats_use_nonzero_values_only(
    weighted_config: WeightedSignalConfig,
) -> None:
    stats = interaction_representation_stats(
        (_row("u1", "a", views=1), _row("u1", "b")),
        weighted_config=weighted_config,
    )
    view = next(row for row in stats if row["representation"] == REPRESENTATION_VIEW_COUNT)
    assert view["nonzero_pair_count"] == 1
    assert view["density"] == 0.5
    assert view["mean_nonzero"] == 1.0


def test_event_overlap_and_binary_deep_event_counts() -> None:
    overlap = event_signal_overlap(
        (
            _row("u1", "a", views=1, favorites=1, carts=1, purchases=1),
            _row("u1", "b", views=1),
        )
    )
    assert overlap["favoriteplus_outside_view"] == 0
    assert overlap["purchase_outside_view"] == 0
    assert overlap["purchase_outside_favoriteplus"] == 0
    assert overlap["max_favorite_count_per_pair"] == 1
    assert overlap["max_cart_count_per_pair"] == 1
    assert overlap["max_purchase_count_per_pair"] == 1


def test_result_artifacts_are_reproducible(tmp_path: Path) -> None:
    source_dir = tmp_path / "dataset"
    source_dir.mkdir()
    (source_dir / "manifest.json").write_text('{"dataset_version":"recommendation_dataset_v1"}\n', encoding="utf-8")
    metric = {
        "popularity_signal": "total_view_count",
        "evaluation_task": "viewplus",
        "split": "test",
        "exclude_seen": True,
        "k": 10,
        "eligible_users": 1,
        "recall": 1.0,
        "ndcg": 1.0,
        "hit_rate": 1.0,
        "precision": 0.1,
    }
    representation = {
        "representation": "binary_viewplus",
        "nonzero_pair_count": 1,
        "density": 0.5,
        "mean_nonzero": 1.0,
        "std_nonzero": 0.0,
        "min_nonzero": 1.0,
        "p25_nonzero": 1.0,
        "median_nonzero": 1.0,
        "p75_nonzero": 1.0,
        "max_nonzero": 1.0,
    }
    richness = {
        "signal": "view",
        "popularity_signal": "total_view_count",
        "nonzero_pair_count": 1,
        "user_coverage": 1,
        "item_coverage": 1,
        "density": 0.5,
        "test_purchase_recall_at_10": 1.0,
        "test_purchase_ndcg_at_10": 1.0,
    }
    stability = {
        "popularity_signal": "total_view_count",
        "evaluation_task": "viewplus",
        "validation_recall_at_10": 1.0,
        "test_recall_at_10": 1.0,
        "recall_difference_test_minus_validation": 0.0,
        "validation_ndcg_at_10": 1.0,
        "test_ndcg_at_10": 1.0,
        "ndcg_difference_test_minus_validation": 0.0,
    }
    result = PopularityExperimentResult(
        dataset_manifest={"dataset_version": "recommendation_dataset_v1"},
        weighted_config=WeightedSignalConfig(),
        metrics=(metric,),
        purchase_cross_signal_metrics=(metric,),
        representation_stats=(representation,),
        signal_richness=(richness,),
        stability=(stability,),
        heavy_user={"top_1_percent_raw_share": 1.0},
        signal_overlap={"view_pairs": 1},
    )
    first = export_popularity_results(
        result,
        tmp_path / "first",
        dataset_dir=source_dir,
        exclude_seen=True,
    )
    second = export_popularity_results(
        result,
        tmp_path / "second",
        dataset_dir=source_dir,
        exclude_seen=True,
    )
    assert first == second
    for artifact in first["artifacts"]:
        assert (tmp_path / "first" / artifact).read_bytes() == (
            tmp_path / "second" / artifact
        ).read_bytes()
