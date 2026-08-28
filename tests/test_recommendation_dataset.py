from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ml.data.recommendation_config import (
    TASK_FAVORITEPLUS,
    TASK_PURCHASE,
    TASK_VIEWPLUS,
    TEST_WINDOW,
    TRAIN_WINDOW,
    VALIDATION_WINDOW,
)
from ml.data.recommendation_dataset import (
    FrozenObservedWorld,
    InteractionFacts,
    ObservedEvent,
    ProductSnapshot,
    aggregate_events,
    all_item_candidate_sets,
    build_dataset_bundle,
    derive_task_state,
    events_in_window,
    impressed_candidate_sets,
    positive_seen_items,
    relevance_sets,
)
from ml.data.recommendation_export import TASK_COLUMNS, export_recommendation_dataset
from ml.data.recommendation_validation import validate_dataset_bundle
from ml.evaluation.metrics import (
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


UTC = timezone.utc


def _event(
    number: int,
    event_type: str,
    timestamp: datetime,
    *,
    user_id: str = "user-1",
    product_id: str = "product-1",
) -> ObservedEvent:
    return ObservedEvent(
        event_id=f"event-{number}",
        session_id="session-1",
        user_id=user_id,
        product_id=product_id,
        event_type=event_type,
        timestamp=timestamp,
        source_version="test_events",
        source_seed=42,
    )


def test_master_user_item_aggregation_preserves_event_facts() -> None:
    events = (
        _event(1, "impression", datetime(2026, 1, 10, 10, tzinfo=UTC)),
        _event(2, "view", datetime(2026, 1, 10, 10, 1, tzinfo=UTC)),
        _event(3, "favorite", datetime(2026, 1, 10, 10, 2, tzinfo=UTC)),
        _event(4, "add_to_cart", datetime(2026, 1, 10, 10, 3, tzinfo=UTC)),
        _event(5, "purchase", datetime(2026, 1, 10, 10, 4, tzinfo=UTC)),
        _event(6, "impression", datetime(2026, 1, 10, 10, 5, tzinfo=UTC)),
    )

    facts = aggregate_events(events)[("user-1", "product-1")]

    assert (
        facts.impression_count,
        facts.view_count,
        facts.favorite_count,
        facts.cart_count,
        facts.purchase_count,
    ) == (2, 1, 1, 1, 1)
    assert facts.first_impression_at == events[0].timestamp
    assert facts.last_impression_at == events[-1].timestamp
    assert facts.first_interaction_at == events[0].timestamp
    assert facts.last_interaction_at == events[-1].timestamp


@pytest.mark.parametrize(
    ("task", "facts", "expected"),
    (
        (TASK_VIEWPLUS, InteractionFacts(impression_count=1), "observed_non_conversion"),
        (TASK_VIEWPLUS, InteractionFacts(view_count=1), "positive"),
        (TASK_FAVORITEPLUS, InteractionFacts(view_count=1), "observed_non_conversion"),
        (TASK_FAVORITEPLUS, InteractionFacts(cart_count=1), "positive"),
        (TASK_PURCHASE, InteractionFacts(view_count=1), "observed_non_conversion"),
        (TASK_PURCHASE, InteractionFacts(purchase_count=1), "positive"),
        (TASK_PURCHASE, InteractionFacts(), "unknown"),
    ),
)
def test_task_state_derivation_is_mutually_exclusive(
    task: str,
    facts: InteractionFacts,
    expected: str,
) -> None:
    state = derive_task_state(task, facts)
    assert state.state == expected
    assert sum(
        (state.is_positive, state.is_observed_non_conversion, state.is_unknown)
    ) == 1


def test_no_forced_negative_label_or_event_weight_column() -> None:
    assert "negative" not in TASK_COLUMNS
    assert "weight" not in TASK_COLUMNS
    assert "is_observed_non_conversion" in TASK_COLUMNS
    assert "is_unknown" in TASK_COLUMNS


def test_temporal_split_boundaries_are_half_open_and_non_overlapping() -> None:
    events = (
        _event(1, "view", datetime(2026, 1, 1, tzinfo=UTC)),
        _event(2, "view", datetime(2026, 1, 21, tzinfo=UTC)),
        _event(3, "view", datetime(2026, 1, 26, tzinfo=UTC)),
        _event(4, "view", datetime(2026, 1, 31, tzinfo=UTC)),
    )
    assert [event.event_id for event in events_in_window(events, TRAIN_WINDOW)] == [
        "event-1"
    ]
    assert [
        event.event_id for event in events_in_window(events, VALIDATION_WINDOW)
    ] == ["event-2"]
    assert [event.event_id for event in events_in_window(events, TEST_WINDOW)] == [
        "event-3"
    ]


def test_future_purchase_does_not_leak_into_train_facts() -> None:
    world = FrozenObservedWorld(
        user_ids=("user-1",),
        products=(ProductSnapshot("product-1", "available"),),
        events=(
            _event(1, "impression", datetime(2026, 1, 10, tzinfo=UTC)),
            _event(2, "view", datetime(2026, 1, 10, 0, 1, tzinfo=UTC)),
            _event(3, "purchase", datetime(2026, 1, 28, tzinfo=UTC)),
        ),
    )
    bundle = build_dataset_bundle(world)
    train = bundle.split_facts["train"][("user-1", "product-1")]

    assert train.view_count == 1
    assert train.purchase_count == 0
    assert derive_task_state(TASK_PURCHASE, train).state == "observed_non_conversion"
    assert bundle.relevance["test"][TASK_PURCHASE]["user-1"] == ["product-1"]


def test_relevance_eligibility_seen_and_impressed_sets() -> None:
    events = (
        _event(1, "impression", datetime(2026, 1, 10, tzinfo=UTC)),
        _event(2, "view", datetime(2026, 1, 10, 0, 1, tzinfo=UTC)),
        _event(3, "favorite", datetime(2026, 1, 10, 0, 2, tzinfo=UTC)),
    )
    facts = aggregate_events(events)

    relevance = relevance_sets(events, ("user-1", "user-2"), TASK_FAVORITEPLUS)
    assert relevance == {"user-1": ["product-1"], "user-2": []}
    assert sum(bool(items) for items in relevance.values()) == 1
    assert positive_seen_items(facts, ("user-1", "user-2"), TASK_VIEWPLUS) == {
        "user-1": ["product-1"],
        "user-2": [],
    }
    assert impressed_candidate_sets(events, ("user-1", "user-2")) == {
        "user-1": ["product-1"],
        "user-2": [],
    }


def test_candidate_sets_follow_task_status_policy() -> None:
    products = (
        ProductSnapshot("available", "available"),
        ProductSnapshot("soon", "coming_soon"),
        ProductSnapshot("sold", "sold_out"),
        ProductSnapshot("missing", "unavailable"),
    )
    candidates = all_item_candidate_sets(products)

    assert candidates[TASK_VIEWPLUS]["product_ids"] == [
        "available",
        "soon",
        "sold",
        "missing",
    ]
    assert candidates[TASK_FAVORITEPLUS]["product_ids"] == [
        "available",
        "soon",
        "sold",
        "missing",
    ]
    assert candidates[TASK_PURCHASE]["product_ids"] == ["available"]


def test_small_dataset_build_and_manifest_are_reproducible(tmp_path: Path) -> None:
    world = FrozenObservedWorld(
        user_ids=("user-1", "user-2"),
        products=(
            ProductSnapshot("product-1", "available"),
            ProductSnapshot("product-2", "sold_out"),
        ),
        events=(
            _event(1, "impression", datetime(2026, 1, 10, tzinfo=UTC)),
            _event(2, "view", datetime(2026, 1, 10, 0, 1, tzinfo=UTC)),
            _event(3, "purchase", datetime(2026, 1, 28, tzinfo=UTC)),
        ),
    )
    bundle_one = build_dataset_bundle(world)
    bundle_two = build_dataset_bundle(world)
    assert bundle_one == bundle_two

    source = tmp_path / "source.csv"
    source.write_text("frozen\n", encoding="utf-8")
    source_paths = {
        name: source
        for name in ("products", "users", "session_events", "engagement")
    }
    manifest_one = export_recommendation_dataset(
        bundle_one, tmp_path / "one", source_paths=source_paths
    )
    manifest_two = export_recommendation_dataset(
        bundle_two, tmp_path / "two", source_paths=source_paths
    )

    assert manifest_one == manifest_two
    assert manifest_one["event_weights"] is None
    assert "true-negative" in manifest_one["true_negative_policy"]
    for artifact in manifest_one["artifacts"]:
        assert (tmp_path / "one" / artifact).read_bytes() == (
            tmp_path / "two" / artifact
        ).read_bytes()


def test_metric_helpers_for_perfect_random_and_empty_rankings() -> None:
    relevant = {"a", "b"}
    perfect = evaluate_ranking(["a", "b", "c"], relevant, k=2)
    assert perfect == {
        "recall": 1.0,
        "ndcg": 1.0,
        "hit_rate": 1.0,
        "precision": 1.0,
    }

    assert recall_at_k(["x", "a", "y"], relevant, 2) == 0.5
    assert ndcg_at_k(["x", "a", "y"], relevant, 2) == pytest.approx(
        (1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)
    )
    assert hit_rate_at_k(["x", "a", "y"], relevant, 2) == 1.0
    assert precision_at_k(["x", "a", "y"], relevant, 2) == 0.5

    assert precision_at_k([], relevant, 5) == 0.0
    assert hit_rate_at_k([], relevant, 10) == 0.0
    assert recall_at_k([], relevant, 20) == 0.0
    assert ndcg_at_k([], relevant, 20) == 0.0
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(["a"], set(), 10)
    with pytest.raises(ValueError, match="undefined"):
        ndcg_at_k(["a"], set(), 10)


def test_frozen_snapshot_event_counts_and_full_validation() -> None:
    from ml.data.recommendation_dataset import load_frozen_observed_world

    world = load_frozen_observed_world(
        product_path="data/synthetic_product_v1_seed42.csv",
        user_path="data/synthetic_user_v1_seed42.csv",
        interaction_path="data/synthetic_event_v1_seed42.csv",
        engagement_path="data/synthetic_engagement_v1_seed42.csv",
    )
    bundle = build_dataset_bundle(world)
    validate_dataset_bundle(bundle)

    assert len(world.events) == 131_772
    assert sum(facts.impression_count for facts in bundle.full_facts.values()) == 93_710
    assert sum(facts.view_count for facts in bundle.full_facts.values()) == 30_537
    assert sum(facts.favorite_count for facts in bundle.full_facts.values()) == 4_385
    assert sum(facts.cart_count for facts in bundle.full_facts.values()) == 1_881
    assert sum(facts.purchase_count for facts in bundle.full_facts.values()) == 1_259
