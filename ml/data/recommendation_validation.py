from __future__ import annotations

from collections import Counter

from ml.data.recommendation_config import SPLIT_WINDOWS, STATES, TASKS
from ml.data.recommendation_dataset import (
    EVENT_TYPES,
    InteractionFacts,
    RecommendationDatasetBundle,
    all_user_item_pairs,
    derive_task_state,
    positive_event_types,
)


class RecommendationDatasetValidationError(ValueError):
    """Raised when the observed-fact dataset violates its temporal contract."""


def validate_dataset_bundle(bundle: RecommendationDatasetBundle) -> None:
    errors = []
    world = bundle.world
    if len(world.user_ids) != 1000 or len(world.products) != 200:
        errors.append("frozen world must contain 1,000 users and 200 products")

    split_membership = Counter()
    for window in SPLIT_WINDOWS:
        split_events = bundle.split_events[window.name]
        for event in split_events:
            split_membership[event.event_id] += 1
            if not window.start <= event.timestamp < window.end_exclusive:
                errors.append(f"{event.event_id}: outside {window.name} boundary")
        expected_counts = Counter(event.event_type for event in split_events)
        fact_counts = Counter()
        for facts in bundle.split_facts[window.name].values():
            fact_counts.update(
                {
                    "impression": facts.impression_count,
                    "view": facts.view_count,
                    "favorite": facts.favorite_count,
                    "add_to_cart": facts.cart_count,
                    "purchase": facts.purchase_count,
                }
            )
            if facts.first_interaction_at is None or facts.last_interaction_at is None:
                errors.append(f"{window.name}: observed facts lack timestamps")
            elif not (
                window.start
                <= facts.first_interaction_at
                <= facts.last_interaction_at
                < window.end_exclusive
            ):
                errors.append(f"{window.name}: first/last timestamp leakage")
        if expected_counts != fact_counts:
            errors.append(f"{window.name}: aggregated Event counts do not reconcile")

    expected_event_ids = {event.event_id for event in world.events}
    if set(split_membership) != expected_event_ids:
        errors.append("some Events are missing from temporal splits")
    if any(count != 1 for count in split_membership.values()):
        errors.append("an Event belongs to more than one temporal split")

    full_counts = Counter(event.event_type for event in world.events)
    aggregated_counts = Counter()
    for facts in bundle.full_facts.values():
        aggregated_counts.update(
            {
                "impression": facts.impression_count,
                "view": facts.view_count,
                "favorite": facts.favorite_count,
                "add_to_cart": facts.cart_count,
                "purchase": facts.purchase_count,
            }
        )
        if (
            facts.first_interaction_at is None
            or facts.last_interaction_at is None
            or facts.first_interaction_at > facts.last_interaction_at
        ):
            errors.append("full master first/last timestamps are inconsistent")
    if full_counts != aggregated_counts or set(full_counts) != set(EVENT_TYPES):
        errors.append("full master counts do not reconcile with raw Events")

    empty = InteractionFacts()
    pair_count = 0
    for pair in all_user_item_pairs(world.user_ids, world.products):
        pair_count += 1
        train_facts = bundle.split_facts["train"].get(pair, empty)
        for task in TASKS:
            state = derive_task_state(task, train_facts)
            flags = (
                state.is_positive,
                state.is_observed_non_conversion,
                state.is_unknown,
            )
            if sum(flags) != 1 or state.state not in STATES:
                errors.append(f"{task}: three-state flags are not mutually exclusive")
    if pair_count != 200_000:
        errors.append("user-item universe must contain exactly 200,000 pairs")

    for split in ("validation", "test"):
        split_events = bundle.split_events[split]
        for task in TASKS:
            expected = Counter()
            relevant_types = positive_event_types(task)
            for event in split_events:
                if event.event_type in relevant_types:
                    expected[(event.user_id, event.product_id)] = 1
            actual = {
                (user_id, product_id)
                for user_id, products in bundle.relevance[split][task].items()
                for product_id in products
            }
            if actual != set(expected):
                errors.append(f"{split}/{task}: relevance set mismatch")

    purchase_candidates = set(
        bundle.all_item_candidates["purchase"]["product_ids"]
    )
    available = {
        product.product_id for product in world.products if product.status == "available"
    }
    if purchase_candidates != available:
        errors.append("Purchase candidate set must contain available products only")

    if errors:
        displayed = errors[:50]
        remaining = len(errors) - len(displayed)
        suffix = f"\n- ... and {remaining} more" if remaining else ""
        raise RecommendationDatasetValidationError(
            "Recommendation dataset validation failed:\n- "
            + "\n- ".join(displayed)
            + suffix
        )
