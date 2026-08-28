from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from ml.data.recommendation_config import (
    ENGAGEMENT_VERSION,
    PRODUCT_VERSION,
    SESSION_EVENT_VERSION,
    STATE_OBSERVED_NON_CONVERSION,
    STATE_POSITIVE,
    STATE_UNKNOWN,
    TASK_FAVORITEPLUS,
    TASK_PURCHASE,
    TASK_VIEWPLUS,
    TASKS,
    USER_VERSION,
    SPLIT_WINDOWS,
    SplitWindow,
)


EVENT_IMPRESSION = "impression"
EVENT_VIEW = "view"
EVENT_FAVORITE = "favorite"
EVENT_CART = "add_to_cart"
EVENT_PURCHASE = "purchase"
EVENT_TYPES = (
    EVENT_IMPRESSION,
    EVENT_VIEW,
    EVENT_FAVORITE,
    EVENT_CART,
    EVENT_PURCHASE,
)


@dataclass(frozen=True)
class ProductSnapshot:
    product_id: str
    status: str


@dataclass(frozen=True)
class ObservedEvent:
    event_id: str
    session_id: str
    user_id: str
    product_id: str
    event_type: str
    timestamp: datetime
    source_version: str
    source_seed: int


@dataclass(frozen=True)
class FrozenObservedWorld:
    user_ids: tuple[str, ...]
    products: tuple[ProductSnapshot, ...]
    events: tuple[ObservedEvent, ...]


@dataclass(frozen=True)
class RecommendationDatasetBundle:
    world: FrozenObservedWorld
    full_facts: dict[tuple[str, str], "InteractionFacts"]
    split_events: dict[str, tuple[ObservedEvent, ...]]
    split_facts: dict[str, dict[tuple[str, str], "InteractionFacts"]]
    relevance: dict[str, dict[str, dict[str, list[str]]]]
    train_seen_items: dict[str, dict[str, list[str]]]
    impressed_candidates: dict[str, dict[str, list[str]]]
    all_item_candidates: dict[str, dict[str, object]]


@dataclass
class InteractionFacts:
    impression_count: int = 0
    view_count: int = 0
    favorite_count: int = 0
    cart_count: int = 0
    purchase_count: int = 0
    first_impression_at: datetime | None = None
    first_view_at: datetime | None = None
    first_favorite_at: datetime | None = None
    first_cart_at: datetime | None = None
    first_purchase_at: datetime | None = None
    last_impression_at: datetime | None = None
    last_view_at: datetime | None = None
    last_favorite_at: datetime | None = None
    last_cart_at: datetime | None = None
    last_purchase_at: datetime | None = None
    first_interaction_at: datetime | None = None
    last_interaction_at: datetime | None = None

    def add(self, event: ObservedEvent) -> None:
        count_name = {
            EVENT_IMPRESSION: "impression_count",
            EVENT_VIEW: "view_count",
            EVENT_FAVORITE: "favorite_count",
            EVENT_CART: "cart_count",
            EVENT_PURCHASE: "purchase_count",
        }[event.event_type]
        first_name = {
            EVENT_IMPRESSION: "first_impression_at",
            EVENT_VIEW: "first_view_at",
            EVENT_FAVORITE: "first_favorite_at",
            EVENT_CART: "first_cart_at",
            EVENT_PURCHASE: "first_purchase_at",
        }[event.event_type]
        last_name = first_name.replace("first_", "last_")
        setattr(self, count_name, getattr(self, count_name) + 1)
        if getattr(self, first_name) is None:
            setattr(self, first_name, event.timestamp)
        setattr(self, last_name, event.timestamp)
        if self.first_interaction_at is None:
            self.first_interaction_at = event.timestamp
        self.last_interaction_at = event.timestamp

    @property
    def was_impressed(self) -> bool:
        return self.impression_count > 0

    @property
    def was_viewed(self) -> bool:
        return self.view_count > 0

    @property
    def was_favorited(self) -> bool:
        return self.favorite_count > 0

    @property
    def was_carted(self) -> bool:
        return self.cart_count > 0

    @property
    def was_purchased(self) -> bool:
        return self.purchase_count > 0


@dataclass(frozen=True)
class TaskState:
    task: str
    state: str
    is_positive: bool
    is_observed_non_conversion: bool
    is_unknown: bool


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _load_users(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if any(
        row["user_generation_version"] != USER_VERSION
        or int(row["generation_seed"]) != 42
        for row in rows
    ):
        raise ValueError("User snapshot provenance does not match the frozen world")
    user_ids = tuple(sorted(row["user_id"] for row in rows))
    if len(user_ids) != 1000 or len(set(user_ids)) != len(user_ids):
        raise ValueError("Expected exactly 1,000 unique frozen users")
    return user_ids


def _load_products(path: Path) -> tuple[ProductSnapshot, ...]:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if any(
        row["catalog_version"] != PRODUCT_VERSION or int(row["generation_seed"]) != 42
        for row in rows
    ):
        raise ValueError("Product snapshot provenance does not match the frozen world")
    products = tuple(
        sorted(
            (ProductSnapshot(row["name"], row["status"]) for row in rows),
            key=lambda product: product.product_id,
        )
    )
    if len(products) != 200 or len({product.product_id for product in products}) != 200:
        raise ValueError("Expected exactly 200 unique frozen products")
    return products


def _load_events(path: Path, *, expected_version: str) -> list[ObservedEvent]:
    events = []
    with path.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            if row["generation_version"] != expected_version:
                raise ValueError(f"Unexpected event version in {path}")
            seed = int(row["generation_seed"])
            if seed != 42:
                raise ValueError(f"Unexpected event seed in {path}")
            events.append(
                ObservedEvent(
                    event_id=row["event_id"],
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                    product_id=row["product_id"],
                    event_type=row["event_type"],
                    timestamp=_parse_timestamp(row["timestamp"]),
                    source_version=expected_version,
                    source_seed=seed,
                )
            )
    return events


def load_frozen_observed_world(
    *,
    product_path: str | Path,
    user_path: str | Path,
    interaction_path: str | Path,
    engagement_path: str | Path,
) -> FrozenObservedWorld:
    user_ids = _load_users(Path(user_path))
    products = _load_products(Path(product_path))
    events = [
        *_load_events(Path(interaction_path), expected_version=SESSION_EVENT_VERSION),
        *_load_events(Path(engagement_path), expected_version=ENGAGEMENT_VERSION),
    ]
    events.sort(key=lambda event: (event.timestamp, event.event_id))
    valid_users = set(user_ids)
    valid_products = {product.product_id for product in products}
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Raw Event IDs must be unique")
    for event in events:
        if event.user_id not in valid_users or event.product_id not in valid_products:
            raise ValueError(f"Event references an unknown user/product: {event.event_id}")
        if event.event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported Event type: {event.event_type}")
    return FrozenObservedWorld(user_ids, products, tuple(events))


def events_in_window(
    events: Iterable[ObservedEvent],
    window: SplitWindow,
) -> tuple[ObservedEvent, ...]:
    return tuple(
        event
        for event in events
        if window.start <= event.timestamp < window.end_exclusive
    )


def aggregate_events(
    events: Iterable[ObservedEvent],
) -> dict[tuple[str, str], InteractionFacts]:
    facts: dict[tuple[str, str], InteractionFacts] = defaultdict(InteractionFacts)
    for event in sorted(events, key=lambda value: (value.timestamp, value.event_id)):
        facts[(event.user_id, event.product_id)].add(event)
    return dict(facts)


def all_user_item_pairs(
    user_ids: Iterable[str],
    products: Iterable[ProductSnapshot],
) -> Iterator[tuple[str, str]]:
    product_ids = tuple(product.product_id for product in products)
    for user_id in user_ids:
        for product_id in product_ids:
            yield user_id, product_id


def derive_task_state(task: str, facts: InteractionFacts) -> TaskState:
    if task == TASK_VIEWPLUS:
        positive = facts.was_viewed
        observed = facts.was_impressed and not positive
    elif task == TASK_FAVORITEPLUS:
        positive = facts.was_favorited or facts.was_carted or facts.was_purchased
        observed = facts.was_viewed and not positive
    elif task == TASK_PURCHASE:
        positive = facts.was_purchased
        observed = facts.was_viewed and not positive
    else:
        raise ValueError(f"Unknown task: {task}")
    unknown = not positive and not observed
    state = (
        STATE_POSITIVE
        if positive
        else STATE_OBSERVED_NON_CONVERSION
        if observed
        else STATE_UNKNOWN
    )
    return TaskState(task, state, positive, observed, unknown)


def positive_event_types(task: str) -> frozenset[str]:
    if task == TASK_VIEWPLUS:
        return frozenset({EVENT_VIEW})
    if task == TASK_FAVORITEPLUS:
        return frozenset({EVENT_FAVORITE, EVENT_CART, EVENT_PURCHASE})
    if task == TASK_PURCHASE:
        return frozenset({EVENT_PURCHASE})
    raise ValueError(f"Unknown task: {task}")


def relevance_sets(
    events: Iterable[ObservedEvent],
    user_ids: Iterable[str],
    task: str,
) -> dict[str, list[str]]:
    relevant_types = positive_event_types(task)
    values: dict[str, set[str]] = {user_id: set() for user_id in user_ids}
    for event in events:
        if event.event_type in relevant_types:
            values[event.user_id].add(event.product_id)
    return {user_id: sorted(items) for user_id, items in values.items()}


def positive_seen_items(
    facts: dict[tuple[str, str], InteractionFacts],
    user_ids: Iterable[str],
    task: str,
) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {user_id: [] for user_id in user_ids}
    for (user_id, product_id), pair_facts in sorted(facts.items()):
        if derive_task_state(task, pair_facts).is_positive:
            values[user_id].append(product_id)
    return values


def impressed_candidate_sets(
    events: Iterable[ObservedEvent],
    user_ids: Iterable[str],
) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {user_id: set() for user_id in user_ids}
    for event in events:
        if event.event_type == EVENT_IMPRESSION:
            values[event.user_id].add(event.product_id)
    return {user_id: sorted(items) for user_id, items in values.items()}


def all_item_candidate_sets(
    products: Iterable[ProductSnapshot],
) -> dict[str, dict[str, object]]:
    products = tuple(products)
    all_ids = [product.product_id for product in products]
    available_ids = [
        product.product_id for product in products if product.status == "available"
    ]
    return {
        TASK_VIEWPLUS: {
            "policy": "all_catalog_products",
            "excluded_statuses": [],
            "product_ids": all_ids,
        },
        TASK_FAVORITEPLUS: {
            "policy": "all_catalog_products",
            "excluded_statuses": [],
            "product_ids": all_ids,
        },
        TASK_PURCHASE: {
            "policy": "available_products_only",
            "excluded_statuses": ["coming_soon", "sold_out", "unavailable"],
            "product_ids": available_ids,
        },
    }


def build_dataset_bundle(world: FrozenObservedWorld) -> RecommendationDatasetBundle:
    split_events = {
        window.name: events_in_window(world.events, window) for window in SPLIT_WINDOWS
    }
    split_facts = {
        split: aggregate_events(events) for split, events in split_events.items()
    }
    relevance = {
        split: {
            task: relevance_sets(events, world.user_ids, task) for task in TASKS
        }
        for split, events in split_events.items()
        if split in {"validation", "test"}
    }
    train_seen_items = {
        task: positive_seen_items(split_facts["train"], world.user_ids, task)
        for task in TASKS
    }
    impressed_candidates = {
        split: impressed_candidate_sets(events, world.user_ids)
        for split, events in split_events.items()
    }
    return RecommendationDatasetBundle(
        world=world,
        full_facts=aggregate_events(world.events),
        split_events=split_events,
        split_facts=split_facts,
        relevance=relevance,
        train_seen_items=train_seen_items,
        impressed_candidates=impressed_candidates,
        all_item_candidates=all_item_candidate_sets(world.products),
    )
