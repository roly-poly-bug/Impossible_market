from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from backend.app.db.models import EventType, ProductStatus
from synthetic_data.engagement_config import (
    CART_FAVORITE_BONUS,
    CART_IMPULSIVENESS_WEIGHT,
    CART_INTERCEPT,
    CART_NOISE_STDDEV,
    CART_POPULARITY_WEIGHT,
    CART_PREFERENCE_WEIGHT,
    CART_PRICE_PENALTY,
    CART_REPEAT_WEIGHT,
    CART_DELAY_PROBABILITY,
    DEFAULT_ENGAGEMENT_SEED,
    ENGAGEMENT_GENERATION_VERSION,
    FAVORITE_DELAY_PROBABILITY,
    FAVORITE_INTERCEPT,
    FAVORITE_LUXURY_WEIGHT,
    FAVORITE_NOISE_STDDEV,
    FAVORITE_NOVELTY_WEIGHT,
    FAVORITE_PREFERENCE_WEIGHT,
    FAVORITE_RARITY_WEIGHT,
    FAVORITE_REPEAT_WEIGHT,
    MAX_DELAY_DAYS,
    PURCHASE_CART_BONUS,
    PURCHASE_DELAY_PROBABILITY,
    PURCHASE_FAVORITE_BONUS,
    PURCHASE_IMPULSIVENESS_WEIGHT,
    PURCHASE_INTERCEPT,
    PURCHASE_NOISE_STDDEV,
    PURCHASE_POPULARITY_WEIGHT,
    PURCHASE_PREFERENCE_WEIGHT,
    PURCHASE_PRICE_PENALTY,
    PURCHASE_REPEAT_WEIGHT,
)
from synthetic_data.event_generator import SyntheticEventRecord
from synthetic_data.interaction_config import INTERACTION_GENERATION_VERSION
from synthetic_data.product_generator import SyntheticProductRecord
from synthetic_data.session_generator import SyntheticSessionRecord
from synthetic_data.user_generator import SyntheticUserRecord


@dataclass(frozen=True)
class SyntheticEngagementRecord:
    event_id: str
    session_id: str
    user_id: str
    product_id: str
    product_category: str
    event_type: EventType
    timestamp: datetime
    source_view_event_id: str
    source_view_session_id: str
    conversion_timing: str
    preference_match: float
    initial_popularity: float
    price_log10: float
    over_budget: bool
    impulsiveness: float
    view_count: int
    had_favorite_before: bool
    had_cart_before: bool
    probability: float
    generation_version: str
    generation_seed: int
    upstream_generation_version: str
    upstream_generation_seed: int


@dataclass
class _UserItemState:
    view_count: int = 0
    favorite_target: int | None = None
    cart_target: int | None = None
    purchase_target: int | None = None


@dataclass(frozen=True)
class _Draft:
    target_session: SyntheticSessionRecord
    source_view: SyntheticEventRecord
    event_type: EventType
    view_count: int
    had_favorite_before: bool
    had_cart_before: bool
    probability: float
    timing_weight: float


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _price_friction(user: SyntheticUserRecord, view: SyntheticEventRecord) -> float:
    gap = max(0.0, view.price_log10 - user.budget_log10)
    return user.price_sensitivity * min(gap, 8.0)


def _favorite_probability(
    user: SyntheticUserRecord,
    product: SyntheticProductRecord,
    view: SyntheticEventRecord,
    view_count: int,
    rng: random.Random,
) -> float:
    utility = (
        FAVORITE_INTERCEPT
        + FAVORITE_PREFERENCE_WEIGHT * (view.preference_match - 0.5)
        + FAVORITE_NOVELTY_WEIGHT * (product.attributes["novelty"] - 0.5)
        + FAVORITE_RARITY_WEIGHT * (product.rarity - 0.5)
        + FAVORITE_LUXURY_WEIGHT * (product.attributes["luxury"] - 0.5)
        + FAVORITE_REPEAT_WEIGHT * math.log1p(view_count)
        + rng.gauss(0.0, FAVORITE_NOISE_STDDEV)
    )
    return _logistic(utility)


def _cart_probability(
    user: SyntheticUserRecord,
    view: SyntheticEventRecord,
    view_count: int,
    has_favorite: bool,
    rng: random.Random,
) -> float:
    utility = (
        CART_INTERCEPT
        + CART_PREFERENCE_WEIGHT * (view.preference_match - 0.5)
        + CART_IMPULSIVENESS_WEIGHT * (user.impulsiveness - 0.5)
        + CART_REPEAT_WEIGHT * math.log1p(view_count)
        + CART_FAVORITE_BONUS * has_favorite
        - CART_PRICE_PENALTY * _price_friction(user, view)
        + CART_POPULARITY_WEIGHT * (view.initial_popularity - 0.5)
        + rng.gauss(0.0, CART_NOISE_STDDEV)
    )
    return _logistic(utility)


def _purchase_probability(
    user: SyntheticUserRecord,
    view: SyntheticEventRecord,
    view_count: int,
    has_favorite: bool,
    has_cart: bool,
    rng: random.Random,
) -> float:
    utility = (
        PURCHASE_INTERCEPT
        + PURCHASE_PREFERENCE_WEIGHT * (view.preference_match - 0.5)
        + PURCHASE_IMPULSIVENESS_WEIGHT * (user.impulsiveness - 0.5)
        + PURCHASE_REPEAT_WEIGHT * math.log1p(view_count)
        + PURCHASE_FAVORITE_BONUS * has_favorite
        + PURCHASE_CART_BONUS * has_cart
        - PURCHASE_PRICE_PENALTY * _price_friction(user, view)
        + PURCHASE_POPULARITY_WEIGHT * (view.initial_popularity - 0.5)
        + rng.gauss(0.0, PURCHASE_NOISE_STDDEV)
    )
    return _logistic(utility)


def _target_session_index(
    user_sessions: list[SyntheticSessionRecord],
    current_index: int,
    minimum_index: int,
    source_timestamp: datetime,
    delay_probability: float,
    rng: random.Random,
) -> int:
    base_index = max(current_index, minimum_index)
    if base_index > current_index:
        return base_index
    future = [
        index
        for index in range(current_index + 1, len(user_sessions))
        if user_sessions[index].started_at > source_timestamp
        and user_sessions[index].started_at - source_timestamp <= timedelta(days=MAX_DELAY_DAYS)
    ]
    if future and rng.random() < delay_probability:
        weights = [1.0 / (index - current_index) for index in future]
        return rng.choices(future, weights=weights, k=1)[0]
    return current_index


def _assign_timestamps(
    drafts: list[_Draft],
    base_events: list[SyntheticEventRecord],
    seed: int,
) -> list[SyntheticEngagementRecord]:
    drafts_by_session: dict[str, list[_Draft]] = defaultdict(list)
    base_by_session: dict[str, list[SyntheticEventRecord]] = defaultdict(list)
    for draft in drafts:
        drafts_by_session[draft.target_session.session_id].append(draft)
    for event in base_events:
        base_by_session[event.session_id].append(event)

    stage_order = {
        EventType.FAVORITE: 0,
        EventType.ADD_TO_CART: 1,
        EventType.PURCHASE: 2,
    }
    records = []
    for session_id, session_drafts in drafts_by_session.items():
        ordered = sorted(
            session_drafts,
            key=lambda draft: (
                draft.source_view.timestamp,
                stage_order[draft.event_type],
                draft.source_view.product_id,
            ),
        )
        session = ordered[0].target_session
        latest_base = max(event.timestamp for event in base_by_session[session_id])
        usable = session.ended_at - latest_base
        if usable <= timedelta(microseconds=len(ordered)):
            raise RuntimeError(f"Not enough Session time for engagement events: {session_id}")
        weight_total = sum(draft.timing_weight for draft in ordered) + 1.0
        cumulative = 0.0
        for draft in ordered:
            cumulative += draft.timing_weight
            timestamp = latest_base + usable * (cumulative / weight_total)
            view = draft.source_view
            records.append(
                SyntheticEngagementRecord(
                    event_id="",
                    session_id=session_id,
                    user_id=view.user_id,
                    product_id=view.product_id,
                    product_category=view.product_category,
                    event_type=draft.event_type,
                    timestamp=timestamp,
                    source_view_event_id=view.event_id,
                    source_view_session_id=view.session_id,
                    conversion_timing=(
                        "same_session" if session_id == view.session_id else "later_session"
                    ),
                    preference_match=view.preference_match,
                    initial_popularity=view.initial_popularity,
                    price_log10=view.price_log10,
                    over_budget=view.over_budget,
                    impulsiveness=0.0,
                    view_count=draft.view_count,
                    had_favorite_before=draft.had_favorite_before,
                    had_cart_before=draft.had_cart_before,
                    probability=round(draft.probability, 6),
                    generation_version=ENGAGEMENT_GENERATION_VERSION,
                    generation_seed=seed,
                    upstream_generation_version=INTERACTION_GENERATION_VERSION,
                    upstream_generation_seed=view.generation_seed,
                )
            )

    records.sort(key=lambda record: (record.timestamp, record.session_id, record.product_id))
    return records


def generate_engagement_events(
    sessions: list[SyntheticSessionRecord],
    base_events: list[SyntheticEventRecord],
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
    *,
    seed: int = DEFAULT_ENGAGEMENT_SEED,
) -> list[SyntheticEngagementRecord]:
    """Generate deep engagement from the frozen Impression/View event stream."""
    rng = random.Random(seed)
    users_by_id = {user.user_id: user for user in users}
    products_by_id = {product.name: product for product in products}
    sessions_by_user: dict[str, list[SyntheticSessionRecord]] = defaultdict(list)
    for session in sessions:
        sessions_by_user[session.user_id].append(session)
    for user_sessions in sessions_by_user.values():
        user_sessions.sort(key=lambda session: session.started_at)
    session_indices = {
        session.session_id: index
        for user_sessions in sessions_by_user.values()
        for index, session in enumerate(user_sessions)
    }

    states: dict[tuple[str, str], _UserItemState] = defaultdict(_UserItemState)
    drafts: list[_Draft] = []
    views = sorted(
        (event for event in base_events if event.event_type == EventType.VIEW),
        key=lambda event: event.timestamp,
    )
    for view in views:
        user = users_by_id[view.user_id]
        product = products_by_id[view.product_id]
        user_sessions = sessions_by_user[view.user_id]
        current_index = session_indices[view.session_id]
        state = states[(view.user_id, view.product_id)]
        state.view_count += 1
        if state.purchase_target is not None:
            continue

        if state.favorite_target is None and state.cart_target is None:
            probability = _favorite_probability(user, product, view, state.view_count, rng)
            if rng.random() < probability:
                target = _target_session_index(
                    user_sessions,
                    current_index,
                    current_index,
                    view.timestamp,
                    FAVORITE_DELAY_PROBABILITY,
                    rng,
                )
                state.favorite_target = target
                drafts.append(
                    _Draft(
                        user_sessions[target],
                        view,
                        EventType.FAVORITE,
                        state.view_count,
                        False,
                        False,
                        probability,
                        rng.uniform(0.7, 1.5),
                    )
                )

        favorite_precedes = state.favorite_target is not None
        if state.cart_target is None and product.status == ProductStatus.AVAILABLE:
            probability = _cart_probability(
                user,
                view,
                state.view_count,
                favorite_precedes,
                rng,
            )
            if rng.random() < probability:
                minimum = state.favorite_target if favorite_precedes else current_index
                target = _target_session_index(
                    user_sessions,
                    current_index,
                    minimum,
                    view.timestamp,
                    CART_DELAY_PROBABILITY,
                    rng,
                )
                state.cart_target = target
                drafts.append(
                    _Draft(
                        user_sessions[target],
                        view,
                        EventType.ADD_TO_CART,
                        state.view_count,
                        favorite_precedes,
                        False,
                        probability,
                        rng.uniform(0.8, 1.7),
                    )
                )

        cart_precedes = state.cart_target is not None
        if product.status == ProductStatus.AVAILABLE:
            probability = _purchase_probability(
                user,
                view,
                state.view_count,
                favorite_precedes,
                cart_precedes,
                rng,
            )
            if rng.random() < probability:
                minimum = max(
                    current_index,
                    state.favorite_target if favorite_precedes else current_index,
                    state.cart_target if cart_precedes else current_index,
                )
                target = _target_session_index(
                    user_sessions,
                    current_index,
                    minimum,
                    view.timestamp,
                    PURCHASE_DELAY_PROBABILITY,
                    rng,
                )
                state.purchase_target = target
                drafts.append(
                    _Draft(
                        user_sessions[target],
                        view,
                        EventType.PURCHASE,
                        state.view_count,
                        favorite_precedes,
                        cart_precedes,
                        probability,
                        rng.uniform(1.0, 2.0),
                    )
                )

    assigned = _assign_timestamps(drafts, base_events, seed)
    records = []
    for index, record in enumerate(assigned, start=1):
        records.append(
            replace(
                record,
                event_id=f"synthetic-engagement-v1-{index:08d}",
                impulsiveness=users_by_id[record.user_id].impulsiveness,
            )
        )
    return records
