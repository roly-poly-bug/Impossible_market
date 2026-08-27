from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone

from backend.app.db.models import SessionEntryType
from synthetic_data.interaction_config import (
    DEFAULT_INTERACTION_SEED,
    DEFAULT_SIMULATION_END,
    DEFAULT_SIMULATION_START,
    INTERACTION_GENERATION_VERSION,
    MAX_IMPRESSIONS_PER_SESSION,
    MIN_IMPRESSIONS_PER_SESSION,
    SESSION_ACTIVITY_RATE,
    SESSION_BASE_RATE,
)
from synthetic_data.user_generator import SyntheticUserRecord


@dataclass(frozen=True)
class SimulationWindow:
    start: datetime
    end_exclusive: datetime

    @property
    def end_inclusive(self) -> datetime:
        return self.end_exclusive - timedelta(microseconds=1)


@dataclass(frozen=True)
class SyntheticSessionRecord:
    session_id: str
    user_id: str
    started_at: datetime
    ended_at: datetime
    entry_type: SessionEntryType
    impression_target: int
    generation_version: str
    generation_seed: int


def simulation_window(start_date: date, end_date: date) -> SimulationWindow:
    if end_date < start_date:
        raise ValueError("simulation end date must not precede start date")
    return SimulationWindow(
        start=datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        end_exclusive=datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc),
    )


def _poisson(rate: float, rng: random.Random) -> int:
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _entry_type(user: SyntheticUserRecord, rng: random.Random) -> SessionEntryType:
    exploration = user.exploration_tendency
    weights = (
        34,
        32 - 8 * exploration,
        14 + 18 * exploration,
        20 - 10 * exploration,
    )
    return rng.choices(tuple(SessionEntryType), weights=weights, k=1)[0]


def _impression_target(
    user: SyntheticUserRecord,
    entry_type: SessionEntryType,
    rng: random.Random,
) -> int:
    entry_adjustment = {
        SessionEntryType.HOME: 0,
        SessionEntryType.CATEGORY: 2,
        SessionEntryType.SEARCH_LIKE: 3,
        SessionEntryType.DIRECT: -2,
    }[entry_type]
    target = round(7 + 14 * user.activity_level + entry_adjustment + rng.gauss(0.0, 3.0))
    return min(MAX_IMPRESSIONS_PER_SESSION, max(MIN_IMPRESSIONS_PER_SESSION, target))


def _session_start(window: SimulationWindow, rng: random.Random) -> datetime:
    day_count = (window.end_exclusive.date() - window.start.date()).days
    day_offset = rng.randrange(day_count)
    hour = rng.choices(
        range(24),
        weights=(1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 7, 7, 7, 7, 8, 10, 12, 13, 12, 9, 6, 3),
        k=1,
    )[0]
    return window.start + timedelta(
        days=day_offset,
        hours=hour,
        minutes=rng.randrange(60),
        seconds=rng.randrange(60),
    )


def generate_sessions(
    users: list[SyntheticUserRecord],
    *,
    seed: int = DEFAULT_INTERACTION_SEED,
    start_date: date = DEFAULT_SIMULATION_START,
    end_date: date = DEFAULT_SIMULATION_END,
) -> tuple[SimulationWindow, list[SyntheticSessionRecord]]:
    """Generate deterministic, time-ordered Session records without Events or DB access."""
    window = simulation_window(start_date, end_date)
    rng = random.Random(seed)
    drafts = []

    for user in users:
        session_count = max(
            1,
            _poisson(SESSION_BASE_RATE + SESSION_ACTIVITY_RATE * user.activity_level, rng),
        )
        user_drafts = []
        for _ in range(session_count):
            entry_type = _entry_type(user, rng)
            impression_target = _impression_target(user, entry_type, rng)
            duration_seconds = max(
                impression_target * 25 + 60,
                min(2100, max(45, round(rng.lognormvariate(math.log(360), 0.65)))),
            )
            started_at = _session_start(window, rng)
            latest_start = window.end_exclusive - timedelta(seconds=duration_seconds + 1)
            started_at = min(started_at, latest_start)
            user_drafts.append(
                SyntheticSessionRecord(
                    session_id="",
                    user_id=user.user_id,
                    started_at=started_at,
                    ended_at=started_at + timedelta(seconds=duration_seconds),
                    entry_type=entry_type,
                    impression_target=impression_target,
                    generation_version=INTERACTION_GENERATION_VERSION,
                    generation_seed=seed,
                )
            )
        user_drafts.sort(key=lambda record: record.started_at)
        drafts.extend(user_drafts)

    drafts.sort(key=lambda record: (record.started_at, record.user_id))
    sessions = [
        replace(record, session_id=f"synthetic-session-v1-{index:07d}")
        for index, record in enumerate(drafts, start=1)
    ]
    return window, sessions

