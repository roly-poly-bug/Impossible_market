from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from synthetic_data.event_generator import SyntheticEventRecord
from synthetic_data.session_generator import SyntheticSessionRecord


SESSION_CSV_COLUMNS = (
    "session_id",
    "user_id",
    "started_at",
    "ended_at",
    "duration_seconds",
    "entry_type",
    "impression_count",
    "generation_version",
    "generation_seed",
)

EVENT_CSV_COLUMNS = (
    "event_id",
    "session_id",
    "user_id",
    "product_id",
    "product_category",
    "event_type",
    "timestamp",
    "exposure_source",
    "generation_version",
    "generation_seed",
)


def _timestamp(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def export_sessions_csv(
    sessions: Iterable[SyntheticSessionRecord],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SESSION_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for session in sessions:
            writer.writerow(
                {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "started_at": _timestamp(session.started_at),
                    "ended_at": _timestamp(session.ended_at),
                    "duration_seconds": int(
                        (session.ended_at - session.started_at).total_seconds()
                    ),
                    "entry_type": session.entry_type.value,
                    "impression_count": session.impression_target,
                    "generation_version": session.generation_version,
                    "generation_seed": session.generation_seed,
                }
            )
    return destination


def export_events_csv(
    events: Iterable[SyntheticEventRecord],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=EVENT_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "session_id": event.session_id,
                    "user_id": event.user_id,
                    "product_id": event.product_id,
                    "product_category": event.product_category,
                    "event_type": event.event_type.value,
                    "timestamp": _timestamp(event.timestamp),
                    "exposure_source": event.exposure_source,
                    "generation_version": event.generation_version,
                    "generation_seed": event.generation_seed,
                }
            )
    return destination
