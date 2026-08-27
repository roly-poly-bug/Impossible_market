from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from synthetic_data.engagement_generator import SyntheticEngagementRecord


ENGAGEMENT_CSV_COLUMNS = (
    "event_id",
    "session_id",
    "user_id",
    "product_id",
    "product_category",
    "event_type",
    "timestamp",
    "source_view_event_id",
    "source_view_session_id",
    "conversion_timing",
    "generation_version",
    "generation_seed",
    "upstream_generation_version",
    "upstream_generation_seed",
)


def export_engagement_csv(
    events: Iterable[SyntheticEngagementRecord],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=ENGAGEMENT_CSV_COLUMNS,
            lineterminator="\n",
        )
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
                    "timestamp": event.timestamp.isoformat().replace("+00:00", "Z"),
                    "source_view_event_id": event.source_view_event_id,
                    "source_view_session_id": event.source_view_session_id,
                    "conversion_timing": event.conversion_timing,
                    "generation_version": event.generation_version,
                    "generation_seed": event.generation_seed,
                    "upstream_generation_version": event.upstream_generation_version,
                    "upstream_generation_seed": event.upstream_generation_seed,
                }
            )
    return destination
