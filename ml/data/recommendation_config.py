from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


DATASET_VERSION = "recommendation_dataset_v1"
DEFAULT_DATASET_SEED = 42
PRODUCT_VERSION = "synthetic_product_v1"
USER_VERSION = "synthetic_user_v1"
SESSION_EVENT_VERSION = "synthetic_session_event_v1"
ENGAGEMENT_VERSION = "synthetic_engagement_v1"


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start: datetime
    end_exclusive: datetime


TRAIN_WINDOW = SplitWindow(
    "train",
    datetime(2026, 1, 1, tzinfo=timezone.utc),
    datetime(2026, 1, 21, tzinfo=timezone.utc),
)
VALIDATION_WINDOW = SplitWindow(
    "validation",
    datetime(2026, 1, 21, tzinfo=timezone.utc),
    datetime(2026, 1, 26, tzinfo=timezone.utc),
)
TEST_WINDOW = SplitWindow(
    "test",
    datetime(2026, 1, 26, tzinfo=timezone.utc),
    datetime(2026, 1, 31, tzinfo=timezone.utc),
)
SPLIT_WINDOWS = (TRAIN_WINDOW, VALIDATION_WINDOW, TEST_WINDOW)

TASK_VIEWPLUS = "viewplus"
TASK_FAVORITEPLUS = "favoriteplus"
TASK_PURCHASE = "purchase"
TASKS = (TASK_VIEWPLUS, TASK_FAVORITEPLUS, TASK_PURCHASE)

STATE_POSITIVE = "positive"
STATE_OBSERVED_NON_CONVERSION = "observed_non_conversion"
STATE_UNKNOWN = "unknown"
STATES = (STATE_POSITIVE, STATE_OBSERVED_NON_CONVERSION, STATE_UNKNOWN)
