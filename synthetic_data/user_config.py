from datetime import datetime, timezone

from synthetic_data.config import CATALOG_VERSION


USER_GENERATION_VERSION = "synthetic_user_v1"
DEFAULT_USER_COUNT = 1000
MAX_USER_COUNT = 100_000
DEFAULT_USER_SEED = 42
USER_PREFERENCE_NOISE_STDDEV = 0.16
MIXED_PREFERENCE_PROBABILITY = 0.30
SYNTHETIC_USER_CREATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
PRODUCT_CATALOG_VERSION = CATALOG_VERSION

PREFERENCE_PRODUCT_MAPPING = {
    "danger_preference": "danger",
    "luxury_preference": "luxury",
    "novelty_preference": "novelty",
    "historical_preference": "historical_value",
    "technology_preference": "technology_level",
    "nature_preference": "natural_significance",
    "fantasy_preference": "fantasy_level",
    "space_preference": "space_affinity",
    "power_preference": "power",
}

PREFERENCE_NAMES = tuple(PREFERENCE_PRODUCT_MAPPING)

