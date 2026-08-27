from datetime import date


INTERACTION_GENERATION_VERSION = "synthetic_session_event_v1"
DEFAULT_INTERACTION_SEED = 42
DEFAULT_SIMULATION_START = date(2026, 1, 1)
DEFAULT_SIMULATION_END = date(2026, 1, 30)

SESSION_BASE_RATE = 1.4
SESSION_ACTIVITY_RATE = 7.3
MIN_IMPRESSIONS_PER_SESSION = 5
MAX_IMPRESSIONS_PER_SESSION = 30

EXPOSURE_SOURCE_WEIGHTS = {
    "preference": 50,
    "popular": 20,
    "exploration": 20,
    "random": 10,
}

VIEW_INTERCEPT = -1.05
VIEW_PREFERENCE_WEIGHT = 1.65
VIEW_POPULARITY_WEIGHT = 0.65
VIEW_NOVELTY_WEIGHT = 0.35
VIEW_EXPLORATION_WEIGHT = 0.65
VIEW_PRICE_PENALTY = 0.13
VIEW_NOISE_STDDEV = 0.68

ARCHETYPE_PRIMARY_CATEGORIES = {
    "Space Enthusiast": frozenset({"Space"}),
    "History Collector": frozenset({"History", "Art & Culture"}),
    "Tech Futurist": frozenset({"Technology"}),
    "Nature Explorer": frozenset({"Geography", "Creatures"}),
    "Fantasy Lover": frozenset({"Fantasy"}),
    "Power Seeker": frozenset({"Abstract & Phenomena", "Technology", "Fantasy"}),
    "Luxury Collector": frozenset({"Art & Culture", "History"}),
    "Thrill Seeker": frozenset({"Creatures", "Geography", "Abstract & Phenomena"}),
}
