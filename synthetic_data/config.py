CATALOG_VERSION = "synthetic_product_v1"
DEFAULT_PRODUCT_COUNT = 200
MAX_PRODUCT_COUNT = 200
DEFAULT_SEED = 42
ATTRIBUTE_NOISE_STDDEV = 0.065

ATTRIBUTE_NAMES = (
    "danger",
    "luxury",
    "novelty",
    "historical_value",
    "technology_level",
    "natural_significance",
    "fantasy_level",
    "space_affinity",
    "power",
)

PARENT_CATEGORY_TARGETS = {
    "Space": 30,
    "History": 30,
    "Creatures": 25,
    "Technology": 30,
    "Geography": 25,
    "Fantasy": 25,
    "Art & Culture": 20,
    "Abstract & Phenomena": 15,
}

TAG_VOCABULARY = frozenset(
    {
        "space",
        "historic",
        "prehistoric",
        "natural",
        "artificial",
        "technology",
        "fantasy",
        "mysterious",
        "dangerous",
        "luxury",
        "exclusive",
        "rare",
        "collectible",
        "scientific",
        "cultural",
        "powerful",
        "massive",
        "portable",
        "habitable",
        "legendary",
        "unexplored",
        "beautiful",
        "destructive",
        "valuable",
        "impossible",
    }
)

STATUS_PROPORTIONS = {
    "available": 0.85,
    "sold_out": 0.07,
    "coming_soon": 0.05,
    "unavailable": 0.03,
}
