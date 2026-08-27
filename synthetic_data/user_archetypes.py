from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserArchetype:
    name: str
    weight: int
    preference_prototype: dict[str, float]
    budget_log10: float
    price_sensitivity: float
    popularity_preference: float
    exploration_tendency: float
    impulsiveness: float
    activity_level: float


def preferences(
    danger: float,
    luxury: float,
    novelty: float,
    historical: float,
    technology: float,
    nature: float,
    fantasy: float,
    space: float,
    power: float,
) -> dict[str, float]:
    return {
        "danger_preference": danger,
        "luxury_preference": luxury,
        "novelty_preference": novelty,
        "historical_preference": historical,
        "technology_preference": technology,
        "nature_preference": nature,
        "fantasy_preference": fantasy,
        "space_preference": space,
        "power_preference": power,
    }


USER_ARCHETYPES = (
    UserArchetype(
        "Curious Generalist",
        15,
        preferences(.50, .53, .62, .52, .52, .53, .51, .48, .54),
        14.5,
        .55,
        .55,
        .67,
        .48,
        .55,
    ),
    UserArchetype(
        "Eclectic Browser",
        13,
        preferences(.58, .50, .70, .48, .56, .57, .60, .55, .57),
        15.0,
        .46,
        .48,
        .82,
        .58,
        .66,
    ),
    UserArchetype(
        "Space Enthusiast",
        9,
        preferences(.50, .56, .72, .38, .66, .50, .35, .86, .58),
        20.5,
        .42,
        .58,
        .62,
        .45,
        .57,
    ),
    UserArchetype(
        "History Collector",
        9,
        preferences(.38, .68, .52, .88, .34, .50, .38, .24, .55),
        16.5,
        .66,
        .43,
        .38,
        .31,
        .44,
    ),
    UserArchetype(
        "Tech Futurist",
        10,
        preferences(.48, .58, .75, .35, .87, .30, .40, .47, .64),
        13.0,
        .49,
        .61,
        .64,
        .53,
        .62,
    ),
    UserArchetype(
        "Nature Explorer",
        10,
        preferences(.46, .38, .62, .48, .28, .88, .30, .35, .50),
        12.5,
        .61,
        .35,
        .72,
        .38,
        .59,
    ),
    UserArchetype(
        "Fantasy Lover",
        9,
        preferences(.55, .56, .72, .48, .35, .36, .87, .32, .68),
        12.0,
        .45,
        .52,
        .66,
        .57,
        .53,
    ),
    UserArchetype(
        "Power Seeker",
        8,
        preferences(.67, .62, .64, .48, .55, .42, .55, .44, .88),
        18.5,
        .30,
        .68,
        .43,
        .72,
        .65,
    ),
    UserArchetype(
        "Luxury Collector",
        9,
        preferences(.32, .87, .58, .60, .48, .42, .48, .38, .62),
        16.5,
        .56,
        .72,
        .34,
        .34,
        .47,
    ),
    UserArchetype(
        "Thrill Seeker",
        8,
        preferences(.87, .42, .74, .32, .48, .62, .44, .38, .75),
        11.5,
        .28,
        .46,
        .74,
        .82,
        .70,
    ),
)

ARCHETYPE_BY_NAME = {archetype.name: archetype for archetype in USER_ARCHETYPES}
