from dataclasses import dataclass


@dataclass(frozen=True)
class CategorySpec:
    parent: str
    name: str
    slug: str
    target_count: int
    prototype: dict[str, float]
    base_log10_price: float
    base_tags: tuple[str, ...]
    reality_weights: dict[str, float]
    featured_names: tuple[str, ...]
    generated_subject: str


def prototype(
    danger: float,
    luxury: float,
    novelty: float,
    historical_value: float,
    technology_level: float,
    natural_significance: float,
    fantasy_level: float,
    space_affinity: float,
    power: float,
) -> dict[str, float]:
    return {
        "danger": danger,
        "luxury": luxury,
        "novelty": novelty,
        "historical_value": historical_value,
        "technology_level": technology_level,
        "natural_significance": natural_significance,
        "fantasy_level": fantasy_level,
        "space_affinity": space_affinity,
        "power": power,
    }


CATEGORY_SPECS = (
    CategorySpec("Space", "Planet", "planet", 8, prototype(.75, .90, .85, .55, .15, .98, .15, 1.0, .90), 19.0, ("space", "natural", "massive", "rare"), {"real": .8, "speculative": .2}, ("Mars", "Venus", "Jupiter", "Neptune"), "Exoplanet"),
    CategorySpec("Space", "Satellite", "satellite", 7, prototype(.55, .85, .80, .60, .25, .90, .10, 1.0, .70), 17.0, ("space", "natural", "scientific", "exclusive"), {"real": .85, "speculative": .15}, ("Moon", "Europa", "Titan"), "Moon"),
    CategorySpec("Space", "Star & Cosmic Object", "star-cosmic-object", 7, prototype(.95, .95, .95, .35, .05, 1.0, .25, 1.0, 1.0), 24.0, ("space", "massive", "powerful", "destructive"), {"real": .75, "speculative": .25}, ("The Sun", "Andromeda Galaxy", "Halley's Comet"), "Cosmic Object"),
    CategorySpec("Space", "Spacecraft", "spacecraft", 8, prototype(.70, .85, .80, .65, .95, .10, .10, 1.0, .75), 13.0, ("space", "technology", "scientific", "exclusive"), {"real": .55, "speculative": .45}, ("International Space Station", "Apollo 11", "Voyager 1"), "Starship"),
    CategorySpec("History", "Empire & Nation", "empire-nation", 10, prototype(.75, .90, .75, 1.0, .30, .40, .20, .05, .95), 20.0, ("historic", "cultural", "powerful", "massive"), {"historical": .9, "real": .1}, ("Roman Empire", "Byzantine Empire", "Kingdom of Kush"), "Lost Realm"),
    CategorySpec("History", "Historical Artifact", "historical-artifact", 10, prototype(.35, .85, .80, 1.0, .25, .25, .25, .05, .65), 10.0, ("historic", "collectible", "rare", "valuable"), {"historical": .9, "real": .1}, ("Rosetta Stone", "Antikythera Mechanism", "Crown of Charlemagne"), "Ancient Relic"),
    CategorySpec("History", "Monument & Architecture", "monument-architecture", 10, prototype(.25, .85, .65, .95, .25, .45, .15, .05, .80), 14.0, ("historic", "cultural", "massive", "beautiful"), {"historical": .8, "real": .2}, ("Great Pyramid", "Colosseum", "Hanging Gardens"), "Monument"),
    CategorySpec("Creatures", "Extinct Creature", "extinct-creature", 9, prototype(.90, .55, .90, .85, .05, .95, .15, 0.0, .70), 9.0, ("prehistoric", "natural", "dangerous", "rare"), {"historical": .8, "real": .2}, ("Tyrannosaurus Rex", "Dodo", "Woolly Mammoth"), "Extinct Beast"),
    CategorySpec("Creatures", "Mythical Creature", "mythical-creature", 8, prototype(.85, .75, 1.0, .65, .10, .50, 1.0, .10, .90), 10.0, ("fantasy", "legendary", "powerful", "mysterious"), {"fictional": 1.0}, ("Phoenix", "Dragon", "Kraken"), "Mythical Creature"),
    CategorySpec("Creatures", "Extraordinary Animal", "extraordinary-animal", 8, prototype(.65, .65, .80, .35, .10, .95, .25, 0.0, .65), 8.0, ("natural", "rare", "beautiful", "powerful"), {"real": .7, "speculative": .3}, ("Immortal Jellyfish", "Giant Blue Whale", "White Lion"), "Extraordinary Animal"),
    CategorySpec("Technology", "Impossible Technology", "impossible-technology", 10, prototype(.70, .90, 1.0, .30, 1.0, .05, .65, .25, 1.0), 12.0, ("technology", "impossible", "powerful", "mysterious"), {"speculative": .8, "fictional": .2}, ("Time Machine", "Teleportation Device", "Matter Replicator"), "Impossible Device"),
    CategorySpec("Technology", "Vehicle", "vehicle", 10, prototype(.65, .80, .70, .45, .85, .15, .20, .15, .75), 9.0, ("technology", "artificial", "luxury", "powerful"), {"real": .5, "speculative": .5}, ("Flying Car", "Personal Submarine", "Supersonic Train"), "Concept Vehicle"),
    CategorySpec("Technology", "Machine & Device", "machine-device", 10, prototype(.45, .70, .75, .40, .90, .10, .25, .10, .80), 8.0, ("technology", "artificial", "scientific", "valuable"), {"real": .45, "speculative": .55}, ("Quantum Computer", "Weather Machine", "Dream Recorder"), "Experimental Machine"),
    CategorySpec("Geography", "Island", "island", 8, prototype(.25, .80, .75, .55, .10, .95, .25, 0.0, .65), 13.0, ("natural", "exclusive", "beautiful", "unexplored"), {"real": .75, "fictional": .15, "speculative": .1}, ("Private Island", "Easter Island", "Surtsey"), "Hidden Island"),
    CategorySpec("Geography", "Ocean & Sea", "ocean-sea", 8, prototype(.80, .85, .80, .65, .05, 1.0, .20, 0.0, .90), 20.0, ("natural", "massive", "dangerous", "unexplored"), {"real": .85, "fictional": .1, "speculative": .05}, ("Pacific Ocean", "Mediterranean Sea", "Mariana Trench"), "Uncharted Sea"),
    CategorySpec("Geography", "Land & Territory", "land-territory", 9, prototype(.40, .90, .70, .75, .10, .95, .15, 0.0, .95), 21.0, ("natural", "massive", "powerful", "valuable"), {"real": .8, "fictional": .1, "speculative": .1}, ("Antarctica", "Sahara Desert", "Amazon Rainforest"), "Remote Territory"),
    CategorySpec("Fantasy", "Magical Item", "magical-item", 9, prototype(.60, .90, .95, .65, .10, .20, 1.0, .10, .85), 9.0, ("fantasy", "mysterious", "legendary", "collectible"), {"fictional": .9, "speculative": .1}, ("Dragon Egg", "Philosopher's Stone", "Invisibility Cloak"), "Enchanted Artifact"),
    CategorySpec("Fantasy", "Superpower", "superpower", 8, prototype(.75, .75, 1.0, .25, .15, .10, 1.0, .15, 1.0), 8.0, ("fantasy", "powerful", "impossible", "exclusive"), {"fictional": .85, "speculative": .15}, ("Flight", "Telepathy", "Invulnerability"), "Impossible Ability"),
    CategorySpec("Fantasy", "Fictional Object", "fictional-object", 8, prototype(.55, .85, .95, .55, .45, .15, .95, .20, .85), 9.0, ("fantasy", "collectible", "legendary", "valuable"), {"fictional": .95, "speculative": .05}, ("Excalibur", "One Ring", "Pandora's Box"), "Legendary Object"),
    CategorySpec("Art & Culture", "Artwork", "artwork", 10, prototype(.10, .95, .70, .85, .15, .20, .40, 0.0, .55), 9.0, ("cultural", "beautiful", "collectible", "valuable"), {"real": .65, "historical": .35}, ("Mona Lisa", "The Starry Night", "David"), "Masterpiece"),
    CategorySpec("Art & Culture", "Cultural Treasure", "cultural-treasure", 10, prototype(.20, .90, .65, 1.0, .15, .25, .35, 0.0, .70), 11.0, ("cultural", "historic", "rare", "valuable"), {"historical": .7, "real": .3}, ("Imperial Regalia", "Library of Alexandria", "Terracotta Army"), "Cultural Treasure"),
    CategorySpec("Abstract & Phenomena", "Natural Phenomenon", "natural-phenomenon", 5, prototype(.90, .60, .90, .30, .05, 1.0, .40, .10, .90), 17.0, ("natural", "powerful", "beautiful", "dangerous"), {"real": .75, "speculative": .25}, ("Northern Lights", "A Solar Eclipse", "A Perfect Storm"), "Natural Phenomenon"),
    CategorySpec("Abstract & Phenomena", "Concept", "concept", 5, prototype(.40, .80, 1.0, .75, .20, .30, .80, .05, 1.0), 18.0, ("impossible", "mysterious", "powerful", "valuable"), {"abstract": 1.0}, ("Luck", "Silence", "Infinite Patience"), "Abstract Concept"),
    CategorySpec("Abstract & Phenomena", "Time & Reality", "time-reality", 5, prototype(.85, .95, 1.0, .70, .70, .10, 1.0, .40, 1.0), 22.0, ("impossible", "mysterious", "powerful", "exclusive"), {"abstract": .55, "speculative": .45}, ("One Extra Hour Per Day", "A Second Timeline", "Yesterday"), "Reality Fragment"),
)


NAME_ADJECTIVES = (
    "Crimson",
    "Silent",
    "Golden",
    "Obsidian",
    "Luminous",
    "Forgotten",
    "Celestial",
    "Emerald",
    "Midnight",
    "Eternal",
    "Sovereign",
    "Hidden",
    "Titanic",
    "Last",
    "Quantum",
    "Impossible",
    "Ancient",
    "Radiant",
    "Uncharted",
    "Private",
)


FEATURED_DESCRIPTIONS = {
    "Moon": "Earth's only natural satellite, offered with tides and phases included.",
    "Mars": "A rust-colored planet with excellent views and a demanding commute.",
    "Pacific Ocean": "The world's largest ocean, including its depths and weather systems.",
    "Tyrannosaurus Rex": "A reconstructed apex predator requiring a very sturdy enclosure.",
    "Time Machine": "A prototype temporal vehicle whose warranty begins before purchase.",
    "Roman Empire": "A vast historical civilization supplied with roads and administrative overhead.",
    "International Space Station": "A legendary orbital laboratory with panoramic windows and zero gravity.",
    "Luck": "A transferable measure of improbable good fortune with unpredictable side effects.",
    "One Extra Hour Per Day": "A private twenty-fifth hour added to every day, visible only to its owner.",
}
