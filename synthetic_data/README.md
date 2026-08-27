# Synthetic Data Generators

## Synthetic Product Generator v1

The v1 generator creates a reproducible 200-product Impossible Market catalog
with structured ground truth. It uses only local Python code and never calls an
LLM or external API.

## Pipeline

```text
Category prototype
      + deterministic random noise
      + curated local vocabulary
      ↓
SyntheticProductRecord objects
      ↓
Validation
      ↓ (only with --write-db)
SQLite upsert
```

Generation and database persistence are separate. `product_generator.py`
returns structured Python records; `database.py` owns SQLAlchemy mapping;
`export.py` owns deterministic offline serialization; `generate_products.py`
is the CLI. `catalog_quality.py` computes audit statistics without implementing
a recommendation model.

## Catalog Contract

- Version: `synthetic_product_v1`
- Default count: exactly 200
- Default seed: `42`
- Count supported by v1: 1–200
- Product names: curated names plus descriptive local templates; always unique
- Tags: 3–6 values drawn only from the fixed vocabulary
- Attributes: the same nine normalized axes on every product
- Prices: positive arbitrary-precision decimal integers

The seven original examples—Moon, Mars, Pacific Ocean, Tyrannosaurus Rex, Time
Machine, Roman Empire, and International Space Station—are included in the 200.

## Category Distribution

```text
Space                 30
History               30
Creatures             25
Technology            30
Geography             25
Fantasy               25
Art & Culture         20
Abstract & Phenomena  15
```

These eight parent categories contain the 24 requested child categories. Slugs
use lowercase kebab-case; ampersands become `and` in parent slugs.

## Fixed Tag Vocabulary

```text
space historic prehistoric natural artificial technology fantasy mysterious
dangerous luxury exclusive rare collectible scientific cultural powerful
massive portable habitable legendary unexplored beautiful destructive valuable
impossible
```

Each child category supplies semantically appropriate base tags. Attribute
thresholds may add compatible tags, but never vocabulary-external values.

## Common Attributes

Every product has exactly these values in the inclusive range `0..1`:

```text
danger luxury novelty historical_value technology_level
natural_significance fantasy_level space_affinity power
```

Each child category defines a readable nine-dimensional prototype. Generation
adds Gaussian noise with standard deviation `0.065`, clips each value to
`0..1`, and rounds to four decimals. Products in one category are therefore
similar without being identical.

`rarity` means scarcity inside the catalog world. `novelty` means how unfamiliar
the item feels to a user. Rarity is correlated with novelty and luxury but has
independent noise, so the values are not interchangeable.

## Reality and Status

`reality_type` is one of `real`, `historical`, `fictional`, `abstract`, or
`speculative`. Categories define reality-type probabilities, while iconic names
such as Moon, Roman Empire, Time Machine, and Luck have explicit semantics.

Status allocation is deterministic for a requested count. At 200 products it is:

```text
available 170 | sold_out 14 | coming_soon 10 | unavailable 6
```

## Price Rule

Price is generated in logarithmic space:

```text
log10(price) = child_category_base
             + 2.0 × (rarity - 0.5)
             + 1.5 × (luxury - 0.5)
             + 1.7 × (power - 0.5)
             + 1.0 × (historical_value - 0.5)
             + Gaussian noise(0, 0.42)
```

The result is bounded to `10^5..10^30`, converted under a 50-digit Decimal
context, and rounded to a whole monetary unit. Category bases make portable
objects cheaper than territories, planets, cosmic objects, and reality itself.

## CLI and Reproducibility

Generate, validate, and print a summary without modifying SQLite:

```bash
python -m synthetic_data.generate_products --count 200 --seed 42
```

Write to the configured DB:

```bash
python -m synthetic_data.generate_products --count 200 --seed 42 --write-db
```

The same generator version, count, and seed produces equal records. The DB
stores both `catalog_version` and `generation_seed`. Re-running the same catalog
updates matching unique names without duplicates.

Export a pandas-friendly flat CSV and structured JSON snapshot without changing
SQLite:

```bash
python -m synthetic_data.generate_products --count 200 --seed 42 \
  --export-csv data/synthetic_product_v1_seed42.csv \
  --export-json data/synthetic_product_v1_seed42.json
```

CSV tags use `|` as a delimiter; the fixed tag vocabulary contains no pipe
characters. Prices are serialized as exact base-10 strings in both formats so
large values do not lose precision. JSON has top-level version, seed, count,
and attribute-name metadata plus nested category, tags, and attributes.

The checked-in seed-42 snapshots are deliberately small, reproducible reference
data for offline experiments. Regenerate the quality report with:

```bash
python -m synthetic_data.catalog_quality \
  --count 200 \
  --seed 42 \
  --output docs/synthetic_product_v1_quality.md
```

The report records descriptive statistics, category means, Pearson
correlations, rarity/price/tag distributions, and representative cosine-neighbor
checks. It only audits the feature space; it is not a recommender.

When another seed or product set is already marked `synthetic_product_v1`, the
writer stops. Replacement requires explicit authorization:

```bash
python -m synthetic_data.generate_products --count 200 --seed 43 --write-db --replace-existing
```

This deletes and recreates only the v1 synthetic products. It refuses replacement
if those products already have Event rows. It never resets unrelated DB content.

## Validation and Summary

Before any DB write, validation checks count, unique names, category membership,
3–6 valid tags, all nine attributes and ranges, rarity, reality type, status,
positive exact price, non-empty descriptions, and catalog version.

The CLI reports parent-category, reality-type and status distributions; minimum,
median and maximum prices; and all nine attribute means.

## Synthetic User Generator v1

The user generator creates 1,000 reproducible simulator profiles by default.
It uses ten overlapping archetypes rather than nine independent uniform draws:

```text
archetype prototype
      + optional 20–40% secondary-archetype mixture
      + Gaussian individual noise
      v
nine normalized user preferences
```

The archetypes are Curious Generalist, Eclectic Browser, Space Enthusiast,
History Collector, Tech Futurist, Nature Explorer, Fantasy Lover, Power Seeker,
Luxury Collector, and Thrill Seeker. Curious Generalist has 15% weight,
Eclectic Browser 13%, and the other types 8–10%. Approximately 30% of generated
users receive a secondary archetype, so labels remain meaningful without making
ten perfectly separable clusters.

The fixed preference-to-product mapping is:

```text
danger_preference      -> danger
luxury_preference      -> luxury
novelty_preference     -> novelty
historical_preference  -> historical_value
technology_preference  -> technology_level
nature_preference      -> natural_significance
fantasy_preference     -> fantasy_level
space_preference       -> space_affinity
power_preference       -> power
```

Each profile also stores continuous `budget_log10`, `price_sensitivity`,
`popularity_preference`, `exploration_tendency`, `impulsiveness`, and
`activity_level`. Budget and activity tiers are readable derived labels; future
simulation should use the continuous values. Price sensitivity is generated
independently from budget, allowing cautious wealthy users and impulsive users
with lower budgets. Synthetic `created_at` values start from a fixed UTC epoch
and advance by user index so exported records remain byte-reproducible.

Generate and summarize without touching SQLite:

```bash
python -m synthetic_data.generate_users --count 1000 --seed 42
```

Export deterministic offline snapshots:

```bash
python -m synthetic_data.generate_users --count 1000 --seed 42 \
  --export-csv data/synthetic_user_v1_seed42.csv \
  --export-json data/synthetic_user_v1_seed42.json
```

Write explicitly to SQLite:

```bash
python -m synthetic_data.generate_users --count 1000 --seed 42 --write-db
```

Re-running the same version/count/seed updates the same identities without
duplicates. A different population requires `--replace-existing`. Replacement
is refused if an affected user already has a Session or Event row.

`User` remains the stable database identity. `SyntheticUserProfile` is a
one-to-one table with nine fixed preference columns, behavioral fields, and
version/seed provenance. The preferences are hidden world state used by a future
simulator; a recommendation model must not automatically receive them.

Regenerate the population audit with:

```bash
python -m synthetic_data.user_quality \
  --count 1000 \
  --seed 42 \
  --output docs/synthetic_user_v1_quality.md
```

The audit checks descriptive statistics, archetype means, correlations,
nearest-prototype separability, budget coverage of the frozen product catalog,
and representative preference/product alignment. It does not create Sessions,
Events, observed datasets, or recommendation output.

## Synthetic Session / Impression / View Generator v1

The interaction generator combines the frozen 200-product and 1,000-user worlds
over `2026-01-01` through `2026-01-30` UTC. Activity level controls a noisy,
overlapping Poisson session count. Sessions contain 5–30 unique product
Impressions and probabilistic Views with strictly increasing timestamps.

Exposure slots use a deterministic 50% preference, 20% popular, 20%
exploration, and 10% random allocation (integer rounding varies an individual
session by at most a slot). Preference match uses centered user preferences and
catalog-standardized product attributes rather than compressed raw cosine.
Initial popularity is an internal deterministic prior derived from rarity,
luxury, novelty, and a product-name hash. A weak continuity term makes products
near the last viewed item's category/attributes slightly easier to expose.

View probability is a noisy logistic utility combining preference match,
popularity, novelty, exploration, and soft log-price friction. Price above the
user budget never forces probability to zero. This is a simulator rule, not a
production recommender.

Generate and export without changing SQLite:

```bash
python -m synthetic_data.generate_sessions_events \
  --seed 42 \
  --start-date 2026-01-01 \
  --end-date 2026-01-30 \
  --export-sessions-csv data/synthetic_session_v1_seed42.csv \
  --export-events-csv data/synthetic_event_v1_seed42.csv
```

Persist only after validation with `--write-db`. Re-running the same frozen
population is idempotent. `--replace-existing` replaces only compatible
`synthetic_session_event_v1` Impression/View rows and refuses to erase unrelated
or later-funnel events.

The CSV snapshots are deterministic but comparatively large, so they are
Git-ignored and regenerated from the command above. The quality audit is tracked
at `docs/synthetic_session_event_v1_quality.md` and can be regenerated with:

```bash
python -m synthetic_data.interaction_quality
```

Exposure is not preference. `not viewed` is not equivalent to `disliked`, and
`not exposed` means no behavioral label was observed. Favorite, add-to-cart,
purchase, and every recommendation model remain outside v1.
