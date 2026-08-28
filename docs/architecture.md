# Architecture

Impossible Market starts as two local processes: a React browser application and
a FastAPI service. The API owns persistence and will later own the boundary to
ML inference. This prevents UI code from depending on model implementation
details.

The initial domain model reserves four core entities:

- `User`: a known visitor identity.
- `Product`: an item that can receive interactions.
- `Session`: a browsing period, optionally linked to a user.
- `Event`: an impression, view, favorite, add-to-cart, or purchase interaction.

Synthetic simulation metadata extends `User` without turning identity rows into
wide experiment records:

```text
User 1 ----- 0..1 SyntheticUserProfile
                       |
                       +-- fixed nine-column preference vector
                       +-- budget and behavioral tendencies
                       `-- generator provenance
```

The fixed columns make a `(1000, 9)` user matrix straightforward to load. This
is intentionally different from flexible `ProductAttribute` rows: the v1 user
preference vocabulary is fixed and benefits from direct typed columns.

## Product Metadata Model

```text
Category
  |  parent_id (optional self-reference)
  |
  `----< Product >----< product_tags >---- Tag
             |
             `----< ProductAttribute
```

- `Product` owns identity, description, exact price, rarity, lifecycle status,
  image URL, timestamps, and one category foreign key.
- `Category` is a stable classification. Its optional `parent_id` supports trees
  such as Space → Satellite without requiring navigation UI yet.
- `Tag` is a flexible human-readable label. Products and tags have a
  many-to-many relationship with a composite primary key preventing duplicate
  associations.
- `ProductAttribute` is a unique `(product_id, attribute_name)` numeric feature.
  Values are normalized to `0..1`, making synthetic ground truth and later
  analysis predictable while allowing new axes without schema changes.

`ProductSpecification` is intentionally deferred. User-facing specifications
such as `diameter = 3474 km` need string values and units, while
`ProductAttribute` represents normalized experimental features. Combining them
now would blur those responsibilities. A future minimal specification table can
contain `product_id`, `key`, `value`, and optional `unit` when the catalog needs
a specifications UI.

SQLite is the local default. SQLAlchemy models keep a future PostgreSQL migration
possible. FastAPI creates missing tables at startup, while a separate idempotent
command seeds the development catalog. Schema migrations will be introduced with
Alembic when the schema starts evolving beyond this learning-stage setup.

Prices cross every boundary as exact decimal values: SQLite stores their decimal
text, SQLAlchemy presents `Decimal`, the API emits strings, and React formats the
integer component using `BigInt`. This avoids precision loss for values larger
than JavaScript's safe integer range.

## Future Recommendation Boundaries

Product metadata can be transformed into features for a future content-based
recommender. Collaborative filtering may instead learn only from event logs and
need none of these true attributes. During synthetic-data experiments,
`ProductAttribute` may act as hidden ground truth used to generate user choices;
that does not imply every trained model is allowed to observe those attributes.

No embedding, similarity calculation, training data, model, or inference API is
part of the current metadata layer.

## Recommendation Dataset v1 Boundary

The Recommendation Dataset Builder lives under `ml/data/` because it transforms
frozen observations into reusable experiment inputs; it does not belong to the
synthetic world's data-generating mechanism or to the web API.

```text
Frozen Product/User snapshots + frozen Event snapshots
                         |
                         v
              observed-fact aggregation
                         |
          +--------------+---------------+
          v              v               v
        View+        Favorite+       Purchase-only
          |              |               |
          +------ half-open UTC splits --+
                         |
                         v
          Train tables + relevance/candidate sets
```

The interaction matrix records Events, counts, and timestamps. It never imports
synthetic user preference vectors, hidden product attributes, preference-match
ground truth, archetype prototypes, or future events. It assigns no fixed
strength weight and creates no true-negative label.

Each task derives one of `positive`, `observed_non_conversion`, or `unknown`.
Implicit Feedback has no reliable true negative: non-conversion is merely an
observed prerequisite without a conversion, while Unknown means that even the
prerequisite was not observed. A later sampled negative would not be a true
negative. Sampling remains owned by each future experiment.

Train is `[2026-01-01T00:00:00Z, 2026-01-21T00:00:00Z)`, Validation is
`[2026-01-21T00:00:00Z, 2026-01-26T00:00:00Z)`, and Test is
`[2026-01-26T00:00:00Z, 2026-01-31T00:00:00Z)`. Aggregation occurs independently
inside each interval, so a future conversion cannot change an earlier state.
See `docs/recommendation_dataset_v1.md` for the complete contract.

## Synthetic Product Generator v1

The generator remains outside both the web application and `ml/`:

```text
Readable child-category prototypes
              |
              +-- nine normalized attributes + clipped noise
              +-- fixed tag vocabulary
              +-- local name/description templates
              +-- metadata-driven log-price rule
              v
      200 SyntheticProductRecord values
              |
          validation
              |
      explicit SQLAlchemy writer
              v
            SQLite
```

Pure generation can later feed JSON/CSV export or offline experiments without a
database. Persistence is a separate explicit step. `catalog_version` and
`generation_seed` preserve provenance, while a name unique constraint and
same-catalog checks provide idempotency. Replacing a different v1 catalog
requires an explicit flag and affects only that catalog version.

These attributes are structured simulator ground truth, not an implemented ML
feature pipeline. Future synthetic users may react to them; future collaborative
models may observe only the resulting events.

## Synthetic User Generator v1

```text
ProductAttribute ground truth
            |
            v
Archetype prototype + optional secondary prototype + individual noise
            |
            v
SyntheticUserProfile hidden ground truth
            |
            +-- budget_log10 / budget tier
            +-- price sensitivity / popularity preference
            +-- exploration / impulsiveness
            `-- activity level / activity tier
            |
            v
      future Session / Event simulator
            |
            v
       observed interactions
            |
            v
 future recommendation experiment
```

`SyntheticUserProfile` is simulator-only information. A future collaborative
filtering model may train on observed Event rows without receiving these true
preferences. This preserves the boundary between hidden data-generating causes
and features available to a learned model.

Generation, validation, export, analysis, and SQLAlchemy persistence are
separate modules. The same user version, count, and seed produces identical
records and byte-identical snapshots. Replacement of a different v1 population
is explicit and is refused once any affected User has a Session or Event row.
The next section documents the now-implemented first layer of those downstream
records; the profile remains hidden ground truth rather than a model input.

## Synthetic Session / Impression / View Generator v1

```text
Frozen Product v1 + Frozen User v1
                  |
                  v
         Session generation
                  |
                  v
 preference / popular / exploration / random exposure mix
                  |
                  v
          Impression events
                  |
                  v
 noisy preference + price + popularity + exploration utility
                  |
                  v
              View events
                  |
                  v
        validation and quality audit
```

Generation, validation, summary, CSV export, and SQLAlchemy persistence are
separate modules. Sessions and Events carry stable IDs plus
`generation_version` and `generation_seed`; database writes are explicit and
idempotent for one frozen population. An additive startup compatibility step
adds the new nullable interaction columns to an older local SQLite database.

Exposure is an observed opportunity, not the user's true preference. A product
that was not viewed may have been ignored under noise, price friction, or its
position in a mixed exposure set. A product that was not exposed has no feedback
label at all. Future recommendation experiments must preserve this distinction
to avoid treating missing exposure as negative preference.

The v1 generator emits only `impression` and `view`. It does not implement a
ranking/recommendation model, recommendation API, or favorite/cart/purchase
funnel.

## Synthetic Engagement Generator v1

```text
Frozen Impression / View
           |
           v
user-item repeated-interest state
           |
           +-- Favorite utility: preference / novelty / rarity / luxury
           +-- Cart utility: preference / price / impulsiveness / Favorite
           `-- Purchase utility: stronger price / state / impulsiveness
           |
           v
same-session or later-session engagement Events
```

The engagement layer is separate from View generation and carries its own
`synthetic_engagement_v1` provenance. It appends Favorite, Add-to-Cart, and
Purchase rows to the existing generic Event table. No new domain table is
needed. Stable Event IDs make same-population writes idempotent, and an explicit
replacement affects only engagement-version rows.

Each deep Event points to an existing user, product, and Session and is derived
from an earlier View of that user-item pair. Favorite/Cart/Purchase are each
limited to one occurrence per user-item. Cart and Purchase require an available
product, while Favorite does not. Later-Session placement provides simple state
continuity without introducing an application-level state machine.

View, Favorite, Cart, and Purchase are different observed intent levels. None is
identical to true preference: exposure bias, product status, price, budget,
impulsiveness, repeated interest, prior state, and noise affect what is logged.
Recommendation event weights, datasets, models, Orders, and payment processing
remain downstream concerns.
