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
