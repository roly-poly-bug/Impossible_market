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

SQLite is the local default. SQLAlchemy models keep a future PostgreSQL migration
possible. FastAPI creates missing tables at startup, while a separate idempotent
command seeds the development catalog. Schema migrations will be introduced with
Alembic when the schema starts evolving beyond this learning-stage setup.

Prices cross every boundary as exact decimal values: SQLite stores their decimal
text, SQLAlchemy presents `Decimal`, the API emits strings, and React formats the
integer component using `BigInt`. This avoids precision loss for values larger
than JavaScript's safe integer range.
