# Impossible Market

A fictional e-commerce platform for learning how recommendation systems and
machine-learning inference connect to a web application.

> The marketplace for things you should never be able to buy.

## Goals

- Build an understandable full-stack e-commerce application.
- Keep the frontend, backend, database, and ML responsibilities explicit.
- Call future Python ML inference through a REST API.
- Experiment with synthetic user behavior data safely and reproducibly.
- Implement recommendation models gradually as a separate learning project.

## Architecture

```text
React catalog  --HTTP/JSON-->  FastAPI product API  -->  SQLite
                                      ^                  ^
                                      |                  |
                           synthetic generators --------+
                                      |
                                      +--> future ML inference layer
```

The frontend never talks directly to the database or loads an ML model. The
backend owns those integrations and exposes them through REST endpoints.

## Tech Stack

- **Frontend:** React 18, Vite, JavaScript
- **Backend/API:** Python 3.11+, FastAPI, Uvicorn
- **Database:** SQLite and SQLAlchemy (designed to allow a later PostgreSQL move)
- **Testing:** pytest and FastAPI TestClient
- **Future ML:** Python, with PyTorch or scikit-learn chosen per experiment

React with Vite keeps the UI setup small and makes the frontend/backend boundary
visible. FastAPI is beginner-friendly, provides interactive API documentation,
and can later call Python inference code without adding another language or
service. SQLite requires no local database server while SQLAlchemy keeps the
data-access layer portable.

## Directory Structure

```text
Impossible_market/
|-- frontend/          # React user interface
|-- backend/           # FastAPI application, database models, and seed command
|   `-- data/          # Local SQLite database (database files are Git-ignored)
|-- ml/                # Future ML experiments (no model implementation yet)
|   |-- data/
|   |-- models/
|   |-- training/
|   |-- evaluation/
|   `-- inference/
|-- synthetic_data/    # Reproducible structured catalog generation
|-- tests/             # Backend API tests
|-- docs/              # Architecture and development notes
|-- AGENTS.md
`-- README.md
```

## Setup

### Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer (includes npm)

### Backend

From the repository root:

```bash
python -m venv .venv
```

Activate it on PowerShell and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements-dev.txt
```

Copy `backend/.env.example` to `backend/.env` only if you want to override the
defaults. Then start the API:

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

The health endpoint is at <http://localhost:8000/api/health> and interactive API
documentation is at <http://localhost:8000/docs>.

On startup, FastAPI safely creates missing tables in
`backend/data/impossible_market.db`. It does not automatically add products.

Seed the small development catalog explicitly from the repository root:

```bash
python -m backend.app.db.seed_products
```

The command can be run repeatedly: existing products are detected by their
unique names and are not duplicated. Because this project does not use Alembic
yet, the command detects the earlier product schema, preserves that DB as a
timestamped backup, and rebuilds the development catalog when needed. Database
files and backups are ignored by Git.

Generate and validate the 200-product synthetic catalog without changing the DB:

```bash
python -m synthetic_data.generate_products --count 200 --seed 42
```

Export the same validated catalog for offline analysis without touching SQLite:

```bash
python -m synthetic_data.generate_products --count 200 --seed 42 \
  --export-csv data/synthetic_product_v1_seed42.csv \
  --export-json data/synthetic_product_v1_seed42.json
```

The small seed-42 CSV and JSON snapshots are tracked as the reproducible v1
reference catalog. CSV stores one product per row, exact prices as base-10 text,
and pipe-delimited tags. JSON preserves nested category, tag, attribute, and
catalog metadata. See [the catalog quality report](docs/synthetic_product_v1_quality.md)
for distributions, correlations, category structure, and feature-space sanity checks.

Generate the deterministic 1,000-user hidden-ground-truth population:

```bash
python -m synthetic_data.generate_users --count 1000 --seed 42
```

Export it for offline simulation work without changing SQLite:

```bash
python -m synthetic_data.generate_users --count 1000 --seed 42 \
  --export-csv data/synthetic_user_v1_seed42.csv \
  --export-json data/synthetic_user_v1_seed42.json
```

Write the validated users and profiles explicitly:

```bash
python -m synthetic_data.generate_users --count 1000 --seed 42 --write-db
```

The user generator creates overlapping archetype-based preference vectors,
log-scale budgets, price/popularity/exploration/impulsiveness tendencies, and
activity levels. A different existing v1 population requires
`--replace-existing`; replacement is refused if affected users already have
Session or Event rows. See the
[Synthetic User quality report](docs/synthetic_user_v1_quality.md).

Generate the frozen 30-day Session / Impression / View population and export
offline CSV files without changing SQLite:

```bash
python -m synthetic_data.generate_sessions_events \
  --seed 42 \
  --start-date 2026-01-01 \
  --end-date 2026-01-30 \
  --export-sessions-csv data/synthetic_session_v1_seed42.csv \
  --export-events-csv data/synthetic_event_v1_seed42.csv
```

Add `--write-db` to persist the validated records. Repeating the same version
and seed is idempotent; replacing another interaction population requires
`--replace-existing`. The larger Session/Event snapshots are reproducible build
artifacts and are Git-ignored. See the
[Session/Event quality report](docs/synthetic_session_event_v1_quality.md).

Generate Favorite, Add-to-Cart, and Purchase events from that frozen View stream:

```bash
python -m synthetic_data.generate_engagement_events \
  --seed 42 \
  --export-csv data/synthetic_engagement_v1_seed42.csv \
  --quality-report docs/synthetic_engagement_v1_quality.md
```

Add `--write-db` only after the frozen Product/User/Session/Event world has been
written. The same engagement version and seed is idempotent; replacement needs
`--replace-existing`. The engagement CSV is reproducible and Git-ignored. See
the [Engagement quality report](docs/synthetic_engagement_v1_quality.md).

Build the observed-fact Recommendation Dataset v1 from all four frozen
snapshots:

```bash
python -m ml.data.build_recommendation_dataset \
  --dataset-version recommendation_dataset_v1 \
  --output-dir data/recommendation_v1
```

This produces a 200,000-row full user-item fact table, three Train task tables,
Validation/Test relevance sets, task-specific Train seen items, and impressed
candidate sets. Large generated files are ignored; the manifest and quality
report remain trackable. See the
[Recommendation Dataset v1 design](docs/recommendation_dataset_v1.md) and
[generated quality audit](docs/recommendation_dataset_v1_quality.md).

Run the first non-personalized Train-only control experiment:

```bash
python -m ml.experiments.run_popularity_baselines \
  --dataset-dir data/recommendation_v1 \
  --output-dir results/popularity_v1
```

Popularity Baseline v1 compares raw, unique-User, log-transformed, Favorite+,
Cart, Purchase, and configurable weighted signals using identical candidates,
seen exclusion, relevance sets, and Top-K metrics. It also audits six possible
future user-item matrix representations without training Matrix Factorization.
See the [Popularity Baseline v1 quality report](docs/popularity_baseline_v1_quality.md).

Install the separate ML runtime dependencies and run the fixed Matrix
Factorization v1 experiment:

```bash
python -m pip install -r ml/requirements.txt
python -m ml.experiments.run_matrix_factorization_v1 \
  --dataset-dir data/recommendation_v1 \
  --popularity-dir results/popularity_v1 \
  --output-dir results/matrix_factorization_v1
```

The experiment trains bias-free BCE and BPR Matrix Factorization from Binary
View+ positives and four seeded Random Unknown samples per positive. Validation
Purchase NDCG@10 selects checkpoints; Test is evaluated once afterward. See the
[Matrix Factorization v1 quality report](docs/matrix_factorization_v1_quality.md).

Write the validated catalog explicitly:

```bash
python -m synthetic_data.generate_products --count 200 --seed 42 --write-db
```

The seven familiar development products are deterministic members of the 200
products and are updated in place. Re-running the same seed is idempotent. If a
different seed or product set already exists, the command refuses to change it
unless `--replace-existing` is explicitly supplied. That option replaces only
products marked `synthetic_product_v1`; it does not reset the whole database.
See `synthetic_data/README.md` for the generation contract and distributions.

### Frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The catalog calls the backend product API and
renders the stored products. Selecting a card opens `/products/{id}`, which
loads that product from the detail endpoint.

## Product Database and API

The `products` table contains the product identity and commercial fields:
`id`, `name`, `description`, `category_id`, `price`, `rarity`, `image_url`,
`status`, `reality_type`, generator provenance, `created_at`, and `updated_at`.
Product names are unique, and rarity is constrained to the range `0` through `1`.

SQLite cannot store arbitrary-precision decimal values natively, so prices are
stored as exact base-10 text while SQLAlchemy exposes them as Python `Decimal`
objects. The API also serializes prices as strings. The React frontend formats
the integer portion with `BigInt`, avoiding JavaScript `Number` precision loss
for extremely expensive artifacts.

Available endpoints:

- `GET /api/products` — lightweight list; category remains a display string
- `GET /api/products/{product_id}` — detail with category, tags, and attributes, or `404`
- `GET /api/health` — backend availability check

### Product Metadata

The metadata tables have deliberately different responsibilities:

- `Category` gives each product one stable classification and supports a
  nullable parent category for future hierarchies.
- `Tag` supplies flexible, human-readable labels through the `product_tags`
  many-to-many table. Tags are appropriate for display and filtering.
- `ProductAttribute` stores named, normalized numeric values from `0` to `1`.
  These flexible feature axes do not become fixed columns on `Product`.
- `Product` stays focused on the item itself and points to its category.

The seven-item seed creates categories, tags, associations, and product
attributes idempotently. If it encounters the earlier string-category schema,
the seed command moves the existing local DB to a timestamped
`*.pre-metadata-*.db` backup before rebuilding and restoring the seven seeded
products. Run the same seed command after updating this branch:

```bash
python -m backend.app.db.seed_products
```

Product attributes may later serve two distinct experiment roles:

```text
Product metadata  --> Content-based recommendation inputs
Event logs        --> Collaborative-filtering inputs (metadata may be unused)

ProductAttribute  --> hidden simulator ground truth
                  --> synthetic user behavior generation
```

The interaction and engagement generators are synthetic-world mechanisms, not
recommendation models. Exposure deliberately mixes preference, popularity,
exploration, and random products. Therefore `not viewed` does not mean
`disliked`, and no label exists for a product that was never exposed. Likewise,
View, Favorite, Cart, and Purchase represent different intent levels rather
than interchangeable positive labels.

### Synthetic User Ground Truth

`User` stores a stable identity. Its optional one-to-one
`SyntheticUserProfile` stores the fixed nine preference columns, budget and
behavior tendencies, activity, and generator provenance. These preferences are
hidden simulator ground truth. They may generate future behavior but are not
automatically exposed to a collaborative-filtering model trained on observed
Event rows.

## Environment Variables

The checked-in `.env.example` files document all current options. Real `.env`
files are ignored by Git.

- `IMPOSSIBLE_MARKET_DATABASE_URL`: backend SQLAlchemy database URL
- `IMPOSSIBLE_MARKET_CORS_ORIGINS`: comma-separated allowed frontend origins
- `VITE_API_BASE_URL`: URL used by the browser to reach the backend

All variables have local-development defaults, so `.env` files are optional.

## Tests

After installing backend development dependencies:

```bash
python -m pytest
```

## Current Status

- Minimal React landing page
- FastAPI `GET /api/health` endpoint
- SQLite-backed product catalog and idempotent seven-product seed
- Hierarchical categories, many-to-many tags, and normalized product attributes
- Deterministic `synthetic_product_v1` generator for 200 structured products
- Reproducible CSV/JSON v1 reference snapshot and catalog quality audit
- Deterministic `synthetic_user_v1` generator for 1,000 overlapping archetype profiles
- Reproducible user CSV/JSON snapshot and population quality audit
- Deterministic `synthetic_session_event_v1` generator for a 30-day browsing window
- Reproducible Session/Event CSV export, validation, idempotent DB writer, and quality audit
- Deterministic `synthetic_engagement_v1` Favorite/Cart/Purchase generator and funnel audit
- Deterministic `recommendation_dataset_v1` builder with UTC temporal splits,
  observed-fact three-state task tables, relevance/candidate sets, and ranking metrics
- Train-only `popularity_baseline_v1` with cross-signal evaluation and interaction
  representation statistics; no personalized model yet
- Fixed-config PyTorch BCE/BPR `matrix_factorization_v1` with validation-only
  early stopping, full-ranking evaluation, and personalization diagnostics
- Fixed-config BCE MF signal comparison across Binary View, Log View confidence,
  Favorite+, Weighted Implicit, and Purchase-only representations
- FastAPI product list and detail endpoints
- React product cards, progressive catalog display, metadata-aware detail pages,
  and loading/error/empty states
- Initial SQLAlchemy domain models for users, products, sessions, and events
- No Order/payment flow, recommendation model, or inference implementation yet

Run the signal-representation comparison with:

```bash
python -m ml.experiments.run_mf_signal_representation_v1
```

It writes `results/mf_signal_representation_v1/`; interpretation is documented
in `docs/mf_signal_representation_v1_quality.md`.

Run the fixed Weighted BCE MF negative-sampling comparison with:

```bash
python -m ml.experiments.run_mf_negative_sampling_v1
```

It compares frozen Random Unknown with Exposed Non-conversion and fixed 50/50
Mixed sampling. Results are written to `results/mf_negative_sampling_v1/`.

Run the fixed Weighted/Exposed MF Bias comparison with `python -m
ml.experiments.run_mf_bias_v1`. Results are written to `results/mf_bias_v1/`.

Run the fixed Item-Bias MF positive-signal comparison with:

```bash
python -m ml.experiments.run_mf_cart_signal_v1
```

It compares Existing Weighted, Cart+, Favorite+Cart+, and one pre-fixed
Cart-centered Weighted signal while holding architecture and sampling fixed.
Results are written to `results/mf_cart_signal_v1/`; interpretation is in
`docs/mf_cart_signal_v1_quality.md`.

Run the fixed MF capacity comparison with:

```bash
python -m ml.experiments.run_mf_latent_dim_v1
```

It compares latent dimensions 8, 16, 32, and 64 while keeping the selected
Existing Weighted, Exposed-sampling, Item-Bias setup fixed. Results are written
to `results/mf_latent_dim_v1/`; interpretation is in
`docs/mf_latent_dim_v1_quality.md`.

Run the fixed MF objective comparison with:

```bash
python -m ml.experiments.run_mf_objective_v2
```

It compares the frozen best dim-8 Item-Bias MF under pointwise BCE and
confidence-weighted BPR while sharing all Train comparisons. Results are in
`results/mf_objective_v2/`; the decision is documented in
`docs/mf_objective_v2_quality.md`.

Run the post-hoc MF and Train Cart Popularity fusion experiment with:

```bash
python -m ml.experiments.run_mf_cart_hybrid_v1
```

It keeps the best MF checkpoint frozen, z-scores both score sources, and selects
one of five pre-fixed alpha values using Validation Purchase NDCG@10. Results
are in `results/mf_cart_hybrid_v1/`; interpretation is in
`docs/mf_cart_hybrid_v1_quality.md`.

## Planned ML Features

Potential future learning milestones include collaborative filtering, matrix
factorization, content-based and hybrid recommendation, semantic search,
transformer recommenders, and learning to rank. They are intentionally not
implemented yet and will be designed collaboratively before code is added.
