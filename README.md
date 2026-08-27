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
React frontend  --HTTP/JSON-->  FastAPI backend  -->  SQLite
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
|-- backend/           # FastAPI application and database models
|-- ml/                # Future ML experiments (no model implementation yet)
|   |-- data/
|   |-- models/
|   |-- training/
|   |-- evaluation/
|   `-- inference/
|-- synthetic_data/    # Future synthetic-data generators
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

### Frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The page calls the backend health endpoint and
shows whether the connection succeeded.

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
- Frontend-to-backend connectivity indicator
- Initial SQLAlchemy domain models for users, products, sessions, and events
- No generated dataset, recommendation model, or inference implementation yet

## Planned ML Features

Potential future learning milestones include collaborative filtering, matrix
factorization, content-based and hybrid recommendation, semantic search,
transformer recommenders, and learning to rank. They are intentionally not
implemented yet and will be designed collaboratively before code is added.
