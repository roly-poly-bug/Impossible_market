from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import settings
from backend.app.db.base import Base


def _sqlite_connect_args(database_url: str) -> dict[str, bool]:
    return {"check_same_thread": False} if database_url.startswith("sqlite") else {}


engine = create_engine(
    settings.database_url,
    connect_args=_sqlite_connect_args(settings.database_url),
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _database_path() -> Path | None:
    if settings.database_url.startswith("sqlite:///./"):
        return Path(settings.database_url.removeprefix("sqlite:///./"))
    return None


def _has_legacy_product_schema() -> bool:
    database_path = _database_path()
    if database_path is None or not database_path.exists():
        return False

    inspector = inspect(engine)
    if "products" not in inspector.get_table_names():
        return False
    column_names = {column["name"] for column in inspector.get_columns("products")}
    return "category" in column_names and "category_id" not in column_names


def _upgrade_additive_product_columns() -> None:
    """Apply the small pre-Alembic additive upgrade used by local dev databases."""
    inspector = inspect(engine)
    if "products" not in inspector.get_table_names():
        return

    column_names = {column["name"] for column in inspector.get_columns("products")}
    statements = []
    if "reality_type" not in column_names:
        statements.append(
            "ALTER TABLE products ADD COLUMN reality_type VARCHAR(32) "
            "NOT NULL DEFAULT 'speculative'"
        )
    if "catalog_version" not in column_names:
        statements.append("ALTER TABLE products ADD COLUMN catalog_version VARCHAR(64)")
    if "generation_seed" not in column_names:
        statements.append("ALTER TABLE products ADD COLUMN generation_seed INTEGER")

    if statements:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_products_reality_type ON products (reality_type)"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_products_catalog_version ON products (catalog_version)"
            )


def _upgrade_additive_interaction_columns() -> None:
    """Add pre-Alembic Session/Event provenance columns to local databases."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    statements = []
    if "sessions" in table_names:
        session_columns = {column["name"] for column in inspector.get_columns("sessions")}
        for name, sql_type in (
            ("ended_at", "DATETIME"),
            ("entry_type", "VARCHAR(32)"),
            ("generation_version", "VARCHAR(64)"),
            ("generation_seed", "INTEGER"),
        ):
            if name not in session_columns:
                statements.append(f"ALTER TABLE sessions ADD COLUMN {name} {sql_type}")
    if "events" in table_names:
        event_columns = {column["name"] for column in inspector.get_columns("events")}
        for name, sql_type in (
            ("event_key", "VARCHAR(64)"),
            ("exposure_source", "VARCHAR(32)"),
            ("generation_version", "VARCHAR(64)"),
            ("generation_seed", "INTEGER"),
        ):
            if name not in event_columns:
                statements.append(f"ALTER TABLE events ADD COLUMN {name} {sql_type}")

    if statements:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    if "sessions" in table_names or "events" in table_names:
        with engine.begin() as connection:
            if "sessions" in table_names:
                connection.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_sessions_generation_version "
                    "ON sessions (generation_version)"
                )
            if "events" in table_names:
                connection.exec_driver_sql(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_events_event_key ON events (event_key)"
                )
                connection.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_events_generation_version "
                    "ON events (generation_version)"
                )


def backup_legacy_database() -> Path | None:
    """Move a pre-metadata development DB aside so it remains recoverable."""
    if not _has_legacy_product_schema():
        return None

    database_path = _database_path()
    if database_path is None:
        return None

    engine.dispose()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = database_path.with_name(
        f"{database_path.stem}.pre-metadata-{timestamp}{database_path.suffix}"
    )
    database_path.replace(backup_path)
    return backup_path


def init_database() -> None:
    """Create the local data directory and any tables that do not exist yet."""
    database_path = _database_path()
    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)

    if _has_legacy_product_schema():
        raise RuntimeError(
            "The local product schema predates metadata support. "
            "Run `python -m backend.app.db.seed_products` to back it up and rebuild it."
        )

    _upgrade_additive_product_columns()
    _upgrade_additive_interaction_columns()

    # Importing models registers their tables on Base.metadata.
    from backend.app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()
