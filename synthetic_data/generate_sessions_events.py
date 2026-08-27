import argparse
from datetime import date
from pathlib import Path

from backend.app.db.session import SessionLocal, init_database
from synthetic_data.config import DEFAULT_PRODUCT_COUNT, DEFAULT_SEED
from synthetic_data.event_generator import generate_events
from synthetic_data.interaction_config import (
    DEFAULT_INTERACTION_SEED,
    DEFAULT_SIMULATION_END,
    DEFAULT_SIMULATION_START,
)
from synthetic_data.interaction_database import write_interactions
from synthetic_data.interaction_export import export_events_csv, export_sessions_csv
from synthetic_data.interaction_summary import format_interaction_summary, summarize_interactions
from synthetic_data.interaction_validation import validate_interactions
from synthetic_data.product_generator import generate_catalog
from synthetic_data.session_generator import generate_sessions
from synthetic_data.user_config import DEFAULT_USER_COUNT, DEFAULT_USER_SEED
from synthetic_data.user_generator import generate_users


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Synthetic Sessions and Impression/View Events v1."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_INTERACTION_SEED)
    parser.add_argument("--start-date", type=_date, default=DEFAULT_SIMULATION_START)
    parser.add_argument("--end-date", type=_date, default=DEFAULT_SIMULATION_END)
    parser.add_argument("--export-sessions-csv", type=Path)
    parser.add_argument("--export-events-csv", type=Path)
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace v1 interactions only when no downstream Event types exist.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.replace_existing and not args.write_db:
        parser.error("--replace-existing requires --write-db")

    products = generate_catalog(count=DEFAULT_PRODUCT_COUNT, seed=DEFAULT_SEED)
    users = generate_users(count=DEFAULT_USER_COUNT, seed=DEFAULT_USER_SEED)
    window, sessions = generate_sessions(
        users,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    events = generate_events(sessions, users, products, seed=args.seed)
    validate_interactions(
        sessions,
        events,
        users,
        products,
        window,
        expected_seed=args.seed,
    )
    print(format_interaction_summary(summarize_interactions(sessions, events, users)))

    if args.export_sessions_csv:
        print(f"\nSession CSV export complete: {export_sessions_csv(sessions, args.export_sessions_csv)}")
    if args.export_events_csv:
        print(f"Event CSV export complete: {export_events_csv(events, args.export_events_csv)}")

    if args.write_db:
        init_database()
        with SessionLocal() as database:
            result = write_interactions(
                database,
                sessions,
                events,
                users,
                products,
                window,
                replace_existing=args.replace_existing,
            )
        print(
            "\nDatabase write complete: "
            f"sessions {result.sessions_created} created / {result.sessions_updated} updated / "
            f"{result.sessions_deleted} deleted; events {result.events_created} created / "
            f"{result.events_updated} updated / {result.events_deleted} deleted."
        )


if __name__ == "__main__":
    main()
