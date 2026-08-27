import argparse
from pathlib import Path

from backend.app.db.session import SessionLocal, init_database
from synthetic_data.user_config import DEFAULT_USER_COUNT, DEFAULT_USER_SEED
from synthetic_data.user_database import write_users
from synthetic_data.user_export import export_users_csv, export_users_json
from synthetic_data.user_generator import format_user_summary, generate_users, summarize_users


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Impossible Market synthetic users v1."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_USER_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_USER_SEED)
    parser.add_argument("--export-csv", type=Path, help="Export a flat UTF-8 CSV snapshot.")
    parser.add_argument("--export-json", type=Path, help="Export a structured UTF-8 JSON snapshot.")
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Write validated synthetic identities and profiles to the configured database.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace a different v1 population only when it has no Session or Event rows.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.replace_existing and not args.write_db:
        parser.error("--replace-existing requires --write-db")

    records = generate_users(count=args.count, seed=args.seed)
    print(format_user_summary(summarize_users(records)))

    if args.export_csv:
        print(f"\nCSV export complete: {export_users_csv(records, args.export_csv)}")
    if args.export_json:
        print(f"JSON export complete: {export_users_json(records, args.export_json)}")

    if args.write_db:
        init_database()
        with SessionLocal() as database:
            result = write_users(
                database,
                records,
                replace_existing=args.replace_existing,
            )
        print(
            "\nDatabase write complete: "
            f"{result.created} created, {result.updated} updated, {result.deleted} replaced."
        )


if __name__ == "__main__":
    main()
