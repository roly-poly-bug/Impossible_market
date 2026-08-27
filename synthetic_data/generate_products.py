import argparse
from pathlib import Path

from backend.app.db.session import SessionLocal, init_database
from synthetic_data.config import DEFAULT_PRODUCT_COUNT, DEFAULT_SEED
from synthetic_data.database import write_catalog
from synthetic_data.export import export_catalog_csv, export_catalog_json
from synthetic_data.product_generator import format_summary, generate_catalog, summarize_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic Impossible Market synthetic product catalog v1."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Write the validated records to the configured SQLite database.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace an existing v1 catalog when its seed or product set differs.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Export the validated catalog as a flat UTF-8 CSV snapshot.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Export the validated catalog as structured UTF-8 JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.replace_existing and not args.write_db:
        parser.error("--replace-existing requires --write-db")

    records = generate_catalog(count=args.count, seed=args.seed)
    print(format_summary(summarize_catalog(records)))

    if args.export_csv:
        exported_csv = export_catalog_csv(records, args.export_csv)
        print(f"\nCSV export complete: {exported_csv}")
    if args.export_json:
        exported_json = export_catalog_json(records, args.export_json)
        print(f"JSON export complete: {exported_json}")

    if args.write_db:
        init_database()
        with SessionLocal() as database:
            result = write_catalog(
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
