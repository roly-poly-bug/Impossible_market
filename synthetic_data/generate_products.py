import argparse

from backend.app.db.session import SessionLocal, init_database
from synthetic_data.config import DEFAULT_PRODUCT_COUNT, DEFAULT_SEED
from synthetic_data.database import write_catalog
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.replace_existing and not args.write_db:
        parser.error("--replace-existing requires --write-db")

    records = generate_catalog(count=args.count, seed=args.seed)
    print(format_summary(summarize_catalog(records)))

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
