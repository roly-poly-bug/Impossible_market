from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.db.session import SessionLocal, init_database
from synthetic_data.config import DEFAULT_PRODUCT_COUNT, DEFAULT_SEED
from synthetic_data.engagement_config import DEFAULT_ENGAGEMENT_SEED
from synthetic_data.engagement_database import write_engagement_events
from synthetic_data.engagement_export import export_engagement_csv
from synthetic_data.engagement_generator import generate_engagement_events
from synthetic_data.engagement_quality import analyze_engagement, render_engagement_report
from synthetic_data.engagement_validation import validate_engagement_events
from synthetic_data.event_generator import generate_events
from synthetic_data.interaction_config import DEFAULT_SIMULATION_END, DEFAULT_SIMULATION_START
from synthetic_data.product_generator import generate_catalog
from synthetic_data.session_generator import generate_sessions
from synthetic_data.user_config import DEFAULT_USER_COUNT, DEFAULT_USER_SEED
from synthetic_data.user_generator import generate_users


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Favorite/Cart/Purchase Events from frozen Views."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_ENGAGEMENT_SEED)
    parser.add_argument("--export-csv", type=Path)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
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
        seed=42,
        start_date=DEFAULT_SIMULATION_START,
        end_date=DEFAULT_SIMULATION_END,
    )
    base_events = generate_events(sessions, users, products, seed=42)
    engagement_events = generate_engagement_events(
        sessions,
        base_events,
        users,
        products,
        seed=args.seed,
    )
    validate_engagement_events(
        engagement_events,
        base_events,
        sessions,
        users,
        products,
        window,
        expected_seed=args.seed,
    )
    analysis = analyze_engagement(base_events, engagement_events, users, products)
    counts = analysis["counts"]
    conversion = analysis["conversion"]
    print(
        f"Impressions: {counts['impressions']}\nViews: {counts['views']}\n"
        f"Favorites: {counts['favorites']}\nCarts: {counts['carts']}\n"
        f"Purchases: {counts['purchases']}\n"
        f"View -> Favorite: {conversion['view_to_favorite']:.2%}\n"
        f"View -> Cart: {conversion['view_to_cart']:.2%}\n"
        f"View -> Purchase: {conversion['view_to_purchase']:.2%}\n"
        f"Cart -> Purchase: {conversion['cart_to_purchase']:.2%}"
    )

    if args.export_csv:
        destination = export_engagement_csv(engagement_events, args.export_csv)
        print(f"Engagement CSV export complete: {destination}")
    if args.quality_report:
        args.quality_report.parent.mkdir(parents=True, exist_ok=True)
        args.quality_report.write_text(render_engagement_report(analysis), encoding="utf-8")
        print(f"Quality report complete: {args.quality_report}")
    if args.write_db:
        init_database()
        with SessionLocal() as database:
            result = write_engagement_events(
                database,
                engagement_events,
                base_events,
                sessions,
                users,
                products,
                window,
                replace_existing=args.replace_existing,
            )
        print(
            "Database write complete: "
            f"{result.events_created} created / {result.events_updated} updated / "
            f"{result.events_deleted} deleted."
        )


if __name__ == "__main__":
    main()
