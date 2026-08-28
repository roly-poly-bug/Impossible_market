from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.recommendation_config import DATASET_VERSION
from ml.data.recommendation_dataset import (
    build_dataset_bundle,
    load_frozen_observed_world,
)
from ml.data.recommendation_export import export_recommendation_dataset
from ml.data.recommendation_quality import (
    analyze_recommendation_dataset,
    render_quality_report,
)
from ml.data.recommendation_validation import validate_dataset_bundle


DEFAULT_PRODUCT_PATH = Path("data/synthetic_product_v1_seed42.csv")
DEFAULT_USER_PATH = Path("data/synthetic_user_v1_seed42.csv")
DEFAULT_INTERACTION_PATH = Path("data/synthetic_event_v1_seed42.csv")
DEFAULT_ENGAGEMENT_PATH = Path("data/synthetic_engagement_v1_seed42.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build observed-fact recommendation datasets without true-negative labels."
    )
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--output-dir", type=Path, default=Path("data/recommendation_v1"))
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("docs/recommendation_dataset_v1_quality.md"),
    )
    parser.add_argument("--product-snapshot", type=Path, default=DEFAULT_PRODUCT_PATH)
    parser.add_argument("--user-snapshot", type=Path, default=DEFAULT_USER_PATH)
    parser.add_argument(
        "--interaction-snapshot",
        type=Path,
        default=DEFAULT_INTERACTION_PATH,
    )
    parser.add_argument(
        "--engagement-snapshot",
        type=Path,
        default=DEFAULT_ENGAGEMENT_PATH,
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.dataset_version != DATASET_VERSION:
        parser.error(f"only {DATASET_VERSION} is supported")

    source_paths = {
        "products": args.product_snapshot,
        "users": args.user_snapshot,
        "session_events": args.interaction_snapshot,
        "engagement": args.engagement_snapshot,
    }
    world = load_frozen_observed_world(
        product_path=args.product_snapshot,
        user_path=args.user_snapshot,
        interaction_path=args.interaction_snapshot,
        engagement_path=args.engagement_snapshot,
    )
    bundle = build_dataset_bundle(world)
    validate_dataset_bundle(bundle)
    analysis = analyze_recommendation_dataset(bundle)
    export_recommendation_dataset(
        bundle,
        args.output_dir,
        source_paths=source_paths,
    )
    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
    args.quality_report.write_text(
        render_quality_report(analysis),
        encoding="utf-8",
    )

    overall = analysis["overall"]
    print(
        f"Dataset: {DATASET_VERSION}\n"
        f"Users: {overall['users']}\n"
        f"Items: {overall['items']}\n"
        f"Raw Events: {overall['events']}\n"
        f"Observed pairs: {overall['observed_pairs']}\n"
        f"Output: {args.output_dir}\n"
        f"Quality report: {args.quality_report}"
    )


if __name__ == "__main__":
    main()
