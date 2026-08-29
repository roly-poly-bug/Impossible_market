from __future__ import annotations

import argparse
from pathlib import Path

from ml.baselines.popularity import WeightedSignalConfig
from ml.experiments.popularity_experiment import run_popularity_experiment
from ml.experiments.popularity_export import export_popularity_results
from ml.experiments.popularity_report import render_popularity_quality_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic Train-only popularity baselines."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/recommendation_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/popularity_v1"),
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("docs/popularity_baseline_v1_quality.md"),
    )
    parser.add_argument(
        "--include-seen",
        action="store_true",
        help="Optional comparison mode. Main v1 results exclude same-task Train positives.",
    )
    parser.add_argument("--view-weight", type=float, default=1.0)
    parser.add_argument("--favorite-weight", type=float, default=3.0)
    parser.add_argument("--cart-weight", type=float, default=5.0)
    parser.add_argument("--purchase-weight", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weighted_config = WeightedSignalConfig(
        view_weight=args.view_weight,
        favorite_weight=args.favorite_weight,
        cart_weight=args.cart_weight,
        purchase_weight=args.purchase_weight,
    )
    exclude_seen = not args.include_seen
    result = run_popularity_experiment(
        args.dataset_dir,
        exclude_seen=exclude_seen,
        weighted_config=weighted_config,
    )
    export_popularity_results(
        result,
        args.output_dir,
        dataset_dir=args.dataset_dir,
        exclude_seen=exclude_seen,
    )
    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
    args.quality_report.write_text(
        render_popularity_quality_report(result),
        encoding="utf-8",
    )
    print(
        "Experiment: popularity_baseline_v1\n"
        "Dataset: recommendation_dataset_v1\n"
        f"Exclude seen: {exclude_seen}\n"
        f"Metric rows: {len(result.metrics)}\n"
        f"Output: {args.output_dir}\n"
        f"Quality report: {args.quality_report}"
    )


if __name__ == "__main__":
    main()
