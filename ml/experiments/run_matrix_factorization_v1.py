from __future__ import annotations

import argparse
from pathlib import Path

from ml.experiments.mf_experiment import run_mf_experiment
from ml.experiments.mf_export import export_mf_results
from ml.experiments.mf_report import render_mf_quality_report
from ml.training.mf_trainer import MFTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fixed-config BCE and BPR MF v1.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/recommendation_v1"))
    parser.add_argument("--popularity-dir", type=Path, default=Path("results/popularity_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/matrix_factorization_v1"))
    parser.add_argument("--quality-report", type=Path, default=Path("docs/matrix_factorization_v1_quality.md"))
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MFTrainingConfig(
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    result = run_mf_experiment(args.dataset_dir, args.popularity_dir, config=config)
    export_mf_results(
        result,
        args.output_dir,
        dataset_dir=args.dataset_dir,
        popularity_dir=args.popularity_dir,
    )
    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
    args.quality_report.write_text(render_mf_quality_report(result), encoding="utf-8")
    print(
        "Experiment: matrix_factorization_v1\n"
        f"BCE best epoch: {result.training['bce'].best_epoch}\n"
        f"BPR best epoch: {result.training['bpr'].best_epoch}\n"
        f"Output: {args.output_dir}\n"
        f"Quality report: {args.quality_report}"
    )


if __name__ == "__main__":
    main()
