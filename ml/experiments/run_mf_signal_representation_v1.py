from __future__ import annotations

import argparse
from pathlib import Path

from ml.experiments.mf_signal_experiment import run_signal_experiment
from ml.experiments.mf_signal_export import export_signal_results
from ml.training.mf_trainer import MFTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-config BCE MF signal comparison")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/recommendation_v1"))
    parser.add_argument("--popularity-dir", type=Path, default=Path("results/popularity_v1"))
    parser.add_argument("--mf-v1-dir", type=Path, default=Path("results/matrix_factorization_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mf_signal_representation_v1"))
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
    result = run_signal_experiment(
        args.dataset_dir, args.popularity_dir, args.mf_v1_dir, config=config
    )
    manifest = export_signal_results(
        result,
        args.output_dir,
        dataset_dir=args.dataset_dir,
        popularity_dir=args.popularity_dir,
        mf_v1_dir=args.mf_v1_dir,
    )
    print(f"Wrote {args.output_dir} ({len(manifest['representations'])} representations)")


if __name__ == "__main__":
    main()
