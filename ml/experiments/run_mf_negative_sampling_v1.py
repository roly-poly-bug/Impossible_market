from __future__ import annotations

import argparse
from pathlib import Path

from ml.experiments.mf_negative_sampling_experiment import run_negative_sampling_experiment
from ml.experiments.mf_negative_sampling_export import export_negative_sampling_results
from ml.training.mf_trainer import MFTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed Weighted BCE MF non-positive sampling")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/recommendation_v1"))
    parser.add_argument("--popularity-dir", type=Path, default=Path("results/popularity_v1"))
    parser.add_argument("--signal-results-dir", type=Path, default=Path("results/mf_signal_representation_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mf_negative_sampling_v1"))
    args = parser.parse_args()
    config = MFTrainingConfig()
    result = run_negative_sampling_experiment(
        args.dataset_dir, args.popularity_dir, args.signal_results_dir, config=config
    )
    manifest = export_negative_sampling_results(
        result,
        args.output_dir,
        dataset_dir=args.dataset_dir,
        popularity_dir=args.popularity_dir,
        signal_results_dir=args.signal_results_dir,
    )
    print(f"Wrote {args.output_dir} ({len(manifest['strategies'])} strategies)")


if __name__ == "__main__":
    main()
