from ml.experiments.mf_objective_experiment import run_objective_experiment
from ml.experiments.mf_objective_export import export_objective_results


def main() -> None:
    result = run_objective_experiment("data/recommendation_v1", "results/popularity_v1", "results/mf_latent_dim_v1")
    export_objective_results(
        result,
        "results/mf_objective_v2",
        dataset_dir="data/recommendation_v1",
        latent_results_dir="results/mf_latent_dim_v1",
    )
    print("Wrote results/mf_objective_v2")


if __name__ == "__main__":
    main()
