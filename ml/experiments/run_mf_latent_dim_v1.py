from ml.experiments.mf_latent_dim_experiment import run_latent_dim_experiment
from ml.experiments.mf_latent_dim_export import export_latent_dim_results


def main() -> None:
    result = run_latent_dim_experiment("data/recommendation_v1", "results/popularity_v1", "results/mf_bias_v1")
    export_latent_dim_results(result, "results/mf_latent_dim_v1", dataset_dir="data/recommendation_v1", bias_results_dir="results/mf_bias_v1")
    print("Wrote results/mf_latent_dim_v1")


if __name__ == "__main__":
    main()
