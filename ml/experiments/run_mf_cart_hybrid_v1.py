from ml.experiments.mf_cart_hybrid_experiment import run_hybrid_experiment
from ml.experiments.mf_cart_hybrid_export import export_hybrid_results


def main() -> None:
    result = run_hybrid_experiment("data/recommendation_v1", "results/popularity_v1", "results/mf_latent_dim_v1", "results/mf_objective_v2")
    export_hybrid_results(result, "results/mf_cart_hybrid_v1", dataset_dir="data/recommendation_v1", latent_results_dir="results/mf_latent_dim_v1")
    print("Wrote results/mf_cart_hybrid_v1")


if __name__ == "__main__":
    main()
