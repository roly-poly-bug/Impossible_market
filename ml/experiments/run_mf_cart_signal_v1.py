from ml.experiments.mf_cart_signal_experiment import run_cart_signal_experiment
from ml.experiments.mf_cart_signal_export import export_cart_signal_results


def main() -> None:
    result = run_cart_signal_experiment("data/recommendation_v1", "results/popularity_v1", "results/mf_bias_v1")
    export_cart_signal_results(result, "results/mf_cart_signal_v1", dataset_dir="data/recommendation_v1", bias_results_dir="results/mf_bias_v1")
    print("Wrote results/mf_cart_signal_v1")


if __name__ == "__main__":
    main()
