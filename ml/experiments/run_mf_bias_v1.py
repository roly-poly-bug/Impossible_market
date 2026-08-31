from pathlib import Path
from ml.experiments.mf_bias_experiment import run_bias_experiment
from ml.experiments.mf_bias_export import export_bias_results
def main():
    result=run_bias_experiment("data/recommendation_v1","results/popularity_v1","results/mf_negative_sampling_v1")
    export_bias_results(result,"results/mf_bias_v1",dataset_dir="data/recommendation_v1",popularity_dir="results/popularity_v1",negative_results_dir="results/mf_negative_sampling_v1")
    print("Wrote results/mf_bias_v1")
if __name__=="__main__":main()
