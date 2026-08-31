from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from ml.evaluation.matrix_factorization import evaluate_model, load_evaluation_data, model_diagnostics
from ml.evaluation.mf_bias import bias_diagnostics
from ml.evaluation.mf_signal import SignalFinalTestEvaluator
from ml.experiments.mf_negative_sampling_experiment import _cart_comparison, _read_csv
from ml.representations.mf_signal import WEIGHTED, RepresentationData, load_representation
from ml.training.mf_bias_trainer import BiasTrainingResult, train_bias_model
from ml.training.mf_negative_sampling import EXPOSED_NON_CONVERSION, sample_non_positives
from ml.training.mf_trainer import MFTrainingConfig


NO_BIAS, ITEM_BIAS, USER_ITEM_BIAS = "no_bias", "item_bias", "user_item_bias"
MODELS = (NO_BIAS, ITEM_BIAS, USER_ITEM_BIAS)
TRAINED_MODELS = (ITEM_BIAS, USER_ITEM_BIAS)
EXPERIMENT_VERSION = "mf_bias_v1"


@dataclass
class BiasExperimentResult:
    config: MFTrainingConfig; data: RepresentationData
    training: dict[str, BiasTrainingResult]; histories: dict[str, list[dict[str, object]]]
    metrics: dict[str, list[dict[str, object]]]; diagnostics: dict[str, dict[str, object]]
    comparison: list[dict[str, object]]; bias_rows: list[dict[str, object]]; personalization: list[dict[str, object]]


def _frozen_control(path: Path):
    history = [{k: float(v) for k,v in row.items()} for row in _read_csv(path/"training_history.csv")]
    metrics=[]
    for row in _read_csv(path/"metrics.csv"):
        metrics.append({"task":row["task"],"split":row["split"],"k":int(row["k"]),"eligible_users":int(row["eligible_users"]),"recall":float(row["recall"]),"ndcg":float(row["ndcg"]),"hit_rate":float(row["hit_rate"]),"precision":float(row["precision"])})
    diag=json.loads((path/"diagnostics.json").read_text(encoding="utf-8")); diag["source"]="frozen no-bias Exposed control; Test not re-evaluated"
    return history,metrics,diag


def run_bias_experiment(dataset_dir, popularity_dir, negative_results_dir, *, config=None):
    settings=config or MFTrainingConfig(); root=Path(dataset_dir)
    base=load_representation(root, WEIGHTED, negative_ratio=4, seed=42)
    exposed=sample_non_positives(base.indexed, EXPOSED_NON_CONVERSION, sample_ratio=4, seed=42)
    data=replace(base, sampled=exposed.sampled); evaluation=load_evaluation_data(root,data.indexed)
    training={name:train_bias_model(name,data,evaluation,settings) for name in TRAINED_MODELS}
    validation={name:evaluate_model(value.model,data.indexed,evaluation,split="validation")[0] for name,value in training.items()}
    test=SignalFinalTestEvaluator().evaluate({name:(value.model,data.indexed,evaluation) for name,value in training.items()})
    h0,m0,d0=_frozen_control(Path(negative_results_dir)/"exposed_non_conversion")
    histories={NO_BIAS:h0,**{n:v.history for n,v in training.items()}}
    metrics={NO_BIAS:m0,**{n:[*validation[n],*test[n][0]] for n in TRAINED_MODELS}}
    diagnostics={NO_BIAS:d0}
    bias_rows=[]
    for name in TRAINED_MODELS:
        diag=model_diagnostics(training[name].model,data.indexed,evaluation,test[name][1])
        bd=bias_diagnostics(training[name].model,data.indexed,evaluation,root/"train_viewplus.csv"); diag["bias_analysis"]=bd
        diagnostics[name]=diag
        for component in ("item_bias","user_bias"):
            if component in bd:
                bias_rows.append({"model":name,"component":component,**bd[component]})
    cart=_cart_comparison(Path(popularity_dir)); comparison=[]
    for k in (5,10,20):
        row=cart[k]; comparison.append({"model":"cart_popularity","k":k,"eligible_users":int(row["eligible_users"]),"recall":float(row["recall"]),"ndcg":float(row["ndcg"]),"hit_rate":float(row["hit_rate"]),"precision":float(row["precision"])})
        for name in MODELS:
            value=next(r for r in metrics[name] if r["split"]=="test" and r["task"]=="purchase" and r["k"]==k)
            comparison.append({"model":name,**{key:value[key] for key in ("k","eligible_users","recall","ndcg","hit_rate","precision")}})
    personalization=[{"model":name,"unique_purchase_top10_lists":diagnostics[name]["unique_purchase_top10_lists"],"average_pairwise_top10_overlap":diagnostics[name]["average_pairwise_top10_overlap"],"average_cart_popularity_top10_overlap":diagnostics[name]["average_cart_popularity_top10_overlap"],"recommended_item_cart_score_mean":diagnostics[name]["recommended_item_cart_score"]["mean"]} for name in MODELS]
    return BiasExperimentResult(settings,data,training,histories,metrics,diagnostics,comparison,bias_rows,personalization)
