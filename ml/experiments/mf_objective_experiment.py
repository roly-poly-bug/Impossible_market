from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ml.evaluation.matrix_factorization import evaluate_model, load_evaluation_data, model_diagnostics
from ml.evaluation.mf_bias import bias_diagnostics
from ml.evaluation.mf_latent_dim import history_group_metrics, item_popularity_group_metrics, parameter_counts
from ml.evaluation.mf_objective import pairwise_margin_diagnostics
from ml.evaluation.mf_signal import SignalFinalTestEvaluator
from ml.experiments.mf_negative_sampling_experiment import _cart_comparison
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.representations.mf_cart_signal import EXISTING_WEIGHTED, load_cart_signal
from ml.training.mf_objective import ObjectiveTrainingResult, train_weighted_bpr
from ml.training.mf_trainer import MFTrainingConfig


OBJECTIVES=("bce","bpr"); EXPERIMENT_VERSION="mf_objective_v2"


@dataclass
class ObjectiveExperimentResult:
    config: MFTrainingConfig; data: object; bpr_training: ObjectiveTrainingResult; models: dict
    histories: dict; metrics: dict; diagnostics: dict; runtime: float
    comparison: list; margins: list; personalization: list; bias_rows: list; decomposition: list
    embeddings: list; history_groups: list; popularity_groups: list; capacity: list


def _load_bce(path, users, items):
    payload=torch.load(path,map_location="cpu",weights_only=True);model=BiasedMatrixFactorization(users,items,8,"item_bias");model.load_state_dict(payload["state_dict"]);model.eval();return model


def run_objective_experiment(dataset_dir,popularity_dir,latent_results_dir,*,config=None):
    settings=config or MFTrainingConfig(latent_dim=8); data,_=load_cart_signal(dataset_dir,EXISTING_WEIGHTED,sample_ratio=4,seed=42)
    evaluation=load_evaluation_data(dataset_dir,data.indexed)
    started=time.perf_counter(); bpr=train_weighted_bpr(data,evaluation,settings); runtime=time.perf_counter()-started
    bce=_load_bce(Path(latent_results_dir)/"checkpoints"/"dim8_best.pt",len(data.indexed.user_ids),len(data.indexed.item_ids));models={"bce":bce,"bpr":bpr.model}
    validation={name:evaluate_model(model,data.indexed,evaluation,split="validation")[0] for name,model in models.items()}
    test=SignalFinalTestEvaluator().evaluate({name:(model,data.indexed,evaluation) for name,model in models.items()})
    import csv,json
    with (Path(latent_results_dir)/"dim8"/"training_history.csv").open(encoding="utf-8",newline="") as f: bce_history=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    histories={"bce":bce_history,"bpr":bpr.history};metrics={name:[*validation[name],*test[name][0]] for name in OBJECTIVES}
    diagnostics={};margins=[];personalization=[];bias_rows=[];decomposition=[];embeddings=[];history_groups=[];popularity_groups=[];capacity=[]
    for name in OBJECTIVES:
        model=models[name];rankings=test[name][1];diag=model_diagnostics(model,data.indexed,evaluation,rankings);diag["bias_analysis"]=bias_diagnostics(model,data.indexed,evaluation,Path(dataset_dir)/"train_viewplus.csv");diagnostics[name]=diag
        margins.append({"objective":name,**pairwise_margin_diagnostics(model,data.sampled.triples)})
        personalization.append({"objective":name,"unique_purchase_top10_lists":diag["unique_purchase_top10_lists"],"average_pairwise_top10_overlap":diag["average_pairwise_top10_overlap"],"average_cart_popularity_top10_overlap":diag["average_cart_popularity_top10_overlap"],"recommended_item_cart_score_mean":diag["recommended_item_cart_score"]["mean"]})
        bias=diag["bias_analysis"];corr=bias["item_bias_correlations"];bias_rows.append({"objective":name,**bias["item_bias"],"cart_pearson":corr["cart"]["pearson"],"cart_spearman":corr["cart"]["spearman"],"purchase_pearson":corr["purchase"]["pearson"],"purchase_spearman":corr["purchase"]["spearman"]})
        dec=bias["score_decomposition"];decomposition.append({"objective":name,"personal_variance":dec["personal_variance"],"item_bias_variance":dec["item_bias_variance"],"bias_to_personal_variance_ratio":dec["item_bias_to_personal_variance_ratio"]})
        embeddings.append({"objective":name,**{f"user_norm_{k}":diag["user_embedding_norm"][k] for k in ("mean","std","min","median","max")},**{f"item_norm_{k}":diag["item_embedding_norm"][k] for k in ("mean","std","min","median","max")},**{f"score_{k}":diag["score_distribution"][k] for k in ("mean","std","min","median","max")}})
        history_groups.extend({"objective":name,**row} for row in history_group_metrics(data.indexed,evaluation,rankings));popularity_groups.extend({"objective":name,**row} for row in item_popularity_group_metrics(evaluation,rankings))
        capacity.append({"objective":name,**parameter_counts(model),"training_runtime_seconds":runtime if name=="bpr" else None,"epoch_runtime_seconds":runtime/len(histories[name]) if name=="bpr" else None})
    cart=_cart_comparison(Path(popularity_dir));comparison=[]
    for k in (5,10,20):
        row=cart[k];comparison.append({"model":"cart_popularity","k":k,**{key:int(row[key]) if key=="eligible_users" else float(row[key]) for key in ("eligible_users","recall","ndcg","hit_rate","precision")}})
        for name in OBJECTIVES:
            m=next(row for row in metrics[name] if row["split"]=="test" and row["task"]=="purchase" and row["k"]==k);comparison.append({"model":name,**{key:m[key] for key in ("k","eligible_users","recall","ndcg","hit_rate","precision")}})
    return ObjectiveExperimentResult(settings,data,bpr,models,histories,metrics,diagnostics,runtime,comparison,margins,personalization,bias_rows,decomposition,embeddings,history_groups,popularity_groups,capacity)
