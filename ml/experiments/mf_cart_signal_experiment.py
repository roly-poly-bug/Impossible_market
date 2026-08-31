from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ml.evaluation.matrix_factorization import evaluate_model, load_evaluation_data, model_diagnostics
from ml.evaluation.mf_bias import bias_diagnostics
from ml.evaluation.mf_cart_signal import fallback_group_diagnostic, signal_alignment
from ml.evaluation.mf_signal import SignalFinalTestEvaluator
from ml.experiments.mf_bias_experiment import _frozen_control
from ml.experiments.mf_negative_sampling_experiment import _cart_comparison
from ml.representations.mf_cart_signal import (
    CARTPLUS,
    CART_CENTERED_WEIGHTED,
    EXISTING_WEIGHTED,
    FAVORITE_CARTPLUS,
    SIGNALS,
    load_cart_signal,
)
from ml.representations.mf_signal import RepresentationData
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.training.mf_bias_trainer import BiasTrainingResult, train_bias_model
from ml.training.mf_negative_sampling import NegativeSamplingResult
from ml.training.mf_trainer import MFTrainingConfig


TRAINED_SIGNALS = (CARTPLUS, FAVORITE_CARTPLUS, CART_CENTERED_WEIGHTED)
EXPERIMENT_VERSION = "mf_cart_signal_v1"


@dataclass
class CartSignalExperimentResult:
    config: MFTrainingConfig
    data: dict[str, RepresentationData]
    sampling: dict[str, NegativeSamplingResult]
    training: dict[str, BiasTrainingResult]
    histories: dict[str, list[dict[str, object]]]
    metrics: dict[str, list[dict[str, object]]]
    diagnostics: dict[str, dict[str, object]]
    comparison: list[dict[str, object]]
    coverage: list[dict[str, object]]
    fallback: list[dict[str, object]]
    personalization: list[dict[str, object]]
    bias_rows: list[dict[str, object]]
    alignment: list[dict[str, object]]


def run_cart_signal_experiment(
    dataset_dir: str | Path,
    popularity_dir: str | Path,
    bias_results_dir: str | Path,
    *,
    config: MFTrainingConfig | None = None,
) -> CartSignalExperimentResult:
    settings = config or MFTrainingConfig()
    root = Path(dataset_dir)
    data: dict[str, RepresentationData] = {}
    sampling: dict[str, NegativeSamplingResult] = {}
    evaluations = {}
    for signal in SIGNALS:
        data[signal], sampling[signal] = load_cart_signal(root, signal, sample_ratio=4, seed=42)
        evaluations[signal] = load_evaluation_data(root, data[signal].indexed)

    training = {
        signal: train_bias_model("item_bias", data[signal], evaluations[signal], settings)
        for signal in TRAINED_SIGNALS
    }
    validation = {
        signal: evaluate_model(result.model, data[signal].indexed, evaluations[signal], split="validation")[0]
        for signal, result in training.items()
    }
    control_payload = torch.load(Path(bias_results_dir) / "checkpoints" / "item_bias_best.pt", map_location="cpu", weights_only=True)
    control_model = BiasedMatrixFactorization(
        len(data[EXISTING_WEIGHTED].indexed.user_ids), len(data[EXISTING_WEIGHTED].indexed.item_ids), settings.latent_dim, "item_bias"
    )
    control_model.load_state_dict(control_payload["state_dict"])
    control_model.eval()
    test = SignalFinalTestEvaluator().evaluate({
        EXISTING_WEIGHTED: (control_model, data[EXISTING_WEIGHTED].indexed, evaluations[EXISTING_WEIGHTED]),
        **{
            signal: (result.model, data[signal].indexed, evaluations[signal])
            for signal, result in training.items()
        },
    })

    control_history, control_metrics, control_diagnostics = _frozen_control(Path(bias_results_dir) / "item_bias")
    control_diagnostics["source"] = "frozen MF Bias v1 Item Bias control; Test not re-evaluated"
    control_validation = [row for row in control_metrics if row["split"] == "validation"]
    histories = {EXISTING_WEIGHTED: control_history, **{name: value.history for name, value in training.items()}}
    metrics = {EXISTING_WEIGHTED: [*control_validation, *test[EXISTING_WEIGHTED][0]], **{name: [*validation[name], *test[name][0]] for name in TRAINED_SIGNALS}}
    control_diagnostics = model_diagnostics(control_model, data[EXISTING_WEIGHTED].indexed, evaluations[EXISTING_WEIGHTED], test[EXISTING_WEIGHTED][1])
    control_diagnostics["source"] = "frozen MF Bias v1 Item Bias checkpoint; evaluated in the single final Test pass"
    control_diagnostics["bias_analysis"] = bias_diagnostics(control_model, data[EXISTING_WEIGHTED].indexed, evaluations[EXISTING_WEIGHTED], root / "train_viewplus.csv")
    diagnostics: dict[str, dict[str, object]] = {EXISTING_WEIGHTED: control_diagnostics}
    fallback = []
    bias_rows = []
    for signal in TRAINED_SIGNALS:
        model = training[signal].model
        diag = model_diagnostics(model, data[signal].indexed, evaluations[signal], test[signal][1])
        diag["bias_analysis"] = bias_diagnostics(model, data[signal].indexed, evaluations[signal], root / "train_viewplus.csv")
        diag["fallback_groups"] = fallback_group_diagnostic(data[signal].indexed, evaluations[signal], test[signal][1])
        diagnostics[signal] = diag
        fallback.extend({"signal": signal, **row} for row in diag["fallback_groups"])
        item = diag["bias_analysis"]["item_bias"]
        corr = diag["bias_analysis"]["item_bias_correlations"]
        bias_rows.append({
            "signal": signal, **item,
            "cart_pearson": corr["cart"]["pearson"], "cart_spearman": corr["cart"]["spearman"],
            "purchase_pearson": corr["purchase"]["pearson"], "purchase_spearman": corr["purchase"]["spearman"],
        })
    control_bias = diagnostics[EXISTING_WEIGHTED]["bias_analysis"]
    control_item = control_bias["item_bias"]
    control_corr = control_bias["item_bias_correlations"]
    bias_rows.insert(0, {
        "signal": EXISTING_WEIGHTED, **control_item,
        "cart_pearson": control_corr["cart"]["pearson"], "cart_spearman": control_corr["cart"]["spearman"],
        "purchase_pearson": control_corr["purchase"]["pearson"], "purchase_spearman": control_corr["purchase"]["spearman"],
    })

    fallback.extend(
        {"signal": EXISTING_WEIGHTED, **row}
        for row in fallback_group_diagnostic(data[EXISTING_WEIGHTED].indexed, evaluations[EXISTING_WEIGHTED], test[EXISTING_WEIGHTED][1])
    )

    cart = _cart_comparison(Path(popularity_dir))
    comparison = []
    for k in (5, 10, 20):
        row = cart[k]
        comparison.append({"model": "cart_popularity", "k": k, **{key: float(row[key]) if key != "eligible_users" else int(row[key]) for key in ("eligible_users", "recall", "ndcg", "hit_rate", "precision")}})
        for signal in SIGNALS:
            value = next(row for row in metrics[signal] if row["split"] == "test" and row["task"] == "purchase" and row["k"] == k)
            comparison.append({"model": signal, **{key: value[key] for key in ("k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")}})

    coverage = []
    alignment = []
    for signal in SIGNALS:
        align = signal_alignment(data[signal].indexed, evaluations[signal])
        alignment.append({"signal": signal, **align})
        cov = data[signal].coverage
        coverage.append({
            "signal": signal,
            "positive_pair_count": cov["positive_pair_count"],
            "positive_user_count": cov["training_user_count"],
            "positive_item_count": cov["training_item_count"],
            "density": cov["density"],
            "train_positive_zero_user_count": cov["cold_user_count"],
            "test_purchase_eligible_users": align["test_purchase_eligible_users"],
            "test_eligible_with_train_positive": align["users_with_train_positive"],
            "test_eligible_without_train_positive": align["users_without_train_positive"],
            "test_fallback_rate": align["users_without_train_positive"] / align["test_purchase_eligible_users"],
        })

    personalization = []
    for signal in SIGNALS:
        diag = diagnostics[signal]
        personalization.append({
            "signal": signal,
            "unique_purchase_top10_lists": diag["unique_purchase_top10_lists"],
            "average_pairwise_top10_overlap": diag["average_pairwise_top10_overlap"],
            "average_cart_popularity_top10_overlap": diag["average_cart_popularity_top10_overlap"],
            "recommended_item_cart_score_mean": diag["recommended_item_cart_score"]["mean"],
        })
    return CartSignalExperimentResult(settings, data, sampling, training, histories, metrics, diagnostics, comparison, coverage, fallback, personalization, bias_rows, alignment)
