from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median, pstdev

import torch

from ml.evaluation.matrix_factorization import EvaluationData
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.training.mf_data import IndexedInteractions


def _summary(values: list[float]) -> dict[str, float]:
    return {"mean": fmean(values), "std": pstdev(values), "min": min(values), "median": median(values), "max": max(values)}


def _pearson(left: list[float], right: list[float]) -> float:
    lm, rm = fmean(left), fmean(right)
    numerator = sum((a-lm)*(b-rm) for a,b in zip(left,right))
    denominator = math.sqrt(sum((a-lm)**2 for a in left) * sum((b-rm)**2 for b in right))
    return numerator / denominator if denominator else 0.0


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]: ranks[index] = rank
        start = end
    return ranks


def _correlations(signal: list[float], target: list[float]) -> dict[str, float]:
    return {"pearson": _pearson(signal, target), "spearman": _pearson(_ranks(signal), _ranks(target))}


def load_train_counts(path: str | Path, indexed: IndexedInteractions) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    item = {name: [0.0]*len(indexed.item_ids) for name in ("view", "cart", "purchase")}
    user = {name: [0.0]*len(indexed.user_ids) for name in ("view_pairs", "deep_events")}
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            u, i = indexed.user_to_index[row["user_id"]], indexed.item_to_index[row["product_id"]]
            view, favorite, cart, purchase = (float(row[k]) for k in ("view_count","favorite_count","cart_count","purchase_count"))
            item["view"][i] += view; item["cart"][i] += cart; item["purchase"][i] += purchase
            user["view_pairs"][u] += float(view > 0)
            user["deep_events"][u] += favorite + cart + purchase
    return item, user


def bias_diagnostics(model: BiasedMatrixFactorization, indexed: IndexedInteractions, evaluation: EvaluationData, train_path: str | Path) -> dict[str, object]:
    item_counts, user_counts = load_train_counts(train_path, indexed)
    item_bias = model.item_bias.weight.detach().squeeze(-1).cpu().tolist()
    result: dict[str, object] = {
        "item_bias": _summary(item_bias),
        "item_bias_correlations": {name: _correlations(item_bias, values) for name, values in item_counts.items()},
    }
    if model.user_bias is not None:
        values = model.user_bias.weight.detach().squeeze(-1).cpu().tolist()
        result["user_bias"] = _summary(values)
        result["user_bias_correlations"] = {name: _correlations(values, counts) for name, counts in user_counts.items()}
    with torch.no_grad():
        users = torch.arange(len(indexed.user_ids))
        personal = model.user_embeddings(users) @ model.item_embeddings.weight.T
        candidates = torch.tensor([indexed.item_to_index[item] for item in evaluation.candidates["purchase"]])
        personal_values = personal[:, candidates].flatten().cpu().tolist()
        item_values = model.item_bias.weight.squeeze(-1)[candidates].repeat(len(users)).cpu().tolist()
    result["score_decomposition"] = {
        "personal_component": _summary(personal_values),
        "item_bias_component": _summary(item_values),
        "personal_variance": pstdev(personal_values) ** 2,
        "item_bias_variance": pstdev(item_values) ** 2,
        "item_bias_to_personal_variance_ratio": (pstdev(item_values) ** 2) / (pstdev(personal_values) ** 2),
    }
    return result
