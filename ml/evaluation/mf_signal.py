from __future__ import annotations

from statistics import fmean

from ml.evaluation.matrix_factorization import EvaluationData, evaluate_model
from ml.evaluation.metrics import evaluate_ranking
from ml.models.matrix_factorization import MatrixFactorization
from ml.training.mf_data import IndexedInteractions


class SignalFinalTestEvaluator:
    def __init__(self) -> None:
        self._used = False

    def evaluate(
        self,
        experiments: dict[str, tuple[MatrixFactorization, IndexedInteractions, EvaluationData]],
    ) -> dict[str, tuple[list[dict[str, object]], dict[str, list[str]]]]:
        if self._used:
            raise RuntimeError("Test evaluation is allowed only once per experiment run")
        self._used = True
        return {
            name: evaluate_model(model, indexed, evaluation, split="test")
            for name, (model, indexed, evaluation) in experiments.items()
        }


def purchase_cold_start_diagnostic(
    indexed: IndexedInteractions,
    evaluation: EvaluationData,
    purchase_top10: dict[str, list[str]],
) -> dict[str, object]:
    relevance = evaluation.relevance["test"]["purchase"]
    eligible_users = sorted(user for user, items in relevance.items() if items)
    fallback_users = [
        user
        for user in eligible_users
        if indexed.user_to_index[user] in indexed.cold_user_indices
    ]
    learned_users = [user for user in eligible_users if user not in set(fallback_users)]

    def summarize(users: list[str]) -> dict[str, float | int | None]:
        values = [
            evaluate_ranking(purchase_top10[user], relevance[user], k=10)
            for user in users
        ]
        return {
            "user_count": len(users),
            "recall_at_10": fmean(value["recall"] for value in values) if values else None,
            "ndcg_at_10": fmean(value["ndcg"] for value in values) if values else None,
            "hit_rate_at_10": fmean(value["hit_rate"] for value in values) if values else None,
            "precision_at_10": fmean(value["precision"] for value in values) if values else None,
        }

    return {
        "train_positive_zero_user_count": len(indexed.cold_user_indices),
        "test_eligible_purchase_user_count": len(eligible_users),
        "test_eligible_with_zero_train_positive": len(fallback_users),
        "fallback_usage_count": len(fallback_users),
        "fallback_policy": "Train-only Cart popularity",
        "fallback_users": summarize(fallback_users),
        "learned_users": summarize(learned_users),
    }
