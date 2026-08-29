from __future__ import annotations

from collections.abc import Iterable

from ml.baselines.popularity import (
    SIGNAL_CART,
    SIGNAL_FAVORITEPLUS,
    SIGNAL_LOG_VIEW,
    SIGNAL_PURCHASE,
    SIGNAL_TOTAL_VIEW,
    SIGNAL_UNIQUE_VIEW_USERS,
    SIGNAL_WEIGHTED,
)
from ml.experiments.popularity_experiment import (
    TASK_MATCHED_SIGNALS,
    PopularityExperimentResult,
)


def _table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> list[str]:
    headers = tuple(headers)
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _pct(value: object) -> str:
    return f"{float(value):.4%}"


def _best_rows(result: PopularityExperimentResult) -> list[dict[str, object]]:
    test_k10 = [
        row for row in result.metrics if row["split"] == "test" and row["k"] == 10
    ]
    output = []
    for task in ("viewplus", "favoriteplus", "purchase"):
        rows = [row for row in test_k10 if row["evaluation_task"] == task]
        best_ndcg = max(
            rows,
            key=lambda row: (row["ndcg"], row["recall"], row["popularity_signal"]),
        )
        best_recall = max(
            rows,
            key=lambda row: (row["recall"], row["ndcg"], row["popularity_signal"]),
        )
        output.append(
            {
                "task": task,
                "best_ndcg_signal": best_ndcg["popularity_signal"],
                "best_ndcg": best_ndcg["ndcg"],
                "best_ndcg_recall": best_ndcg["recall"],
                "best_recall_signal": best_recall["popularity_signal"],
                "best_recall": best_recall["recall"],
                "best_recall_ndcg": best_recall["ndcg"],
            }
        )
    return output


def render_popularity_quality_report(result: PopularityExperimentResult) -> str:
    weights = result.weighted_config
    lines = [
        "# Popularity Baseline v1 Quality Report",
        "",
        "This report evaluates non-personalized global rankings built only from the Recommendation Dataset v1 Train split.",
        "",
        "## Experiment contract",
        "",
        "- Experiment: `popularity_baseline_v1`",
        "- Dataset: `recommendation_dataset_v1`, seed `42`",
        "- Score source: Train only; Validation/Test events never affect popularity scores.",
        "- Personalization: none. Every eligible User starts from the same global ranking.",
        "- Main evaluation: task-specific candidates, task-specific Train seen exclusion enabled, identical relevance and metrics across signals.",
        "- Tie-break: score descending, then `product_id` ascending.",
        "- Cold candidates remain with score 0.",
        "- View+/Favorite+ candidates: all 200 Products; Purchase candidates: 170 `available` Products.",
        "",
        "A zero interaction representation means no positive signal was observed for that representation. It is not a true negative, and Unknown is not converted to a negative label.",
        "",
        "## Popularity signals",
        "",
        f"- `{SIGNAL_TOTAL_VIEW}`: sum of all Train View events, including repeated Views.",
        f"- `{SIGNAL_UNIQUE_VIEW_USERS}`: number of Train Users with at least one View.",
        f"- `{SIGNAL_LOG_VIEW}`: sum of `log1p(view_count_ui)` across Users.",
        f"- `{SIGNAL_FAVORITEPLUS}`: unique Train user-item pairs with Favorite, Cart, or Purchase. Pair counting avoids triple-crediting the same funnel progression.",
        f"- `{SIGNAL_CART}`: Train Add-to-Cart count.",
        f"- `{SIGNAL_PURCHASE}`: Train Purchase count.",
        f"- `{SIGNAL_WEIGHTED}`: `{weights.view_weight}*log1p(View) + {weights.favorite_weight}*Favorite + {weights.cart_weight}*Cart + {weights.purchase_weight}*Purchase` summed over user-item pairs.",
        "",
        "The weighted coefficients are a v1 hypothesis, not optimized values or ground truth.",
        "",
        "## Task-matched results",
        "",
    ]
    task_matched = [
        row
        for row in result.metrics
        if row["popularity_signal"] == TASK_MATCHED_SIGNALS[row["evaluation_task"]]
    ]
    lines.extend(
        _table(
            ("task", "split", "K", "Recall", "NDCG", "HitRate", "Precision", "eligible"),
            (
                (
                    str(row["evaluation_task"]),
                    str(row["split"]),
                    str(row["k"]),
                    _pct(row["recall"]),
                    _pct(row["ndcg"]),
                    _pct(row["hit_rate"]),
                    _pct(row["precision"]),
                    str(row["eligible_users"]),
                )
                for row in task_matched
            ),
        )
    )
    lines.extend(("", "## Purchase cross-signal results at K=10", ""))
    purchase = [
        row for row in result.purchase_cross_signal_metrics if row["k"] == 10
    ]
    lines.extend(
        _table(
            ("signal", "split", "Recall@10", "NDCG@10", "HitRate@10", "Precision@10", "eligible"),
            (
                (
                    str(row["popularity_signal"]),
                    str(row["split"]),
                    _pct(row["recall"]),
                    _pct(row["ndcg"]),
                    _pct(row["hit_rate"]),
                    _pct(row["precision"]),
                    str(row["eligible_users"]),
                )
                for row in purchase
            ),
        )
    )
    lines.extend(("", "## All-task cross-signal results at K=10", ""))
    all_k10 = [row for row in result.metrics if row["k"] == 10]
    lines.extend(
        _table(
            ("signal", "task", "split", "Recall@10", "NDCG@10", "eligible"),
            (
                (
                    str(row["popularity_signal"]),
                    str(row["evaluation_task"]),
                    str(row["split"]),
                    _pct(row["recall"]),
                    _pct(row["ndcg"]),
                    str(row["eligible_users"]),
                )
                for row in all_k10
            ),
        )
    )
    lines.extend(("", "## Best Test baseline by task", ""))
    best_rows = _best_rows(result)
    lines.extend(
        _table(
            ("task", "best NDCG signal", "NDCG@10", "Recall@10", "best Recall signal", "Recall@10", "NDCG@10"),
            (
                (
                    str(row["task"]),
                    str(row["best_ndcg_signal"]),
                    _pct(row["best_ndcg"]),
                    _pct(row["best_ndcg_recall"]),
                    str(row["best_recall_signal"]),
                    _pct(row["best_recall"]),
                    _pct(row["best_recall_ndcg"]),
                )
                for row in best_rows
            ),
        )
    )
    lines.extend(("", "## Interaction representation statistics (nonzero values)", ""))
    lines.extend(
        _table(
            ("representation", "pairs", "density", "mean", "std", "min", "p25", "median", "p75", "max"),
            (
                (
                    str(row["representation"]),
                    f"{int(row['nonzero_pair_count']):,}",
                    _pct(row["density"]),
                    f"{float(row['mean_nonzero']):.4f}",
                    f"{float(row['std_nonzero']):.4f}",
                    f"{float(row['min_nonzero']):.4f}",
                    f"{float(row['p25_nonzero']):.4f}",
                    f"{float(row['median_nonzero']):.4f}",
                    f"{float(row['p75_nonzero']):.4f}",
                    f"{float(row['max_nonzero']):.4f}",
                )
                for row in result.representation_stats
            ),
        )
    )
    lines.extend(("", "## Signal richness versus future Purchase", ""))
    lines.extend(
        _table(
            ("signal", "pairs", "users", "items", "density", "Test Purchase Recall@10", "NDCG@10"),
            (
                (
                    str(row["signal"]),
                    f"{int(row['nonzero_pair_count']):,}",
                    f"{int(row['user_coverage']):,}",
                    f"{int(row['item_coverage']):,}",
                    _pct(row["density"]),
                    _pct(row["test_purchase_recall_at_10"]),
                    _pct(row["test_purchase_ndcg_at_10"]),
                )
                for row in result.signal_richness
            ),
        )
    )
    heavy = result.heavy_user
    overlap = result.signal_overlap
    largest_shift = max(
        result.stability,
        key=lambda row: abs(float(row["ndcg_difference_test_minus_validation"])),
    )
    largest_recall_shift = max(
        result.stability,
        key=lambda row: abs(float(row["recall_difference_test_minus_validation"])),
    )
    lines.extend(
        (
            "",
            "## Heavy User and Raw-vs-Log audit",
            "",
            f"- Top 1% observed-activity Users contribute {_pct(heavy['top_1_percent_raw_share'])} of raw View strength and {_pct(heavy['top_1_percent_log_share'])} after per-pair log1p compression.",
            f"- Maximum per-User strength changes from {float(heavy['max_raw_strength']):.1f} raw to {float(heavy['max_log_strength']):.4f} log strength.",
            "- Activity deciles are computed only from observed Train Views; the hidden synthetic activity tier is not loaded.",
            "",
            "## Event overlap audit",
            "",
            f"- View pairs: {overlap['view_pairs']:,}; Favorite+ pairs: {overlap['favoriteplus_pairs']:,}; Purchase pairs: {overlap['purchase_pairs']:,}.",
            f"- Favorite+ outside View: {overlap['favoriteplus_outside_view']}; Purchase outside View: {overlap['purchase_outside_view']}; Purchase outside Favorite+: {overlap['purchase_outside_favoriteplus']}.",
            f"- Maximum Favorite/Cart/Purchase count per pair: {overlap['max_favorite_count_per_pair']}/{overlap['max_cart_count_per_pair']}/{overlap['max_purchase_count_per_pair']}.",
            "",
            "## Validation/Test stability",
            "",
            f"The largest absolute NDCG@10 shift is `{largest_shift['popularity_signal']}` on `{largest_shift['evaluation_task']}`: {_pct(largest_shift['validation_ndcg_at_10'])} Validation to {_pct(largest_shift['test_ndcg_at_10'])} Test (difference {float(largest_shift['ndcg_difference_test_minus_validation']):+.4f}). No Test result was used to tune a score or weight.",
            f"The largest absolute Recall@10 shift is `{largest_recall_shift['popularity_signal']}` on `{largest_recall_shift['evaluation_task']}`: {_pct(largest_recall_shift['validation_recall_at_10'])} Validation to {_pct(largest_recall_shift['test_recall_at_10'])} Test (difference {float(largest_recall_shift['recall_difference_test_minus_validation']):+.4f}).",
            "",
            "## Interpretation for the next phase",
            "",
            "- Popularity is a non-personalized control: all Users receive the same global order before task-specific seen exclusion.",
            "- View is weak but rich; Purchase is strong but sparse. Log View reduces repeat-view dominance while retaining broad coverage.",
            "- Weighted implicit v1 should remain a candidate, not the assumed answer. Matrix Factorization v1 should compare binary View+, log View count, Favorite+, Purchase-only, and weighted implicit under the same split and evaluation policy.",
            "- Recommended MF v1 sequence: establish binary View+ as the simplest dense control, compare log View as the count-aware primary candidate, then add Favorite+ and weighted implicit as challengers. Purchase-only is too sparse to be the sole first representation.",
            "- Do not make weighted implicit the only initial MF input: Cart popularity beat weighted v1 on Test Purchase, and weighted Purchase Recall@10 moved more between Validation and Test. Keep the weights configurable and compare representations before tuning them.",
            "- Hidden User preferences, Product ground-truth attributes, archetypes, preference_match, and Validation/Test Events are absent from scoring and representations.",
            "",
        )
    )
    return "\n".join(lines)
