from __future__ import annotations

from ml.experiments.mf_experiment import MFExperimentResult


def _pct(value: object) -> str:
    return f"{float(value):.4%}"


def _table(headers, rows) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _metric(result: MFExperimentResult, model: str, split: str, task: str, k: int = 10):
    return next(
        row
        for row in result.metrics[model]
        if row["split"] == split and row["task"] == task and row["k"] == k
    )


def render_mf_quality_report(result: MFExperimentResult) -> str:
    config = result.config
    lines = [
        "# Matrix Factorization v1 Quality Report",
        "",
        "## Scope and semantics",
        "",
        "BCE MF learns target 1 for Train Binary View+ and target 0 for sampled Unknown using `BCEWithLogitsLoss`. Target 0 is a sampled training non-positive, not true dislike.",
        "",
        "BPR MF learns `score(u, positive) > score(u, sampled unknown)` with `-logsigmoid(score_positive-score_unknown)`. This is a pairwise training assumption, not proof that the Unknown item is disliked.",
        "",
        "Observed Non-conversion pairs are excluded from v1 sampling. Hidden user preferences, Product attributes, archetypes, preference_match, Validation events, and Test events are not training features.",
        "",
        "## Fixed training configuration",
        "",
        f"- Architecture: bias-free `user_embedding · item_embedding`; latent dimension `{config.latent_dim}`",
        f"- Positive signal: Train Binary View+ ({len(result.indexed.positive_pairs):,} pairs)",
        f"- Sampling: `{config.negative_ratio}` distinct Random Unknown items per positive, seed `{config.seed}` ({len(result.sampled.triples):,} BPR triples)",
        f"- Adam: learning rate `{config.learning_rate}`, weight decay `{config.weight_decay}`",
        f"- Batch size `{config.batch_size}`, max epochs `{config.max_epochs}`, patience `{config.patience}`",
        "- Early stopping and checkpoint selection: Validation Purchase NDCG@10 only",
        "- Test: one final full-ranking pass after both best checkpoints were fixed",
        "- Candidate and task-specific seen policies: unchanged from Recommendation Dataset v1",
        "",
        "## Training behavior",
        "",
    ]
    lines.extend(
        _table(
            ("model", "epochs run", "best epoch", "first loss", "best-epoch loss", "best Val Purchase NDCG@10"),
            (
                (
                    model,
                    str(len(training.history)),
                    str(training.best_epoch),
                    f"{float(training.history[0]['train_loss']):.6f}",
                    f"{float(training.history[training.best_epoch - 1]['train_loss']):.6f}",
                    _pct(training.best_validation_purchase_ndcg_at_10),
                )
                for model, training in result.training.items()
            ),
        )
    )
    lines.extend(("", "## Validation and Test results at K=10", ""))
    lines.extend(
        _table(
            ("model", "split", "task", "Recall@10", "NDCG@10", "HitRate@10", "Precision@10", "eligible"),
            (
                (
                    model,
                    split,
                    task,
                    _pct(row["recall"]),
                    _pct(row["ndcg"]),
                    _pct(row["hit_rate"]),
                    _pct(row["precision"]),
                    str(row["eligible_users"]),
                )
                for model in ("bce", "bpr")
                for split in ("validation", "test")
                for task in ("purchase", "viewplus", "favoriteplus")
                for row in (_metric(result, model, split, task),)
            ),
        )
    )
    lines.extend(("", "## Purchase Test comparison", ""))
    lines.extend(
        _table(
            ("model", "K", "Recall", "NDCG", "HitRate", "Precision", "eligible"),
            (
                (
                    str(row["model"]),
                    str(row["k"]),
                    _pct(row["recall"]),
                    _pct(row["ndcg"]),
                    _pct(row["hit_rate"]),
                    _pct(row["precision"]),
                    str(row["eligible_users"]),
                )
                for row in result.comparison
            ),
        )
    )
    lines.extend(("", "## Coverage and diagnostics", ""))
    lines.extend(
        (
            f"- Train View+ zero-interaction Users: `{len(result.indexed.cold_user_indices)}`. They use the Train-only Cart popularity fallback rather than untrained random user embeddings.",
            f"- Train View+ zero-interaction Items: `{len(result.indexed.cold_item_indices)}`.",
            "- All evaluation scores use full task candidate sets, never sampled ranking.",
            "",
        )
    )
    for model in ("bce", "bpr"):
        diagnostics = result.diagnostics[model]
        lines.extend(
            (
                f"### {model.upper()} diagnostics",
                "",
                f"- User embedding norm mean/std/min/median/max: {diagnostics['user_embedding_norm']['mean']:.4f} / {diagnostics['user_embedding_norm']['std']:.4f} / {diagnostics['user_embedding_norm']['min']:.4f} / {diagnostics['user_embedding_norm']['median']:.4f} / {diagnostics['user_embedding_norm']['max']:.4f}",
                f"- Item embedding norm mean/std/min/median/max: {diagnostics['item_embedding_norm']['mean']:.4f} / {diagnostics['item_embedding_norm']['std']:.4f} / {diagnostics['item_embedding_norm']['min']:.4f} / {diagnostics['item_embedding_norm']['median']:.4f} / {diagnostics['item_embedding_norm']['max']:.4f}",
                f"- Score mean/std/min/median/max: {diagnostics['score_distribution']['mean']:.4f} / {diagnostics['score_distribution']['std']:.4f} / {diagnostics['score_distribution']['min']:.4f} / {diagnostics['score_distribution']['median']:.4f} / {diagnostics['score_distribution']['max']:.4f}",
                f"- Unique Purchase Top10 lists: {diagnostics['unique_purchase_top10_lists']}/{diagnostics['purchase_evaluation_users']}; average pairwise overlap: {_pct(diagnostics['average_pairwise_top10_overlap'])}",
                f"- Mean overlap with Cart Popularity Top10: {_pct(diagnostics['average_cart_popularity_top10_overlap'])}",
                f"- All embedding and score values finite: `{diagnostics['all_values_finite']}`",
                "",
            )
        )
    bce = _metric(result, "bce", "test", "purchase")
    bpr = _metric(result, "bpr", "test", "purchase")
    better = "bce" if float(bce["ndcg"]) >= float(bpr["ndcg"]) else "bpr"
    lines.extend(
        (
            "## Observed issues",
            "",
            "- Neither fixed-config MF objective beats Cart Popularity on Purchase Test Recall@10 or NDCG@10.",
            "- BPR has the stronger Validation Purchase NDCG@10 but drops more on Test, so the Validation advantage is not stable in this single temporal split.",
            "- BCE recommendations are personalized by list identity, but their high average pairwise Top10 overlap shows a strong shared-ranking component.",
            "- The 29 Users without Train View+ positives require the documented Train-only popularity fallback.",
            "",
            "## Interpretation and next phase",
            "",
            f"- `{better.upper()}` is the recommended main v1 objective because it has the higher Purchase Test Recall@10 and NDCG@10 of the two fixed objectives. This is a single fixed-config result, not a hyperparameter conclusion.",
            "- Compare both MF results with Cart Popularity before claiming personalization improved ranking quality.",
            "- The next single representation experiment should be Log View: it preserves broad View coverage while compressing repeats. Favorite+ can follow; Weighted should remain later because it changes multiple assumptions at once.",
            "- No parameter was changed after seeing Test. Test was not used for early stopping or model selection.",
            "",
        )
    )
    return "\n".join(lines)
