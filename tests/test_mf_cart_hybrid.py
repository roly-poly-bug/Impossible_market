from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ml.evaluation.mf_cart_hybrid import ALPHAS, FinalHybridTestEvaluator, HybridRanker, hybrid_score, select_alpha, zscore
from ml.evaluation.matrix_factorization import load_evaluation_data
from ml.experiments.mf_cart_signal_export import _write_csv
from ml.experiments.mf_cart_hybrid_experiment import _load_model
from ml.representations.mf_cart_signal import EXISTING_WEIGHTED, load_cart_signal


def _fixture():
    data, _ = load_cart_signal("data/recommendation_v1", EXISTING_WEIGHTED, sample_ratio=4, seed=42)
    evaluation = load_evaluation_data("data/recommendation_v1", data.indexed)
    model = _load_model(Path("results/mf_latent_dim_v1/checkpoints/dim8_best.pt"), len(data.indexed.user_ids), len(data.indexed.item_ids))
    return data, evaluation, HybridRanker(model, data.indexed, evaluation)


def test_cart_popularity_is_loaded_from_train_only():
    source = Path("ml/evaluation/matrix_factorization.py").read_text(encoding="utf-8")
    assert 'root / "train_viewplus.csv"' in source
    assert "validation" not in source[source.index("train_rows ="):source.index("return EvaluationData")]
    assert "test" not in source[source.index("train_rows ="):source.index("return EvaluationData")]


def test_mf_checkpoint_is_frozen_and_no_training_code_is_called():
    checkpoint = Path("results/mf_latent_dim_v1/checkpoints/dim8_best.pt")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    data, _, _ = _fixture()
    after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source = Path("ml/experiments/mf_cart_hybrid_experiment.py").read_text(encoding="utf-8")
    assert before == after
    assert "optimizer" not in source and "backward(" not in source and "train_bias_model" not in source
    assert len(data.indexed.item_ids) == 200


def test_zscore_normalization_correctness():
    values = zscore(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(values.mean(), torch.tensor(0.0, dtype=torch.float64))
    assert torch.isclose(values.std(unbiased=False), torch.tensor(1.0, dtype=torch.float64))


def test_zscore_zero_std_is_safe():
    assert torch.equal(zscore(torch.tensor([4.0, 4.0])), torch.zeros(2, dtype=torch.float64))


def test_hybrid_score_correctness():
    mf = torch.tensor([-1.0, 1.0]); cart = torch.tensor([1.0, -1.0])
    assert torch.equal(hybrid_score(mf, cart, 0.25), torch.tensor([0.5, -0.5]))


def test_alpha_endpoints_equal_mf_and_cart_rankings():
    data, evaluation, ranker = _fixture()
    user = next(user for user in data.indexed.user_ids if data.indexed.user_to_index[user] not in data.indexed.cold_user_indices)
    seen = set(evaluation.seen["purchase"].get(user, ()))
    candidates, mf_z, _ = ranker.score_components(user, "purchase")
    expected_mf = sorted((item for item in candidates if item not in seen), key=lambda item: (-float(ranker.mf_scores[data.indexed.user_to_index[user], data.indexed.item_to_index[item]]), item))
    expected_cart = [item for item in evaluation.cart_rankings["purchase"] if item not in seen]
    assert ranker.ranking(user, "purchase", 1.0) == expected_mf
    assert ranker.ranking(user, "purchase", 0.0) == expected_cart


def test_ranking_is_deterministic():
    data, _, ranker = _fixture(); user = data.indexed.user_ids[0]
    assert ranker.ranking(user, "purchase", 0.5) == ranker.ranking(user, "purchase", 0.5)


def test_alpha_selection_is_validation_only_with_documented_ties():
    rows = [{"task": "purchase", "split": "validation", "k": 10, "alpha": alpha, "ndcg": 0.2, "recall": 0.3} for alpha in ALPHAS]
    assert select_alpha(rows) == 1.0
    rows[2]["ndcg"] = 0.21
    assert select_alpha(rows) == 0.5


def test_test_guard_prevents_alpha_sweep():
    evaluator = FinalHybridTestEvaluator()
    evaluator._used = True
    try:
        evaluator.evaluate(None, 0.5)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Test evaluator must be single-use")


def test_candidate_seen_and_fallback_policies_are_shared():
    data, evaluation, ranker = _fixture()
    cold = data.indexed.user_ids[next(iter(data.indexed.cold_user_indices))]
    seen = set(evaluation.seen["purchase"].get(cold, ()))
    assert ranker.ranking(cold, "purchase", 0.5) == [item for item in evaluation.cart_rankings["purchase"] if item not in seen]
    assert set(ranker.score_components(data.indexed.user_ids[0], "purchase")[0]) == set(evaluation.candidates["purchase"])
    assert not (seen & set(ranker.ranking(cold, "purchase", 0.5)))


def test_fixed_alpha_grid_has_no_fine_search():
    assert ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_csv_artifact_reproducibility(tmp_path: Path):
    rows = [{"alpha": 0.5, "ndcg": 0.1}]
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_csv(first, rows, ("alpha", "ndcg")); _write_csv(second, rows, ("alpha", "ndcg"))
    assert first.read_bytes() == second.read_bytes()
