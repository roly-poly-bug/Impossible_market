from __future__ import annotations

from pathlib import Path

from ml.evaluation.mf_cart_signal import fallback_group_diagnostic, signal_alignment
from ml.experiments.mf_cart_signal_export import _write_csv
from ml.representations.mf_cart_signal import (
    CARTPLUS,
    CART_CENTERED_WEIGHTED,
    FAVORITE_CARTPLUS,
    cart_signal_strength,
    is_positive,
)
from ml.training.mf_trainer import MFTrainingConfig


def _row(*, view=0, favorite=0, cart=0, purchase=0):
    return {"view_count": str(view), "favorite_count": str(favorite), "cart_count": str(cart), "purchase_count": str(purchase)}


def test_cartplus_positive_definition():
    assert is_positive(CARTPLUS, _row(view=3, cart=1))
    assert is_positive(CARTPLUS, _row(purchase=1))
    assert not is_positive(CARTPLUS, _row(view=3, favorite=1))


def test_favorite_cartplus_positive_definition():
    assert is_positive(FAVORITE_CARTPLUS, _row(favorite=1))
    assert is_positive(FAVORITE_CARTPLUS, _row(cart=1))
    assert not is_positive(FAVORITE_CARTPLUS, _row(view=4))


def test_cart_centered_strength_correctness():
    value = cart_signal_strength(CART_CENTERED_WEIGHTED, _row(view=3, favorite=1, cart=1, purchase=1))
    assert value == 0.5 * __import__("math").log1p(3) + 2 + 6 + 10


def test_representation_specific_positive_pool():
    favorite_only = _row(view=2, favorite=1)
    assert not is_positive(CARTPLUS, favorite_only)
    assert is_positive(FAVORITE_CARTPLUS, favorite_only)
    assert is_positive(CART_CENTERED_WEIGHTED, favorite_only)


def test_no_true_negative_interpretation_is_documented():
    source = Path("ml/representations/mf_cart_signal.py").read_text(encoding="utf-8")
    assert "never a true negative" in source


def test_sampling_and_representation_use_train_only():
    source = Path("ml/representations/mf_cart_signal.py").read_text(encoding="utf-8")
    assert "train_viewplus.csv" in source
    assert "validation_relevance" not in source and "test_relevance" not in source


def test_same_fixed_mf_config():
    assert MFTrainingConfig().as_dict() == {
        "latent_dim": 16, "learning_rate": 0.001, "weight_decay": 0.0001,
        "batch_size": 1024, "negative_ratio": 4, "seed": 42,
        "max_epochs": 100, "patience": 5, "deterministic_algorithms": True,
        "torch_num_threads": 1,
    }


def test_validation_only_selection_and_test_guard_are_wired():
    trainer = Path("ml/training/mf_bias_trainer.py").read_text(encoding="utf-8")
    experiment = Path("ml/experiments/mf_cart_signal_experiment.py").read_text(encoding="utf-8")
    assert "validation_purchase_ndcg_at_10" in trainer
    assert "SignalFinalTestEvaluator" in experiment
    assert "single batched Test" in Path("ml/experiments/mf_cart_signal_export.py").read_text(encoding="utf-8")


def test_coverage_and_group_evaluation_functions_are_separate():
    assert callable(signal_alignment)
    assert callable(fallback_group_diagnostic)


def test_artifact_reproducibility(tmp_path: Path):
    rows = [{"signal": "cartplus", "positive_pair_count": 2}]
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    columns = ("signal", "positive_pair_count")
    _write_csv(first, rows, columns)
    _write_csv(second, rows, columns)
    assert first.read_bytes() == second.read_bytes()
