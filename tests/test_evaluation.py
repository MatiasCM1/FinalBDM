import json

import numpy as np
import pandas as pd
import pytest

from tpf.evaluation.metrics import evaluate
from tpf.evaluation.thresholds import choose_thresholds


def test_all_negative_predictions():
    result = evaluate([0, 1, 0, 1], [0.0, 0.0, 0.0, 0.0])
    assert result["f1_fraud"] == 0
    assert result["mcc"] == 0
    assert result["balanced_accuracy"] == 0.5
    assert result["auc_roc"] == 0.5


def test_weighted_metrics_and_reliability_curve():
    result = evaluate([0, 1], [0.1, 0.8], sample_weight=[9, 1], n_bins=1)
    assert result["brier"] == pytest.approx(0.013)
    assert result["ece"] == pytest.approx(0.07)
    assert result["reliability_curve"][0]["fraction_positive"] == pytest.approx(0.1)


def test_rules_and_mcc_on_validation():
    y_val = [0, 0, 1, 1]
    p_val = [0.1, 0.2, 0.3, 0.4]
    rules = choose_thresholds(y_val, p_val, pi_ctx=0.25)
    assert rules == {"R0": 0.5, "R1": 0.25, "R2": 0.3}
    assert choose_thresholds(y_val, p_val, pi_ctx=0.5)["R1"] == 0.5
    assert evaluate(y_val, p_val, rules["R2"])["mcc"] == 1.0


def test_invalid_probabilities():
    with pytest.raises(ValueError, match="probabilidades"):
        evaluate([0, 1], [0.2, 1.5])


def test_runner_does_not_infer_twice(tmp_path, monkeypatch):
    import tpf.evaluation.runner as runner

    y = np.tile([0, 1], 50)
    X = pd.DataFrame({"x": np.linspace(-2, 2, 100)})
    parts = {
        "train": np.arange(60),
        "val": np.arange(60, 80),
        "test": np.arange(80, 100),
    }
    monkeypatch.setattr(runner, "load_dataset", lambda name: (X, y))
    monkeypatch.setattr(runner, "load_split", lambda name, seed: parts)
    calls = []

    def predict_fn(X_ctx, y_ctx, X_val, X_test, cat_cols):
        calls.append(1)
        assert len(X_ctx) == 20 and len(y_ctx) == 20
        return np.full(len(X_val), 0.6), np.full(len(X_test), 0.6)

    kw = dict(
        dataset="toy",
        seed=0,
        strategy="balanced",
        m=20,
        model_id="toy_model",
        predict_fn=predict_fn,
        results_dir=tmp_path,
    )
    first = runner.run_once(**kw)
    second = runner.run_once(**kw)
    assert first["status"] == "complete"
    assert second["status"] == "cached"
    assert len(calls) == 1
    out = tmp_path / "toy" / "seed0" / "balanced_m20_toy_model"
    with np.load(out / "predictions.npz") as data:
        assert len(data["p_val"]) == 20 and len(data["p_test"]) == 20
    scores = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(scores["rules"]) == {"R0", "R1", "R2"}
    runner.score_saved_run(out)
    assert len(calls) == 1
