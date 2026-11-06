"""Sanity checks on metrics, parsing and probes with synthetic data (no dataset needed).

  PYTHONPATH=src pytest -q tests
"""
import numpy as np
import pandas as pd

from prefbias import metrics as M
from prefbias.data import parse_turns, swap_ab, add_lengths
from prefbias.probes import make_verbose_probe


def test_uniform_log_loss_is_ln3():
    y = np.array([0, 1, 2, 0])
    assert abs(M.log_loss(y, np.full((4, 3), 1 / 3)) - np.log(3)) < 1e-9


def test_position_invariant_model():
    rng = np.random.default_rng(0)
    P = rng.dirichlet([1, 1, 1], 100)
    P_swap = M.unswap(P)  # a perfectly invariant model
    assert M.position_consistency(P, P_swap) == 1.0
    assert abs(M.position_bias(P, P_swap)) < 1e-12


def test_always_a_model_is_fully_position_biased():
    P = np.tile([0.8, 0.1, 0.1], (50, 1))
    assert M.position_consistency(P, P) == 0.0
    assert abs(M.position_bias(P, P) - 0.7) < 1e-9


def test_logistic_slope_recovers_truth():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000)
    q = 1 / (1 + np.exp(-(0.2 + 1.5 * x)))
    assert abs(M.soft_logistic_slope(x, q) - 1.5) < 0.01
    y = (rng.random(20000) < q).astype(float)
    assert abs(M.soft_logistic_slope(x, y) - 1.5) < 0.1


def test_verbosity_attack():
    P = np.tile([0.4, 0.4, 0.2], (4, 1))
    Q = np.array([[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2], [0.5, 0.3, 0.2]])
    r = M.verbosity_attack(P, Q, np.array([0, 1, 0, 1]))
    # rows 0 and 1 gain; only row 1 flips (row 0's padded side A was already the argmax of the A/B tie)
    assert r["verb_success"] == 0.5 and r["verb_flip"] == 0.25


def test_parse_turns_handles_null_and_surrogates():
    assert parse_turns('["hi", null]') == ["hi", ""]
    assert parse_turns('["\\ud83d bad"]')[0].endswith("bad")
    assert parse_turns(float("nan")) == []


def _toy():
    df = pd.DataFrame({"id": [1, 2], "prompt_text": ["q1", "q2"], "turns_a": [["Short. Answer."], ["x"]],
                       "turns_b": [["Longer answer here. Two sentences."], ["y"]], "label": [0, 2],
                       "model_a": ["m1", "m2"], "model_b": ["m3", "m4"]})
    df["resp_a"] = df.turns_a.map(lambda t: t[0]); df["resp_b"] = df.turns_b.map(lambda t: t[0])
    return add_lengths(df)


def test_swap_ab():
    s = swap_ab(_toy())
    assert s.label.tolist() == [1, 2] and s.resp_a[0].startswith("Longer") and s.model_a[0] == "m3"


def test_verbose_probe_only_pads_one_side():
    df = _toy()
    v = make_verbose_probe(df, "restate", seed=0)
    for i, s in enumerate(v.inflated_side):
        grown = v.loc[i, ["len_a", "len_b"]].to_numpy() - df.loc[i, ["len_a", "len_b"]].to_numpy()
        assert grown[s] > 0 and grown[1 - s] == 0
