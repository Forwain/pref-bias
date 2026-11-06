"""Predictive metrics and bias metrics.

All probability arrays P have shape (n, 3) with columns [P(A wins), P(B wins), P(tie)].
"""
import numpy as np
from scipy.optimize import minimize

EPS = 1e-15


def normalize(P):
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return P / P.sum(1, keepdims=True)


# ---------------------------------------------------------------- predictive

def log_loss(y, P):
    P = normalize(P)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def accuracy(y, P):
    return float(np.mean(np.asarray(P).argmax(1) == y))


def ece(y, P, n_bins=15):
    """Expected calibration error of the top-class confidence (Guo et al., 2017)."""
    P = normalize(P)
    conf, pred = P.max(1), P.argmax(1)
    correct = pred == y
    edges = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(err)


# ---------------------------------------------------------------- position bias

def unswap(P_swap):
    """Map predictions made on the swapped pair back to the original A/B order."""
    return np.asarray(P_swap)[:, [1, 0, 2]]


def position_consistency(P, P_swap):
    """Share of pairs whose predicted winner is unchanged after swapping A and B."""
    return float(np.mean(np.asarray(P).argmax(1) == unswap(P_swap).argmax(1)))


def position_bias(P, P_swap):
    """Mean preference for whichever response sits in slot A, averaged over both orders.
    0 for a position-invariant model; > 0 means slot A is favoured."""
    P, Q = np.asarray(P), np.asarray(P_swap)
    return float(np.mean(0.5 * ((P[:, 0] - P[:, 1]) + (Q[:, 0] - Q[:, 1]))))


def human_position_bias(y):
    """The same quantity for the human labels: P(A wins) - P(B wins)."""
    y = np.asarray(y)
    return float(np.mean(y == 0) - np.mean(y == 1))


# ---------------------------------------------------------------- length bias

def soft_logistic_slope(x, q):
    """Fit q ~ sigmoid(a + b x) with soft targets q in [0, 1]; return b."""
    x, q = np.asarray(x, float), np.asarray(q, float)

    def nll(w):
        z = w[0] + w[1] * x
        # log(1 + e^z) computed stably
        lse = np.logaddexp(0, z)
        loss = np.mean(lse - q * z)
        s = 1 / (1 + np.exp(-z))
        g = np.array([np.mean(s - q), np.mean((s - q) * x)])
        return loss, g

    res = minimize(nll, np.zeros(2), jac=True, method="L-BFGS-B")
    return float(res.x[1])


def length_sensitivity(log_len_ratio, y, P):
    """How strongly the win probability tracks log(len_A / len_B), on non-tie pairs.

    b_human: slope fitted on the human labels.
    b_model: slope fitted on the model's P(A) / (P(A) + P(B)).
    excess = b_model - b_human. > 0 means the model leans on length more than people do,
    which is the evidence for shortcut use (people themselves prefer longer answers).
    """
    y = np.asarray(y)
    P = normalize(P)
    m = y != 2
    x = np.asarray(log_len_ratio)[m]
    b_h = soft_logistic_slope(x, (y[m] == 0).astype(float))
    q = P[m, 0] / (P[m, 0] + P[m, 1])
    b_m = soft_logistic_slope(x, q)
    return {"len_slope_human": b_h, "len_slope_model": b_m, "len_slope_excess": b_m - b_h}


def verbosity_attack(P, P_inflated, inflated_side):
    """Effect of padding one response with content-free text.

    inflated_side: array of 0 (A was padded) or 1 (B was padded).
    gain: mean change in P(padded side wins); success: share where it went up;
    flip: share where the predicted winner switched to the padded side.
    """
    P, Q = normalize(P), normalize(P_inflated)
    idx = np.arange(len(P))
    s = np.asarray(inflated_side)
    gain = Q[idx, s] - P[idx, s]
    flip = (P.argmax(1) != s) & (Q.argmax(1) == s)
    return {"verb_gain": float(gain.mean()), "verb_success": float(np.mean(gain > 1e-6)),
            "verb_flip": float(flip.mean())}


# ---------------------------------------------------------------- self-preference

def self_preference(P, y, model_a, model_b, family):
    """For pairs where exactly one side comes from `family` (e.g. "qwen") and humans did not tie:
    model's mean P(family side wins | not tie) minus the human win rate of that side."""
    fa = np.char.find(np.char.lower(np.asarray(model_a, dtype=str)), family) >= 0
    fb = np.char.find(np.char.lower(np.asarray(model_b, dtype=str)), family) >= 0
    y = np.asarray(y)
    m = (fa ^ fb) & (y != 2)
    if m.sum() == 0:
        return {"self_pref_n": 0, "self_pref_model": np.nan, "self_pref_human": np.nan, "self_pref_excess": np.nan}
    P = normalize(P)[m]
    side = np.where(fa[m], 0, 1)
    q = P[np.arange(len(P)), side] / (P[:, 0] + P[:, 1])
    human = float(np.mean(y[m] == side))
    return {"self_pref_n": int(m.sum()), "self_pref_model": float(q.mean()),
            "self_pref_human": human, "self_pref_excess": float(q.mean() - human)}


# ---------------------------------------------------------------- uncertainty

def bootstrap_ci(stat_fn, n, B=200, seed=0, alpha=0.05):
    """Percentile bootstrap CI. stat_fn takes an index array and returns a float."""
    rng = np.random.default_rng(seed)
    vals = [stat_fn(rng.integers(0, n, n)) for _ in range(B)]
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
