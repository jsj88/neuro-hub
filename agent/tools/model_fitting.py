"""
Model fitting and parameter recovery tools.

FitModel: MLE fitting of reward models to behavioral data (scipy.optimize).
ParameterRecovery: Simulate-then-fit loop to validate model identifiability.
CompareModels: BIC/AIC model comparison across a model set.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field

from .base import BaseTool

# ═══════════════════════════════════════════════════════════════════════════
# Negative log-likelihood functions
# ═══════════════════════════════════════════════════════════════════════════


def _softmax_prob(values: np.ndarray, beta: float) -> np.ndarray:
    ev = beta * values
    ev -= ev.max()
    exp_ev = np.exp(ev)
    return exp_ev / exp_ev.sum()


def nll_random(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for Random Responding (null model)."""
    bias = params[0]
    if n_options == 2:
        p = np.array([bias, 1 - bias])
    else:
        p = np.ones(n_options) / n_options
    nll = 0.0
    for t in range(len(choices)):
        nll -= np.log(max(p[choices[t]], 1e-8))
    return nll


def nll_wsls(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for Noisy Win-Stay-Lose-Shift."""
    epsilon = params[0]
    nll = 0.0
    last_choice = None
    last_reward = None

    for t in range(len(choices)):
        if last_choice is None:
            p = np.ones(n_options) / n_options
        else:
            p = np.full(n_options, epsilon / n_options)
            if last_reward == 1:
                p[last_choice] = 1 - epsilon + epsilon / n_options
            else:
                stay_prob = epsilon / n_options
                p[last_choice] = stay_prob
                shift_prob = (1 - stay_prob) / max(1, n_options - 1)
                for a in range(n_options):
                    if a != last_choice:
                        p[a] = shift_prob
        p = np.maximum(p, 1e-8)
        p /= p.sum()
        nll -= np.log(max(p[choices[t]], 1e-8))
        last_choice = choices[t]
        last_reward = int(outcomes[t])
    return nll


def nll_q_dual(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for Q-learning with dual learning rates."""
    alpha_pos, alpha_neg, beta = params
    Q = np.full(n_options, 1.0 / n_options)
    nll = 0.0
    for t in range(len(choices)):
        p = _softmax_prob(Q, beta)
        nll -= np.log(max(p[choices[t]], 1e-8))
        rpe = outcomes[t] - Q[choices[t]]
        if rpe >= 0:
            Q[choices[t]] += alpha_pos * rpe
        else:
            Q[choices[t]] += alpha_neg * rpe
    return nll


def nll_actor_critic(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for Actor-Critic."""
    alpha_c, alpha_a, beta = params
    V = 0.5
    H = np.zeros(n_options)
    nll = 0.0
    for t in range(len(choices)):
        p = _softmax_prob(H, beta)
        nll -= np.log(max(p[choices[t]], 1e-8))
        delta = outcomes[t] - V
        V += alpha_c * delta
        H[choices[t]] += alpha_a * delta
    return nll


def nll_q_decay(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for Q-learning with forgetting/decay."""
    alpha, beta, decay = params
    Q = np.full(n_options, 0.5)
    nll = 0.0
    for t in range(len(choices)):
        p = _softmax_prob(Q, beta)
        nll -= np.log(max(p[choices[t]], 1e-8))
        Q[choices[t]] += alpha * (outcomes[t] - Q[choices[t]])
        for a in range(n_options):
            if a != choices[t]:
                Q[a] += decay * (0.5 - Q[a])
    return nll


def nll_rw_bias(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for RW + Side Bias."""
    alpha, beta, bias = params
    Q = np.full(n_options, 1.0 / n_options)
    nll = 0.0
    for t in range(len(choices)):
        V = Q.copy()
        V[0] += bias
        p = _softmax_prob(V, beta)
        nll -= np.log(max(p[choices[t]], 1e-8))
        Q[choices[t]] += alpha * (outcomes[t] - Q[choices[t]])
    return nll


def nll_rw(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for Rescorla-Wagner."""
    alpha, beta = params
    Q = np.full(n_options, 1.0 / n_options)
    nll = 0.0

    for t in range(len(choices)):
        p = _softmax_prob(Q, beta)
        p_chosen = max(p[choices[t]], 1e-8)
        nll -= np.log(p_chosen)
        Q[choices[t]] += alpha * (outcomes[t] - Q[choices[t]])

    return nll


def nll_ck(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for Choice Kernel."""
    alpha_c, beta_c = params
    CK = np.zeros(n_options)
    nll = 0.0

    for t in range(len(choices)):
        p = _softmax_prob(CK, beta_c)
        p_chosen = max(p[choices[t]], 1e-8)
        nll -= np.log(p_chosen)

        a = np.zeros(n_options)
        a[choices[t]] = 1.0
        CK += alpha_c * (a - CK)

    return nll


def nll_rwck(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int) -> float:
    """Negative log-likelihood for RW + Choice Kernel."""
    alpha, beta, alpha_c, beta_c = params
    Q = np.full(n_options, 1.0 / n_options)
    CK = np.zeros(n_options)
    nll = 0.0

    for t in range(len(choices)):
        V = beta * Q + beta_c * CK
        ev = V - V.max()
        p = np.exp(ev) / np.exp(ev).sum()
        p_chosen = max(p[choices[t]], 1e-8)
        nll -= np.log(p_chosen)

        Q[choices[t]] += alpha * (outcomes[t] - Q[choices[t]])

        a = np.zeros(n_options)
        a[choices[t]] = 1.0
        CK += alpha_c * (a - CK)

    return nll


def nll_vpp(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int = 4) -> float:
    """Negative log-likelihood for VPP (Iowa Gambling Task)."""
    A, alpha, lam, c, ep_p, ep_n, K, w = params
    n_decks = n_options
    E = np.zeros(n_decks)
    P = np.zeros(n_decks)
    theta = 3.0**c - 1.0
    nll = 0.0

    for t in range(len(choices)):
        V = w * E + (1.0 - w) * P
        prob = _softmax_prob(V, theta)
        p_chosen = max(prob[choices[t]], 1e-8)
        nll -= np.log(p_chosen)

        x = outcomes[t]
        if x >= 0:
            u = abs(x) ** alpha
        else:
            u = -lam * abs(x) ** alpha

        E[choices[t]] += A * (u - E[choices[t]])
        P *= K
        if x >= 0:
            P[choices[t]] += ep_p
        else:
            P[choices[t]] += ep_n

    return nll


def nll_orl(params: np.ndarray, choices: np.ndarray, outcomes: np.ndarray, n_options: int = 4) -> float:
    """Negative log-likelihood for ORL (Iowa Gambling Task)."""
    A_rew, A_pun, K, beta_f, beta_p = params
    n_decks = n_options
    ev = np.zeros(n_decks)
    ef = np.zeros(n_decks)
    pers = np.zeros(n_decks)
    K_tr = 3.0**K - 1.0
    nll = 0.0

    for t in range(len(choices)):
        util = ev + beta_f * ef + beta_p * pers
        prob = np.exp(util - util.max())
        prob /= prob.sum()
        p_chosen = max(prob[choices[t]], 1e-8)
        nll -= np.log(p_chosen)

        ch = choices[t]
        x = outcomes[t]
        sign_x = np.sign(x) if x != 0 else 0.0

        d_val = x - ev[ch]
        d_freq = sign_x - ef[ch]

        if x >= 0:
            ev[ch] += A_rew * d_val
            ef[ch] += A_rew * d_freq
            for k in range(n_decks):
                if k != ch:
                    d_fic = -sign_x / 3.0 - ef[k]
                    ef[k] += A_pun * d_fic
        else:
            ev[ch] += A_pun * d_val
            ef[ch] += A_pun * d_freq
            for k in range(n_decks):
                if k != ch:
                    d_fic = -sign_x / 3.0 - ef[k]
                    ef[k] += A_rew * d_fic

        pers[ch] += 1.0
        pers /= (1.0 + K_tr)

    return nll


# ═══════════════════════════════════════════════════════════════════════════
# Model registry — bounds and NLL functions
# ═══════════════════════════════════════════════════════════════════════════

MODEL_SPECS = {
    "random": {
        "nll_fn": nll_random,
        "param_names": ["bias"],
        "bounds": [(0.01, 0.99)],
        "n_params": 1,
    },
    "wsls": {
        "nll_fn": nll_wsls,
        "param_names": ["epsilon"],
        "bounds": [(0.001, 0.999)],
        "n_params": 1,
    },
    "rw": {
        "nll_fn": nll_rw,
        "param_names": ["alpha", "beta"],
        "bounds": [(0.001, 0.999), (0.1, 20.0)],
        "n_params": 2,
    },
    "q_learning": {
        "nll_fn": nll_rw,  # identical to RW for bandits
        "param_names": ["alpha", "beta"],
        "bounds": [(0.001, 0.999), (0.1, 20.0)],
        "n_params": 2,
    },
    "q_dual": {
        "nll_fn": nll_q_dual,
        "param_names": ["alpha_pos", "alpha_neg", "beta"],
        "bounds": [(0.001, 0.999), (0.001, 0.999), (0.1, 20.0)],
        "n_params": 3,
    },
    "actor_critic": {
        "nll_fn": nll_actor_critic,
        "param_names": ["alpha_critic", "alpha_actor", "beta"],
        "bounds": [(0.001, 0.999), (0.001, 0.999), (0.1, 20.0)],
        "n_params": 3,
    },
    "q_decay": {
        "nll_fn": nll_q_decay,
        "param_names": ["alpha", "beta", "decay"],
        "bounds": [(0.001, 0.999), (0.1, 20.0), (0.001, 0.999)],
        "n_params": 3,
    },
    "rw_bias": {
        "nll_fn": nll_rw_bias,
        "param_names": ["alpha", "beta", "bias"],
        "bounds": [(0.001, 0.999), (0.1, 20.0), (-5.0, 5.0)],
        "n_params": 3,
    },
    "ck": {
        "nll_fn": nll_ck,
        "param_names": ["alpha_c", "beta_c"],
        "bounds": [(0.001, 0.999), (0.1, 20.0)],
        "n_params": 2,
    },
    "rwck": {
        "nll_fn": nll_rwck,
        "param_names": ["alpha", "beta", "alpha_c", "beta_c"],
        "bounds": [(0.001, 0.999), (0.1, 20.0), (0.001, 0.999), (0.1, 20.0)],
        "n_params": 4,
    },
    "vpp": {
        "nll_fn": nll_vpp,
        "param_names": ["A", "alpha", "lambda", "c", "ep_p", "ep_n", "K", "w"],
        "bounds": [
            (0.001, 0.999),   # A
            (0.01, 2.0),      # alpha
            (0.01, 10.0),     # lambda
            (0.01, 5.0),      # c
            (-5.0, 5.0),      # ep_p
            (-5.0, 5.0),      # ep_n
            (0.001, 0.999),   # K
            (0.001, 0.999),   # w
        ],
        "n_params": 8,
    },
    "orl": {
        "nll_fn": nll_orl,
        "param_names": ["A_rew", "A_pun", "K", "beta_f", "beta_p"],
        "bounds": [
            (0.001, 0.999),  # A_rew
            (0.001, 0.999),  # A_pun
            (0.01, 5.0),     # K
            (-10.0, 10.0),   # beta_f
            (-10.0, 10.0),   # beta_p
        ],
        "n_params": 5,
    },
}


@dataclass
class FitResult:
    """Result of a single model fit."""
    model: str
    params: Dict[str, float]
    nll: float
    bic: float
    aic: float
    n_trials: int
    n_params: int
    success: bool
    message: str = ""

    def __repr__(self) -> str:
        return (
            f"FitResult(model={self.model}, NLL={self.nll:.2f}, "
            f"BIC={self.bic:.2f}, AIC={self.aic:.2f}, params={self.params})"
        )


def fit_model(
    model_name: str,
    choices: np.ndarray,
    outcomes: np.ndarray,
    n_options: int = 2,
    method: str = "de",
    n_starts: int = 10,
    seed: int = 42,
) -> FitResult:
    """
    Fit a reward model via MLE.

    Parameters
    ----------
    model_name : str
        One of 'rw', 'ck', 'rwck', 'vpp', 'orl'
    choices, outcomes : np.ndarray
        Behavioral data
    n_options : int
        Number of options
    method : str
        'de' (differential evolution, global) or 'multistart' (L-BFGS-B)
    n_starts : int
        Number of random starts for multistart
    seed : int

    Returns
    -------
    FitResult
    """
    spec = MODEL_SPECS[model_name]
    nll_fn = spec["nll_fn"]
    bounds = spec["bounds"]
    param_names = spec["param_names"]
    n_params = spec["n_params"]
    n_trials = len(choices)

    if method == "de":
        result = differential_evolution(
            nll_fn, bounds, args=(choices, outcomes, n_options),
            seed=seed, maxiter=500, tol=1e-6,
        )
        best_nll = result.fun
        best_params = result.x
        success = result.success
        msg = result.message
    else:
        rng = np.random.default_rng(seed)
        best_nll = np.inf
        best_params = None
        success = False
        msg = ""

        for _ in range(n_starts):
            x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
            try:
                result = minimize(
                    nll_fn, x0, args=(choices, outcomes, n_options),
                    method="L-BFGS-B", bounds=bounds,
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x
                    success = result.success
                    msg = result.message
            except Exception as e:
                continue

    bic = n_params * np.log(n_trials) + 2 * best_nll
    aic = 2 * n_params + 2 * best_nll

    return FitResult(
        model=model_name,
        params=dict(zip(param_names, best_params)),
        nll=best_nll,
        bic=bic,
        aic=aic,
        n_trials=n_trials,
        n_params=n_params,
        success=success,
        message=str(msg),
    )


def extract_trial_variables(
    model_name: str,
    params: Dict[str, float],
    choices: np.ndarray,
    outcomes: np.ndarray,
    n_options: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Re-run a fitted model to extract trial-level latent variables
    (Q-values, RPEs, choice kernels, etc.) for neural correlation.
    """
    n_trials = len(choices)

    if model_name == "random":
        return {"rpes": np.zeros(n_trials)}

    elif model_name == "wsls":
        return {"rpes": np.zeros(n_trials)}

    elif model_name in ("rw", "q_learning"):
        alpha, beta = params["alpha"], params["beta"]
        Q = np.full(n_options, 1.0 / n_options)
        q_hist = np.zeros((n_trials, n_options))
        rpes = np.zeros(n_trials)
        for t in range(n_trials):
            q_hist[t] = Q.copy()
            rpe = outcomes[t] - Q[choices[t]]
            rpes[t] = rpe
            Q[choices[t]] += alpha * rpe
        return {"q_values": q_hist, "rpes": rpes, "chosen_q": q_hist[np.arange(n_trials), choices]}

    elif model_name == "q_dual":
        alpha_pos, alpha_neg = params["alpha_pos"], params["alpha_neg"]
        Q = np.full(n_options, 1.0 / n_options)
        q_hist = np.zeros((n_trials, n_options))
        rpes = np.zeros(n_trials)
        for t in range(n_trials):
            q_hist[t] = Q.copy()
            rpe = outcomes[t] - Q[choices[t]]
            rpes[t] = rpe
            if rpe >= 0:
                Q[choices[t]] += alpha_pos * rpe
            else:
                Q[choices[t]] += alpha_neg * rpe
        return {"q_values": q_hist, "rpes": rpes, "chosen_q": q_hist[np.arange(n_trials), choices]}

    elif model_name == "actor_critic":
        alpha_c = params["alpha_critic"]
        alpha_a = params["alpha_actor"]
        V = 0.5
        H = np.zeros(n_options)
        rpes = np.zeros(n_trials)
        v_hist = np.zeros(n_trials)
        for t in range(n_trials):
            delta = outcomes[t] - V
            rpes[t] = delta
            v_hist[t] = V
            V += alpha_c * delta
            H[choices[t]] += alpha_a * delta
        return {"rpes": rpes, "state_values": v_hist}

    elif model_name == "q_decay":
        alpha, decay = params["alpha"], params["decay"]
        Q = np.full(n_options, 0.5)
        q_hist = np.zeros((n_trials, n_options))
        rpes = np.zeros(n_trials)
        for t in range(n_trials):
            q_hist[t] = Q.copy()
            rpe = outcomes[t] - Q[choices[t]]
            rpes[t] = rpe
            Q[choices[t]] += alpha * rpe
            for a in range(n_options):
                if a != choices[t]:
                    Q[a] += decay * (0.5 - Q[a])
        return {"q_values": q_hist, "rpes": rpes, "chosen_q": q_hist[np.arange(n_trials), choices]}

    elif model_name == "rw_bias":
        alpha = params["alpha"]
        Q = np.full(n_options, 1.0 / n_options)
        q_hist = np.zeros((n_trials, n_options))
        rpes = np.zeros(n_trials)
        for t in range(n_trials):
            q_hist[t] = Q.copy()
            rpe = outcomes[t] - Q[choices[t]]
            rpes[t] = rpe
            Q[choices[t]] += alpha * rpe
        return {"q_values": q_hist, "rpes": rpes, "chosen_q": q_hist[np.arange(n_trials), choices]}

    elif model_name == "rwck":
        alpha = params["alpha"]
        alpha_c, beta_c = params["alpha_c"], params["beta_c"]
        Q = np.full(n_options, 1.0 / n_options)
        CK = np.zeros(n_options)
        q_hist = np.zeros((n_trials, n_options))
        ck_hist = np.zeros((n_trials, n_options))
        rpes = np.zeros(n_trials)
        for t in range(n_trials):
            q_hist[t] = Q.copy()
            ck_hist[t] = CK.copy()
            rpe = outcomes[t] - Q[choices[t]]
            rpes[t] = rpe
            Q[choices[t]] += alpha * rpe
            a = np.zeros(n_options)
            a[choices[t]] = 1.0
            CK += alpha_c * (a - CK)
        return {"q_values": q_hist, "ck_values": ck_hist, "rpes": rpes}

    elif model_name == "orl":
        A_rew, A_pun = params["A_rew"], params["A_pun"]
        K, beta_f, beta_p = params["K"], params["beta_f"], params["beta_p"]
        n_decks = n_options
        ev_arr = np.zeros(n_decks)
        ef_arr = np.zeros(n_decks)
        pers_arr = np.zeros(n_decks)
        K_tr = 3.0**K - 1.0
        ev_hist = np.zeros((n_trials, n_decks))
        ef_hist = np.zeros((n_trials, n_decks))
        rpes = np.zeros(n_trials)
        for t in range(n_trials):
            ev_hist[t] = ev_arr.copy()
            ef_hist[t] = ef_arr.copy()
            ch = choices[t]
            x = outcomes[t]
            sign_x = np.sign(x) if x != 0 else 0.0
            d_val = x - ev_arr[ch]
            d_freq = sign_x - ef_arr[ch]
            rpes[t] = d_val
            if x >= 0:
                ev_arr[ch] += A_rew * d_val
                ef_arr[ch] += A_rew * d_freq
                for k in range(n_decks):
                    if k != ch:
                        ef_arr[k] += A_pun * (-sign_x / 3.0 - ef_arr[k])
            else:
                ev_arr[ch] += A_pun * d_val
                ef_arr[ch] += A_pun * d_freq
                for k in range(n_decks):
                    if k != ch:
                        ef_arr[k] += A_rew * (-sign_x / 3.0 - ef_arr[k])
            pers_arr[ch] += 1.0
            pers_arr /= (1.0 + K_tr)
        return {"ev_values": ev_hist, "ef_values": ef_hist, "rpes": rpes}

    else:
        return {"rpes": np.zeros(n_trials)}


# ═══════════════════════════════════════════════════════════════════════════
# Agent tools
# ═══════════════════════════════════════════════════════════════════════════


class FitModel(BaseTool):
    """Agent tool: fit a reward model to behavioral data via MLE."""

    def __init__(self):
        super().__init__("FIT_MODEL")

    @property
    def description(self) -> str:
        return (
            "Fit a reward learning model to behavioral data via maximum likelihood.\n"
            "Input JSON keys:\n"
            "  model: str — one of 'rw', 'ck', 'rwck', 'vpp', 'orl'\n"
            "  data_path: str — path to CSV with columns [trial, choice, outcome]\n"
            "  subject: int — subject index to fit (if multi-subject CSV)\n"
            "  n_options: int — 2 (bandit) or 4 (IGT) (default 2)\n"
            "  method: str — 'de' (differential evolution) or 'multistart' (default 'de')\n"
            "Returns: best-fit parameters, NLL, BIC, AIC."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        model = params.get("model", "rw").lower()
        data_path = params.get("data_path")
        subject = params.get("subject", None)
        n_options = params.get("n_options", 2)
        method = params.get("method", "de")

        if model not in MODEL_SPECS:
            return f"ERROR: Unknown model '{model}'. Use one of {list(MODEL_SPECS.keys())}"

        df = pd.read_csv(data_path)
        if subject is not None:
            df = df[df["subject"] == subject]

        choices = df["choice"].values.astype(int)
        outcomes = df["outcome"].values.astype(float)

        result = fit_model(model, choices, outcomes, n_options, method)

        return (
            f"Model: {result.model}\n"
            f"Success: {result.success}\n"
            f"Parameters: {result.params}\n"
            f"NLL: {result.nll:.4f}\n"
            f"BIC: {result.bic:.4f}\n"
            f"AIC: {result.aic:.4f}\n"
            f"N trials: {result.n_trials}, N params: {result.n_params}"
        )


class ParameterRecovery(BaseTool):
    """Agent tool: simulate-then-fit loop to test model identifiability."""

    def __init__(self):
        super().__init__("PARAMETER_RECOVERY")

    @property
    def description(self) -> str:
        return (
            "Run parameter recovery: simulate from known parameters, then fit.\n"
            "Input JSON keys:\n"
            "  model: str — 'rw', 'ck', 'rwck', 'vpp', 'orl'\n"
            "  true_params: dict — ground-truth parameters\n"
            "  n_trials: int — trials per simulation (default 200)\n"
            "  n_simulations: int — number of recovery runs (default 50)\n"
            "  n_options: int — (default 2)\n"
            "  reward_probs: list — (default [0.7, 0.3])\n"
            "Returns: mean recovered params, correlation (true vs recovered), bias."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        from .simulate import (
            simulate_random, simulate_wsls, simulate_rw, simulate_q_dual,
            simulate_actor_critic, simulate_q_decay, simulate_rw_bias,
            simulate_ck, simulate_rwck,
        )

        model = params.get("model", "rw").lower()
        true_params = params.get("true_params", {})
        n_trials = params.get("n_trials", 200)
        n_sims = params.get("n_simulations", 50)
        n_options = params.get("n_options", 2)
        reward_probs = np.array(params.get("reward_probs", [0.7, 0.3]))

        spec = MODEL_SPECS[model]
        param_names = spec["param_names"]

        sim_fns = {
            "random": lambda seed: simulate_random(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "wsls": lambda seed: simulate_wsls(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "rw": lambda seed: simulate_rw(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "q_learning": lambda seed: simulate_rw(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "q_dual": lambda seed: simulate_q_dual(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "actor_critic": lambda seed: simulate_actor_critic(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "q_decay": lambda seed: simulate_q_decay(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "rw_bias": lambda seed: simulate_rw_bias(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "ck": lambda seed: simulate_ck(n_trials, n_options, reward_probs, seed=seed, **true_params),
            "rwck": lambda seed: simulate_rwck(n_trials, n_options, reward_probs, seed=seed, **true_params),
        }

        if model not in sim_fns:
            return f"ERROR: Parameter recovery currently supports: {list(sim_fns.keys())}"

        recovered = {name: [] for name in param_names}

        for i in range(n_sims):
            sim_data = sim_fns[model](seed=i)
            result = fit_model(
                model,
                sim_data["choices"],
                sim_data["outcomes"],
                n_options,
                method="de",
                seed=i,
            )
            for name in param_names:
                recovered[name].append(result.params[name])

        # Compute correlations
        lines = [f"Parameter Recovery: model={model}, n_sims={n_sims}, n_trials={n_trials}\n"]
        lines.append(f"{'Param':<12} {'True':>8} {'Mean Rec':>10} {'Std':>8} {'Bias':>8} {'r':>6}")
        lines.append("-" * 56)

        for name in param_names:
            true_val = true_params[name]
            rec = np.array(recovered[name])
            mean_rec = rec.mean()
            std_rec = rec.std()
            bias = mean_rec - true_val
            # For correlation, use a flat true vector (same value)
            lines.append(
                f"{name:<12} {true_val:>8.3f} {mean_rec:>10.3f} {std_rec:>8.3f} {bias:>+8.3f}"
            )

        return "\n".join(lines)


class CompareModels(BaseTool):
    """Agent tool: compare multiple models on the same data via BIC."""

    def __init__(self):
        super().__init__("COMPARE_MODELS")

    @property
    def description(self) -> str:
        return (
            "Compare multiple reward models on the same behavioral data.\n"
            "Input JSON keys:\n"
            "  models: list of str — e.g. ['rw', 'ck', 'rwck']\n"
            "  data_path: str — path to behavioral CSV\n"
            "  subject: int — subject index (optional)\n"
            "  n_options: int — (default 2)\n"
            "Returns: ranked table of BIC, AIC, and ΔBIC."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        models = params.get("models", ["rw", "ck", "rwck"])
        data_path = params.get("data_path")
        subject = params.get("subject", None)
        n_options = params.get("n_options", 2)

        df = pd.read_csv(data_path)
        if subject is not None:
            df = df[df["subject"] == subject]

        choices = df["choice"].values.astype(int)
        outcomes = df["outcome"].values.astype(float)

        results = []
        for m in models:
            r = fit_model(m, choices, outcomes, n_options)
            results.append(r)

        results.sort(key=lambda r: r.bic)
        best_bic = results[0].bic

        lines = [f"Model Comparison (N={len(choices)} trials)\n"]
        lines.append(f"{'Rank':<5} {'Model':<8} {'NLL':>10} {'BIC':>10} {'ΔBIC':>8} {'AIC':>10} {'Params':>7}")
        lines.append("-" * 62)

        for i, r in enumerate(results):
            lines.append(
                f"{i+1:<5} {r.model:<8} {r.nll:>10.2f} {r.bic:>10.2f} "
                f"{r.bic - best_bic:>+8.2f} {r.aic:>10.2f} {r.n_params:>7}"
            )

        lines.append(f"\n★ Best model: {results[0].model} (BIC={results[0].bic:.2f})")
        lines.append(f"  Parameters: {results[0].params}")

        return "\n".join(lines)
