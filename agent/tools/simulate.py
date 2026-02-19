"""
Behavioral and neural data simulation tools.

SimulateBehavior: Forward-simulate any reward model (RW, CK, RW+CK, VPP, ORL)
    to generate synthetic trial-level behavioral data.
SimulateNeural: Generate synthetic EEG epochs with parameterized REWP signals
    coupled to model-derived RPEs.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .base import BaseTool

# ═══════════════════════════════════════════════════════════════════════════
# Behavioral model simulators (pure functions)
# ═══════════════════════════════════════════════════════════════════════════


def softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """Softmax choice probabilities."""
    ev = beta * values
    ev -= ev.max()  # numerical stability
    exp_ev = np.exp(ev)
    return exp_ev / exp_ev.sum()


def simulate_random(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    bias: float = 0.5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Random Responding model (M1) — null/baseline model.
    No learning; agent chooses with fixed bias.

    Parameters
    ----------
    bias : float
        Probability of choosing option 0 (for 2-option tasks).
    """
    rng = np.random.default_rng(seed)
    if n_options == 2:
        p = np.array([bias, 1 - bias])
    else:
        p = np.ones(n_options) / n_options

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)

    for t in range(n_trials):
        c = rng.choice(n_options, p=p)
        choices[t] = c
        outcomes[t] = float(rng.random() < reward_probs[c])

    return {
        "choices": choices,
        "outcomes": outcomes,
        "rpes": np.zeros(n_trials),
    }


def simulate_wsls(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    epsilon: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Noisy Win-Stay-Lose-Shift model (M2).
    Heuristic strategy with noise parameter epsilon.
    """
    rng = np.random.default_rng(seed)
    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)

    last_choice = None
    last_reward = None

    for t in range(n_trials):
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
        c = rng.choice(n_options, p=p)
        choices[t] = c
        outcomes[t] = float(rng.random() < reward_probs[c])

        last_choice = c
        last_reward = int(outcomes[t])

    return {
        "choices": choices,
        "outcomes": outcomes,
        "rpes": np.zeros(n_trials),
    }


def simulate_q_learning(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    alpha: float,
    beta: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Q-learning agent (single learning rate).
    For bandit tasks, equivalent to RW. Included for naming clarity.
    Q(a) += alpha * (r - Q(a)); softmax action selection.
    """
    return simulate_rw(n_trials, n_options, reward_probs, alpha, beta, seed)


def simulate_q_dual(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    beta: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Q-learning with dual learning rates (Frank et al., 2007).
    Separate learning rates for positive vs negative prediction errors.
    Maps to dopaminergic asymmetry: D1 (alpha+) vs D2 (alpha-).
    """
    rng = np.random.default_rng(seed)
    Q = np.full(n_options, 1.0 / n_options)

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    q_history = np.zeros((n_trials, n_options))
    rpes = np.zeros(n_trials)

    for t in range(n_trials):
        p = softmax(Q, beta)
        q_history[t] = Q.copy()

        c = rng.choice(n_options, p=p)
        choices[t] = c
        r = float(rng.random() < reward_probs[c])
        outcomes[t] = r

        rpe = r - Q[c]
        rpes[t] = rpe

        if rpe >= 0:
            Q[c] += alpha_pos * rpe
        else:
            Q[c] += alpha_neg * rpe

    return {
        "choices": choices,
        "outcomes": outcomes,
        "q_values": q_history,
        "rpes": rpes,
    }


def simulate_actor_critic(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    alpha_critic: float,
    alpha_actor: float,
    beta: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Actor-Critic agent.

    Critic learns state value V(s), Actor learns preferences H(a).
    TD error delta = r - V drives both updates.
    Neuroscience: Critic ~ ventral striatum, Actor ~ dorsal striatum,
    delta ~ phasic dopamine.
    """
    rng = np.random.default_rng(seed)
    V = 0.5   # critic: state value (single state for bandit)
    H = np.zeros(n_options)  # actor: action preferences

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    rpes = np.zeros(n_trials)

    for t in range(n_trials):
        p = softmax(H, beta)
        c = rng.choice(n_options, p=p)
        choices[t] = c
        r = float(rng.random() < reward_probs[c])
        outcomes[t] = r

        delta = r - V
        rpes[t] = delta
        V += alpha_critic * delta
        H[c] += alpha_actor * delta

    return {
        "choices": choices,
        "outcomes": outcomes,
        "rpes": rpes,
    }


def simulate_q_decay(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    alpha: float,
    beta: float,
    decay: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Q-learning with forgetting/decay.
    Unchosen options decay toward 0.5 (initial value).
    Common in reversal learning paradigms.
    """
    rng = np.random.default_rng(seed)
    Q = np.full(n_options, 0.5)

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    q_history = np.zeros((n_trials, n_options))
    rpes = np.zeros(n_trials)

    for t in range(n_trials):
        p = softmax(Q, beta)
        q_history[t] = Q.copy()

        c = rng.choice(n_options, p=p)
        choices[t] = c
        r = float(rng.random() < reward_probs[c])
        outcomes[t] = r

        rpe = r - Q[c]
        rpes[t] = rpe
        Q[c] += alpha * rpe

        for a in range(n_options):
            if a != c:
                Q[a] += decay * (0.5 - Q[a])

    return {
        "choices": choices,
        "outcomes": outcomes,
        "q_values": q_history,
        "rpes": rpes,
    }


def simulate_rw_bias(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    alpha: float,
    beta: float,
    bias: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate RW + Side Bias model (M6 from Wilson & Collins).
    Adds a constant bias to option 0's value.
    """
    rng = np.random.default_rng(seed)
    Q = np.full(n_options, 0.5)

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    q_history = np.zeros((n_trials, n_options))
    rpes = np.zeros(n_trials)

    for t in range(n_trials):
        V = Q.copy()
        V[0] += bias
        p = softmax(V, beta)
        q_history[t] = Q.copy()

        c = rng.choice(n_options, p=p)
        choices[t] = c
        r = float(rng.random() < reward_probs[c])
        outcomes[t] = r

        rpe = r - Q[c]
        rpes[t] = rpe
        Q[c] += alpha * rpe

    return {
        "choices": choices,
        "outcomes": outcomes,
        "q_values": q_history,
        "rpes": rpes,
    }


def simulate_rw(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    alpha: float,
    beta: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Rescorla-Wagner (M3) agent.

    Parameters
    ----------
    n_trials : int
    n_options : int
        Number of options (2 for bandit, 4 for IGT)
    reward_probs : np.ndarray
        (n_options,) probability of reward per option
    alpha : float
        Learning rate
    beta : float
        Inverse temperature
    seed : int

    Returns
    -------
    dict with keys: choices, outcomes, q_values, rpes, choice_probs
    """
    rng = np.random.default_rng(seed)
    Q = np.full(n_options, 0.5)

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    q_history = np.zeros((n_trials, n_options))
    rpes = np.zeros(n_trials)
    probs = np.zeros((n_trials, n_options))

    for t in range(n_trials):
        p = softmax(Q, beta)
        probs[t] = p

        c = rng.choice(n_options, p=p)
        choices[t] = c

        r = float(rng.random() < reward_probs[c])
        outcomes[t] = r

        rpe = r - Q[c]
        rpes[t] = rpe
        q_history[t] = Q.copy()

        Q[c] += alpha * rpe

    return {
        "choices": choices,
        "outcomes": outcomes,
        "q_values": q_history,
        "rpes": rpes,
        "choice_probs": probs,
    }


def simulate_ck(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    alpha_c: float,
    beta_c: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Simulate Choice Kernel (M4) — perseveration only, no reward learning."""
    rng = np.random.default_rng(seed)
    CK = np.zeros(n_options)

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    ck_history = np.zeros((n_trials, n_options))

    for t in range(n_trials):
        p = softmax(CK, beta_c)
        c = rng.choice(n_options, p=p)
        choices[t] = c
        outcomes[t] = float(rng.random() < reward_probs[c])
        ck_history[t] = CK.copy()

        # Update: chosen action indicator
        a = np.zeros(n_options)
        a[c] = 1.0
        CK += alpha_c * (a - CK)

    return {"choices": choices, "outcomes": outcomes, "ck_values": ck_history}


def simulate_rwck(
    n_trials: int,
    n_options: int,
    reward_probs: np.ndarray,
    alpha: float,
    beta: float,
    alpha_c: float,
    beta_c: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Simulate RW + Choice Kernel (M5)."""
    rng = np.random.default_rng(seed)
    Q = np.full(n_options, 0.5)
    CK = np.zeros(n_options)

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    q_history = np.zeros((n_trials, n_options))
    ck_history = np.zeros((n_trials, n_options))
    rpes = np.zeros(n_trials)

    for t in range(n_trials):
        V = beta * Q + beta_c * CK
        ev = V - V.max()
        p = np.exp(ev) / np.exp(ev).sum()

        c = rng.choice(n_options, p=p)
        choices[t] = c

        r = float(rng.random() < reward_probs[c])
        outcomes[t] = r

        rpe = r - Q[c]
        rpes[t] = rpe
        q_history[t] = Q.copy()
        ck_history[t] = CK.copy()

        Q[c] += alpha * rpe

        a = np.zeros(n_options)
        a[c] = 1.0
        CK += alpha_c * (a - CK)

    return {
        "choices": choices,
        "outcomes": outcomes,
        "q_values": q_history,
        "ck_values": ck_history,
        "rpes": rpes,
    }


def simulate_vpp(
    n_trials: int,
    payoffs: np.ndarray,
    A: float,
    alpha: float,
    lam: float,
    c: float,
    ep_p: float,
    ep_n: float,
    K: float,
    w: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Value-Plus-Perseverance (VPP) for IGT.

    Parameters
    ----------
    payoffs : np.ndarray
        (n_trials, n_decks=4) net payoff schedule
    A, alpha, lam, c, ep_p, ep_n, K, w : float
        VPP parameters (see reward_learning_models.docx)
    """
    rng = np.random.default_rng(seed)
    n_decks = 4
    E = np.zeros(n_decks)
    P = np.zeros(n_decks)
    theta = 3.0**c - 1.0

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    e_history = np.zeros((n_trials, n_decks))
    p_history = np.zeros((n_trials, n_decks))
    rpes = np.zeros(n_trials)

    for t in range(n_trials):
        V = w * E + (1.0 - w) * P
        prob = softmax(V, theta)

        ch = rng.choice(n_decks, p=prob)
        choices[t] = ch
        x = payoffs[t, ch]
        outcomes[t] = x

        # Prospect utility
        if x >= 0:
            u = abs(x) ** alpha
        else:
            u = -lam * abs(x) ** alpha

        rpe = u - E[ch]
        rpes[t] = rpe
        e_history[t] = E.copy()
        p_history[t] = P.copy()

        # Update expectancy (chosen only)
        E[ch] += A * rpe

        # Update perseverance
        P *= K  # decay all
        if x >= 0:
            P[ch] += ep_p
        else:
            P[ch] += ep_n

    return {
        "choices": choices,
        "outcomes": outcomes,
        "expectancies": e_history,
        "perseverances": p_history,
        "rpes": rpes,
    }


def simulate_orl(
    n_trials: int,
    payoffs: np.ndarray,
    A_rew: float,
    A_pun: float,
    K: float,
    beta_f: float,
    beta_p: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Outcome-Representation Learning (ORL) for IGT.

    Parameters
    ----------
    payoffs : np.ndarray
        (n_trials, n_decks=4) net payoff schedule
    A_rew, A_pun, K, beta_f, beta_p : float
        ORL parameters (see reward_learning_models.docx)
    """
    rng = np.random.default_rng(seed)
    n_decks = 4
    ev = np.zeros(n_decks)
    ef = np.zeros(n_decks)
    pers = np.zeros(n_decks)
    K_tr = 3.0**K - 1.0

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    ev_history = np.zeros((n_trials, n_decks))
    ef_history = np.zeros((n_trials, n_decks))
    pers_history = np.zeros((n_trials, n_decks))
    rpes = np.zeros(n_trials)

    for t in range(n_trials):
        util = ev + beta_f * ef + beta_p * pers
        prob = np.exp(util - util.max())
        prob /= prob.sum()

        ch = rng.choice(n_decks, p=prob)
        choices[t] = ch
        x = payoffs[t, ch]
        outcomes[t] = x

        sign_x = np.sign(x)

        # Prediction errors
        d_val = x - ev[ch]
        d_freq = sign_x - ef[ch]
        rpes[t] = d_val

        ev_history[t] = ev.copy()
        ef_history[t] = ef.copy()
        pers_history[t] = pers.copy()

        # Value update — chosen deck
        if x >= 0:
            ev[ch] += A_rew * d_val
            ef[ch] += A_rew * d_freq
            # Fictive update — unchosen decks
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

        # Perseverance update
        pers[ch] += 1.0
        pers /= (1.0 + K_tr)

    return {
        "choices": choices,
        "outcomes": outcomes,
        "ev_values": ev_history,
        "ef_values": ef_history,
        "perseverances": pers_history,
        "rpes": rpes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tool wrappers for the Coscientist agent
# ═══════════════════════════════════════════════════════════════════════════

# Default IGT payoff schedule (simplified Bechara 1994)
def _default_igt_payoffs(n_trials: int = 100, seed: int = 42) -> np.ndarray:
    """Generate a simple IGT-like payoff matrix (n_trials, 4 decks)."""
    rng = np.random.default_rng(seed)
    payoffs = np.zeros((n_trials, 4))
    # Deck A: high gain, high frequent loss (bad)
    payoffs[:, 0] = 100 + rng.choice([-150, -200, -250, -300, -350, 0, 0, 0, 0, 0], n_trials)
    # Deck B: high gain, rare but large loss (bad)
    payoffs[:, 1] = 100
    payoffs[rng.random(n_trials) < 0.1, 1] -= 1250
    # Deck C: low gain, low frequent loss (good)
    payoffs[:, 2] = 50 + rng.choice([-25, -50, -75, 0, 0, 0, 0, 0, 0, 0], n_trials)
    # Deck D: low gain, rare but small loss (good)
    payoffs[:, 3] = 50
    payoffs[rng.random(n_trials) < 0.1, 3] -= 250
    return payoffs


class SimulateBehavior(BaseTool):
    """Agent tool: simulate behavioral data from any reward model."""

    MODEL_REGISTRY = [
        "random", "wsls", "rw", "q_learning", "q_dual",
        "ck", "rwck", "rw_bias", "q_decay",
        "actor_critic", "vpp", "orl",
    ]

    def __init__(self):
        super().__init__("SIMULATE_BEHAVIOR")

    @property
    def description(self) -> str:
        return (
            "Simulate behavioral data from a reward learning model.\n"
            "Input JSON keys:\n"
            "  model: str — one of 'rw', 'ck', 'rwck', 'vpp', 'orl'\n"
            "  params: dict — model parameters (e.g. {'alpha': 0.3, 'beta': 5.0})\n"
            "  n_trials: int — number of trials (default 200)\n"
            "  n_subjects: int — number of simulated subjects (default 1)\n"
            "  n_options: int — 2 for bandit, 4 for IGT (default 2)\n"
            "  reward_probs: list — per-option reward probability (bandit models)\n"
            "  seed: int — random seed (default 42)\n"
            "Returns: summary statistics + saves CSV to output_dir."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        model = params.get("model", "rw").lower()
        model_params = params.get("params", {})
        n_trials = params.get("n_trials", 200)
        n_subjects = params.get("n_subjects", 1)
        n_options = params.get("n_options", 2)
        reward_probs = np.array(params.get("reward_probs", [0.7, 0.3]))
        seed = params.get("seed", 42)
        output_dir = params.get("output_dir", "./results/simulated")

        import os
        os.makedirs(output_dir, exist_ok=True)

        all_results = []

        for subj in range(n_subjects):
            subj_seed = seed + subj

            if model == "random":
                res = simulate_random(
                    n_trials, n_options, reward_probs,
                    bias=model_params.get("bias", 0.5),
                    seed=subj_seed,
                )
            elif model == "wsls":
                res = simulate_wsls(
                    n_trials, n_options, reward_probs,
                    epsilon=model_params.get("epsilon", 0.1),
                    seed=subj_seed,
                )
            elif model in ("rw", "q_learning"):
                res = simulate_rw(
                    n_trials, n_options, reward_probs,
                    alpha=model_params.get("alpha", 0.3),
                    beta=model_params.get("beta", 5.0),
                    seed=subj_seed,
                )
            elif model == "q_dual":
                res = simulate_q_dual(
                    n_trials, n_options, reward_probs,
                    alpha_pos=model_params.get("alpha_pos", 0.5),
                    alpha_neg=model_params.get("alpha_neg", 0.1),
                    beta=model_params.get("beta", 5.0),
                    seed=subj_seed,
                )
            elif model == "actor_critic":
                res = simulate_actor_critic(
                    n_trials, n_options, reward_probs,
                    alpha_critic=model_params.get("alpha_critic", 0.3),
                    alpha_actor=model_params.get("alpha_actor", 0.3),
                    beta=model_params.get("beta", 5.0),
                    seed=subj_seed,
                )
            elif model == "q_decay":
                res = simulate_q_decay(
                    n_trials, n_options, reward_probs,
                    alpha=model_params.get("alpha", 0.3),
                    beta=model_params.get("beta", 5.0),
                    decay=model_params.get("decay", 0.1),
                    seed=subj_seed,
                )
            elif model == "rw_bias":
                res = simulate_rw_bias(
                    n_trials, n_options, reward_probs,
                    alpha=model_params.get("alpha", 0.3),
                    beta=model_params.get("beta", 5.0),
                    bias=model_params.get("bias", 0.0),
                    seed=subj_seed,
                )
            elif model == "ck":
                res = simulate_ck(
                    n_trials, n_options, reward_probs,
                    alpha_c=model_params.get("alpha_c", 0.3),
                    beta_c=model_params.get("beta_c", 3.0),
                    seed=subj_seed,
                )
            elif model == "rwck":
                res = simulate_rwck(
                    n_trials, n_options, reward_probs,
                    alpha=model_params.get("alpha", 0.3),
                    beta=model_params.get("beta", 5.0),
                    alpha_c=model_params.get("alpha_c", 0.2),
                    beta_c=model_params.get("beta_c", 2.0),
                    seed=subj_seed,
                )
            elif model == "vpp":
                payoffs = _default_igt_payoffs(n_trials, subj_seed)
                res = simulate_vpp(
                    n_trials, payoffs,
                    A=model_params.get("A", 0.1),
                    alpha=model_params.get("alpha", 0.8),
                    lam=model_params.get("lambda", 2.0),
                    c=model_params.get("c", 2.0),
                    ep_p=model_params.get("ep_p", 0.5),
                    ep_n=model_params.get("ep_n", -0.5),
                    K=model_params.get("K", 0.3),
                    w=model_params.get("w", 0.6),
                    seed=subj_seed,
                )
            elif model == "orl":
                payoffs = _default_igt_payoffs(n_trials, subj_seed)
                res = simulate_orl(
                    n_trials, payoffs,
                    A_rew=model_params.get("A_rew", 0.3),
                    A_pun=model_params.get("A_pun", 0.1),
                    K=model_params.get("K", 1.0),
                    beta_f=model_params.get("beta_f", 2.0),
                    beta_p=model_params.get("beta_p", 1.0),
                    seed=subj_seed,
                )
            else:
                return f"ERROR: Unknown model '{model}'. Use one of {self.MODEL_REGISTRY}"

            # Build DataFrame
            df = pd.DataFrame({
                "subject": subj,
                "trial": np.arange(n_trials),
                "choice": res["choices"],
                "outcome": res["outcomes"],
            })
            if "rpes" in res:
                df["rpe"] = res["rpes"]
            if "q_values" in res:
                for k in range(res["q_values"].shape[1]):
                    df[f"Q_{k}"] = res["q_values"][:, k]
            all_results.append(df)

        combined = pd.concat(all_results, ignore_index=True)
        out_path = os.path.join(output_dir, f"sim_{model}_{n_subjects}subj.csv")
        combined.to_csv(out_path, index=False)

        # Summary
        mean_reward = combined["outcome"].mean()
        opt_choice = (combined["choice"] == np.argmax(reward_probs)).mean() if model in ("rw", "ck", "rwck") else None

        summary = (
            f"Simulated {n_subjects} subject(s) × {n_trials} trials with model={model}.\n"
            f"Parameters: {model_params}\n"
            f"Mean reward: {mean_reward:.3f}\n"
        )
        if opt_choice is not None:
            summary += f"Optimal choice rate: {opt_choice:.3f}\n"
        summary += f"Saved to: {out_path}"

        return summary


class SimulateNeural(BaseTool):
    """Agent tool: generate synthetic EEG epochs with model-coupled REWP."""

    def __init__(self):
        super().__init__("SIMULATE_NEURAL")

    @property
    def description(self) -> str:
        return (
            "Generate synthetic EEG epochs with a parameterized REWP signal.\n"
            "The REWP amplitude can be coupled to trial-level RPEs from a behavioral model.\n"
            "Input JSON keys:\n"
            "  n_epochs: int — number of epochs (default 200)\n"
            "  n_channels: int — number of channels (default 64)\n"
            "  sfreq: float — sampling frequency in Hz (default 200)\n"
            "  tmin: float — epoch start in seconds (default -0.2)\n"
            "  tmax: float — epoch end in seconds (default 0.8)\n"
            "  rewp_amplitude: float — base REWP signal in µV (default 3.0)\n"
            "  rewp_tmin: float — REWP window start (default 0.240)\n"
            "  rewp_tmax: float — REWP window end (default 0.340)\n"
            "  rpe_coupling: float — RPE-to-REWP scaling factor (default 0.0 = no coupling)\n"
            "  rpes: list — trial-level RPEs to modulate REWP amplitude\n"
            "  noise_std: float — Gaussian noise std (default 10.0)\n"
            "  reward_fraction: float — fraction of reward trials (default 0.5)\n"
            "  seed: int — random seed (default 42)\n"
            "Returns: summary of created TMazeEEGData object."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        n_epochs = params.get("n_epochs", 200)
        n_channels = params.get("n_channels", 64)
        sfreq = params.get("sfreq", 200.0)
        tmin = params.get("tmin", -0.2)
        tmax = params.get("tmax", 0.8)
        rewp_amp = params.get("rewp_amplitude", 3.0)
        rewp_tmin = params.get("rewp_tmin", 0.240)
        rewp_tmax = params.get("rewp_tmax", 0.340)
        rpe_coupling = params.get("rpe_coupling", 0.0)
        rpes = params.get("rpes", None)
        noise_std = params.get("noise_std", 10.0)
        reward_frac = params.get("reward_fraction", 0.5)
        seed = params.get("seed", 42)

        rng = np.random.default_rng(seed)

        n_times = int((tmax - tmin) * sfreq)
        times = np.linspace(tmin, tmax, n_times)

        # Labels: 1 = reward, 0 = no-reward
        n_reward = int(n_epochs * reward_frac)
        labels = np.array([1] * n_reward + [0] * (n_epochs - n_reward))
        rng.shuffle(labels)

        # Base noise
        data = rng.normal(0, noise_std, (n_epochs, n_channels, n_times))

        # FCz-like channels (first 5)
        fcz_idx = list(range(min(5, n_channels)))
        rewp_mask = (times >= rewp_tmin) & (times <= rewp_tmax)

        for i in range(n_epochs):
            if labels[i] == 1:
                amp = rewp_amp
                if rpe_coupling > 0 and rpes is not None and i < len(rpes):
                    amp += rpe_coupling * rpes[i]
                data[i, fcz_idx[:, None] if isinstance(fcz_idx, np.ndarray) else fcz_idx, :][:, rewp_mask] += amp

        # Build channel names
        ch_names_64 = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
            'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
            'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
            'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
            'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6',
            'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
            'PO3', 'POz', 'PO4', 'FCz'
        ]
        channels = ch_names_64[:n_channels]

        # Store as module-level cache so neural tools can access it
        self._last_data = {
            "data": data,
            "times": times,
            "labels": labels,
            "channels": channels,
            "sfreq": sfreq,
            "n_epochs": n_epochs,
            "n_channels": n_channels,
        }

        summary = (
            f"Created synthetic EEG data: {n_epochs} epochs × {n_channels} channels × {n_times} times.\n"
            f"Sampling freq: {sfreq} Hz | Time window: [{tmin}, {tmax}]s\n"
            f"REWP signal: {rewp_amp} µV in [{rewp_tmin}, {rewp_tmax}]s on frontocentral channels.\n"
            f"RPE coupling: {rpe_coupling} | Noise std: {noise_std} µV\n"
            f"Reward trials: {int(labels.sum())}/{n_epochs}\n"
            f"Data shape: {data.shape}"
        )
        return summary
