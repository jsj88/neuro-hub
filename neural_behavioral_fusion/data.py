"""
Data preparation utilities: simulation and real data loading.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def simulate_fusion_dataset(
    n_subjects=20,
    n_trials=100,
    alpha=0.3,
    beta=5.0,
    rpe_coupling=2.0,
    noise_std=5.0,
    baseline_mean=0.0,
    baseline_sd=2.0,
    seed=42,
):
    """Simulate a multi-subject RPE-REWP fusion dataset.

    For each subject:
    1. Run Rescorla-Wagner to generate choices, outcomes, RPEs
    2. Generate REWP = subject_baseline + rpe_coupling * RPE + noise

    Parameters
    ----------
    n_subjects : int
        Number of simulated subjects.
    n_trials : int
        Trials per subject.
    alpha : float
        Learning rate (or mean for between-subject variability).
    beta : float
        Inverse temperature.
    rpe_coupling : float
        True RPE-to-REWP coupling coefficient.
    noise_std : float
        Gaussian noise on REWP.
    baseline_mean : float
        Mean of per-subject REWP baseline.
    baseline_sd : float
        SD of per-subject REWP baseline.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: subject, trial, choice, outcome, reward, rpe, rewp, condition.
    """
    # Import simulate_rw from the agent tools
    try:
        neuro_hub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if neuro_hub not in sys.path:
            sys.path.insert(0, neuro_hub)
        from agent.tools.simulate import simulate_rw
    except ImportError:
        simulate_rw = _fallback_simulate_rw

    rng = np.random.default_rng(seed)
    reward_probs = np.array([0.7, 0.3])

    all_rows = []
    for subj_idx in range(n_subjects):
        subj_seed = seed + subj_idx
        # Per-subject variability in learning rate
        subj_alpha = np.clip(rng.normal(alpha, 0.05), 0.01, 0.99)

        sim = simulate_rw(
            n_trials=n_trials,
            n_options=2,
            reward_probs=reward_probs,
            alpha=subj_alpha,
            beta=beta,
            seed=subj_seed,
        )

        rpes = sim["rpes"]
        outcomes = sim["outcomes"]

        # Generate REWP with subject-specific baseline
        subj_baseline = rng.normal(baseline_mean, baseline_sd)
        rewp = subj_baseline + rpe_coupling * rpes + rng.normal(0, noise_std, n_trials)

        subj_df = pd.DataFrame({
            "subject": f"S{subj_idx:03d}",
            "trial": np.arange(n_trials),
            "choice": sim["choices"],
            "outcome": outcomes,
            "reward": (outcomes > 0).astype(int),
            "rpe": rpes,
            "rewp": rewp,
            "condition": np.where(outcomes > 0, "reward", "noreward"),
        })
        # Add z-scored trial for rpe_trial model
        subj_df["trial_z"] = (subj_df["trial"] - subj_df["trial"].mean()) / subj_df["trial"].std()

        all_rows.append(subj_df)

    return pd.concat(all_rows, ignore_index=True)


def _fallback_simulate_rw(n_trials, n_options, reward_probs, alpha, beta, seed=42):
    """Standalone RW simulation if agent tools not importable."""
    rng = np.random.default_rng(seed)
    Q = np.full(n_options, 0.5)
    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials)
    rpes = np.zeros(n_trials)

    for t in range(n_trials):
        ev = beta * Q
        ev -= ev.max()
        p = np.exp(ev) / np.exp(ev).sum()
        c = rng.choice(n_options, p=p)
        choices[t] = c
        r = float(rng.random() < reward_probs[c])
        outcomes[t] = r
        rpe = r - Q[c]
        rpes[t] = rpe
        Q[c] += alpha * rpe

    return {"choices": choices, "outcomes": outcomes, "rpes": rpes}


def load_real_fusion_data(
    behavior_csv=None,
    eeg_dir=None,
    channels=None,
    time_window=(0.240, 0.340),
    sfreq=200.0,
):
    """Load real T-maze behavioral data and optionally merge EEG.

    Parameters
    ----------
    behavior_csv : str or None
        Path to behavioral CSV. Defaults to sample_data/merged_clean_V3_scaled.csv.
    eeg_dir : str or None
        Path to directory with per-subject EEG epoch files (.npy or .fif).
        If None, uses ERP columns already present in the CSV.
    channels : list of str or None
        EEG channels to extract. Defaults to ["FCz", "Cz", "Pz"].
    time_window : tuple
        (tmin, tmax) in seconds for REWP amplitude extraction.
    sfreq : float
        Sampling frequency of EEG data.

    Returns
    -------
    pd.DataFrame
        DataFrame ready for LME with 'subject', 'trial', 'reward', 'rewp', etc.
    """
    if behavior_csv is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        behavior_csv = os.path.join(base, "sample_data", "merged_clean_V3_scaled.csv")

    df = pd.read_csv(behavior_csv)

    # Rename SubjectCode -> subject for statsmodels grouping
    if "SubjectCode" in df.columns:
        df = df.rename(columns={"SubjectCode": "subject"})

    # Recode Reward: 1 -> 1 (reward), 2 -> 0 (no reward)
    if "Reward" in df.columns:
        df["reward"] = df["Reward"].map({1: 1, 2: 0})

    # Z-score trial within subject for rpe_trial model
    if "Trial" in df.columns:
        df["trial_z"] = df.groupby("subject")["Trial"].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    # Default channels from CSV
    if channels is None:
        channels = ["FCz", "Cz", "Pz"]

    # If EEG columns are already in CSV (they are for the real data),
    # use FCz as default REWP
    erp_cols_present = [c for c in channels if c in df.columns]

    if erp_cols_present and eeg_dir is None:
        # Use the first available channel as default rewp
        df["rewp"] = df[erp_cols_present[0]]
    elif eeg_dir is not None:
        # Load EEG epoch files and extract REWP amplitudes
        df = _merge_eeg_epochs(df, eeg_dir, channels, time_window, sfreq)

    return df


def _merge_eeg_epochs(df, eeg_dir, channels, time_window, sfreq):
    """Merge EEG epoch data from files into the behavioral DataFrame."""
    tmin_samp = int(time_window[0] * sfreq)
    tmax_samp = int(time_window[1] * sfreq)

    for subj in df["subject"].unique():
        epoch_file = os.path.join(eeg_dir, f"{subj}_epochs.npy")
        if not os.path.exists(epoch_file):
            continue

        epochs = np.load(epoch_file)  # (n_trials, n_channels, n_times)
        # Extract mean amplitude in REWP window for first channel
        rewp_amp = epochs[:, 0, tmin_samp:tmax_samp].mean(axis=1)

        subj_mask = df["subject"] == subj
        n_match = min(subj_mask.sum(), len(rewp_amp))
        idx = df.index[subj_mask][:n_match]
        df.loc[idx, "rewp"] = rewp_amp[:n_match]

    return df
