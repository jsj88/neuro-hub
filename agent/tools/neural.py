"""
Neural analysis tools — wrappers around the existing tmaze-analysis pipeline.

RunTemporalDecoding: Time-resolved SVM/LDA decoding via MNE SlidingEstimator.
RunREWP: Reward Positivity (REWP) focused temporal analysis on FCz channels.
"""

import sys
import os
from typing import Any, Dict
import numpy as np

from .base import BaseTool

# Add tmaze-analysis to path so we can import from the existing pipeline
_TMAZE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "tmaze-analysis")
if _TMAZE_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_TMAZE_ROOT))


def _build_eeg_data(data_dict: Dict) -> Any:
    """Build a TMazeEEGData from a dict (e.g., from SimulateNeural)."""
    from core.containers import TMazeEEGData

    return TMazeEEGData(
        data=data_dict["data"],
        times=data_dict["times"],
        labels=data_dict["labels"],
        condition_names=["NoReward", "Reward"],
        channels=data_dict["channels"],
        sfreq=data_dict["sfreq"],
        subject_id=data_dict.get("subject_id", "sim-001"),
    )


class RunTemporalDecoding(BaseTool):
    """Agent tool: run time-resolved decoding on EEG data."""

    def __init__(self):
        super().__init__("RUN_TEMPORAL_DECODING")

    @property
    def description(self) -> str:
        return (
            "Run time-resolved SVM/LDA decoding on EEG data.\n"
            "Input JSON keys:\n"
            "  source: str — 'simulated' (uses last SimulateNeural output) or 'file'\n"
            "  file_path: str — path to .fif epoch file (if source='file')\n"
            "  classifier: str — 'svm' or 'lda' (default 'svm')\n"
            "  cv_folds: int — number of cross-validation folds (default 5)\n"
            "  scoring: str — 'roc_auc' or 'accuracy' (default 'roc_auc')\n"
            "Returns: peak decoding time, peak score, significant clusters."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        from classification.temporal import temporal_decoding, find_significant_times

        source = params.get("source", "simulated")
        classifier = params.get("classifier", "svm")
        cv_folds = params.get("cv_folds", 5)
        scoring = params.get("scoring", "roc_auc")

        if source == "simulated":
            # Import from agent context — the SimulateNeural tool stores data on itself
            # The coscientist.py dispatcher passes shared_state
            data_dict = params.get("_simulated_data")
            if data_dict is None:
                return "ERROR: No simulated data found. Run SIMULATE_NEURAL first."
            eeg_data = _build_eeg_data(data_dict)

        elif source == "file":
            file_path = params.get("file_path")
            if not file_path:
                return "ERROR: file_path required when source='file'"
            try:
                from io.loaders import TMazeEEGLoader
                loader = TMazeEEGLoader(
                    condition_mapping={"MazeReward": 1, "MazeNoReward": 0},
                    tmin=-0.2, tmax=0.8,
                )
                eeg_data = loader.load(file_path)
            except Exception as e:
                return f"ERROR loading file: {e}"
        else:
            return f"ERROR: Unknown source '{source}'. Use 'simulated' or 'file'."

        # Run decoding
        result = temporal_decoding(
            eeg_data,
            classifier_type=classifier,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1,
            verbose=False,
        )

        # Find significant windows
        sig_mask, clusters = find_significant_times(result, chance_level=0.5, alpha=0.05)

        # Format output
        cluster_str = ", ".join(
            [f"[{c[0]*1000:.0f}–{c[1]*1000:.0f}ms]" for c in clusters]
        ) if clusters else "none"

        return (
            f"Temporal Decoding Results ({classifier.upper()}, {scoring}, {cv_folds}-fold CV):\n"
            f"  Data: {eeg_data.n_epochs} epochs × {eeg_data.n_channels} ch × {eeg_data.n_times} times\n"
            f"  Peak score: {result.peak_score:.4f} at {result.peak_time*1000:.0f}ms\n"
            f"  Onset: {result.onset_time*1000:.0f}ms\n"
            f"  Significant clusters: {cluster_str}\n"
            f"  Chance level: 0.5 | Significant time points: {sig_mask.sum()}/{len(sig_mask)}"
        )


class RunREWP(BaseTool):
    """Agent tool: REWP-focused temporal analysis on FCz channels."""

    def __init__(self):
        super().__init__("RUN_REWP")

    @property
    def description(self) -> str:
        return (
            "Run REWP (Reward Positivity) temporal analysis focused on FCz channels (0–500ms).\n"
            "Input JSON keys:\n"
            "  source: str — 'simulated' or 'file'\n"
            "  file_path: str — path to .fif epoch file (if source='file')\n"
            "  classifier: str — 'svm' or 'lda' (default 'svm')\n"
            "  cv_folds: int — (default 5)\n"
            "Returns: REWP-window decoding results."
        )

    def __call__(self, params: Dict[str, Any]) -> str:
        from classification.temporal import rewp_temporal_analysis

        source = params.get("source", "simulated")
        classifier = params.get("classifier", "svm")
        cv_folds = params.get("cv_folds", 5)

        if source == "simulated":
            data_dict = params.get("_simulated_data")
            if data_dict is None:
                return "ERROR: No simulated data found. Run SIMULATE_NEURAL first."
            eeg_data = _build_eeg_data(data_dict)
        elif source == "file":
            file_path = params.get("file_path")
            try:
                from io.loaders import TMazeEEGLoader
                loader = TMazeEEGLoader(
                    condition_mapping={"MazeReward": 1, "MazeNoReward": 0},
                    tmin=-0.2, tmax=0.8,
                )
                eeg_data = loader.load(file_path)
            except Exception as e:
                return f"ERROR loading file: {e}"
        else:
            return f"ERROR: Unknown source '{source}'."

        result = rewp_temporal_analysis(
            eeg_data,
            fcz_only=True,
            classifier_type=classifier,
            cv=cv_folds,
            n_jobs=-1,
        )

        cluster_str = ", ".join(
            [f"[{c[0]*1000:.0f}–{c[1]*1000:.0f}ms]" for c in (result.significant_clusters or [])]
        ) or "none"

        in_rewp = (
            result.peak_time is not None
            and 0.240 <= result.peak_time <= 0.340
        )

        return (
            f"REWP Analysis ({classifier.upper()}, FCz-region, 0–500ms):\n"
            f"  Peak score: {result.peak_score:.4f} at {result.peak_time*1000:.0f}ms\n"
            f"  Peak in REWP window (240–340ms): {'YES ✓' if in_rewp else 'NO ✗'}\n"
            f"  Significant clusters: {cluster_str}\n"
            f"  Onset: {result.onset_time*1000:.0f}ms" if result.onset_time else ""
        )
