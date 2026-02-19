"""
Pre-written task strings for common Neuro-Coscientist workflows.

Usage:
    from agent.tasks.example_tasks import TASKS
    agent.run(TASKS["parameter_recovery_rw"])
"""

TASKS = {
    # ── Behavioral Simulation ──────────────────────────────────────────
    "simulate_rw": (
        "Simulate 50 subjects with the Rescorla-Wagner model "
        "(alpha=0.3, beta=5.0) on a 2-armed bandit task with 200 trials. "
        "Reward probabilities: [0.7, 0.3]. Save the behavioral data."
    ),

    "simulate_orl_igt": (
        "Simulate 30 subjects with the ORL model on the Iowa Gambling Task "
        "(A_rew=0.3, A_pun=0.1, K=1.0, beta_f=2.0, beta_p=1.0). "
        "100 trials per subject. Save the data."
    ),

    # ── Parameter Recovery ─────────────────────────────────────────────
    "parameter_recovery_rw": (
        "Run parameter recovery for the Rescorla-Wagner model. "
        "True parameters: alpha=0.3, beta=5.0. "
        "Simulate 50 subjects with 200 trials each, then fit each. "
        "Report recovery quality (bias, correlation)."
    ),

    "parameter_recovery_rwck": (
        "Run parameter recovery for the RW+CK model. "
        "True parameters: alpha=0.3, beta=5.0, alpha_c=0.2, beta_c=2.0. "
        "50 simulations × 200 trials. Report recovery for all 4 parameters."
    ),

    # ── Model Comparison ──────────────────────────────────────────────
    "model_comparison_bandit": (
        "Simulate data from the RW model (alpha=0.3, beta=5.0, 200 trials). "
        "Then fit RW, CK, and RW+CK to the same data. "
        "Compare with BIC. Does the correct model win? "
        "Plot the BIC comparison."
    ),

    # ── Neural + Behavioral Pipeline ──────────────────────────────────
    "full_pipeline_simulated": (
        "Run the full simulated pipeline:\n"
        "1. Simulate behavioral data: RW model, alpha=0.3, beta=5.0, 200 trials.\n"
        "2. Simulate EEG data: 200 epochs, 64 channels, REWP=3µV, with RPEs coupled to REWP amplitude (coupling=2.0).\n"
        "3. Run temporal decoding on the simulated EEG.\n"
        "4. Run REWP analysis on FCz channels.\n"
        "5. Correlate trial-level RPEs with FCz REWP amplitude.\n"
        "6. Plot the temporal decoding time course.\n"
        "7. Plot the RPE-REWP correlation.\n"
        "Report all results and stop."
    ),

    "rewp_rpe_correlation": (
        "Test whether RPEs from a Rescorla-Wagner model correlate with REWP amplitude.\n"
        "1. Simulate 200 trials of RW behavior (alpha=0.3, beta=5.0).\n"
        "2. Simulate EEG with REWP amplitude modulated by RPEs (coupling=2.0).\n"
        "3. Correlate RPEs with FCz amplitude in the 240–340ms window.\n"
        "Report the Pearson r, p-value, and effect size."
    ),

    # ── Real Data Templates ───────────────────────────────────────────
    "fit_real_subject": (
        "Load behavioral data from {data_path} for subject {subject_id}. "
        "Fit RW, CK, and RW+CK models. Compare BIC. "
        "Extract RPEs from the best model. "
        "If EEG data is available at {eeg_path}, correlate RPEs with REWP amplitude."
    ),

    "group_model_comparison": (
        "For each subject in {data_path}:\n"
        "1. Fit RW, CK, RW+CK models.\n"
        "2. Record BIC for each.\n"
        "Find the group-level winning model (most subjects or lowest mean BIC). "
        "Report the winner and the distribution of individual best models."
    ),
}
