# Neuroscience Research Hub

An integrated neuroscience research platform combining autonomous AI agents, reward learning models, neural decoding, and meta-analysis toolkits.

## Structure

```
neuro-hub/
├── agent/                  # Neuro-Coscientist: autonomous AI agent
│   ├── coscientist.py      # Main agent (LLM-orchestrated tool loop)
│   ├── tools/
│   │   ├── simulate.py     # 12 reward learning model simulators
│   │   ├── model_fitting.py# MLE fitting, param recovery, model comparison
│   │   ├── neural.py       # EEG temporal decoding, REWP extraction
│   │   ├── correlate.py    # Neuro-model correlation (RPE ↔ EEG)
│   │   └── plot.py         # Publication-quality figures
│   ├── tasks/              # Pre-built task templates
│   ├── prompts/            # System prompts for the agent
│   └── run.py              # CLI entry point
├── examples/               # Runnable sample scripts (start here!)
├── tmaze-analysis/         # T-Maze EEG-fMRI classification toolkit
├── neural-decoding/        # Multimodal neural decoding (SVM, searchlight, etc.)
├── meta-analysis/          # AI-powered literature meta-analysis
├── methods/                # Documented analysis techniques by modality
│   ├── behavior/           # T-maze, open field, fear conditioning
│   ├── imaging/            # Calcium imaging, fMRI, optical
│   └── ephys/              # Electrophysiology, spike sorting, LFP
├── papers/                 # Paper notes and extracted methods
├── scripts/                # Utility scripts
└── resources/              # Reference materials
```

## Quick Start

```bash
cd ~/neuro-hub

# Run the examples
python examples/01_simulate_and_fit.py      # Simulate + fit + compare models
python examples/02_parameter_recovery.py     # Validate model identifiability
python examples/03_neural_fusion.py          # Link RPEs to EEG (REWP)
python examples/04_real_data_template.py     # Template for your own data
python examples/05_agent_demo.py             # Run the autonomous agent
```

## Reward Learning Models (12 Implemented)

All models live in `agent/tools/simulate.py` (simulation) and `agent/tools/model_fitting.py` (fitting).

| # | Model | Key | Parameters | Category |
|---|-------|-----|-----------|----------|
| 1 | Random Responding | `random` | bias | Null baseline |
| 2 | Noisy Win-Stay-Lose-Shift | `wsls` | epsilon | Heuristic |
| 3 | Rescorla-Wagner | `rw` | alpha, beta | Model-free RL |
| 4 | Q-Learning | `q_learning` | alpha, beta | Model-free RL |
| 5 | Q-Learning Dual LR | `q_dual` | alpha_pos, alpha_neg, beta | Asymmetric RL |
| 6 | Actor-Critic | `actor_critic` | alpha_critic, alpha_actor, beta | Actor-Critic |
| 7 | Q-Learning + Decay | `q_decay` | alpha, beta, decay | RL w/ forgetting |
| 8 | RW + Side Bias | `rw_bias` | alpha, beta, bias | RL + nuisance |
| 9 | Choice Kernel | `ck` | alpha_c, beta_c | Perseveration |
| 10 | RW + Choice Kernel | `rwck` | alpha, beta, alpha_c, beta_c | RL + Perseveration |
| 11 | Value-Plus-Perseverance | `vpp` | A, alpha, lambda, c, ep_p, ep_n, K, w | IGT model |
| 12 | Outcome Representation Learning | `orl` | Arew, Apun, K, beta_f, beta_p | IGT model |

### Usage

```python
import sys; sys.path.insert(0, '~/neuro-hub')
import numpy as np
from agent.tools.simulate import simulate_q_dual
from agent.tools.model_fitting import fit_model, extract_trial_variables

# Simulate
sim = simulate_q_dual(200, 2, np.array([0.7, 0.3]),
    alpha_pos=0.5, alpha_neg=0.1, beta=5.0)

# Fit
result = fit_model('q_dual', sim['choices'], sim['outcomes'], n_options=2)
print(f"BIC={result.bic:.2f}, params={result.params}")

# Extract trial-level RPEs for neural fusion
latent = extract_trial_variables('q_dual', result.params,
    sim['choices'], sim['outcomes'])
rpes = latent['rpes']  # Use these as EEG/fMRI regressors
```

## Neuro-Coscientist Agent

Autonomous AI agent inspired by [Boiko et al. (2023)](https://doi.org/10.1038/s41586-023-06792-0). The agent orchestrates tools via an LLM to perform multi-step neuroscience analyses.

### Architecture

```
User Task → LLM (GPT-4 / Claude) → Tool Call → Tool Result → LLM → ... → STOP
```

### Available Tools

| Tool | Purpose |
|------|---------|
| `SIMULATE_BEHAVIOR` | Simulate behavioral data from any of 12 reward models |
| `SIMULATE_NEURAL` | Generate synthetic EEG epochs with RPE-coupled REWP |
| `FIT_MODEL` | MLE fitting via differential evolution or multistart |
| `PARAMETER_RECOVERY` | Simulate-then-fit validation loop |
| `COMPARE_MODELS` | BIC/AIC model comparison |
| `RUN_TEMPORAL_DECODING` | Time-resolved SVM/LDA decoding |
| `RUN_REWP` | REWP-focused EEG analysis (FCz, 240-340ms) |
| `CORRELATE_NEURO_MODEL` | Link RPEs to neural signals |
| `PLOT` | Publication-quality figure generation |

### Running the Agent

```bash
# With LLM (requires API key)
python agent/run.py --task "Simulate 50 subjects with RW, fit all models, compare"

# Offline mode (no API key needed)
python agent/run.py --preset full_pipeline_simulated --offline

# List available presets
python agent/run.py --list-presets
```

## Analysis Pipeline

```
1. Simulate → 2. Fit → 3. Compare → 4. Extract RPEs → 5. Neural Fusion
     ↓              ↓          ↓              ↓                ↓
  Behavioral    MLE/DE     BIC/AIC    Trial-level      RPE ↔ REWP
    data        params     ranking     Q-values        correlation
                                       & RPEs         single-trial GLM
```

## T-Maze Analysis Toolkit

EEG-fMRI classification pipeline for T-maze reward learning:
- Temporal decoding (SVM/LDA sliding estimator)
- REWP component analysis (240-340ms, FCz)
- Functional connectivity (coherence, phase-amplitude coupling)
- Deep learning models (EEGNet, CNN, RNN)
- Mixed-effects statistics

## Neural Decoding Toolkit

General-purpose multimodal neuroimaging decoder:
- Classifiers: SVM, Random Forest, LDA, Logistic, Ensemble
- Searchlight and temporal generalization
- Feature selection: ANOVA, RFE, stability selection
- Cross-validation: LORO, LOSO, StratifiedGroupKFold, permutation testing

## Meta-Analysis Toolkit

AI-powered meta-analysis for neuroscience literature:
- **AI Paper Screening**: LLM-based abstract screening
- **AI Data Extraction**: Extract coordinates and effect sizes from papers
- **Coordinate-Based**: ALE/MKDA via NiMARE
- **Effect Size**: Random effects via PyMARE
- **Visualization**: Brain maps, forest plots, funnel plots

## Requirements

```
numpy>=1.23
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
hbayesdm
pymc>=5.10
arviz
cmdstanpy
```

## References

- Wilson & Collins (2019). "Ten simple rules for computational modeling of behavioral data." *eLife*, 8:e49547.
- Boiko et al. (2023). "Autonomous chemical research with large language models." *Nature*, 624, 570-578.
- Ahn et al. (2017). "Revealing neurocomputational mechanisms with the hBayesDM package." *Computational Psychiatry*.
- Frank et al. (2007). "Genetic triple dissociation reveals multiple roles for dopamine in reinforcement learning." *PNAS*.
