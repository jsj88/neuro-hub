"""
Pre-defined LME model formulas for RPE-REWP fusion.

Each model spec is a dict with:
    formula: statsmodels formula string (fixed effects)
    re_formula: random effects formula (None = random intercept only)
    description: human-readable description
"""

MODEL_REGISTRY = {
    "rpe_only": {
        "formula": "rewp ~ rpe",
        "re_formula": None,
        "description": "RPE predicts REWP, random intercept only",
    },
    "rpe_random_slope": {
        "formula": "rewp ~ rpe",
        "re_formula": "~rpe",
        "description": "RPE predicts REWP, random intercept + slope for RPE",
    },
    "rpe_outcome": {
        "formula": "rewp ~ rpe + reward",
        "re_formula": "~rpe",
        "description": "RPE + reward condition as fixed effects, random slope for RPE",
    },
    "rpe_condition": {
        "formula": "rewp ~ rpe * reward",
        "re_formula": "~rpe",
        "description": "RPE x reward interaction, random slope for RPE",
    },
    "rpe_trial": {
        "formula": "rewp ~ rpe + trial_z",
        "re_formula": "~rpe",
        "description": "RPE + z-scored trial order, random slope for RPE",
    },
    "rpe_stay_shift": {
        "formula": "rewp ~ rpe * C(PES_PRS_index)",
        "re_formula": "~rpe",
        "description": "RPE x stay/shift interaction (T-maze specific)",
    },
    "rpe_velocity": {
        "formula": "rewp ~ rpe * C(Vel_Cond)",
        "re_formula": "~rpe",
        "description": "RPE x velocity condition interaction (T-maze specific)",
    },
}


def get_model_spec(name):
    """Get a model specification by name.

    Parameters
    ----------
    name : str
        Model name from MODEL_REGISTRY.

    Returns
    -------
    dict
        Model specification with formula, re_formula, description.

    Raises
    ------
    KeyError
        If model name not found.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


def list_models():
    """Print all available model specifications."""
    for name, spec in MODEL_REGISTRY.items():
        print(f"  {name:25s}  {spec['description']}")
        print(f"{'':27s}  formula: {spec['formula']}")
        print(f"{'':27s}  RE: {spec['re_formula'] or '~1 (intercept only)'}")
        print()
