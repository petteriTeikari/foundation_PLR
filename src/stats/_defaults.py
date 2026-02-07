"""
Centralized default constants for statistical computations.

These values mirror configs/defaults.yaml and serve as function-signature
defaults. If the config value changes, update the constant here and all
downstream functions pick it up automatically.

Config source: configs/defaults.yaml → CLS_EVALUATION
"""

# Bootstrap parameters (CLS_EVALUATION.BOOTSTRAP)
DEFAULT_N_BOOTSTRAP: int = 1000  # CLS_EVALUATION.BOOTSTRAP.n_iterations
DEFAULT_CI_LEVEL: float = 0.95  # CLS_EVALUATION.BOOTSTRAP.alpha_CI

# Glaucoma screening parameters (CLS_EVALUATION.glaucoma_params)
DEFAULT_TPAUC_SENSITIVITY: float = (
    0.862  # CLS_EVALUATION.glaucoma_params.tpAUC_sensitivity
)
DEFAULT_TPAUC_SPECIFICITY: float = (
    0.821  # CLS_EVALUATION.glaucoma_params.tpAUC_specificity
)
