"""
Uncertainty propagation analysis for PLR feature to classifier predictions.

Implements Monte Carlo simulation to propagate feature uncertainties through
the classification pipeline, enabling analysis of clinical decision stability.

Cross-references:
- planning/uncertainty-propagation-analysis/
- appendix-literature-review/section-08-biostatistics.tex

Key analysis questions:
1. How often does uncertainty change clinical decisions?
2. What pupillometer accuracy is required for stable decisions?
3. How robust must SAMv3 pupil segmentation be?

References:
- Saltelli, A. (2002). Sensitivity Analysis in Practice.
- Sobol, I.M. (2001). Global sensitivity indices.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger

from ._types import StatsResult
from ._validation import validate_array, validate_positive

__all__ = [
    "UncertaintyResult",
    "SensitivityResult",
    "DecisionStabilityResult",
    "monte_carlo_classifier_uncertainty",
    "clinical_decision_stability",
    "sensitivity_analysis_delta",
    "compute_required_accuracy",
]


@dataclass
class UncertaintyResult(StatsResult):
    """
    Result of Monte Carlo uncertainty propagation.

    Attributes
    ----------
    prediction_mean : np.ndarray
        Mean prediction per subject (n_subjects,)
    prediction_std : np.ndarray
        Standard deviation of predictions per subject (n_subjects,)
    prediction_ci_lower : np.ndarray
        Lower CI bound per subject (n_subjects,)
    prediction_ci_upper : np.ndarray
        Upper CI bound per subject (n_subjects,)
    n_simulations : int
        Number of Monte Carlo simulations run
    """

    prediction_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_std: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_ci_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_ci_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    n_simulations: int = 0

    def __repr__(self) -> str:
        n = len(self.prediction_mean)
        return f"UncertaintyResult(n_subjects={n}, n_simulations={self.n_simulations})"


@dataclass
class SensitivityResult(StatsResult):
    """
    Result of sensitivity analysis.

    Attributes
    ----------
    feature_names : List[str]
        Names of features analyzed
    sensitivity_indices : np.ndarray
        Sensitivity index per feature (delta method)
    sensitivity_normalized : np.ndarray
        Normalized sensitivity (sum to 1)
    most_influential : str
        Name of most influential feature
    """

    feature_names: List[str] = field(default_factory=list)
    sensitivity_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    sensitivity_normalized: np.ndarray = field(default_factory=lambda: np.array([]))
    most_influential: str = ""

    def __repr__(self) -> str:
        return (
            f"SensitivityResult(n_features={len(self.feature_names)}, "
            f"most_influential='{self.most_influential}')"
        )


@dataclass
class DecisionStabilityResult(StatsResult):
    """
    Result of clinical decision stability analysis.

    Attributes
    ----------
    decision_stability_pct : float
        Percentage of subjects with stable clinical decisions
    n_unstable : int
        Number of subjects with unstable decisions
    unstable_indices : np.ndarray
        Indices of subjects with unstable decisions
    confidence_required : float
        Minimum prediction confidence for stable decision
    """

    decision_stability_pct: float = 0.0
    n_unstable: int = 0
    unstable_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_required: float = 0.0
    threshold: float = 0.5

    def __repr__(self) -> str:
        return (
            f"DecisionStabilityResult(stability={self.decision_stability_pct:.1f}%, "
            f"n_unstable={self.n_unstable})"
        )


def monte_carlo_classifier_uncertainty(
    features: np.ndarray,
    feature_uncertainties: np.ndarray,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    n_simulations: int = 1000,
    correlation_matrix: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    alpha: float = 0.05,
) -> UncertaintyResult:
    """
    Propagate feature uncertainty through classifier using Monte Carlo simulation.

    For each subject, simulates n_simulations feature vectors by sampling from
    the uncertainty distribution, runs classifier on each, and aggregates
    the prediction distribution.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_subjects, n_features) - point estimates
    feature_uncertainties : np.ndarray
        Standard deviation per feature (n_subjects, n_features) or (n_features,)
    predict_proba_fn : callable
        Function that takes features and returns probabilities: (N, D) -> (N,)
        Should return probability of positive class
    n_simulations : int, default 1000
        Number of Monte Carlo simulations per subject
    correlation_matrix : np.ndarray, optional
        Feature correlation matrix (n_features, n_features)
        If None, assumes features are independent
    random_state : int, optional
        Seed for reproducibility
    alpha : float, default 0.05
        Significance level for CIs (0.05 = 95% CI)

    Returns
    -------
    UncertaintyResult
        Contains prediction distributions and summary statistics

    Raises
    ------
    ValueError
        If features and uncertainties have incompatible shapes

    Notes
    -----
    IMPORTANT LIMITATION: This analysis assumes feature uncertainties are
    approximately Gaussian. For non-Gaussian uncertainties, consider using
    the actual bootstrap distributions if available.

    The pupil segmentation uncertainty is UNKNOWN - we do not have ground truth
    for the SAMv3 algorithm's accuracy. The analysis uses assumed uncertainty
    ranges based on literature values.

    Examples
    --------
    >>> features = np.random.randn(100, 10)  # 100 subjects, 10 features
    >>> uncertainties = np.full((100, 10), 0.1)  # 10% std on all features
    >>> def mock_predict(X):
    ...     return 1 / (1 + np.exp(-X.mean(axis=1)))  # Simple logistic
    >>> result = monte_carlo_classifier_uncertainty(
    ...     features, uncertainties, mock_predict, n_simulations=500
    ... )
    >>> print(f"Mean stability: {result.prediction_std.mean():.4f}")
    """
    features = validate_array(features, name="features")
    feature_uncertainties = validate_array(
        feature_uncertainties, name="feature_uncertainties"
    )
    validate_positive(n_simulations, "n_simulations")

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")

    n_subjects, n_features = features.shape

    # Handle broadcasting of uncertainties
    if feature_uncertainties.ndim == 1:
        if len(feature_uncertainties) != n_features:
            raise ValueError(
                f"1D uncertainties must have length n_features ({n_features}), "
                f"got {len(feature_uncertainties)}"
            )
        feature_uncertainties = np.broadcast_to(
            feature_uncertainties, (n_subjects, n_features)
        )
    elif feature_uncertainties.shape != features.shape:
        raise ValueError(
            f"uncertainties shape {feature_uncertainties.shape} must match "
            f"features shape {features.shape}"
        )

    rng = np.random.default_rng(random_state)

    # Storage for predictions
    all_predictions = np.zeros((n_subjects, n_simulations))

    # Generate correlated samples if correlation matrix provided
    if correlation_matrix is not None:
        if correlation_matrix.shape != (n_features, n_features):
            raise ValueError(
                f"correlation_matrix shape {correlation_matrix.shape} must be "
                f"({n_features}, {n_features})"
            )
        # Cholesky decomposition for correlated sampling
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite, using diagonal")
            L = np.eye(n_features)
    else:
        L = np.eye(n_features)

    logger.info(f"Running {n_simulations} MC simulations for {n_subjects} subjects...")

    for sim_idx in range(n_simulations):
        # Sample random perturbations (standardized)
        z = rng.standard_normal((n_subjects, n_features))

        # Apply correlation structure
        z_corr = z @ L.T

        # Scale by feature-specific uncertainties
        perturbations = z_corr * feature_uncertainties

        # Perturbed features
        features_perturbed = features + perturbations

        # Get predictions
        predictions = predict_proba_fn(features_perturbed)
        all_predictions[:, sim_idx] = predictions

        if (sim_idx + 1) % 100 == 0:
            logger.debug(f"  Completed {sim_idx + 1}/{n_simulations} simulations")

    # Compute summary statistics
    prediction_mean = np.mean(all_predictions, axis=1)
    prediction_std = np.std(all_predictions, axis=1)
    prediction_ci_lower = np.percentile(all_predictions, 100 * alpha / 2, axis=1)
    prediction_ci_upper = np.percentile(all_predictions, 100 * (1 - alpha / 2), axis=1)

    logger.info(
        f"MC simulation complete. Mean prediction std: {prediction_std.mean():.4f}"
    )

    return UncertaintyResult(
        prediction_mean=prediction_mean,
        prediction_std=prediction_std,
        prediction_ci_lower=prediction_ci_lower,
        prediction_ci_upper=prediction_ci_upper,
        n_simulations=n_simulations,
        scalars={
            "mean_prediction_std": float(np.mean(prediction_std)),
            "max_prediction_std": float(np.max(prediction_std)),
            "median_prediction_std": float(np.median(prediction_std)),
        },
        arrays={
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "prediction_ci_lower": prediction_ci_lower,
            "prediction_ci_upper": prediction_ci_upper,
            "all_predictions": all_predictions,
        },
        metadata={
            "n_subjects": n_subjects,
            "n_features": n_features,
            "n_simulations": n_simulations,
            "alpha": alpha,
            "used_correlation": correlation_matrix is not None,
        },
    )


def clinical_decision_stability(
    predictions: np.ndarray,
    threshold: float = 0.5,
    stability_criterion: str = "majority",
    min_confidence: float = 0.6,
) -> DecisionStabilityResult:
    """
    Compute clinical decision stability from Monte Carlo predictions.

    A stable decision is one where the clinical action (treat/don't treat)
    remains consistent despite feature uncertainty.

    Parameters
    ----------
    predictions : np.ndarray
        MC prediction matrix (n_subjects, n_simulations)
    threshold : float, default 0.5
        Decision threshold (predict positive if prob >= threshold)
    stability_criterion : str, default "majority"
        How to define stability:
        - "majority": >=90% of simulations agree on decision
        - "all": 100% of simulations must agree
        - "confidence": mean prediction must be far from threshold
    min_confidence : float, default 0.6
        For "confidence" criterion, minimum distance from threshold needed

    Returns
    -------
    DecisionStabilityResult
        Contains stability metrics and unstable subject indices

    Examples
    --------
    >>> predictions = np.random.beta(2, 5, size=(100, 1000))  # 100 subjects
    >>> result = clinical_decision_stability(predictions, threshold=0.5)
    >>> print(f"{result.decision_stability_pct:.1f}% of decisions are stable")
    """
    predictions = validate_array(predictions, name="predictions")

    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

    n_subjects, n_simulations = predictions.shape

    # Binary decisions for each simulation
    decisions = (predictions >= threshold).astype(int)

    # Proportion voting positive
    prop_positive = np.mean(decisions, axis=1)

    # Determine stability based on criterion
    if stability_criterion == "majority":
        # Stable if >= 90% agree
        is_stable = (prop_positive >= 0.9) | (prop_positive <= 0.1)

    elif stability_criterion == "all":
        # Stable if 100% agree
        is_stable = (prop_positive == 1.0) | (prop_positive == 0.0)

    elif stability_criterion == "confidence":
        # Stable if mean prediction is far from threshold
        mean_pred = np.mean(predictions, axis=1)
        distance_from_threshold = np.abs(mean_pred - threshold)
        confidence_needed = min_confidence - 0.5  # Convert to distance
        is_stable = distance_from_threshold >= confidence_needed

    else:
        raise ValueError(
            f"Unknown stability_criterion '{stability_criterion}'. "
            "Must be 'majority', 'all', or 'confidence'."
        )

    n_stable = np.sum(is_stable)
    stability_pct = 100.0 * n_stable / n_subjects
    unstable_indices = np.where(~is_stable)[0]

    # Compute confidence required for stability (for majority criterion)
    # This is the prediction value where 90% of simulations would agree
    # Given uncertainty, what confidence do we need?
    np.mean(predictions, axis=1)
    std_preds = np.std(predictions, axis=1)
    # For 90% agreement, need mean +/- ~1.64*std to not cross threshold
    # (assuming approximately normal prediction distribution)
    safe_distance = 1.645 * np.median(std_preds)
    confidence_required = 0.5 + safe_distance

    return DecisionStabilityResult(
        decision_stability_pct=stability_pct,
        n_unstable=len(unstable_indices),
        unstable_indices=unstable_indices,
        confidence_required=confidence_required,
        threshold=threshold,
        scalars={
            "stability_pct": stability_pct,
            "n_stable": float(n_stable),
            "n_unstable": float(len(unstable_indices)),
            "confidence_required": confidence_required,
        },
        arrays={
            "is_stable": is_stable,
            "prop_positive": prop_positive,
        },
        metadata={
            "n_subjects": n_subjects,
            "n_simulations": n_simulations,
            "threshold": threshold,
            "stability_criterion": stability_criterion,
        },
    )


def sensitivity_analysis_delta(
    features: np.ndarray,
    feature_uncertainties: np.ndarray,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: Optional[List[str]] = None,
    delta_fraction: float = 0.01,
) -> SensitivityResult:
    """
    Compute sensitivity indices using delta method (local sensitivity).

    Measures how much classifier predictions change when each feature
    is perturbed by a small amount, scaled by feature uncertainty.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_subjects, n_features)
    feature_uncertainties : np.ndarray
        Standard deviation per feature (n_features,) or (n_subjects, n_features)
    predict_proba_fn : callable
        Classifier prediction function
    feature_names : List[str], optional
        Names of features (for reporting)
    delta_fraction : float, default 0.01
        Fraction of uncertainty to perturb (for numerical gradient)

    Returns
    -------
    SensitivityResult
        Contains sensitivity indices per feature

    Notes
    -----
    This is a local sensitivity analysis (around the nominal feature values).
    For global sensitivity, consider Sobol indices with larger uncertainty ranges.

    Examples
    --------
    >>> features = np.random.randn(100, 5)
    >>> uncertainties = np.array([0.1, 0.2, 0.1, 0.3, 0.1])
    >>> def predict(X): return 1 / (1 + np.exp(-X[:, 0] - 2*X[:, 1]))
    >>> result = sensitivity_analysis_delta(
    ...     features, uncertainties, predict,
    ...     feature_names=['A', 'B', 'C', 'D', 'E']
    ... )
    >>> print(f"Most influential: {result.most_influential}")
    """
    features = validate_array(features, name="features")
    feature_uncertainties = validate_array(feature_uncertainties, name="uncertainties")

    n_subjects, n_features = features.shape

    # Handle uncertainty shape
    if feature_uncertainties.ndim == 1:
        if len(feature_uncertainties) != n_features:
            raise ValueError(f"1D uncertainties must have length {n_features}")
        uncertainties_per_feature = feature_uncertainties
    else:
        # Average uncertainty across subjects
        uncertainties_per_feature = np.mean(feature_uncertainties, axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Base predictions
    predict_proba_fn(features)

    # Compute gradient for each feature
    sensitivity_indices = np.zeros(n_features)

    for feat_idx in range(n_features):
        # Perturbation size
        delta = uncertainties_per_feature[feat_idx] * delta_fraction

        if delta < 1e-10:
            sensitivity_indices[feat_idx] = 0.0
            continue

        # Perturbed features
        features_plus = features.copy()
        features_plus[:, feat_idx] += delta

        features_minus = features.copy()
        features_minus[:, feat_idx] -= delta

        # Predictions with perturbation
        pred_plus = predict_proba_fn(features_plus)
        pred_minus = predict_proba_fn(features_minus)

        # Central difference gradient
        gradient = (pred_plus - pred_minus) / (2 * delta)

        # Sensitivity = |gradient| * uncertainty (mean over subjects)
        sensitivity_indices[feat_idx] = np.mean(
            np.abs(gradient) * uncertainties_per_feature[feat_idx]
        )

    # Normalize
    total_sensitivity = np.sum(sensitivity_indices)
    if total_sensitivity > 1e-10:
        sensitivity_normalized = sensitivity_indices / total_sensitivity
    else:
        sensitivity_normalized = np.zeros(n_features)

    # Find most influential
    most_influential_idx = np.argmax(sensitivity_indices)
    most_influential = feature_names[most_influential_idx]

    return SensitivityResult(
        feature_names=feature_names,
        sensitivity_indices=sensitivity_indices,
        sensitivity_normalized=sensitivity_normalized,
        most_influential=most_influential,
        scalars={
            "total_sensitivity": float(total_sensitivity),
            "max_sensitivity": float(np.max(sensitivity_indices)),
        },
        arrays={
            "sensitivity_indices": sensitivity_indices,
            "sensitivity_normalized": sensitivity_normalized,
        },
        metadata={
            "n_features": n_features,
            "n_subjects": n_subjects,
            "delta_fraction": delta_fraction,
            "method": "delta",
        },
    )


def compute_required_accuracy(
    features: np.ndarray,
    feature_uncertainties: np.ndarray,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    target_stability: float = 0.95,
    uncertainty_multipliers: np.ndarray = None,
    n_simulations: int = 500,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Determine required instrument accuracy for target decision stability.

    Iteratively reduces feature uncertainties until target stability is achieved.
    Useful for determining: "How accurate must my pupillometer be to achieve
    95% decision stability?"

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_subjects, n_features)
    feature_uncertainties : np.ndarray
        Current/baseline standard deviations (n_features,)
    predict_proba_fn : callable
        Classifier prediction function
    target_stability : float, default 0.95
        Target decision stability (0.0 to 1.0)
    uncertainty_multipliers : np.ndarray, optional
        Multipliers to test (default: [0.1, 0.25, 0.5, 0.75, 1.0])
    n_simulations : int, default 500
        MC simulations per multiplier level
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    dict
        Contains:
        - 'multipliers': tested uncertainty multipliers
        - 'stabilities': achieved stability at each level
        - 'required_multiplier': multiplier needed for target stability
        - 'required_uncertainties': actual uncertainty values needed

    Examples
    --------
    >>> # Determine how much accuracy improvement is needed
    >>> result = compute_required_accuracy(
    ...     features, baseline_uncertainties, predict_fn,
    ...     target_stability=0.95
    ... )
    >>> print(f"Need {result['required_multiplier']:.2f}x current accuracy")
    """
    if uncertainty_multipliers is None:
        uncertainty_multipliers = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    features = validate_array(features, name="features")
    feature_uncertainties = validate_array(feature_uncertainties, name="uncertainties")

    results_by_multiplier = {}
    stabilities = []

    logger.info(
        f"Computing accuracy requirements for {target_stability:.0%} stability..."
    )

    for mult in uncertainty_multipliers:
        scaled_uncertainties = feature_uncertainties * mult

        # Run MC simulation
        mc_result = monte_carlo_classifier_uncertainty(
            features,
            scaled_uncertainties,
            predict_proba_fn,
            n_simulations=n_simulations,
            random_state=random_state,
        )

        # Check stability
        stability_result = clinical_decision_stability(
            mc_result.arrays["all_predictions"],
            threshold=0.5,
            stability_criterion="majority",
        )

        stability = stability_result.decision_stability_pct / 100.0
        stabilities.append(stability)
        results_by_multiplier[mult] = {
            "stability": stability,
            "n_unstable": stability_result.n_unstable,
        }

        logger.debug(f"  Multiplier {mult:.2f}x: stability = {stability:.1%}")

    stabilities = np.array(stabilities)

    # Find required multiplier (linear interpolation)
    if stabilities[-1] < target_stability:
        # Even lowest uncertainty doesn't achieve target
        required_multiplier = uncertainty_multipliers[0] * (
            target_stability / stabilities[0]
        )
        logger.warning(
            f"Target stability {target_stability:.0%} not achievable with tested range. "
            f"Extrapolated multiplier: {required_multiplier:.3f}"
        )
    elif stabilities[0] >= target_stability:
        # Already achieved at highest uncertainty
        required_multiplier = uncertainty_multipliers[-1]
    else:
        # Interpolate
        idx = np.searchsorted(stabilities, target_stability)
        if idx == 0:
            required_multiplier = uncertainty_multipliers[0]
        else:
            # Linear interpolation
            x0, x1 = uncertainty_multipliers[idx - 1], uncertainty_multipliers[idx]
            y0, y1 = stabilities[idx - 1], stabilities[idx]
            required_multiplier = x0 + (target_stability - y0) * (x1 - x0) / (y1 - y0)

    required_uncertainties = feature_uncertainties * required_multiplier

    return {
        "multipliers": uncertainty_multipliers,
        "stabilities": stabilities,
        "required_multiplier": required_multiplier,
        "required_uncertainties": required_uncertainties,
        "target_stability": target_stability,
        "achieved_at_baseline": stabilities[-1] if len(stabilities) > 0 else 0.0,
        "details": results_by_multiplier,
    }
