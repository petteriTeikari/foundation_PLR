"""Group-level aggregation of PLR decomposition results with confidence intervals.

Phase 4 of the decomposition figure pipeline:
1. Load preprocessed signals from DuckDB
2. Apply decomposition methods per preprocessing config
3. Aggregate across subjects within each preprocessing category
4. Compute bootstrap confidence intervals

This module is designed to work with the per-subject DuckDB created by
scripts/extract_decomposition_signals.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import yaml
from numpy.random import Generator
from numpy.typing import NDArray

from src.stats._defaults import DEFAULT_CI_LEVEL, DEFAULT_N_BOOTSTRAP

from .ged import GEDDecomposition
from .pca_methods import RotatedPCA, SparsePCADecomposition, StandardPCA
from .template_fitting import TemplateFitting

# Load category mapping from config
CATEGORY_MAPPING_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "mlflow_registry"
    / "category_mapping.yaml"
)


def load_category_mapping() -> dict[str, str]:
    """Load preprocessing category mapping from YAML config.

    IMPORTANT: Config is REQUIRED - no fallback mapping.
    This ensures single source of truth from category_mapping.yaml.

    Raises
    ------
    FileNotFoundError
        If category_mapping.yaml does not exist
    ValueError
        If preprocessing_categories section is missing or empty
    """
    if not CATEGORY_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Category mapping config required but not found: {CATEGORY_MAPPING_PATH}\n"
            "This file is the single source of truth for method→category mappings."
        )

    with open(CATEGORY_MAPPING_PATH) as f:
        mapping = yaml.safe_load(f)

    # The config uses outlier_method_categories, not preprocessing_categories
    # Build mapping from exact matches and patterns
    result = {}

    # Get exact matches
    exact = mapping.get("outlier_method_categories", {}).get("exact", {})
    result.update(exact)

    # Note: Pattern-based matching is handled by get_outlier_category() in extraction
    # For aggregation, we only need the exact mapping for validation

    if not result:
        raise ValueError(
            f"No exact category mappings found in {CATEGORY_MAPPING_PATH}\n"
            "Expected 'outlier_method_categories.exact' section."
        )

    return result


def get_preprocessing_categories() -> list[str]:
    """Load preprocessing category display names from config.

    Returns list in canonical order: Ground Truth, Foundation Model, Deep Learning,
    Traditional, Ensemble.
    """
    display_names_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "mlflow_registry"
        / "display_names.yaml"
    )

    if not display_names_path.exists():
        raise FileNotFoundError(
            f"display_names.yaml not found at {display_names_path}\n"
            "This file is required for preprocessing category names."
        )

    with open(display_names_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract category display names in canonical order
    categories = config.get("categories", {})
    order = [
        "ground_truth",
        "foundation_model",
        "deep_learning",
        "traditional",
        "ensemble",
    ]
    return [categories.get(cat_id, {}).get("display_name", cat_id) for cat_id in order]


DecompositionMethod = Literal["template", "pca", "rotated_pca", "sparse_pca", "ged"]
# PreprocessingCategory is loaded from config - see get_preprocessing_categories()
PreprocessingCategory = str  # Runtime values loaded from display_names.yaml


@dataclass
class ComponentTimecourse:
    """Single component's aggregated timecourse with CIs."""

    name: str  # e.g., "phasic", "PC1", "GED1"
    mean: NDArray[np.float64]  # (n_timepoints,)
    ci_lower: NDArray[np.float64]  # (n_timepoints,) - 2.5%
    ci_upper: NDArray[np.float64]  # (n_timepoints,) - 97.5%


@dataclass
class DecompositionResult:
    """Aggregated decomposition result for a preprocessing category."""

    category: PreprocessingCategory
    method: DecompositionMethod
    time_vector: NDArray[np.float64]
    components: list[ComponentTimecourse]
    n_subjects: int
    mean_waveform: NDArray[np.float64]  # Group mean input waveform
    mean_waveform_ci_lower: NDArray[np.float64]
    mean_waveform_ci_upper: NDArray[np.float64]


@dataclass
class DecompositionAggregator:
    """Compute group-level decompositions with bootstrap CIs.

    Parameters
    ----------
    db_path : Path
        Path to the preprocessed signals DuckDB
    n_bootstrap : int
        Number of bootstrap iterations for CIs (default: 1000)
    ci_level : float
        Confidence level (default: 0.95 for 95% CI)
    random_seed : int
        Random seed for reproducibility
    """

    db_path: Path
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    ci_level: float = DEFAULT_CI_LEVEL
    random_seed: int = 42
    _category_mapping: dict[str, str] = field(
        default_factory=load_category_mapping, init=False
    )
    rng: Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        self._category_mapping = load_category_mapping()

    def load_signals_by_category(
        self, category: PreprocessingCategory, limit: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Load all preprocessed signals for a category.

        Returns
        -------
        waveforms : ndarray
            Shape (n_subjects, n_timepoints)
        time_vector : ndarray
            Shape (n_timepoints,)
        """
        conn = duckdb.connect(str(self.db_path), read_only=True)

        # Get configs that belong to this category
        query = """
            SELECT DISTINCT signal, time_vector
            FROM preprocessed_signals
            WHERE preprocessing_category = ?
        """
        params = [category]

        if limit:
            query += f" LIMIT {limit}"

        result = conn.execute(query, params).fetchall()
        conn.close()

        if not result:
            raise ValueError(f"No signals found for category: {category}")

        # Convert to arrays
        waveforms = np.array([row[0] for row in result])
        time_vector = np.array(result[0][1])  # Same for all

        return waveforms, time_vector

    def _bootstrap_mean_ci(
        self, data: NDArray[np.float64], axis: int = 0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Compute mean and bootstrap CI across specified axis.

        Parameters
        ----------
        data : ndarray
            Data array, shape depends on what we're aggregating
        axis : int
            Axis to compute mean across (subjects axis)

        Returns
        -------
        mean, ci_lower, ci_upper : tuple of ndarray
        """
        n_samples = data.shape[axis]
        mean = np.mean(data, axis=axis)

        # Bootstrap
        alpha = 1 - self.ci_level
        bootstrap_means = []

        for _ in range(self.n_bootstrap):
            indices = self.rng.integers(0, n_samples, size=n_samples)
            boot_sample = np.take(data, indices, axis=axis)
            bootstrap_means.append(np.mean(boot_sample, axis=axis))

        bootstrap_means = np.array(bootstrap_means)
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2), axis=0)

        return mean, ci_lower, ci_upper

    def aggregate_template_fitting(
        self, waveforms: NDArray[np.float64], time_vector: NDArray[np.float64]
    ) -> list[ComponentTimecourse]:
        """Apply template fitting and aggregate components.

        Returns component timecourses for: phasic, sustained, pipr
        """
        tf = TemplateFitting()

        # Fit all subjects
        results = tf.fit_batch(waveforms, time_vector)

        # Stack component arrays: (n_subjects, n_timepoints)
        phasic_all = np.array([r.phasic for r in results])
        sustained_all = np.array([r.sustained for r in results])
        pipr_all = np.array([r.pipr for r in results])

        # Compute aggregated means and CIs
        components = []
        for name, data in [
            ("phasic", phasic_all),
            ("sustained", sustained_all),
            ("pipr", pipr_all),
        ]:
            mean, ci_lo, ci_hi = self._bootstrap_mean_ci(data, axis=0)
            components.append(
                ComponentTimecourse(
                    name=name, mean=mean, ci_lower=ci_lo, ci_upper=ci_hi
                )
            )

        return components

    def aggregate_pca(
        self, waveforms: NDArray[np.float64], n_components: int = 3
    ) -> list[ComponentTimecourse]:
        """Apply standard PCA and return component loadings with CIs.

        For PCA, we return the loadings (eigenvectors) which represent
        the temporal patterns, with CIs from bootstrap.
        """
        components = []

        # Bootstrap to get CI on loadings
        all_loadings = []
        for _ in range(self.n_bootstrap):
            indices = self.rng.integers(0, len(waveforms), size=len(waveforms))
            boot_sample = waveforms[indices]

            pca = StandardPCA(n_components=n_components)
            result = pca.fit_transform(boot_sample)

            # Handle sign flips by aligning to first bootstrap
            if all_loadings:
                for i in range(n_components):
                    if np.dot(result.loadings[i], all_loadings[0][i]) < 0:
                        result.loadings[i] *= -1

            all_loadings.append(result.loadings)

        all_loadings = np.array(
            all_loadings
        )  # (n_bootstrap, n_components, n_timepoints)

        # Fit on full data for mean
        pca = StandardPCA(n_components=n_components)
        result = pca.fit_transform(waveforms)

        for i in range(n_components):
            mean = result.loadings[i]

            # Align bootstrap loadings to full-data mean
            for b in range(self.n_bootstrap):
                if np.dot(all_loadings[b, i], mean) < 0:
                    all_loadings[b, i] *= -1

            alpha = 1 - self.ci_level
            ci_lo = np.percentile(all_loadings[:, i, :], 100 * alpha / 2, axis=0)
            ci_hi = np.percentile(all_loadings[:, i, :], 100 * (1 - alpha / 2), axis=0)

            components.append(
                ComponentTimecourse(
                    name=f"PC{i + 1}", mean=mean, ci_lower=ci_lo, ci_upper=ci_hi
                )
            )

        return components

    def aggregate_rotated_pca(
        self, waveforms: NDArray[np.float64], n_components: int = 3
    ) -> list[ComponentTimecourse]:
        """Apply rotated PCA (Promax) and return component loadings with CIs."""
        components = []

        # Bootstrap
        all_loadings = []
        for _ in range(self.n_bootstrap):
            indices = self.rng.integers(0, len(waveforms), size=len(waveforms))
            boot_sample = waveforms[indices]

            rpca = RotatedPCA(n_components=n_components)
            result = rpca.fit_transform(boot_sample)

            # Handle sign flips
            if all_loadings:
                for i in range(n_components):
                    if np.dot(result.loadings[i], all_loadings[0][i]) < 0:
                        result.loadings[i] *= -1

            all_loadings.append(result.loadings)

        all_loadings = np.array(all_loadings)

        # Fit on full data
        rpca = RotatedPCA(n_components=n_components)
        result = rpca.fit_transform(waveforms)

        for i in range(n_components):
            mean = result.loadings[i]

            for b in range(self.n_bootstrap):
                if np.dot(all_loadings[b, i], mean) < 0:
                    all_loadings[b, i] *= -1

            alpha = 1 - self.ci_level
            ci_lo = np.percentile(all_loadings[:, i, :], 100 * alpha / 2, axis=0)
            ci_hi = np.percentile(all_loadings[:, i, :], 100 * (1 - alpha / 2), axis=0)

            components.append(
                ComponentTimecourse(
                    name=f"RC{i + 1}",  # Rotated Component
                    mean=mean,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                )
            )

        return components

    def aggregate_sparse_pca(
        self, waveforms: NDArray[np.float64], n_components: int = 3
    ) -> list[ComponentTimecourse]:
        """Apply sparse PCA and return component loadings with CIs."""
        components = []

        # Bootstrap (fewer iterations due to sparse PCA being slow)
        n_boot = min(self.n_bootstrap, 100)
        all_loadings = []
        for _ in range(n_boot):
            indices = self.rng.integers(0, len(waveforms), size=len(waveforms))
            boot_sample = waveforms[indices]

            spca = SparsePCADecomposition(n_components=n_components, alpha=1.0)
            result = spca.fit_transform(boot_sample)

            # Handle sign flips
            if all_loadings:
                for i in range(n_components):
                    if np.dot(result.loadings[i], all_loadings[0][i]) < 0:
                        result.loadings[i] *= -1

            all_loadings.append(result.loadings)

        all_loadings = np.array(all_loadings)

        # Fit on full data
        spca = SparsePCADecomposition(n_components=n_components, alpha=1.0)
        result = spca.fit_transform(waveforms)

        for i in range(n_components):
            mean = result.loadings[i]

            for b in range(n_boot):
                if np.dot(all_loadings[b, i], mean) < 0:
                    all_loadings[b, i] *= -1

            alpha = 1 - self.ci_level
            ci_lo = np.percentile(all_loadings[:, i, :], 100 * alpha / 2, axis=0)
            ci_hi = np.percentile(all_loadings[:, i, :], 100 * (1 - alpha / 2), axis=0)

            components.append(
                ComponentTimecourse(
                    name=f"SPC{i + 1}",  # Sparse PC
                    mean=mean,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                )
            )

        return components

    def aggregate_ged(
        self,
        waveforms: NDArray[np.float64],
        time_vector: NDArray[np.float64],
        n_components: int = 3,
    ) -> list[ComponentTimecourse]:
        """Apply GED and return component patterns with CIs."""
        components = []

        # Bootstrap
        all_components = []
        for _ in range(self.n_bootstrap):
            indices = self.rng.integers(0, len(waveforms), size=len(waveforms))
            boot_sample = waveforms[indices]

            ged = GEDDecomposition(n_components=n_components)
            result = ged.fit_transform(boot_sample, time_vector)

            # Handle sign flips
            if all_components:
                for i in range(n_components):
                    if np.dot(result.components[i], all_components[0][i]) < 0:
                        result.components[i] *= -1

            all_components.append(result.components)

        all_components = np.array(all_components)

        # Fit on full data
        ged = GEDDecomposition(n_components=n_components)
        result = ged.fit_transform(waveforms, time_vector)

        for i in range(n_components):
            mean = result.components[i]

            for b in range(self.n_bootstrap):
                if np.dot(all_components[b, i], mean) < 0:
                    all_components[b, i] *= -1

            alpha = 1 - self.ci_level
            ci_lo = np.percentile(all_components[:, i, :], 100 * alpha / 2, axis=0)
            ci_hi = np.percentile(
                all_components[:, i, :], 100 * (1 - alpha / 2), axis=0
            )

            components.append(
                ComponentTimecourse(
                    name=f"GED{i + 1}", mean=mean, ci_lower=ci_lo, ci_upper=ci_hi
                )
            )

        return components

    def compute_decomposition(
        self,
        category: PreprocessingCategory,
        method: DecompositionMethod,
        n_components: int = 3,
        limit: int | None = None,
    ) -> DecompositionResult:
        """Compute full decomposition for a category-method combination.

        Parameters
        ----------
        category : str
            Preprocessing category (Ground Truth, Foundation Model, etc.)
        method : str
            Decomposition method (template, pca, rotated_pca, sparse_pca, ged)
        n_components : int
            Number of components to extract (ignored for template fitting)
        limit : int, optional
            Limit number of subjects (for testing)

        Returns
        -------
        DecompositionResult
        """
        # Load signals
        waveforms, time_vector = self.load_signals_by_category(category, limit)
        n_subjects = len(waveforms)

        # Compute mean waveform with CI
        mean_wave, ci_lo, ci_hi = self._bootstrap_mean_ci(waveforms, axis=0)

        # Apply decomposition method
        if method == "template":
            components = self.aggregate_template_fitting(waveforms, time_vector)
        elif method == "pca":
            components = self.aggregate_pca(waveforms, n_components)
        elif method == "rotated_pca":
            components = self.aggregate_rotated_pca(waveforms, n_components)
        elif method == "sparse_pca":
            components = self.aggregate_sparse_pca(waveforms, n_components)
        elif method == "ged":
            components = self.aggregate_ged(waveforms, time_vector, n_components)
        else:
            raise ValueError(f"Unknown decomposition method: {method}")

        return DecompositionResult(
            category=category,
            method=method,
            time_vector=time_vector,
            components=components,
            n_subjects=n_subjects,
            mean_waveform=mean_wave,
            mean_waveform_ci_lower=ci_lo,
            mean_waveform_ci_upper=ci_hi,
        )

    def compute_all_decompositions(
        self,
        categories: list[PreprocessingCategory] | None = None,
        methods: list[DecompositionMethod] | None = None,
        n_components: int = 3,
        limit: int | None = None,
    ) -> dict[tuple[PreprocessingCategory, DecompositionMethod], DecompositionResult]:
        """Compute all category × method combinations.

        Returns dict keyed by (category, method) tuples.
        """
        if categories is None:
            categories = get_preprocessing_categories()
        if methods is None:
            methods = ["template", "pca", "rotated_pca", "sparse_pca", "ged"]

        results = {}
        for category in categories:
            for method in methods:
                print(f"Computing {method} for {category}...")
                try:
                    results[(category, method)] = self.compute_decomposition(
                        category, method, n_components, limit
                    )
                except Exception as e:
                    print(f"  Error: {e}")

        return results
