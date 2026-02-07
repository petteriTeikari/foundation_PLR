"""
TDD Tests for PLR decomposition data loading.

These tests verify:
1. MLflow artifact access
2. Preprocessed signal extraction
3. Data differs between preprocessing configs
4. Group assignment matches raincloud figure

Run with: pytest tests/test_decomposition/test_data_loading.py -v
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.data

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestMLflowArtifactAccess:
    """Phase 0: Verify MLflow artifacts are accessible."""

    MLFLOW_ROOT = Path("/home/petteri/mlruns")
    IMPUTATION_EXPERIMENT = "940304421003085572"

    def test_mlflow_root_exists(self):
        """MLflow root directory exists."""
        assert self.MLFLOW_ROOT.exists(), f"MLflow root not found: {self.MLFLOW_ROOT}"

    def test_imputation_experiment_exists(self):
        """Imputation experiment directory exists."""
        exp_path = self.MLFLOW_ROOT / self.IMPUTATION_EXPERIMENT
        assert exp_path.exists(), f"Imputation experiment not found: {exp_path}"

    def test_can_find_pickle_artifacts(self):
        """At least one imputation pickle artifact exists."""
        exp_path = self.MLFLOW_ROOT / self.IMPUTATION_EXPERIMENT
        pickle_files = list(exp_path.glob("*/artifacts/imputation/*.pickle"))
        assert len(pickle_files) > 0, "No imputation pickle files found"

    def test_can_load_pickle_artifact(self):
        """Can load and parse a pickle artifact."""
        exp_path = self.MLFLOW_ROOT / self.IMPUTATION_EXPERIMENT
        pickle_files = list(exp_path.glob("*/artifacts/imputation/*.pickle"))
        assert len(pickle_files) > 0, "No pickle files to test"

        # Load first pickle
        with open(pickle_files[0], "rb") as f:
            data = pickle.load(f)

        # Structure: {'source_data': {...}, 'model_artifacts': {...}}
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert "source_data" in data, (
            f"Expected 'source_data' key, got keys: {data.keys()}"
        )
        assert "model_artifacts" in data, (
            f"Expected 'model_artifacts' key, got keys: {data.keys()}"
        )

        # Check model_artifacts structure
        model_artifacts = data["model_artifacts"]
        assert "imputation" in model_artifacts, (
            "Expected 'imputation' key in model_artifacts"
        )

        imputation = model_artifacts["imputation"]
        assert "test" in imputation or "train" in imputation, (
            f"Expected train/test splits, got: {imputation.keys()}"
        )

        # Check imputation data structure
        split_key = "test" if "test" in imputation else "train"
        split_data = imputation[split_key]
        assert "imputation_dict" in split_data, (
            "Expected 'imputation_dict' key in split"
        )
        assert "imputation" in split_data["imputation_dict"], (
            "Expected 'imputation' in imputation_dict"
        )
        assert "mean" in split_data["imputation_dict"]["imputation"], (
            "Expected 'mean' in imputation"
        )

        # Mean should be a 3D array (subjects × timepoints × 1)
        mean_signal = split_data["imputation_dict"]["imputation"]["mean"]
        assert isinstance(mean_signal, np.ndarray), (
            f"Expected ndarray, got {type(mean_signal)}"
        )
        assert mean_signal.ndim == 3, (
            f"Expected 3D array, got shape {mean_signal.shape}"
        )
        assert mean_signal.shape[2] == 1, (
            f"Expected last dim=1, got shape {mean_signal.shape}"
        )

    def test_pickle_contains_expected_subjects(self):
        """Pickle contains expected number of subjects (~507 across train+test)."""
        exp_path = self.MLFLOW_ROOT / self.IMPUTATION_EXPERIMENT
        pickle_files = list(exp_path.glob("*/artifacts/imputation/*.pickle"))

        with open(pickle_files[0], "rb") as f:
            data = pickle.load(f)

        imputation = data["model_artifacts"]["imputation"]
        n_subjects = 0
        for split in ["train", "test"]:
            if split in imputation:
                # mean shape: (subjects, timepoints, 1)
                mean_signal = imputation[split]["imputation_dict"]["imputation"]["mean"]
                n_subjects += mean_signal.shape[0]

        # Should have ~507 subjects (all with ground truth)
        assert n_subjects >= 200, f"Expected >=200 subjects, got {n_subjects}"
        assert n_subjects <= 600, f"Expected <=600 subjects, got {n_subjects}"

    def test_pickle_has_subject_codes(self):
        """Pickle contains subject codes in metadata."""
        exp_path = self.MLFLOW_ROOT / self.IMPUTATION_EXPERIMENT
        pickle_files = list(exp_path.glob("*/artifacts/imputation/*.pickle"))

        with open(pickle_files[0], "rb") as f:
            data = pickle.load(f)

        source_data = data["source_data"]
        assert "df" in source_data, (
            f"Expected 'df' key in source_data, got: {source_data.keys()}"
        )

        df = source_data["df"]
        split_key = "test" if "test" in df else "train"

        # Subject codes should be in metadata
        assert "metadata" in df[split_key], "Expected metadata in split"
        assert "subject_code" in df[split_key]["metadata"], (
            "Expected subject_code in metadata"
        )

        subject_codes = df[split_key]["metadata"]["subject_code"]
        # Should be a 2D array where each row is repeated subject code
        assert subject_codes.ndim == 2, (
            f"Expected 2D subject_code, got {subject_codes.ndim}D"
        )

        # Get unique subject codes (first column should be unique per row)
        unique_codes = np.unique(subject_codes[:, 0])
        assert len(unique_codes) > 50, (
            f"Expected >50 unique subjects, got {len(unique_codes)}"
        )


class TestEssentialMetricsConsistency:
    """Phase 1: Verify essential_metrics.csv matches expected structure."""

    @pytest.fixture
    def metrics_path(self):
        return PROJECT_ROOT / "data" / "r_data" / "essential_metrics.csv"

    def test_essential_metrics_exists(self, metrics_path):
        """essential_metrics.csv exists."""
        if not metrics_path.exists():
            pytest.skip(f"Not found: {metrics_path}. Run: make analyze")

    def test_has_required_columns(self, metrics_path):
        """Has outlier_method, imputation_method, classifier columns."""
        if not metrics_path.exists():
            pytest.skip(f"CSV not found: {metrics_path}. Run: make analyze")
        import pandas as pd

        df = pd.read_csv(metrics_path)
        required = ["outlier_method", "imputation_method", "classifier"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_catboost_filter_gives_expected_count(self, metrics_path):
        """Filtering to CatBoost gives ~45 configs per group."""
        if not metrics_path.exists():
            pytest.skip(f"CSV not found: {metrics_path}. Run: make analyze")
        import pandas as pd

        df = pd.read_csv(metrics_path)
        catboost_df = df[df["classifier"] == "CatBoost"]

        # Should have meaningful number of CatBoost configs
        assert len(catboost_df) >= 30, f"Too few CatBoost configs: {len(catboost_df)}"
        assert len(catboost_df) <= 100, f"Too many CatBoost configs: {len(catboost_df)}"


class TestCategoryMapping:
    """Phase 1: Verify category mapping matches raincloud figure."""

    @pytest.fixture
    def category_mapping_path(self):
        return PROJECT_ROOT / "configs" / "mlflow_registry" / "category_mapping.yaml"

    def test_category_mapping_exists(self, category_mapping_path):
        """category_mapping.yaml exists."""
        assert category_mapping_path.exists()

    def test_has_five_groups(self, category_mapping_path):
        """Mapping defines exactly 5 preprocessing groups."""
        with open(category_mapping_path) as f:
            config = yaml.safe_load(f)

        display_groups = config.get("outlier_category_display", {})
        expected_groups = {
            "Ground Truth",
            "Foundation Model",
            "Deep Learning",
            "Traditional",
            "Ensemble",
        }
        actual_groups = set(display_groups.keys())

        assert actual_groups == expected_groups, f"Groups mismatch: {actual_groups}"


class TestRunNameParsing:
    """Phase 1: Test parsing of MLflow run names to extract methods."""

    MLFLOW_ROOT = Path("/home/petteri/mlruns")
    IMPUTATION_EXPERIMENT = "940304421003085572"

    def parse_run_name(self, run_name: str) -> tuple[str, str]:
        """Parse imputation and outlier method from run name.

        Format: {IMPUTATION}_{params}__{OUTLIER}_impPLR_v0.1
        """
        if "__" not in run_name:
            return None, None

        parts = run_name.split("__")
        # Imputation: first part before underscore in first segment
        imp_method = parts[0].split("_")[0]

        # Outlier: second segment, remove _impPLR_v0.1 suffix and -Outlier
        outlier_part = parts[1].replace("_impPLR_v0.1", "").replace("-Outlier", "")

        # Handle cases like "MOMENT_finetune___gt" -> "MOMENT-gt-finetune"
        # and "pupil_gt" -> "pupil-gt"
        outlier_method = self._normalize_outlier_method(outlier_part)

        return imp_method, outlier_method

    def _normalize_outlier_method(self, raw: str) -> str:
        """Normalize outlier method name to canonical form."""
        # Handle pupil_gt -> pupil-gt
        raw = raw.replace("_gt", "-gt").replace("_orig", "-orig")

        # Handle MOMENT_finetune___gt -> MOMENT-gt-finetune
        if raw.startswith("MOMENT_"):
            mode = raw.split("_")[1]  # finetune or zeroshot
            if "___gt" in raw:
                return f"MOMENT-gt-{mode}"
            elif "___orig" in raw:
                return f"MOMENT-orig-{mode}"
            else:
                return f"MOMENT-{mode}"

        # Handle UniTS and TimesNet variants
        if "UniTS" in raw:
            return raw  # Already in correct format
        if "TimesNet" in raw:
            return raw

        return raw

    def test_can_parse_run_names(self):
        """Can parse at least some run names successfully."""
        exp_path = self.MLFLOW_ROOT / self.IMPUTATION_EXPERIMENT

        parsed_count = 0
        for run_dir in exp_path.iterdir():
            if not run_dir.is_dir():
                continue
            tags_file = run_dir / "tags" / "mlflow.runName"
            if not tags_file.exists():
                continue

            with open(tags_file) as f:
                run_name = f.read().strip()

            imp, outlier = self.parse_run_name(run_name)
            if imp and outlier:
                parsed_count += 1

        assert parsed_count > 50, f"Should parse >50 runs, got {parsed_count}"

    def test_parsed_imputation_methods_valid(self):
        """Parsed imputation methods are in expected set."""
        # Valid imputation methods/prefixes per registry
        valid_imputation_prefixes = {
            "SAITS",
            "CSDI",
            "MOMENT",
            "TimesNet",
            "linear",
            "pupil",
            "ensemble",
            "ensembleThresholded",  # Ensemble methods
        }

        exp_path = self.MLFLOW_ROOT / self.IMPUTATION_EXPERIMENT

        found_methods = set()
        for run_dir in exp_path.iterdir():
            if not run_dir.is_dir():
                continue
            tags_file = run_dir / "tags" / "mlflow.runName"
            if not tags_file.exists():
                continue

            with open(tags_file) as f:
                run_name = f.read().strip()

            imp, _ = self.parse_run_name(run_name)
            if imp:
                found_methods.add(imp)

        # Check that each found method starts with a valid prefix
        for method in found_methods:
            has_valid_prefix = any(
                method.startswith(p) for p in valid_imputation_prefixes
            )
            assert has_valid_prefix, f"Unknown imputation method: {method}"

    def test_parsed_outlier_methods_have_category(self):
        """Every valid parsed outlier method maps to a category."""
        exp_path = self.MLFLOW_ROOT / self.IMPUTATION_EXPERIMENT
        category_path = (
            PROJECT_ROOT / "configs" / "mlflow_registry" / "category_mapping.yaml"
        )

        with open(category_path) as f:
            category_config = yaml.safe_load(f)

        # Build category matcher
        import re

        exact_matches = category_config["outlier_method_categories"]["exact"]
        patterns = category_config["outlier_method_categories"]["patterns"]

        def get_category(method: str) -> str:
            if method in exact_matches:
                return exact_matches[method]
            for p in patterns:
                if re.search(p["pattern"], method):
                    return p["category"]
            return "Unknown"

        # Invalid/garbage methods per CLAUDE.md - these are placeholders to filter out
        GARBAGE_METHODS = {"exclude", "anomaly"}

        def is_garbage(method: str) -> bool:
            """Check if method is a garbage placeholder."""
            return any(g in method.lower() for g in GARBAGE_METHODS)

        # Check all parsed outlier methods
        found_methods = set()
        for run_dir in exp_path.iterdir():
            if not run_dir.is_dir():
                continue
            tags_file = run_dir / "tags" / "mlflow.runName"
            if not tags_file.exists():
                continue

            with open(tags_file) as f:
                run_name = f.read().strip()

            _, outlier = self.parse_run_name(run_name)
            if outlier and not is_garbage(outlier):
                found_methods.add(outlier)

        # No valid method should be "Unknown"
        unknown_methods = {m for m in found_methods if get_category(m) == "Unknown"}
        assert not unknown_methods, f"Methods with Unknown category: {unknown_methods}"


class TestPreprocessingGroupMapping:
    """Phase 1: Verify mapping to 5 preprocessing groups."""

    @pytest.fixture
    def category_mapping(self):
        category_path = (
            PROJECT_ROOT / "configs" / "mlflow_registry" / "category_mapping.yaml"
        )
        with open(category_path) as f:
            return yaml.safe_load(f)

    def test_five_groups_exist(self, category_mapping):
        """Exactly 5 preprocessing groups are defined."""
        groups = set(category_mapping["outlier_category_display"].keys())
        expected = {
            "Ground Truth",
            "Foundation Model",
            "Deep Learning",
            "Traditional",
            "Ensemble",
        }
        assert groups == expected

    def test_ground_truth_category_assignment(self, category_mapping):
        """pupil-gt maps to Ground Truth."""
        exact = category_mapping["outlier_method_categories"]["exact"]
        assert exact.get("pupil-gt") == "Ground Truth"

    def test_foundation_model_pattern_works(self, category_mapping):
        """MOMENT and UniTS methods map to Foundation Model."""
        import re

        patterns = category_mapping["outlier_method_categories"]["patterns"]

        fm_pattern = None
        for p in patterns:
            if p["category"] == "Foundation Model":
                fm_pattern = p["pattern"]
                break

        assert fm_pattern is not None
        assert re.search(fm_pattern, "MOMENT-gt-finetune")
        assert re.search(fm_pattern, "UniTS-gt-finetune")
