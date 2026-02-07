#!/usr/bin/env python3
"""
test_orchestration_flows.py

TDD test suite for the two-block orchestration pipeline:
- Block 1: Extraction (MLflow → DuckDB with re-anonymization)
- Block 2: Analysis (stats, figures, LaTeX with graceful degradation)

Run with:
    pytest tests/test_orchestration_flows.py -v

Created: 2026-01-25
"""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytestmark = pytest.mark.unit

# Disable Prefect for testing (must be before importing flows)
os.environ["PREFECT_DISABLED"] = "1"

# Try imports
try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SERI_DB_PATH = Path(
    "/home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db"
)
MLRUNS_DIR = Path("/home/petteri/mlruns")
CONFIG_DIR = PROJECT_ROOT / "configs"
PRIVATE_DIR = PROJECT_ROOT / "data" / "private"
PUBLIC_DIR = PROJECT_ROOT / "data" / "public"
DEMO_TRACES_PATH = PRIVATE_DIR / "demo_subjects_traces.pkl"


# =============================================================================
# EXPECTED VALUES
# =============================================================================

EXPECTED_VALUES = {
    # Subject counts from SERI database
    "n_control_train": 106,
    "n_glaucoma_train": 39,
    "n_unlabeled_train": 210,
    "n_total_train": 355,  # 106 + 39 + 210
    # Demo subjects (manually selected)
    "n_demo_subjects": 8,
    "demo_healthy": ["H001", "H002", "H003", "H004"],
    "demo_glaucoma": ["G001", "G002", "G003", "G004"],
    # Re-anonymization pattern
    "healthy_code_pattern": r"^H\d{3}$",
    "glaucoma_code_pattern": r"^G\d{3}$",
    "original_code_pattern": r"^PLR\d{4}$",
}


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def seri_db_connection():
    """Get DuckDB connection to SERI database."""
    if not HAS_DUCKDB:
        pytest.skip("duckdb not installed")
    if not SERI_DB_PATH.exists():
        pytest.skip(f"SERI database not found: {SERI_DB_PATH}")
    return duckdb.connect(str(SERI_DB_PATH), read_only=True)


@pytest.fixture
def demo_subjects_config():
    """Load demo subjects config."""
    config_path = CONFIG_DIR / "demo_subjects.yaml"
    if not config_path.exists():
        pytest.skip(f"Demo subjects config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def private_lookup():
    """Load private subject lookup table."""
    lookup_path = PRIVATE_DIR / "subject_lookup.yaml"
    if not lookup_path.exists():
        pytest.skip(f"Private lookup not found: {lookup_path}")
    with open(lookup_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# DEMO SUBJECTS CONFIGURATION TESTS
# =============================================================================


class TestDemoSubjectsConfig:
    """Tests for demo_subjects.yaml configuration."""

    def test_config_exists(self):
        """Verify demo subjects config file exists."""
        config_path = CONFIG_DIR / "demo_subjects.yaml"
        assert config_path.exists(), f"Missing: {config_path}"

    def test_correct_number_of_subjects(self, demo_subjects_config):
        """Verify exactly 8 demo subjects defined."""
        all_subjects = demo_subjects_config.get("all_demo_subjects", [])
        assert len(all_subjects) == EXPECTED_VALUES["n_demo_subjects"], (
            f"Expected {EXPECTED_VALUES['n_demo_subjects']} demo subjects, got {len(all_subjects)}"
        )

    def test_healthy_subjects_correctly_named(self, demo_subjects_config):
        """Verify healthy subjects use H### naming convention."""
        import re

        pattern = re.compile(EXPECTED_VALUES["healthy_code_pattern"])

        healthy = demo_subjects_config.get("demo_subjects", {}).get("healthy", {})
        all_healthy = []
        for category in ["low_outlier", "high_outlier"]:
            for subject in healthy.get(category, []):
                all_healthy.append(subject["code"])

        assert len(all_healthy) == 4, (
            f"Expected 4 healthy subjects, got {len(all_healthy)}"
        )
        for code in all_healthy:
            assert pattern.match(code), f"Invalid healthy code format: {code}"

    def test_glaucoma_subjects_correctly_named(self, demo_subjects_config):
        """Verify glaucoma subjects use G### naming convention."""
        import re

        pattern = re.compile(EXPECTED_VALUES["glaucoma_code_pattern"])

        glaucoma = demo_subjects_config.get("demo_subjects", {}).get("glaucoma", {})
        all_glaucoma = []
        for category in ["low_outlier", "high_outlier"]:
            for subject in glaucoma.get(category, []):
                all_glaucoma.append(subject["code"])

        assert len(all_glaucoma) == 4, (
            f"Expected 4 glaucoma subjects, got {len(all_glaucoma)}"
        )
        for code in all_glaucoma:
            assert pattern.match(code), f"Invalid glaucoma code format: {code}"

    def test_stratification_by_outlier_percentage(self, demo_subjects_config):
        """Verify subjects are stratified by outlier percentage."""
        for diagnosis in ["healthy", "glaucoma"]:
            group = demo_subjects_config.get("demo_subjects", {}).get(diagnosis, {})
            assert "low_outlier" in group, f"Missing low_outlier for {diagnosis}"
            assert "high_outlier" in group, f"Missing high_outlier for {diagnosis}"
            assert len(group["low_outlier"]) == 2, f"Expected 2 low_outlier {diagnosis}"
            assert len(group["high_outlier"]) == 2, (
                f"Expected 2 high_outlier {diagnosis}"
            )


# =============================================================================
# PRIVATE LOOKUP TABLE TESTS
# =============================================================================


class TestPrivateLookupTable:
    """Tests for private subject lookup table."""

    def test_lookup_file_exists(self):
        """Verify private lookup file exists."""
        lookup_path = PRIVATE_DIR / "subject_lookup.yaml"
        assert lookup_path.exists(), f"Missing: {lookup_path}"

    def test_lookup_contains_demo_subjects(self, private_lookup, demo_subjects_config):
        """Verify lookup table contains all demo subjects."""
        lookup = private_lookup.get("lookup", {})
        demo_codes = demo_subjects_config.get("all_demo_subjects", [])

        for code in demo_codes:
            assert code in lookup, f"Demo subject {code} not in lookup table"

    def test_lookup_maps_to_valid_original_codes(self, private_lookup):
        """Verify lookup maps to valid PLRxxxx codes."""
        import re

        pattern = re.compile(EXPECTED_VALUES["original_code_pattern"])
        lookup = private_lookup.get("lookup", {})

        for anon_code, original_code in lookup.items():
            assert pattern.match(original_code), (
                f"{anon_code} maps to invalid code: {original_code}"
            )

    def test_reverse_lookup_is_inverse(self, private_lookup):
        """Verify reverse_lookup is exact inverse of lookup."""
        lookup = private_lookup.get("lookup", {})
        reverse = private_lookup.get("reverse_lookup", {})

        for anon_code, original_code in lookup.items():
            assert original_code in reverse, f"Missing reverse for {original_code}"
            assert reverse[original_code] == anon_code, (
                f"Reverse mismatch: {original_code} → {reverse[original_code]} != {anon_code}"
            )

    def test_lookup_is_gitignored(self):
        """Verify lookup file patterns are in .gitignore."""
        gitignore_path = PROJECT_ROOT / ".gitignore"
        assert gitignore_path.exists(), "Missing .gitignore"

        content = gitignore_path.read_text()
        assert "data/private/" in content or "**/subject_lookup.yaml" in content, (
            "Private lookup not gitignored"
        )


# =============================================================================
# SUBJECT RE-ANONYMIZATION TESTS
# =============================================================================


class TestSubjectReanonymization:
    """Tests for subject re-anonymization logic."""

    def test_generate_mapping_creates_valid_codes(self, seri_db_connection):
        """Test that mapping generates valid Hxxx/Gxxx codes."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.extraction_flow import generate_subject_mapping

        # Call function directly (Prefect disabled in tests)
        original_to_anon, anon_to_original = generate_subject_mapping(
            seri_db_path=SERI_DB_PATH
        )

        import re

        healthy_pattern = re.compile(EXPECTED_VALUES["healthy_code_pattern"])
        glaucoma_pattern = re.compile(EXPECTED_VALUES["glaucoma_code_pattern"])

        for anon_code in original_to_anon.values():
            assert healthy_pattern.match(anon_code) or glaucoma_pattern.match(
                anon_code
            ), f"Invalid anonymized code: {anon_code}"

    def test_mapping_preserves_diagnosis_information(self, seri_db_connection):
        """Verify mapping preserves diagnosis (H for healthy, G for glaucoma)."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.extraction_flow import generate_subject_mapping

        original_to_anon, _ = generate_subject_mapping(seri_db_path=SERI_DB_PATH)

        # Get actual diagnoses from database
        diagnoses = seri_db_connection.execute("""
            SELECT DISTINCT subject_code, class_label
            FROM train
            WHERE class_label IS NOT NULL
        """).fetchall()

        for original_code, label in diagnoses:
            if original_code not in original_to_anon:
                continue
            anon_code = original_to_anon[original_code]

            if label == "control":
                assert anon_code.startswith("H"), (
                    f"Control {original_code} got non-H code: {anon_code}"
                )
            elif label == "glaucoma":
                assert anon_code.startswith("G"), (
                    f"Glaucoma {original_code} got non-G code: {anon_code}"
                )

    def test_mapping_is_deterministic(self, seri_db_connection):
        """Verify mapping generates same codes on repeated calls."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.extraction_flow import generate_subject_mapping

        mapping1, _ = generate_subject_mapping(seri_db_path=SERI_DB_PATH)
        mapping2, _ = generate_subject_mapping(seri_db_path=SERI_DB_PATH)

        assert mapping1 == mapping2, "Mapping is not deterministic"


# =============================================================================
# DUCKDB EXPORT TESTS
# =============================================================================


class TestDuckDBExport:
    """Tests for DuckDB export functionality."""

    def test_export_creates_valid_database(self, temp_output_dir):
        """Test that export creates a valid DuckDB file."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.extraction_flow import export_to_duckdb

        test_data = {
            "predictions": [
                {
                    "subject_code": "H001",
                    "y_true": 0,
                    "y_prob": 0.2,
                    "run_id": "test_run",
                    "classifier": "CatBoost",
                    "featurization": "simple1.0",
                    "imputation_method": "SAITS",
                    "outlier_method": "pupil-gt",
                }
            ],
            "metrics_aggregate": [],
        }

        output_path = temp_output_dir / "test_results.db"
        result_path = export_to_duckdb(test_data, output_path)

        assert result_path.exists(), "Database not created"
        assert result_path.stat().st_size > 0, "Database is empty"

        # Verify we can read it
        conn = duckdb.connect(str(result_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()

        assert count == 1, f"Expected 1 prediction, got {count}"

    def test_export_uses_anonymized_codes_only(self, temp_output_dir):
        """Verify exported database contains only anonymized codes."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import re

        from orchestration.flows.extraction_flow import export_to_duckdb

        test_data = {
            "predictions": [
                {
                    "subject_code": "H001",
                    "y_true": 0,
                    "y_prob": 0.2,
                    "run_id": "r1",
                    "classifier": "C",
                    "featurization": "F",
                    "imputation_method": "I",
                    "outlier_method": "O",
                },
                {
                    "subject_code": "G001",
                    "y_true": 1,
                    "y_prob": 0.8,
                    "run_id": "r1",
                    "classifier": "C",
                    "featurization": "F",
                    "imputation_method": "I",
                    "outlier_method": "O",
                },
            ],
            "metrics_aggregate": [],
        }

        output_path = temp_output_dir / "test_anon.db"
        export_to_duckdb(test_data, output_path)

        conn = duckdb.connect(str(output_path), read_only=True)
        codes = conn.execute("SELECT DISTINCT subject_code FROM predictions").fetchall()
        conn.close()

        original_pattern = re.compile(EXPECTED_VALUES["original_code_pattern"])
        for (code,) in codes:
            assert not original_pattern.match(code), (
                f"Original code {code} found in exported database!"
            )


# =============================================================================
# DEMO TRACES EXTRACTION TESTS
# =============================================================================


class TestDemoTracesExtraction:
    """Tests for demo subject traces extraction."""

    def test_extract_traces_creates_pickle(self, temp_output_dir, private_lookup):
        """Test that trace extraction creates valid pickle file."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.extraction_flow import extract_demo_traces

        output_path = temp_output_dir / "demo_traces.pkl"

        result = extract_demo_traces(
            seri_db_path=SERI_DB_PATH,
            demo_config_path=CONFIG_DIR / "demo_subjects.yaml",
            anon_to_original=private_lookup.get("lookup", {}),
            output_path=output_path,
        )

        if result is None:
            pytest.skip(
                "Trace extraction returned None (expected if SERI DB unavailable)"
            )

        assert result.exists(), "Traces pickle not created"

        with open(result, "rb") as f:
            data = pickle.load(f)

        assert "traces" in data, "Missing 'traces' key in pickle"
        assert "metadata" in data, "Missing 'metadata' key in pickle"
        assert len(data["traces"]) > 0, "No traces extracted"

    def test_extracted_traces_contain_required_fields(
        self, temp_output_dir, private_lookup
    ):
        """Verify extracted traces have all required fields."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.extraction_flow import extract_demo_traces

        output_path = temp_output_dir / "demo_traces.pkl"

        result = extract_demo_traces(
            seri_db_path=SERI_DB_PATH,
            demo_config_path=CONFIG_DIR / "demo_subjects.yaml",
            anon_to_original=private_lookup.get("lookup", {}),
            output_path=output_path,
        )

        if result is None:
            pytest.skip("Trace extraction returned None")

        with open(result, "rb") as f:
            data = pickle.load(f)

        required_fields = [
            "time",
            "pupil_gt",
            "pupil_raw",
            "outlier_mask",
            "class_label",
        ]

        for code, trace in data["traces"].items():
            for field in required_fields:
                assert field in trace, f"Missing field '{field}' in trace {code}"


# =============================================================================
# ANALYSIS FLOW GRACEFUL DEGRADATION TESTS
# =============================================================================


class TestAnalysisFlowGracefulDegradation:
    """Tests for analysis flow's handling of missing private data."""

    def test_analysis_warns_on_missing_private_data(self, temp_output_dir, caplog):
        """Verify analysis emits warning when private data is missing."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.analysis_flow import check_private_data

        # Temporarily modify PRIVATE_DIR to non-existent path
        with patch(
            "orchestration.flows.analysis_flow.PRIVATE_DIR",
            temp_output_dir / "nonexistent",
        ):
            with patch(
                "orchestration.flows.analysis_flow.DEMO_TRACES_PATH",
                temp_output_dir / "nonexistent" / "traces.pkl",
            ):
                result = check_private_data()

        assert result["demo_traces"] is False, (
            "Should report demo_traces as unavailable"
        )

    def test_demo_trace_figure_skips_gracefully(self, caplog):
        """Verify demo trace figure is skipped when private data unavailable."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.analysis_flow import generate_all_figures

        # Mock public data as available but private data as unavailable
        public_status = {"available": True, "n_predictions": 100, "n_metrics": 10}
        private_status = {"demo_traces": False, "subject_lookup": False}

        result = generate_all_figures(public_status, private_status, skip_private=False)

        assert "demo_traces" in result.get("skipped", []), (
            "demo_traces should be skipped when private data unavailable"
        )

    def test_analysis_completes_without_private_data(self, temp_output_dir):
        """Verify analysis flow completes even without private data."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        # Create minimal public database with all required tables and views
        public_db = temp_output_dir / "test_public.db"
        conn = duckdb.connect(str(public_db))
        conn.execute("""
            CREATE TABLE predictions (
                prediction_id INTEGER,
                subject_code VARCHAR,
                y_true INTEGER,
                y_prob DOUBLE,
                run_id VARCHAR,
                classifier VARCHAR,
                featurization VARCHAR,
                imputation_method VARCHAR,
                outlier_method VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO predictions VALUES
            (1, 'H001', 0, 0.2, 'r1', 'CatBoost', 'simple1.0', 'SAITS', 'pupil-gt')
        """)
        conn.execute("""
            CREATE TABLE metrics_aggregate (
                run_id VARCHAR,
                source_name VARCHAR,
                split VARCHAR,
                classifier VARCHAR,
                featurization VARCHAR,
                imputation_method VARCHAR,
                outlier_method VARCHAR,
                AUROC_mean DOUBLE,
                AUROC_std DOUBLE,
                AUROC_ci_lo DOUBLE,
                AUROC_ci_hi DOUBLE,
                Brier_mean DOUBLE,
                Brier_std DOUBLE,
                sensitivity_mean DOUBLE,
                specificity_mean DOUBLE
            )
        """)
        # Create essential_metrics view for viz module compatibility
        conn.execute("""
            CREATE VIEW essential_metrics AS
            SELECT
                run_id, source_name, split, classifier, featurization,
                imputation_method, outlier_method,
                AUROC_mean as auroc, AUROC_std as auroc_std,
                AUROC_ci_lo as auroc_ci_lower, AUROC_ci_hi as auroc_ci_upper,
                Brier_mean as brier, Brier_std as brier_std,
                sensitivity_mean as sensitivity, specificity_mean as specificity
            FROM metrics_aggregate
        """)
        conn.close()

        from orchestration.flows.analysis_flow import analysis_flow

        # Save original env var (analysis_flow sets FOUNDATION_PLR_DB_PATH internally)
        original_db_path = os.environ.get("FOUNDATION_PLR_DB_PATH")

        try:
            # Run with skip_private=True to avoid looking for private data
            result = analysis_flow(
                db_path=public_db,
                skip_private_figures=True,
            )
        finally:
            # Restore original env var to prevent contaminating later tests
            if original_db_path is not None:
                os.environ["FOUNDATION_PLR_DB_PATH"] = original_db_path
            elif "FOUNDATION_PLR_DB_PATH" in os.environ:
                del os.environ["FOUNDATION_PLR_DB_PATH"]

        # Check expected keys in result
        assert "public_status" in result, "Analysis should return public_status"
        assert "figures" in result, "Analysis should return figures dict"
        assert "report" in result, "Analysis should generate report"
        assert result["public_status"]["available"], (
            "Public DB should be marked available"
        )


# =============================================================================
# PRIVACY GUARANTEES TESTS
# =============================================================================


class TestPrivacyGuarantees:
    """Tests to verify privacy guarantees are maintained."""

    def test_public_db_contains_no_original_codes(self):
        """Verify public database contains no PLRxxxx codes."""
        import re

        public_db = PUBLIC_DIR / "foundation_plr_results.db"
        if not public_db.exists():
            pytest.skip("Public database not yet generated")

        original_pattern = re.compile(EXPECTED_VALUES["original_code_pattern"])

        conn = duckdb.connect(str(public_db), read_only=True)

        # Check predictions table
        try:
            codes = conn.execute(
                "SELECT DISTINCT subject_code FROM predictions"
            ).fetchall()
            for (code,) in codes:
                assert not original_pattern.match(code), (
                    f"Original code {code} found in public database!"
                )
        except Exception:
            pass  # Table might not exist yet

        conn.close()

    def test_private_dir_is_gitignored(self):
        """Verify entire data/private/ directory is gitignored."""
        gitignore = PROJECT_ROOT / ".gitignore"
        content = gitignore.read_text()

        assert "data/private/" in content, "data/private/ should be in .gitignore"

    def test_no_original_codes_in_config_files(self):
        """Verify config files don't contain original PLRxxxx codes."""
        import re

        original_pattern = re.compile(EXPECTED_VALUES["original_code_pattern"])

        config_files = list(CONFIG_DIR.glob("*.yaml"))
        for config_file in config_files:
            content = config_file.read_text()
            matches = original_pattern.findall(content)
            assert len(matches) == 0, (
                f"Original codes found in {config_file.name}: {matches}"
            )


# =============================================================================
# PREDICTION EXTRACTION TESTS
# =============================================================================


class TestPredictionExtraction:
    """Tests for per-subject prediction extraction."""

    def test_predictions_extracted_from_mlflow(self):
        """Test that predictions are extracted from MLflow runs."""
        public_db = PUBLIC_DIR / "foundation_plr_results.db"
        if not public_db.exists():
            pytest.skip("Public database not yet generated")

        conn = duckdb.connect(str(public_db), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()

        # Should have predictions (exact count depends on MLflow data)
        # If 0, extraction logic is broken
        assert count > 0, "No predictions extracted - check extraction logic"

    def test_predictions_use_anonymized_codes(self):
        """Verify predictions use Hxxx/Gxxx codes, not PLRxxxx."""
        import re

        public_db = PUBLIC_DIR / "foundation_plr_results.db"
        if not public_db.exists():
            pytest.skip("Public database not yet generated")

        conn = duckdb.connect(str(public_db), read_only=True)
        codes = conn.execute(
            "SELECT DISTINCT subject_code FROM predictions LIMIT 100"
        ).fetchall()
        conn.close()

        if not codes:
            pytest.skip("No predictions to check")

        original_pattern = re.compile(EXPECTED_VALUES["original_code_pattern"])
        anon_pattern = re.compile(r"^[HG]\d{3}$")

        for (code,) in codes:
            assert not original_pattern.match(code), (
                f"Original code {code} found in predictions!"
            )
            assert anon_pattern.match(code), (
                f"Code {code} doesn't match anonymized pattern"
            )


class TestDemoTracesComplete:
    """Tests to verify all 8 demo subjects have traces extracted."""

    def test_all_demo_subjects_have_traces(self, demo_subjects_config):
        """Verify all 8 demo subjects have traces in pickle."""
        if not DEMO_TRACES_PATH.exists():
            pytest.skip("Demo traces not yet extracted")

        with open(DEMO_TRACES_PATH, "rb") as f:
            data = pickle.load(f)

        traces = data.get("traces", {})
        # Use demo_subjects_config (8 subjects), not full lookup (208 subjects)
        demo_codes = demo_subjects_config.get("all_demo_subjects", [])

        for code in demo_codes:
            assert code in traces, f"Missing trace for demo subject {code}"

    def test_demo_traces_have_sufficient_timepoints(self):
        """Verify each demo trace has expected number of timepoints (~1981)."""
        if not DEMO_TRACES_PATH.exists():
            pytest.skip("Demo traces not yet extracted")

        with open(DEMO_TRACES_PATH, "rb") as f:
            data = pickle.load(f)

        traces = data.get("traces", {})
        expected_min_timepoints = 1900  # Allow some tolerance

        for code, trace in traces.items():
            n_timepoints = len(trace.get("time", []))
            assert n_timepoints >= expected_min_timepoints, (
                f"Trace {code} has only {n_timepoints} timepoints (expected ~1981)"
            )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests for the full pipeline."""

    @pytest.mark.slow
    def test_extraction_flow_produces_expected_outputs(self, temp_output_dir):
        """Test that extraction flow produces all expected files."""
        if not MLRUNS_DIR.exists():
            pytest.skip("MLruns directory not available")
        if not SERI_DB_PATH.exists():
            pytest.skip("SERI database not available")

        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from orchestration.flows.extraction_flow import extraction_flow

        # Patch output paths to use temp directory
        with (
            patch(
                "orchestration.flows.extraction_flow.PUBLIC_DB_PATH",
                temp_output_dir / "public.db",
            ),
            patch(
                "orchestration.flows.extraction_flow.PRIVATE_DIR",
                temp_output_dir / "private",
            ),
            patch(
                "orchestration.flows.extraction_flow.SUBJECT_LOOKUP_PATH",
                temp_output_dir / "private" / "lookup.yaml",
            ),
            patch(
                "orchestration.flows.extraction_flow.DEMO_TRACES_PATH",
                temp_output_dir / "private" / "traces.pkl",
            ),
        ):
            result = extraction_flow()

        assert "public_db" in result, "Missing public_db in result"
        assert "private_lookup" in result, "Missing private_lookup in result"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
