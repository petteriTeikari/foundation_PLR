"""
Guardrail Tests: Single Data Location Policy

Enforces the canonical data directory structure:
    data/
    ├── public/      # Shareable data (anonymized predictions, metrics)
    ├── private/     # GITIGNORED (subject lookup, PII)
    └── r_data/      # JSON exports for R visualization

BANNED locations for data files:
    - outputs/       # Legacy location (migrate to data/)
    - Root directory # No loose data files
    - manuscripts/   # Keep manuscript repo separate

TDD Approach: These tests WILL FAIL initially. We fix by migrating data.
"""

from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# CANONICAL DATA PATHS
# =============================================================================


def get_canonical_data_dir():
    """Get the canonical data directory."""
    return PROJECT_ROOT / "data"


def get_public_data_dir():
    """Get public data directory (shareable, anonymized)."""
    return get_canonical_data_dir() / "public"


def get_private_data_dir():
    """Get private data directory (gitignored, PII)."""
    return get_canonical_data_dir() / "private"


def get_r_data_dir():
    """Get R data directory (JSON exports for ggplot2)."""
    return get_canonical_data_dir() / "r_data"


# =============================================================================
# DATA FILE PATTERNS
# =============================================================================

# Extensions that indicate data files
DATA_EXTENSIONS = {
    ".db",
    ".duckdb",  # Databases
    ".csv",
    ".tsv",  # Tabular data
    ".json",  # JSON exports
    ".pkl",
    ".pickle",  # Pickled data
    ".parquet",  # Parquet files
    ".yaml",
    ".yml",  # Data manifests (when in data dirs)
}

# Specific filenames that are data files regardless of extension
DATA_FILENAMES = {
    "essential_metrics.csv",
    "foundation_plr_results.db",
    "foundation_plr_results_stratos.db",
    "cd_diagram_data.duckdb",
    "cd_preprocessing_catboost.duckdb",
    "roc_rc_data.json",
    "calibration_data.json",
    "dca_data.json",
    "predictions_top4.json",
    "selective_classification_data.json",
    "shap_feature_importance.json",
    "shap_per_sample.json",
    "vif_analysis.json",
    "featurization_comparison.json",
    "catboost_metrics.json",
    "top10_configs.json",
}

# Files in outputs/ that are NOT data (allowed to stay)
NON_DATA_FILES = {
    "execution.log",
    "EXECUTION_CHECKPOINT.json",  # Checkpoint, not data
}

# Files that are figures, not data (allowed in outputs/)
FIGURE_EXTENSIONS = {".png", ".pdf", ".svg", ".eps", ".tiff"}


# =============================================================================
# BANNED LOCATIONS
# =============================================================================

BANNED_DATA_LOCATIONS = [
    # Legacy outputs directory - migrate to data/
    ("outputs", "data/{public,r_data}/"),
    # Root directory - no loose data files
    (".", "data/{public,private,r_data}/"),
]


def is_data_file(filepath: Path) -> bool:
    """Check if a file is a data file based on extension or name."""
    if filepath.name in NON_DATA_FILES:
        return False
    if filepath.suffix.lower() in FIGURE_EXTENSIONS:
        return False
    if filepath.suffix.lower() in DATA_EXTENSIONS:
        return True
    if filepath.name in DATA_FILENAMES:
        return True
    return False


def find_data_files_in_dir(directory: Path, recursive: bool = False) -> list:
    """Find all data files in a directory."""
    if not directory.exists():
        return []

    data_files = []
    pattern = "**/*" if recursive else "*"

    for filepath in directory.glob(pattern):
        if filepath.is_file() and is_data_file(filepath):
            data_files.append(filepath)

    return data_files


# =============================================================================
# TESTS
# =============================================================================


class TestCanonicalDataStructure:
    """Test that canonical data directory structure exists."""

    def test_data_dir_exists(self):
        """Canonical data/ directory must exist."""
        data_dir = get_canonical_data_dir()
        assert data_dir.exists(), (
            f"MISSING: Canonical data directory not found at {data_dir}\n"
            "FIX: Create the data/ directory structure"
        )

    def test_public_data_dir_exists(self):
        """data/public/ directory must exist."""
        public_dir = get_public_data_dir()
        assert public_dir.exists(), (
            f"MISSING: Public data directory not found at {public_dir}\n"
            "FIX: mkdir -p data/public"
        )

    def test_private_data_dir_exists(self):
        """data/private/ directory must exist."""
        private_dir = get_private_data_dir()
        assert private_dir.exists(), (
            f"MISSING: Private data directory not found at {private_dir}\n"
            "FIX: mkdir -p data/private"
        )

    def test_r_data_dir_exists(self):
        """data/r_data/ directory must exist."""
        r_data_dir = get_r_data_dir()
        # r_data can be in either location during transition
        outputs_r_data = PROJECT_ROOT / "outputs" / "r_data"
        assert r_data_dir.exists() or outputs_r_data.exists(), (
            f"MISSING: R data directory not found at {r_data_dir}\n"
            "FIX: mkdir -p data/r_data"
        )


class TestNoDataInOutputs:
    """Test that data files are migrated from outputs/ to data/."""

    def test_no_duckdb_in_outputs(self):
        """DuckDB files should be in data/public/, not outputs/."""
        outputs_dir = PROJECT_ROOT / "outputs"
        if not outputs_dir.exists():
            pytest.skip("outputs/ directory does not exist (good!)")

        duckdb_files = list(outputs_dir.glob("*.duckdb")) + list(
            outputs_dir.glob("*.db")
        )

        if duckdb_files:
            file_list = "\n  - ".join(str(f.name) for f in duckdb_files)
            pytest.fail(
                f"DATA LOCATION VIOLATION: DuckDB files in outputs/!\n\n"
                f"Found:\n  - {file_list}\n\n"
                f"FIX: Move to data/public/:\n"
                f"  mv outputs/*.db outputs/*.duckdb data/public/"
            )

    def test_no_csv_in_outputs(self):
        """CSV files should be in data/, not outputs/."""
        outputs_dir = PROJECT_ROOT / "outputs"
        if not outputs_dir.exists():
            pytest.skip("outputs/ directory does not exist (good!)")

        csv_files = list(outputs_dir.glob("*.csv"))

        if csv_files:
            file_list = "\n  - ".join(str(f.name) for f in csv_files)
            pytest.fail(
                f"DATA LOCATION VIOLATION: CSV files in outputs/!\n\n"
                f"Found:\n  - {file_list}\n\n"
                f"FIX: Move to data/public/ or data/r_data/:\n"
                f"  mv outputs/*.csv data/public/"
            )

    def test_no_json_data_in_outputs_root(self):
        """JSON data files should be in data/r_data/, not outputs/."""
        outputs_dir = PROJECT_ROOT / "outputs"
        if not outputs_dir.exists():
            pytest.skip("outputs/ directory does not exist (good!)")

        # Check root of outputs/ (not r_data subdirectory)
        json_files = [
            f for f in outputs_dir.glob("*.json") if f.name not in NON_DATA_FILES
        ]

        if json_files:
            file_list = "\n  - ".join(str(f.name) for f in json_files)
            pytest.fail(
                f"DATA LOCATION VIOLATION: JSON files in outputs/ root!\n\n"
                f"Found:\n  - {file_list}\n\n"
                f"FIX: Move to data/r_data/ or keep only execution checkpoint"
            )


class TestNoDataInOutputsRData:
    """Test that outputs/r_data/ is migrated to data/r_data/."""

    def test_r_data_in_canonical_location(self):
        """R data files should be in data/r_data/, not outputs/r_data/."""
        outputs_r_data = PROJECT_ROOT / "outputs" / "r_data"

        if not outputs_r_data.exists():
            # Good - no legacy location
            return

        legacy_files = list(outputs_r_data.glob("*.json")) + list(
            outputs_r_data.glob("*.csv")
        )

        if legacy_files:
            file_list = "\n  - ".join(str(f.name) for f in legacy_files[:10])
            if len(legacy_files) > 10:
                file_list += f"\n  - ... and {len(legacy_files) - 10} more"

            pytest.fail(
                f"DATA LOCATION VIOLATION: R data in legacy location!\n\n"
                f"Found {len(legacy_files)} files in outputs/r_data/:\n  - {file_list}\n\n"
                f"FIX: Move to canonical location:\n"
                f"  mv outputs/r_data/* data/r_data/\n"
                f"  rmdir outputs/r_data"
            )


class TestNoDataInRoot:
    """Test that no data files are in project root."""

    def test_no_duckdb_in_root(self):
        """No DuckDB files should be in project root."""
        root_duckdb = list(PROJECT_ROOT.glob("*.duckdb")) + list(
            PROJECT_ROOT.glob("*.db")
        )

        # Filter out test databases that might be temporary
        root_duckdb = [f for f in root_duckdb if not f.name.startswith("test_")]

        if root_duckdb:
            file_list = "\n  - ".join(str(f.name) for f in root_duckdb)
            pytest.fail(
                f"DATA LOCATION VIOLATION: Database files in project root!\n\n"
                f"Found:\n  - {file_list}\n\n"
                f"FIX: Move to data/public/"
            )

    def test_no_csv_in_root(self):
        """No CSV files should be in project root."""
        root_csv = list(PROJECT_ROOT.glob("*.csv"))

        if root_csv:
            file_list = "\n  - ".join(str(f.name) for f in root_csv)
            pytest.fail(
                f"DATA LOCATION VIOLATION: CSV files in project root!\n\n"
                f"Found:\n  - {file_list}\n\n"
                f"FIX: Move to data/public/ or data/r_data/"
            )


class TestDataPathUsage:
    """Test that code uses canonical data paths via os.path.join()."""

    def test_python_uses_os_path_join(self):
        """Python extraction scripts should use os.path.join(), not '/'."""
        scripts_dir = PROJECT_ROOT / "scripts"
        if not scripts_dir.exists():
            pytest.skip("scripts/ directory not found")

        violations = []

        # Pattern for hardcoded Unix paths to data directories
        import re

        hardcoded_path_pattern = re.compile(
            r'["\'](?:outputs|data)/(?:r_data|public|private)/[^"\']+["\']'
        )

        for py_file in scripts_dir.glob("*.py"):
            content = py_file.read_text()

            for line_num, line in enumerate(content.split("\n"), 1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue

                match = hardcoded_path_pattern.search(line)
                if match:
                    # Check if it's inside os.path.join() - that's OK
                    if "os.path.join" not in line and "Path(" not in line:
                        violations.append(
                            {
                                "file": py_file.name,
                                "line": line_num,
                                "content": line.strip()[:80],
                            }
                        )

        if violations:
            msg = "PATH STYLE VIOLATION: Hardcoded path separators found!\n\n"
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n    {v['content']}\n\n"
            msg += (
                "FIX: Use os.path.join() or pathlib.Path for cross-platform compatibility:\n"
                "  os.path.join(PROJECT_ROOT, 'data', 'r_data', 'file.json')\n"
                "  Path(PROJECT_ROOT) / 'data' / 'r_data' / 'file.json'"
            )
            pytest.fail(msg)


class TestGitignorePrivateData:
    """Test that private data is properly gitignored."""

    def test_private_dir_gitignored(self):
        """data/private/ must be in .gitignore."""
        gitignore_path = PROJECT_ROOT / ".gitignore"

        if not gitignore_path.exists():
            pytest.fail("MISSING: .gitignore file not found!")

        content = gitignore_path.read_text()

        private_patterns = [
            "data/private/",
            "data/private/*",
            "/data/private/",
        ]

        has_private_ignore = any(p in content for p in private_patterns)

        if not has_private_ignore:
            pytest.fail(
                "GITIGNORE VIOLATION: data/private/ not in .gitignore!\n\n"
                "This directory contains PII (subject lookup tables).\n\n"
                "FIX: Add to .gitignore:\n"
                "  data/private/"
            )


class TestDataManifest:
    """Test that data directories have manifest files."""

    def test_r_data_has_manifest(self):
        """data/r_data/ should have a DATA_MANIFEST.yaml."""
        r_data_dir = get_r_data_dir()
        manifest_path = r_data_dir / "DATA_MANIFEST.yaml"

        if not r_data_dir.exists():
            pytest.skip("data/r_data/ not yet created")

        assert manifest_path.exists(), (
            f"MISSING: Data manifest not found at {manifest_path}\n\n"
            "A DATA_MANIFEST.yaml documents what each file contains.\n\n"
            "FIX: Create DATA_MANIFEST.yaml with file descriptions"
        )

    def test_public_data_has_manifest(self):
        """data/public/ should have a DATA_MANIFEST.yaml."""
        public_dir = get_public_data_dir()
        manifest_path = public_dir / "DATA_MANIFEST.yaml"

        if not public_dir.exists():
            pytest.skip("data/public/ not yet created")

        assert manifest_path.exists(), (
            f"MISSING: Data manifest not found at {manifest_path}\n\n"
            "FIX: Create DATA_MANIFEST.yaml with file descriptions"
        )
