"""
Guardrail Tests: No Hardcoded Values in Code

These tests scan Python and R files for banned patterns that indicate
hardcoded values instead of loading from YAML/DB.

TDD Approach: These tests will FAIL initially, then we fix the code.
"""

import re
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# BANNED PATTERNS
# =============================================================================

# Hardcoded AUROC values (e.g., auroc = 0.9110)
AUROC_PATTERN = re.compile(r"auroc\s*[=:]\s*0\.\d{3,}", re.IGNORECASE)

# Hardcoded method names that should come from registry
METHOD_NAME_PATTERNS = [
    re.compile(r'["\']pupil-gt["\']'),
    re.compile(r'["\']MOMENT-gt-finetune["\']'),
    re.compile(r'["\']MOMENT-gt-zeroshot["\']'),
    re.compile(r'["\']ensemble-LOF-MOMENT'),
    re.compile(r'["\']ensembleThresholded-'),
    re.compile(r'["\']UniTS-'),
    re.compile(r'["\']TimesNet-'),
    re.compile(r'["\']OneClassSVM["\']'),
    re.compile(r'["\']LOF["\']'),
    re.compile(r'["\']SAITS["\']'),
    re.compile(r'["\']CSDI["\']'),
]

# Raw hex colors (outside of designated color definition files)
HEX_COLOR_PATTERN = re.compile(r'["\']#[0-9A-Fa-f]{6}["\']')

# Files that ARE allowed to have these patterns
ALLOWED_FILES = {
    # Config files that DEFINE the values
    "colors.yaml",
    "plot_hyperparam_combos.yaml",
    "display_names.yaml",
    "figure_layouts.yaml",
    # Registry files
    "outlier_methods.yaml",
    "imputation_methods.yaml",
    "classifiers.yaml",
    # Test files (can have expected values)
    "test_",
    "conftest.py",
    # This test file itself
    "test_no_hardcoded_values.py",
    # Database extraction (needs to reference method names for mapping)
    "extract_all_configs_to_duckdb.py",
    # Data IO modules that define the registry
    "registry.py",
    "display_names.py",
}


def is_allowed_file(filepath: Path) -> bool:
    """Check if file is allowed to contain banned patterns."""
    name = filepath.name
    for allowed in ALLOWED_FILES:
        if allowed in name:
            return True
    return False


def is_in_comment(line: str, match_start: int) -> bool:
    """Check if match position is inside a comment."""
    # Python comment
    hash_pos = line.find("#")
    if hash_pos != -1 and hash_pos < match_start:
        return True
    # R comment (same as Python)
    return False


# =============================================================================
# TESTS
# =============================================================================


class TestNoHardcodedAUROC:
    """Test that AUROC values are not hardcoded in code."""

    def get_python_files(self):
        """Get all Python source files (excluding tests and configs)."""
        files = []
        for py_file in (PROJECT_ROOT / "src").rglob("*.py"):
            if not is_allowed_file(py_file):
                files.append(py_file)
        return files

    def test_no_hardcoded_auroc_in_python_src(self):
        """Scan src/ Python files for hardcoded AUROC values."""
        violations = []

        for py_file in self.get_python_files():
            content = py_file.read_text()
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                match = AUROC_PATTERN.search(line)
                if match and not is_in_comment(line, match.start()):
                    violations.append(
                        {
                            "file": str(py_file.relative_to(PROJECT_ROOT)),
                            "line": line_num,
                            "content": line.strip()[:80],
                        }
                    )

        if violations:
            msg = "GUARDRAIL VIOLATION: Hardcoded AUROC values found!\n\n"
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n    {v['content']}\n\n"
            msg += "FIX: Load AUROC from YAML config or database query."
            pytest.fail(msg)

    def test_no_hardcoded_auroc_in_scripts(self):
        """Scan scripts/ Python files for hardcoded AUROC values."""
        violations = []

        for py_file in (PROJECT_ROOT / "scripts").rglob("*.py"):
            if is_allowed_file(py_file):
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                match = AUROC_PATTERN.search(line)
                if match and not is_in_comment(line, match.start()):
                    # Allow in SQL queries (they compute AUROC, not hardcode)
                    if "SELECT" in line.upper() or "ORDER BY" in line.upper():
                        continue
                    violations.append(
                        {
                            "file": str(py_file.relative_to(PROJECT_ROOT)),
                            "line": line_num,
                            "content": line.strip()[:80],
                        }
                    )

        if violations:
            msg = "GUARDRAIL VIOLATION: Hardcoded AUROC values in scripts!\n\n"
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n    {v['content']}\n\n"
            pytest.fail(msg)


def is_allowed_context(line: str) -> bool:
    """Check if line is in an allowed context for method names.

    Allowed contexts:
    - Docstrings (triple quotes)
    - Examples (e.g., ...)
    - SQL queries (WHERE, IN, SELECT)
    - DataFrame filtering (df[df["col"] == "value"])
    - Display name conversion (.replace(...))
    - Type hints / function signatures
    - Default parameters (model_name: str = "...")
    """
    stripped = line.strip()

    # Docstrings and examples
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    if "e.g." in line.lower() or "example" in line.lower():
        return True

    # SQL queries
    sql_keywords = ["WHERE", "AND", "OR", "IN (", "SELECT", "FROM"]
    if any(kw in line.upper() for kw in sql_keywords):
        return True

    # DataFrame filtering (similar to SQL)
    if '== "' in line or "== '" in line:
        return True

    # Display name conversion (.replace() calls or their arguments)
    if ".replace(" in line:
        return True
    # Also allow if line is a quoted string argument (inside a replace call)
    if stripped.startswith('"') and stripped.endswith('",'):
        return True

    # Type hints or method identifier examples in docstrings
    if "Method identifier" in line or "method_name" in line:
        return True

    # Default parameters in function signatures
    if 'str = "' in line or "str = '" in line:
        return True

    return False


class TestNoHardcodedMethodNames:
    """Test that method names are loaded from registry, not hardcoded."""

    def test_no_hardcoded_method_names_in_viz(self):
        """Scan visualization code for hardcoded method names.

        This test is intentionally strict but allows certain contexts:
        - Docstrings and examples
        - SQL WHERE clauses (filtering, not defining behavior)
        - Display name conversion (.replace())

        NOTE: Configuration dicts that define WHICH methods to show
        should use config files, not hardcoded strings.
        """
        violations = []

        for py_file in (PROJECT_ROOT / "src" / "viz").rglob("*.py"):
            if is_allowed_file(py_file):
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            # Track if we're in a docstring
            in_docstring = False

            for line_num, line in enumerate(lines, 1):
                # Track docstring state
                if '"""' in line or "'''" in line:
                    quote_count = line.count('"""') + line.count("'''")
                    if quote_count == 1:
                        in_docstring = not in_docstring
                    # If opening and closing on same line, stay outside
                    elif quote_count >= 2:
                        pass  # Same line open/close

                if in_docstring:
                    continue

                # Skip allowed contexts
                if is_allowed_context(line):
                    continue

                for pattern in METHOD_NAME_PATTERNS:
                    match = pattern.search(line)
                    if match and not is_in_comment(line, match.start()):
                        violations.append(
                            {
                                "file": str(py_file.relative_to(PROJECT_ROOT)),
                                "line": line_num,
                                "pattern": pattern.pattern,
                                "content": line.strip()[:80],
                            }
                        )

        if violations:
            msg = (
                "GUARDRAIL VIOLATION: Hardcoded method names in visualization code!\n\n"
            )
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n"
                msg += f"    Pattern: {v['pattern']}\n"
                msg += f"    Line: {v['content']}\n\n"
            msg += "FIX: Load method names from configs/mlflow_registry/ or use display_names module."
            pytest.fail(msg)


class TestNoRawHexColors:
    """Test that colors reference color_definitions, not raw hex."""

    def test_no_raw_hex_in_combo_configs(self):
        """Check that shap_figure_combos uses color_ref, not raw hex."""
        import yaml

        yaml_path = (
            PROJECT_ROOT / "configs" / "VISUALIZATION" / "plot_hyperparam_combos.yaml"
        )
        if not yaml_path.exists():
            pytest.skip("plot_hyperparam_combos.yaml not found")

        config = yaml.safe_load(yaml_path.read_text())

        violations = []

        # Check shap_figure_combos
        shap_combos = config.get("shap_figure_combos", {}).get("configs", [])
        for combo in shap_combos:
            combo_id = combo.get("id", "unknown")
            if "color" in combo:
                color_val = combo["color"]
                if isinstance(color_val, str) and color_val.startswith("#"):
                    violations.append(
                        {
                            "section": "shap_figure_combos",
                            "combo_id": combo_id,
                            "color": color_val,
                        }
                    )

        if violations:
            msg = "GUARDRAIL VIOLATION: Raw hex colors in combo configs!\n\n"
            for v in violations:
                msg += f"  Section: {v['section']}, Combo: {v['combo_id']}\n"
                msg += f'    Has: color: "{v["color"]}"\n'
                msg += '    Should be: color_ref: "--color-xxx"\n\n'
            msg += (
                "FIX: Replace 'color' with 'color_ref' referencing color_definitions."
            )
            pytest.fail(msg)

    def test_no_raw_hex_in_python_viz(self):
        """Check visualization Python files don't have raw hex colors."""
        violations = []

        # Specific files to check (not all - some legitimately define colors)
        files_to_check = [
            PROJECT_ROOT / "src" / "viz" / "featurization_comparison.py",
            PROJECT_ROOT / "src" / "viz" / "foundation_model_dashboard.py",
        ]

        for py_file in files_to_check:
            if not py_file.exists():
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                match = HEX_COLOR_PATTERN.search(line)
                if match and not is_in_comment(line, match.start()):
                    # Allow if it's loading from COLORS dict
                    if "COLORS[" in line or "COLORS.get(" in line:
                        continue
                    violations.append(
                        {
                            "file": str(py_file.relative_to(PROJECT_ROOT)),
                            "line": line_num,
                            "content": line.strip()[:80],
                        }
                    )

        if violations:
            msg = "GUARDRAIL VIOLATION: Raw hex colors in visualization code!\n\n"
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n    {v['content']}\n\n"
            msg += "FIX: Use COLORS dict from plot_config.py or load from YAML."
            pytest.fail(msg)


class TestNoHardcodedInR:
    """Test that R scripts don't have hardcoded values."""

    def test_no_hardcoded_auroc_in_r(self):
        """Scan R figure scripts for hardcoded AUROC values."""
        violations = []

        r_dir = PROJECT_ROOT / "src" / "r" / "figures"
        if not r_dir.exists():
            pytest.skip("R figures directory not found")

        # Pattern for R: auroc = 0.xxx or auroc <- 0.xxx
        r_auroc_pattern = re.compile(r"auroc\s*[=<]-?\s*0\.\d{3,}", re.IGNORECASE)

        for r_file in r_dir.glob("*.R"):
            if "test_" in r_file.name:
                continue

            content = r_file.read_text()
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                match = r_auroc_pattern.search(line)
                if match:
                    violations.append(
                        {
                            "file": str(r_file.relative_to(PROJECT_ROOT)),
                            "line": line_num,
                            "content": line.strip()[:80],
                        }
                    )

        if violations:
            msg = "GUARDRAIL VIOLATION: Hardcoded AUROC values in R scripts!\n\n"
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n    {v['content']}\n\n"
            msg += "FIX: Load values from JSON file in outputs/r_data/."
            pytest.fail(msg)

    def test_no_hardcoded_numeric_vectors_in_r(self):
        """Check for hardcoded numeric vectors that should come from data."""
        violations = []

        r_dir = PROJECT_ROOT / "src" / "r" / "figures"
        if not r_dir.exists():
            pytest.skip("R figures directory not found")

        # Pattern for suspicious numeric vectors: c(0.xxx, 0.xxx, ...)
        numeric_vector_pattern = re.compile(r"c\(\s*0\.\d{2,}.*0\.\d{2,}.*\)")

        for r_file in r_dir.glob("*.R"):
            if "test_" in r_file.name:
                continue

            content = r_file.read_text()
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                match = numeric_vector_pattern.search(line)
                if match:
                    # Check if it looks like metric data
                    if any(
                        kw in line.lower()
                        for kw in ["auroc", "brier", "ci_lo", "ci_hi"]
                    ):
                        violations.append(
                            {
                                "file": str(r_file.relative_to(PROJECT_ROOT)),
                                "line": line_num,
                                "content": line.strip()[:100],
                            }
                        )

        if violations:
            msg = "GUARDRAIL VIOLATION: Hardcoded metric vectors in R scripts!\n\n"
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n    {v['content']}\n\n"
            msg += "FIX: Load data from JSON file, not hardcoded vectors."
            pytest.fail(msg)


class TestExplicitPathSelection:
    """Test that database paths are explicit, not fallback chains."""

    def test_no_fallback_path_chains(self):
        """Check that get_database_path doesn't have multiple fallbacks."""
        plot_config_path = PROJECT_ROOT / "src" / "viz" / "plot_config.py"

        if not plot_config_path.exists():
            pytest.skip("plot_config.py not found")

        content = plot_config_path.read_text()

        # Check for multiple path definitions in a list (indicates fallback)
        if content.count("Path(") > 5 and "possible_paths" in content:
            # Count how many paths are in the fallback list
            fallback_match = re.search(
                r"possible_paths\s*=\s*\[(.*?)\]", content, re.DOTALL
            )
            if fallback_match:
                fallback_content = fallback_match.group(1)
                path_count = fallback_content.count("Path(")
                if path_count > 2:
                    pytest.fail(
                        f"GUARDRAIL VIOLATION: {path_count}-path fallback chain in plot_config.py!\n\n"
                        "Having multiple fallback paths means different environments may load different databases.\n\n"
                        "FIX: Use single canonical path with explicit error if not found."
                    )
