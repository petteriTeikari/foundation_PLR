# Test Coverage Improvement Plan: Artifact Consistency

> **Goal**: Catch discrepancies early between data generation, visualization scripts, and LaTeX artifacts.

## ⚠️ CRITICAL ISSUES FROM EXPERT PANEL REVIEW (2026-01-21)

Before implementing tests, these critical statistical issues must be understood:

### CRITICAL #1: Friedman Test Independence Violation

The current Friedman-Nemenyi test implementation uses n=1000 bootstrap iterations as "datasets", but bootstrap samples are NOT independent (share ~63% overlap). This produces:
- Artificially small p-values (1.48e-171 is suspicious)
- Artificially small critical difference (CD=0.415)
- False confidence in "statistical significance"

**Tests must verify**: Independence of observations OR add appropriate caveats.

### CRITICAL #2: Sample Size Confusion

Different analyses use different n values:
- n=24 (parallel coordinates - requires complete data)
- n=40 (scatter F1 - pairwise complete for F1)
- n=45 (scatter MAE - pairwise complete for MAE)
- n=79 (all preprocessing configs)
- n=1000 (bootstrap iterations - NOT independent!)

**Tests must verify**: Exclusion criteria are documented and consistent.

---

## Executive Summary

During manuscript preparation, reviewer agents identified several discrepancies:
1. Correlation values differed between scripts (0.407 vs 0.421)
2. Subject counts were inconsistent (208 labeled vs 507 total)
3. LaTeX command naming was inconsistent

These issues were caught during review, but should have been caught by automated tests. This plan outlines how to improve test coverage to prevent such discrepancies.

---

## 1. Expert Panel Reviews

### 1.1 Academic Principal Software Engineer (100+ Nature/Science manuscripts)

**Key Concerns:**
- **Reproducibility**: Every number in the manuscript must trace back to source data
- **Single Source of Truth**: Statistics should be computed once and referenced everywhere
- **Version Control**: Data files and derived artifacts must be version-controlled together
- **Audit Trail**: Every change to numbers must be documented

**Recommendations:**
1. Implement a "numbers registry" that all scripts read from
2. Never hardcode statistics - always compute from data
3. Add checksum validation for data files
4. Create a manifest file listing all data→artifact dependencies

### 1.2 Biostatistician Expert

**Key Concerns:**
- **Statistical Validity**: Correlation values must include sample size and p-values
- **Degrees of Freedom**: Different subsets (n=24 vs n=40) yield different correlations
- **Missing Data**: Subject counts must clearly indicate complete vs partial data
- **Confidence Intervals**: Point estimates alone are insufficient

**Recommendations:**
1. All statistics must include: point estimate, CI, sample size, p-value
2. Create validation tests for statistical consistency
3. Document data exclusion criteria (why n=24 in one place, n=40 in another)
4. Add statistical power analysis to detect underpowered comparisons

### 1.3 AI/Applied Math Expert (Foundation Models for Time Series)

**Key Concerns:**
- **Model Reproducibility**: Same seed, same results
- **Pipeline Consistency**: Feature extraction must be deterministic
- **Benchmark Fairness**: All methods must use same data splits
- **Computational Artifacts**: Numerical precision issues

**Recommendations:**
1. Pin random seeds throughout the pipeline
2. Validate that bootstrap iterations are reproducible
3. Add tests for numerical stability (float32 vs float64 differences)
4. Create regression tests for model outputs

---

## 2. Test Architecture

### 2.1 Unit Tests (Current Coverage: ~40%)

```
tests/
├── unit/
│   ├── test_outlier_detection.py    # Individual outlier methods
│   ├── test_imputation.py           # Individual imputation methods
│   ├── test_featurization.py        # Feature extraction
│   └── test_classification.py       # Classification metrics
```

**Target**: 80% line coverage for core modules

### 2.2 Integration Tests (Current Coverage: ~10%)

```
tests/
├── integration/
│   ├── test_pipeline_consistency.py      # End-to-end pipeline tests
│   ├── test_artifact_generation.py       # Data → JSON/CSV consistency
│   ├── test_statistics_consistency.py    # Cross-module statistics match
│   └── test_latex_artifact_sync.py       # LaTeX numbers match JSON
```

**Target**: All statistics referenced in LaTeX have integration tests

### 2.3 Contract Tests (NEW)

```
tests/
├── contracts/
│   ├── test_data_schema.py          # Data file schemas
│   ├── test_json_schema.py          # Generated JSON schemas
│   ├── test_latex_numbers.py        # LaTeX command validation
│   └── test_cross_repo_sync.py      # foundation-PLR ↔ sci-llm-writer
```

---

## 3. Specific Test Cases to Add

### 3.1 Statistics Consistency Tests

```python
# tests/integration/test_statistics_consistency.py

import json
import pytest
from pathlib import Path

MANUSCRIPT_DATA = Path("path/to/sci-llm-writer/manuscripts/foundationPLR")

class TestCorrelationConsistency:
    """Verify correlation values are consistent across all sources."""

    def test_f1_auroc_correlation_matches(self):
        """F1→AUROC correlation must match across parallel_coords and scatter."""

        # Load from parallel coordinates
        parallel = json.loads((MANUSCRIPT_DATA / "figures/generated/data/fig_parallel_preprocessing.json").read_text())

        # Load from scatter plot
        scatter = json.loads((MANUSCRIPT_DATA / "figures/generated/data/fig_preprocessing_vs_auroc.json").read_text())

        # Document the EXPECTED difference (n=24 vs n=40)
        # These are different because parallel_coords requires complete data
        assert parallel["correlations"]["auroc_vs_f1"] == pytest.approx(0.407, abs=0.01)
        assert scatter["panel_a"]["correlation_r"] == pytest.approx(0.421, abs=0.01)

        # The CANONICAL value is from scatter (larger n)
        # This should be documented in the manuscript

    def test_subject_counts_consistent(self):
        """Subject counts must sum correctly."""

        outlier_data = json.loads((MANUSCRIPT_DATA / "figures/generated/data/fig_outlier_easy_vs_hard.json").read_text())

        n_control = outlier_data["by_class"]["control"]["n_subjects"]
        n_glaucoma = outlier_data["by_class"]["glaucoma"]["n_subjects"]

        assert n_control == 152
        assert n_glaucoma == 56
        assert n_control + n_glaucoma == 208  # Labeled subjects

        # Verify against CSV source
        import pandas as pd
        csv_path = MANUSCRIPT_DATA / "data/outlier_difficulty_analysis.csv"
        df = pd.read_csv(csv_path)

        assert len(df[df['class_label'] == 'control']) == n_control
        assert len(df[df['class_label'] == 'glaucoma']) == n_glaucoma
```

### 3.2 LaTeX Numbers Validation Tests

```python
# tests/contracts/test_latex_numbers.py

import re
import json
from pathlib import Path

class TestLatexNumbersMatchSource:
    """Verify LaTeX \providecommand values match source JSON files."""

    @pytest.fixture
    def numbers_tex_content(self):
        tex_path = Path("path/to/numbers.tex")
        return tex_path.read_text()

    @pytest.fixture
    def source_json(self):
        return {
            'preprocessing': json.loads(Path("fig_preprocessing_vs_auroc.json").read_text()),
            'cd': json.loads(Path("cd_preprocessing_comparison.json").read_text()),
            'outlier': json.loads(Path("fig_outlier_easy_vs_hard.json").read_text()),
        }

    def test_correlation_values_match(self, numbers_tex_content, source_json):
        """Correlation values in LaTeX must match JSON source."""

        # Extract from LaTeX
        f1_match = re.search(r'\\providecommand{\\corrOutlierF1AUROC}{([\d.]+)}', numbers_tex_content)
        mae_match = re.search(r'\\providecommand{\\corrImputationMAEAUROC}{([\d.]+)}', numbers_tex_content)

        # Compare to JSON
        expected_f1 = source_json['preprocessing']['panel_a']['correlation_r']
        expected_mae = source_json['preprocessing']['panel_b']['correlation_r']

        assert float(f1_match.group(1)) == pytest.approx(expected_f1, abs=0.001)
        assert float(mae_match.group(1)) == pytest.approx(expected_mae, abs=0.001)

    def test_friedman_statistics_match(self, numbers_tex_content, source_json):
        """Friedman test statistics must match JSON source."""

        chi_match = re.search(r'\\providecommand{\\friedmanChiSq}{([\d.]+)}', numbers_tex_content)
        cd_match = re.search(r'\\providecommand{\\nemenyiCD}{([\d.]+)}', numbers_tex_content)

        expected_chi = source_json['cd']['friedman_stat']
        expected_cd = source_json['cd']['critical_difference']

        assert float(chi_match.group(1)) == pytest.approx(expected_chi, abs=0.01)
        assert float(cd_match.group(1)) == pytest.approx(expected_cd, abs=0.001)
```

### 3.3 Figure Generation Reproducibility Tests

```python
# tests/integration/test_figure_reproducibility.py

import hashlib
from pathlib import Path
import subprocess

class TestFigureReproducibility:
    """Verify figure generation is deterministic."""

    def test_parallel_coordinates_deterministic(self, tmp_path):
        """Running the same script twice produces identical JSON output."""

        # Run twice with same seed
        for i in range(2):
            result = subprocess.run(
                ["python", "src/viz/parallel_coordinates.py", "--seed", "42"],
                capture_output=True
            )
            assert result.returncode == 0

        # Compare JSON outputs (PNG may have metadata differences)
        json1 = Path("figures/generated/data/fig_parallel_preprocessing.json").read_bytes()
        # ... compare with second run

    def test_bootstrap_reproducibility(self):
        """Bootstrap samples with same seed produce identical results."""

        from src.stats.bootstrap import bootstrap_auroc

        results1 = bootstrap_auroc(data, n_iterations=100, seed=42)
        results2 = bootstrap_auroc(data, n_iterations=100, seed=42)

        assert results1 == results2
```

---

## 4. CI/CD Pipeline Integration

### 4.1 GitHub Actions Workflow

```yaml
# .github/workflows/test-artifacts.yml
name: Artifact Consistency Tests

on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'data/**'
  pull_request:

jobs:
  test-consistency:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[test]"

      - name: Run unit tests
        run: pytest tests/unit -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration -v

      - name: Run contract tests
        run: pytest tests/contracts -v

      - name: Verify LaTeX numbers match JSON
        run: python scripts/validate_latex_numbers.py
```

### 4.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-statistics
        name: Validate statistics consistency
        entry: python scripts/validate_statistics.py
        language: python
        files: '\.(json|tex|csv)$'

      - id: check-latex-numbers
        name: Check LaTeX numbers match source
        entry: python scripts/validate_latex_numbers.py
        language: python
        files: 'numbers\.tex$'
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Create `tests/integration/` directory structure
- [ ] Implement `test_statistics_consistency.py`
- [ ] Add JSON schema validation for all generated data files
- [ ] Create `validate_latex_numbers.py` script

### Phase 2: Contract Tests (Week 3-4)
- [ ] Define data contracts for all artifacts
- [ ] Implement cross-repository sync tests
- [ ] Add LaTeX command naming convention tests
- [ ] Create manifest file for data dependencies

### Phase 3: CI/CD Integration (Week 5-6)
- [ ] Set up GitHub Actions workflow
- [ ] Configure pre-commit hooks
- [ ] Add coverage reporting
- [ ] Document test requirements

### Phase 4: Continuous Improvement
- [ ] Review test failures from real workflow
- [ ] Add regression tests for caught bugs
- [ ] Quarterly audit of test coverage
- [ ] Update expert panel with new edge cases

---

## 6. Key Principles

1. **Single Source of Truth**: All statistics originate from one computation
2. **Fail Fast**: Tests should catch errors before they reach the manuscript
3. **Document Discrepancies**: When different n yields different r, document why
4. **Automate Everything**: No manual checking of numbers
5. **Version Lock Data**: Pin exact data versions for reproducibility

---

## 7. Validation Script Template

```python
#!/usr/bin/env python3
"""validate_artifact_consistency.py - Run before each manuscript update."""

import json
import re
import sys
from pathlib import Path

def load_json_data(base_path: Path) -> dict:
    """Load all generated JSON data files."""
    data = {}
    for json_file in base_path.glob("**/*.json"):
        data[json_file.stem] = json.loads(json_file.read_text())
    return data

def extract_latex_numbers(tex_path: Path) -> dict:
    """Extract all \providecommand values from numbers.tex."""
    content = tex_path.read_text()
    pattern = r'\\providecommand{\\(\w+)}{([^}]+)}'
    return {m.group(1): m.group(2) for m in re.finditer(pattern, content)}

def validate_consistency(json_data: dict, latex_numbers: dict) -> list:
    """Check for inconsistencies between JSON and LaTeX."""
    errors = []

    # Example validation rules
    mappings = {
        'corrOutlierF1AUROC': ('fig_preprocessing_vs_auroc', 'panel_a.correlation_r'),
        'friedmanChiSq': ('cd_preprocessing_comparison', 'friedman_stat'),
        'nemenyiCD': ('cd_preprocessing_comparison', 'critical_difference'),
    }

    for latex_key, (json_file, json_path) in mappings.items():
        if latex_key in latex_numbers:
            latex_val = float(latex_numbers[latex_key])
            # Navigate JSON path
            json_val = json_data[json_file]
            for key in json_path.split('.'):
                json_val = json_val[key]

            if abs(latex_val - json_val) > 0.001:
                errors.append(f"{latex_key}: LaTeX={latex_val}, JSON={json_val}")

    return errors

if __name__ == "__main__":
    # ... main validation logic
    pass
```

---

*Document created: 2026-01-21*
*Authored by: Claude Code with Expert Panel Reviews*
