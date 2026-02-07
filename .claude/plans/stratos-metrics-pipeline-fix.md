# STRATOS Metrics Pipeline Fix Plan

**Created:** 2026-01-22
**Status:** Planning
**Goal:** Ensure all STRATOS-required metrics flow from MLflow → DuckDB → JSON
**Reference:** Van Calster et al. 2024 "Performance evaluation of predictive AI models to support medical decisions" (STRATOS Initiative Topic Group 6)

---

## CRITICAL REQUIREMENT

> "We recommend the following measures and plots as essential to report: **AUROC, calibration plot, a clinical utility measure such as net benefit with decision curve analysis, and a plot with probability distributions per outcome category.**"
> — Van Calster et al. 2024

**FAILURE TO INCLUDE THESE = PUBLICATION REJECTION RISK**

---

## Problem Statement

The STRATOS guidelines (Van Calster 2024) **MANDATE** these measures:

### STRATOS Core Set (MANDATORY)

| Domain | Measure | Purpose | Status |
|--------|---------|---------|--------|
| **Discrimination** | AUROC with 95% CI | Ranking ability | ⚠️ Partial |
| **Calibration** | Smoothed calibration plot | Probability accuracy | ⚠️ Partial |
| **Calibration** | Calibration slope | Spread of probabilities | ❌ Missing |
| **Calibration** | Calibration intercept / O:E ratio | Mean calibration | ❌ Missing |
| **Clinical Utility** | Net Benefit | Decision value | ❌ Missing |
| **Clinical Utility** | DCA curves (threshold sweep) | Utility across thresholds | ❌ Missing |
| **Overall** | Probability distributions per class | Understanding predictions | ✅ Available |

### What STRATOS Says NOT to Use

| Measure | Why NOT | Our Status |
|---------|---------|------------|
| F1 score | Improper + ignores TN | ⚠️ Remove from reports |
| AUPRC | Ignores TN, prevalence-dependent | ⚠️ Demote to supplementary |
| pAUROC | No decision-analytic basis | ⚠️ Remove |
| Youden-optimized threshold | Assumes equal costs | ⚠️ Use clinical threshold |

**Current State:**
- MLflow has predictions (y_true, y_prob) that can compute all metrics ✅
- DuckDB only extracts AUROC (loses 20+ metrics) ❌
- JSON files missing: calibration slope/intercept, O:E ratio, DCA curves ❌

---

## Gap Summary

| Metric | MLflow | DuckDB | JSON | Fix Priority |
|--------|--------|--------|------|--------------|
| AUROC | ✅ | ⚠ | ⚠ | P2 |
| AUPR | ✅ | ❌ | ❌ | P3 |
| Brier Score | ✅ | ⚠ | ⚠ | P2 |
| **Calibration Slope** | ✅ | ❌ | ❌ | **P1** |
| **Calibration Intercept** | ✅ | ❌ | ❌ | **P1** |
| **O:E Ratio** | ✅ | ❌ | ❌ | **P1** |
| Calibration Curve | ✅ | ⚠ | ✅ | OK |
| **DCA Curves (full)** | ✅ | ❌ | ❌ | **P1** |
| Net Benefit (@threshold) | ✅ | ⚠ | ❌ | P2 |
| Prob Distributions | ✅ | ❌ | ✅ | OK |

---

## Implementation Plan

### Phase 1: Fix DuckDB Export (Priority 1)

**File:** `src/data_io/duckdb_export.py`

**Task 1.1:** Modify `extract_mlflow_classification_runs()` to extract ALL metrics

```python
# Current (line ~1019): Only extracts AUROC
if "AUROC" in scalars:
    auroc_vals = scalars["AUROC"]

# Fix: Extract all scalar metrics
REQUIRED_METRICS = [
    'AUROC', 'AUPR', 'tpAUC', 'Brier', 'ECE',
    'calibration_slope', 'calibration_intercept', 'e_o_ratio',
    'sensitivity', 'specificity', 'PPV', 'NPV', 'F1', 'accuracy',
    'net_benefit_5pct', 'net_benefit_10pct', 'net_benefit_15pct', 'net_benefit_20pct'
]

for metric_name in REQUIRED_METRICS:
    if metric_name in scalars:
        metric_dict[metric_name.lower()] = extract_scalar(scalars[metric_name])
```

**Task 1.2:** Populate `dca_curves` table

```python
# Extract DCA data from MLflow artifacts
dca_data = load_mlflow_artifact(run_id, 'dca_curves.pkl')
if dca_data:
    for row in dca_data:
        insert_dca_curve(conn, run_id, row['threshold'], row['nb_model'],
                        row['nb_all'], row['nb_none'])
```

**Task 1.3:** Populate `calibration_curves` table with extended metrics

```python
# Extract calibration metrics
cal_data = load_mlflow_artifact(run_id, 'calibration_metrics.pkl')
if cal_data:
    insert_calibration_summary(conn, run_id,
        slope=cal_data['slope'],
        intercept=cal_data['intercept'],
        oe_ratio=cal_data['e_o_ratio'])
```

### Phase 2: Fix JSON Export (Priority 1)

**Task 2.1:** Create `fig_calibration_extended.json`

**File:** New function in `src/viz/calibration_plot.py`

```python
def save_calibration_extended_json(data: dict, output_path: Path):
    """Save extended calibration metrics to JSON."""
    json_data = {
        'calibration_slope': data['slope'],
        'calibration_intercept': data['intercept'],
        'oe_ratio': data['e_o_ratio'],
        'brier_score': data['brier'],
        'scaled_brier': data['scaled_brier'],  # IPA
        'calibration_curve': {
            'observed': data['observed'],
            'expected': data['expected'],
            'bin_counts': data['bin_counts']
        }
    }
    save_json(json_data, output_path)
```

**Task 2.2:** Create `fig_dca_full.json`

**File:** Modify `src/viz/dca_plot.py`

```python
def save_dca_curves_json(data: dict, output_path: Path):
    """Save full DCA curves for all combos."""
    json_data = {
        'thresholds': data['thresholds'].tolist(),  # e.g., [0.01, 0.02, ..., 0.50]
        'combos': {}
    }
    for combo_id, curves in data['combos'].items():
        json_data['combos'][combo_id] = {
            'net_benefit_model': curves['nb_model'].tolist(),
            'net_benefit_all': curves['nb_all'].tolist(),
            'net_benefit_none': curves['nb_none'].tolist(),
            'clinical_utility_index': curves.get('cui', None)
        }
    save_json(json_data, output_path)
```

**Task 2.3:** Update `fig_retained_multi_metric.json` to include comparison data

Add STRATOS comparison table:
```json
{
  "stratos_comparison": {
    "LOF_SAITS": {
      "auroc": 0.806,
      "brier": 0.144,
      "ece": 0.273,
      "net_benefit_15pct": 0.171,
      "calibration_slope": 0.95,
      "calibration_intercept": -0.02,
      "oe_ratio": 1.05
    },
    "MOMENT_SAITS": {
      "auroc": 0.851,
      "brier": 0.131,
      "ece": 0.258,
      "net_benefit_15pct": 0.199,
      "calibration_slope": 0.98,
      "calibration_intercept": -0.01,
      "oe_ratio": 1.02
    }
  }
}
```

### Phase 3: Tests (TDD)

**Task 3.1:** Unit tests for DuckDB export

**File:** `tests/unit/test_duckdb_export_stratos.py`

```python
class TestSTRATOSMetricsExport:
    """Test that all STRATOS-required metrics are exported to DuckDB."""

    def test_extracts_calibration_slope(self, mock_mlflow_run):
        """Calibration slope must be extracted."""
        result = extract_mlflow_classification_runs(mock_mlflow_run)
        assert 'calibration_slope' in result
        assert result['calibration_slope'] is not None

    def test_extracts_calibration_intercept(self, mock_mlflow_run):
        """Calibration intercept must be extracted."""
        result = extract_mlflow_classification_runs(mock_mlflow_run)
        assert 'calibration_intercept' in result

    def test_extracts_oe_ratio(self, mock_mlflow_run):
        """O:E ratio must be extracted."""
        result = extract_mlflow_classification_runs(mock_mlflow_run)
        assert 'e_o_ratio' in result

    def test_extracts_net_benefit_multiple_thresholds(self, mock_mlflow_run):
        """Net benefit at multiple thresholds must be extracted."""
        result = extract_mlflow_classification_runs(mock_mlflow_run)
        assert 'net_benefit_5pct' in result
        assert 'net_benefit_10pct' in result
        assert 'net_benefit_15pct' in result
        assert 'net_benefit_20pct' in result

    def test_dca_curves_table_populated(self, mock_mlflow_run, db_conn):
        """DCA curves table must be populated."""
        export_mlflow_to_duckdb(mock_mlflow_run, db_conn)
        result = db_conn.execute("SELECT COUNT(*) FROM dca_curves").fetchone()
        assert result[0] > 0
```

**Task 3.2:** Unit tests for JSON export

**File:** `tests/unit/test_json_stratos_export.py`

```python
class TestSTRATOSJSONExport:
    """Test that all STRATOS-required metrics are in JSON outputs."""

    def test_calibration_extended_json_has_slope(self, calibration_data):
        """JSON must include calibration slope."""
        json_data = create_calibration_extended_json(calibration_data)
        assert 'calibration_slope' in json_data
        assert isinstance(json_data['calibration_slope'], float)

    def test_calibration_extended_json_has_oe_ratio(self, calibration_data):
        """JSON must include O:E ratio."""
        json_data = create_calibration_extended_json(calibration_data)
        assert 'oe_ratio' in json_data

    def test_dca_json_has_full_curves(self, dca_data):
        """JSON must include full DCA curves across thresholds."""
        json_data = create_dca_curves_json(dca_data)
        assert 'thresholds' in json_data
        assert len(json_data['thresholds']) >= 20  # At least 20 threshold points
        assert 'combos' in json_data
        for combo in json_data['combos'].values():
            assert 'net_benefit_model' in combo
            assert 'net_benefit_all' in combo
            assert 'net_benefit_none' in combo
```

---

## Execution Order

1. **Write tests first** (TDD)
   - `tests/unit/test_duckdb_export_stratos.py`
   - `tests/unit/test_json_stratos_export.py`
   - Run tests → expect failures

2. **Fix DuckDB export** (Phase 1)
   - Modify `src/data_io/duckdb_export.py`
   - Run tests → expect partial pass

3. **Fix JSON export** (Phase 2)
   - Modify `src/viz/calibration_plot.py`
   - Modify `src/viz/dca_plot.py`
   - Run tests → expect full pass

4. **Integration test**
   - Run full pipeline: MLflow → DuckDB → JSON
   - Verify all STRATOS metrics present

5. **Code review**
   - Review for edge cases
   - Check error handling
   - Verify backwards compatibility

---

## Acceptance Criteria

- [ ] DuckDB `metrics_per_fold` table includes all 18 required metrics
- [ ] DuckDB `dca_curves` table is populated
- [ ] JSON includes calibration slope, intercept, O:E ratio
- [ ] JSON includes full DCA curves (20+ threshold points)
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] STRATOS comparison table can be generated from JSON data

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/data_io/duckdb_export.py` | Extract all metrics, populate DCA table |
| `src/viz/calibration_plot.py` | Add extended calibration JSON export |
| `src/viz/dca_plot.py` | Add full DCA curves JSON export |
| `tests/unit/test_duckdb_export_stratos.py` | New test file |
| `tests/unit/test_json_stratos_export.py` | New test file |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| MLflow artifacts missing some metrics | Check artifact structure first, handle gracefully |
| Schema mismatch in DuckDB | Add migration script if needed |
| Backwards compatibility | Keep old JSON structure, add new fields |
| Performance (large DCA curves) | Limit threshold resolution to 50 points |

---

## References

- Van Calster et al. 2024 STRATOS guidelines
- `configs/VISUALIZATION/metrics.yaml` - metric definitions
- `configs/mlflow_registry/metrics/classification.yaml` - MLflow registry
