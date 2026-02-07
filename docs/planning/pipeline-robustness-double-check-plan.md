# Pipeline Robustness Double-Check Plan

**Version:** 1.0.0
**Created:** 2026-02-02
**Status:** COUNCIL REVIEW COMPLETE - Ready for TDD Execution

---

## Executive Summary

This plan synthesizes findings from 5 parallel LLM Council reviewer agents evaluating:
1. Config versioning schemes
2. Pre-commit enforcement strategies
3. Artifact provenance tracking
4. Industry MLOps best practices
5. Test harness gap assessment

**Key Finding:** Foundation PLR's architecture is already **well-aligned with industry best practices**. The gaps identified are **incremental improvements**, not architectural overhauls.

---

## Council Consensus: Recommended Architecture

### The "LabOps" System Design

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FOUNDATION PLR LABOPS ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CONFIG VERSIONING (Two-Tier)                                                    │
│  ═════════════════════════════                                                   │
│                                                                                  │
│  ┌─────────────────────┐     ┌─────────────────────┐                            │
│  │  TIER 1: Semantic   │  +  │  TIER 2: Content    │  =  v1.0.0-8f3a7bc        │
│  │  (human-maintained) │     │  Hash (automatic)   │     (full version)         │
│  └─────────────────────┘     └─────────────────────┘                            │
│                                                                                  │
│  PRE-COMMIT ENFORCEMENT                                                          │
│  ══════════════════════                                                          │
│                                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                     │
│  │ Auto-version │ --> │ Hash verify  │ --> │ CI Validation│                     │
│  │ (zero effort)│     │ (pre-commit) │     │ (unbypassable)│                     │
│  └──────────────┘     └──────────────┘     └──────────────┘                     │
│                                                                                  │
│  ARTIFACT PROVENANCE (Layered)                                                   │
│  ═════════════════════════════                                                   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 1: MLflow        │ Tags: config_version, git_sha, timestamp       │   │
│  ├──────────────────────────────────────────────────────────────────────────┤   │
│  │ Layer 2: DuckDB        │ _provenance_log table with full config hashes  │   │
│  ├──────────────────────────────────────────────────────────────────────────┤   │
│  │ Layer 3: JSON Sidecars │ _provenance field in figure data files         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  TRACEABILITY CHAIN                                                              │
│  ══════════════════                                                              │
│                                                                                  │
│  Figure.png --> JSON sidecar --> DuckDB extraction_id --> MLflow run --> Config │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Hypothesis Evaluation

### Hypothesis 1: Config Versioning Schemes

| Scheme | Publication Repro | Dev Iteration | Automation | Comprehension | **Score** |
|--------|-------------------|---------------|------------|---------------|-----------|
| A. Semantic (v1.2.3) | Medium | High | Low | High | 6.5 |
| B. Date-based (v1.26.2.1) | Medium | Medium | Medium | Medium | 5.5 |
| C. Git SHA only | High | Low | High | Low | 6.0 |
| D. Content Hash | High | High | High | Low | 7.0 |
| **E. Hybrid (Semantic + Hash)** | **High** | **High** | **High** | **High** | **9.0** |

**Council Consensus:** Scheme E (Hybrid) - `v1.0.0-8f3a7bc` format

**Rationale:**
- Human-readable semantic version for communication
- Content hash for cryptographic verification
- Git SHA links to source control
- Automated bumping for content changes

### Hypothesis 2: Pre-commit Enforcement

| Strategy | Enforcement | DevEx | Bypass Resistance | Complexity | **Score** |
|----------|-------------|-------|-------------------|------------|-----------|
| 1. Version field check | Low | Acceptable | Low | Low | 4.0 |
| 2. Content hash in file | High | Poor | High | Medium | 6.0 |
| 3. Separate registry | High | Poor | Medium | High | 5.0 |
| **4. Auto-versioning** | **Medium-High** | **Excellent** | **Medium** | **Medium** | **8.5** |
| 5. CI-only validation | High | Good | High | Low | 7.5 |

**Council Consensus:** Strategy 4 (auto-versioning) + Strategy 5 (CI validation)

**Rationale:**
- Zero manual effort aligns with CLAUDE.md DevEx mandate
- CI safety net catches `--no-verify` bypasses
- Unbypassable without force-push (requires admin)

### Hypothesis 3: Artifact Provenance

| Strategy | Repro Guarantee | Query Efficiency | Storage | Complexity | **Score** |
|----------|-----------------|------------------|---------|------------|-----------|
| 1. MLflow tags only | Medium | Medium | Low | Low | 6.0 |
| **2. DuckDB metadata** | **High** | **High** | **Low** | **Medium** | **8.4** |
| 3. JSON sidecar enhancement | Good | Good | Medium | Low | 7.4 |
| 4. Git LFS manifest | Medium | Low | High | High | 4.2 |
| 5. Content-addressable | High | High | High | High | 7.0 |

**Council Consensus:** Layered approach (MLflow + DuckDB + JSON)

**Rationale:**
- Each artifact type gets appropriate provenance
- DuckDB tables are self-contained and queryable
- JSON sidecars already exist - just enhance them
- Full traceability chain from figure to config

### Hypothesis 4: Additional Systems

| System | Suitability | Complexity | Added Value | **Recommendation** |
|--------|-------------|------------|-------------|-------------------|
| DVC | Medium | High | Medium | **NOT NEEDED** |
| W&B | Low (data privacy) | Medium | Medium | **NOT NEEDED** |
| Sacred | Medium | High | Low | **NOT NEEDED** |
| Kubeflow | Low (overkill) | Very High | Low | **NOT NEEDED** |

**Council Consensus:** No additional systems needed

**Rationale:**
- Foundation PLR already has Hydra + MLflow (best combo)
- Two-block architecture provides strong reproducibility
- Adding systems increases complexity without proportional benefit
- Focus on enhancing existing excellent patterns

---

## Test Harness Gap Analysis

### Critical Gaps (P0)

| Gap | Description | Impact | Test Location |
|-----|-------------|--------|---------------|
| **GAP-01** | No JSON Schema for YAML configs | Invalid configs pass silently | `tests/unit/test_config_schemas.py` |
| **GAP-02** | No config versioning enforcement | Reproducibility risk | pre-commit hook |
| **GAP-04** | No training reproducibility tests | STRATOS compliance risk | `tests/e2e/test_training_reproducibility.py` |
| **GAP-09** | No hash verification for frozen configs | Publication integrity risk | pre-commit hook |

### High Priority Gaps (P1)

| Gap | Description | Impact | Location |
|-----|-------------|--------|----------|
| **GAP-03** | No hyperparam range validation | Invalid HP silently accepted | `tests/unit/test_hyperparam_validation.py` |
| **GAP-05** | No Hydra composition tests | Config edge cases | `tests/integration/test_hydra_composition.py` |
| **GAP-07** | No DuckDB schema evolution | Breaking changes | `tests/integration/test_db_schema.py` |

### Test Pyramid Rebalancing

| Level | Current | Target | Action |
|-------|---------|--------|--------|
| Unit | 43% | 60% | Add config schema, hyperparam validation tests |
| Integration | 22% | 25% | Add Hydra composition, DB schema tests |
| E2E | 3% | 10% | Add training reproducibility tests |
| Guardrail | 27% | 5% | Acceptable (scientific integrity justified) |

---

## Recommended Action Plan (TDD Execution)

### Phase 1: Config Versioning Foundation (Week 1)

**TDD Approach:** Write tests first, then implement

#### Task 1.1: Config Version Tests

```python
# tests/unit/test_config_versioning.py (WRITE FIRST)
class TestConfigVersioning:
    def test_all_configs_have_version_field(self):
        """All YAML configs must have _version field."""
        pass

    def test_version_format_semantic(self):
        """Version must be semantic (MAJOR.MINOR.PATCH)."""
        pass

    def test_content_hash_format(self):
        """Content hash must be 12-char hex."""
        pass
```

#### Task 1.2: Version Manifest Generator

**File:** `scripts/config_versioning.py`

```python
# Implementation after tests pass
def generate_manifest() -> dict:
    """Generate configs/_version_manifest.yaml"""
    pass

def verify_manifest() -> list[str]:
    """Verify current configs match manifest."""
    pass
```

#### Task 1.3: Pre-commit Auto-versioning

**File:** `scripts/auto_version_configs.py`

- Detect content changes (exclude version/hash lines)
- Auto-bump patch version
- Regenerate content hash
- Re-stage modified files

### Phase 2: Pre-commit Hooks (Week 1-2)

#### Task 2.1: Add Config Version Validation Hook

```yaml
# .pre-commit-config.yaml
- id: auto-version-configs
  name: Auto-version config files
  entry: python scripts/auto_version_configs.py
  language: python
  files: configs/.*\.yaml$
```

#### Task 2.2: Add Frozen Config Guard Hook

```yaml
- id: frozen-config-check
  name: Frozen Experiment Config Guard
  entry: python scripts/check_frozen_configs.py
  files: configs/experiment/.*\.yaml$
```

#### Task 2.3: CI Validation Workflow

**File:** `.github/workflows/config-integrity.yml`

- Validate config hashes match declared versions
- Block merge if hashes mismatch
- Catch `--no-verify` bypasses

### Phase 3: Artifact Provenance (Week 2)

#### Task 3.1: DuckDB Provenance Table

**File:** `scripts/extract_all_configs_to_duckdb.py`

```sql
CREATE TABLE IF NOT EXISTS _provenance_log (
    extraction_id VARCHAR PRIMARY KEY,
    extracted_at TIMESTAMP,
    git_sha VARCHAR,
    config_hashes JSON,
    mlflow_runs_included JSON,
    registry_version VARCHAR
);
```

#### Task 3.2: JSON Sidecar Enhancement

**File:** `src/viz/plot_config.py`

```python
def save_figure(fig, name, data=None, ...):
    if data is not None:
        data["_provenance"] = {
            "generated_at": datetime.now().isoformat(),
            "git_sha": get_git_sha(),
            "duckdb_extraction_id": get_extraction_id(),
            "config_files": hash_config_files([...])
        }
```

#### Task 3.3: MLflow Tag Enhancement

- Add `config_version` tag to all training runs
- Add `git_sha` tag
- Add `registry_version` tag

### Phase 4: Test Harness Reinforcement (Week 2-3)

#### Task 4.1: Hyperparam Validation Tests

```python
# tests/unit/test_hyperparam_validation.py
class TestHyperparamValidation:
    def test_catboost_depth_positive(self):
        """CatBoost depth values must be positive integers."""
        pass

    def test_learning_rate_in_valid_range(self):
        """Learning rates must be in (0, 1]."""
        pass

    def test_all_classifiers_have_search_space(self):
        """Each classifier must have matching hyperparam file."""
        pass
```

#### Task 4.2: Training Reproducibility Tests

```python
# tests/e2e/test_training_reproducibility.py
class TestTrainingReproducibility:
    @pytest.mark.slow
    def test_catboost_deterministic_with_seed(self):
        """Same seed = identical CatBoost predictions."""
        pass

    @pytest.mark.slow
    def test_bootstrap_reproducible(self):
        """Bootstrap samples must be reproducible."""
        pass
```

#### Task 4.3: DuckDB Schema Tests

```python
# tests/integration/test_db_schema.py
class TestDuckDBSchema:
    def test_essential_metrics_schema_matches(self):
        """Schema must match expected definition."""
        pass

    def test_provenance_log_exists(self):
        """_provenance_log table must exist."""
        pass
```

### Phase 5: Documentation & Freeze (Week 3)

#### Task 5.1: Create Frozen Registry Marker

**File:** `configs/mlflow_registry/FROZEN_2026-02.yaml`

```yaml
frozen_date: "2026-02-02"
git_commit: "bf29be5..."
outlier_methods: 11
imputation_methods: 8
classifiers: 5
config_manifest_hash: "sha256:..."
```

#### Task 5.2: Update ARCHITECTURE.md

- Document config versioning system
- Document provenance tracking
- Document TDD test structure

#### Task 5.3: Commit uv.lock

```bash
git add uv.lock
git commit -m "chore: Track uv.lock for reproducibility"
```

---

## Implementation Checklist

### Week 1: Foundation

- [ ] Write `tests/unit/test_config_versioning.py` (TDD first)
- [ ] Implement `scripts/config_versioning.py`
- [ ] Write `tests/unit/test_hyperparam_validation.py` (TDD first)
- [ ] Add `_version` field to all config files
- [ ] Generate initial `configs/_version_manifest.yaml`

### Week 2: Enforcement

- [ ] Implement `scripts/auto_version_configs.py`
- [ ] Add pre-commit hook for auto-versioning
- [ ] Implement `scripts/check_frozen_configs.py`
- [ ] Add pre-commit hook for frozen configs
- [ ] Create `.github/workflows/config-integrity.yml`
- [ ] Add `_provenance_log` table to extraction
- [ ] Enhance JSON sidecars with `_provenance`

### Week 3: Testing & Documentation

- [ ] Write `tests/e2e/test_training_reproducibility.py`
- [ ] Write `tests/integration/test_db_schema.py`
- [ ] Create `configs/mlflow_registry/FROZEN_2026-02.yaml`
- [ ] Update ARCHITECTURE.md
- [ ] Commit uv.lock
- [ ] Run full test suite and verify all pass

---

## Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Config hash coverage | % of configs with content hash | 100% |
| Auto-versioning | Pre-commit passes without manual version bump | Always |
| CI enforcement | Hash mismatches caught before merge | 100% |
| Provenance coverage | Artifacts with provenance metadata | 100% |
| Test coverage | Unit test ratio | ≥60% |
| Reproducibility | Same config = same results | Verified |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Auto-versioning creates noise | Medium | Low | Only bump patch version |
| Hash computation slow | Low | Medium | Cache hashes |
| Pre-commit bypass via `--no-verify` | Medium | High | CI catches |
| Migration breaks existing workflow | Low | High | Phased rollout |

---

## Appendix: Council Agent Outputs

### L1: Config Versioning Schemes
- Recommended: Two-tier (semantic + content hash)
- Full version format: `v1.0.0-8f3a7bc`

### L2: Pre-commit Enforcement
- Recommended: Auto-versioning + CI validation
- Zero manual effort, unbypassable enforcement

### L3: Artifact Provenance
- Recommended: Layered (MLflow → DuckDB → JSON)
- Full traceability chain

### L4: Industry Practices
- Finding: Foundation PLR already well-aligned
- Recommendation: No new systems needed

### L5: Test Harness
- Critical gaps: Config schema, versioning, reproducibility
- Rebalancing needed: More unit tests, E2E tests

---

*Generated by Iterated LLM Council - 5 reviewer agents, 2026-02-02*
