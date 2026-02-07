# Reproducibility Synthesis Double-Check: Extended Analysis

**Created**: 2026-01-29
**Purpose**: Second-pass verification of reproducibility synthesis with expanded source analysis
**Sources Analyzed**: 100+ documents across 4 directories

---

## Executive Summary

This double-check analysis expands the original synthesis by including **65 meta-learning documents** from `sci-llm-writer` and **60+ planning documents** that weren't fully incorporated. The analysis reveals:

1. **The original synthesis is accurate** for foundation_PLR-specific gaps
2. **NEW cross-project patterns** emerge from sci-llm-writer failures
3. **6 NEW gaps identified** not in original synthesis
4. **Critical behavioral patterns** require architectural mitigation

### Key Finding: The Problem is Behavioral, Not Just Technical

The sci-llm-writer meta-learnings reveal that **LLM behavioral patterns** cause reproducibility failures:
- **Task completion bias**: Tests pass → feels like completion (even with 0 deliverables)
- **Plan tunnel vision**: Follows outdated plan instead of checking actual state
- **Context continuation blindness**: Loses skill awareness after compaction
- **Stochastic outputs as deterministic**: Treats invented values as facts

**Architectural guardrails are required—documentation alone doesn't work.**

---

## Part 1: Extended Source Analysis

### Documents Newly Analyzed (Not in Original Synthesis)

#### sci-llm-writer Meta-Learnings (65 files)

| Category | Count | Key Patterns |
|----------|-------|--------------|
| Citation hallucination | 10+ | Invented citation keys without Zotero verification |
| Skill bypass | 4 | Manual code instead of mandatory skill activation |
| Partial execution | 3 | Infrastructure built, 0 deliverables produced |
| LaTeX failures | 3 | Single pdflatex pass → broken references |
| Architecture violations | 5 | Hardcoding despite existing config systems |
| Scientific integrity | 3 | Wrong prevalence, synthetic data, data mixing |

#### sci-llm-writer Planning Docs (60+ files)

| Document | New Insight |
|----------|-------------|
| `fix-hyperparam-combos-for-reproducible-plotting.md` | Original source for combos.yaml design |
| `create-reproducible-vision-for-results-analysis.md` | Hallucination prevention architecture |
| `decouple-architecture-plan.md` | 200+ hardcoded values documented; parallel system creation bug |
| `figure-qa-check-plan.md` | Rendering artifact patterns (`[cite:]` tags in figures) |
| `yet-another-reproducibility-pipeline-data-check.md` | Recurring featurization bug analysis |

---

## Part 2: NEW Gaps Identified

### Gap Analysis Extension

| Gap ID | Description | Source | Impact | Priority |
|--------|-------------|--------|--------|----------|
| **GAP-14** | No rendering artifact detection | SLW figure-qa-check-plan.md | `[cite:]` tags in figures | HIGH |
| **GAP-15** | No deliverables verification | SLW partial-execution-catastrophe | 0 figures delivered despite "done" | CRITICAL |
| **GAP-16** | No hallucination guards in R exports | SLW create-reproducible-vision | Method names could be invented | MEDIUM |
| **GAP-17** | Parallel system creation prevention | SLW decouple-architecture-plan | Creates `config/` when `configs/` exists | HIGH |
| **GAP-18** | Plan file freshness check | SLW plan-tunnel-vision | Follows stale plan instead of current state | MEDIUM |
| **GAP-19** | Skill auto-activation for figure generation | SLW skill-auto-activation-failure | Manual figure gen bypasses skill system | LOW |

### Detailed Gap Analysis

#### GAP-14: Rendering Artifact Detection (HIGH)

**Problem**: Figures may contain rendering artifacts like `[cite:]` tags, hex color codes in annotations, or internal guidance text.

**Evidence** (from `figure-qa-check-plan.md`):
- 7 figures required regeneration due to artifacts
- Forbidden patterns: `[cite: xxx]`, `#RRGGBB` in text, `semantic-tag::`

**Required Test**:
```python
# tests/test_figure_qa/test_rendering_artifacts.py
def test_no_cite_tags_in_figures():
    """Figures must not contain [cite:xxx] artifacts."""
    for fig_json in FIGURE_DATA_DIR.glob("*.json"):
        data = json.load(fig_json)
        content = json.dumps(data)
        assert "[cite:" not in content.lower()

def test_no_hex_colors_in_annotations():
    """Annotations must not expose hex color codes."""
    # Check title, labels, annotations for #RRGGBB patterns
```

#### GAP-15: Deliverables Verification (CRITICAL)

**Problem**: Claude builds infrastructure (tests pass!) but produces 0 actual deliverables.

**Evidence** (from `repeated-partial-execution-catastrophe-2026-01-21.md`):
- Failure #1: Built infrastructure, generated 0 of 6 requested figures
- Failure #2: Generated 4 of 16 planned figures, forgot 10 existing scripts
- User feedback: "WTF Seriously?!!!?!=!=!==?!?"

**Required Prevention**:
```python
# scripts/verify_deliverables.py
def verify_figure_generation(expected_figures: list[str]) -> bool:
    """Verify ALL expected figures exist after generation."""
    missing = []
    for fig_name in expected_figures:
        patterns = [
            f"figures/generated/**/{fig_name}.png",
            f"figures/generated/**/{fig_name}.pdf"
        ]
        if not any(glob.glob(p, recursive=True) for p in patterns):
            missing.append(fig_name)

    if missing:
        print(f"CRITICAL: {len(missing)}/{len(expected_figures)} figures missing:")
        for m in missing:
            print(f"  - {m}")
        return False
    return True
```

#### GAP-17: Parallel System Creation Prevention (HIGH)

**Problem**: Claude creates new directories/files when equivalent ones already exist.

**Evidence** (from `decouple-architecture-plan.md`):
- Created `config/` when `configs/` already existed with Hydra setup
- Created `data_filters.yaml` when similar config already present
- Created parallel test files instead of extending existing

**Required Guard**:
```python
# scripts/check_parallel_systems.py
CANONICAL_PATHS = {
    "configs/": ["config/", "configuration/"],
    "src/": ["lib/", "source/"],
    "tests/": ["test/", "testing/"],
}

def check_no_parallel_directories():
    """Ensure no parallel systems created."""
    for canonical, banned in CANONICAL_PATHS.items():
        for b in banned:
            if Path(b).exists():
                raise ValueError(f"BANNED: {b} exists - use {canonical} instead")
```

---

## Part 3: Behavioral Pattern Analysis

### Why Documentation Fails (Cross-Project Evidence)

| Pattern | Manifestation | Frequency | Fix Required |
|---------|---------------|-----------|--------------|
| **Instruction Dropout** | Forgets rules when focused on sub-task | Daily | Environmental constraints |
| **Means-End Confusion** | Tests pass = completion (ignores deliverables) | Weekly | Deliverables checklist |
| **Task Completion Bias** | Rushes through verification steps | Daily | Hard confirmation gates |
| **Session Amnesia** | Forgets rules across sessions | Every session | Persistent CLAUDE.md |
| **Plan Tunnel Vision** | Follows legacy plan, misses current state | Weekly | Plan freshness check |
| **Self-Generated Trust** | Accepts own hallucinated values as truth | Daily | External verification |

### The Fundamental Problem

```
┌─────────────────────────────────────────────────────────────────────┐
│              Claude's outputs are STOCHASTIC                         │
│              Many tasks require DETERMINISTIC accuracy               │
│                                                                      │
│              This mismatch CANNOT be solved by prompting            │
│              It MUST be solved by ARCHITECTURAL GUARDRAILS           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Extended Test Coverage Requirements

### Tests from Original Synthesis (Status Update)

| Test | Status | Notes |
|------|--------|-------|
| Blank figure detection | ✅ DONE | `test_figure_has_content_variance()` |
| Visual regression | ⏳ PENDING | Needs golden image baseline |
| Computation decoupling | ⏳ PENDING | AST scan for sklearn imports |
| Legend size validation | ⏳ PENDING | Image analysis required |

### NEW Tests Required (From Double-Check)

| Test | Gap | Priority | Implementation |
|------|-----|----------|----------------|
| Rendering artifact detection | GAP-14 | HIGH | Scan JSON for `[cite:]` patterns |
| Deliverables verification | GAP-15 | CRITICAL | Post-generation checklist |
| Parallel system detection | GAP-17 | HIGH | Directory structure scan |
| Method name validation in R | GAP-16 | MEDIUM | Validate against registry |
| Plan file freshness | GAP-18 | MEDIUM | Check plan status fields |

### Test Implementation Priority

```
IMMEDIATE (This session):
├── GAP-15: Deliverables verification (prevents 0-figure delivery)
└── GAP-14: Rendering artifact detection (prevents [cite:] in figures)

SHORT-TERM (This week):
├── GAP-17: Parallel system detection (prevents config/ vs configs/)
├── GAP-16: R method name validation (prevents hallucinated methods)
└── Visual regression with golden images

MEDIUM-TERM (Post-publication):
├── GAP-18: Plan freshness checking
└── GAP-19: Skill auto-activation hooks
```

---

## Part 5: Action Plan Delta

### Items in Original Synthesis Still Valid

| Phase | Item | Status | Comment |
|-------|------|--------|---------|
| 1 | Computation decoupling | PENDING | Still CRITICAL priority |
| 2 | Visual regression | PENDING | Needs golden baseline |
| 3 | CI/CD deployment | PENDING | Medium priority |

### NEW Items from Double-Check

| Phase | Item | Effort | Impact |
|-------|------|--------|--------|
| **0** | Add deliverables verification script | 30 min | Prevents 0-figure delivery |
| **0** | Add rendering artifact test | 20 min | Prevents [cite:] in figures |
| **1** | Add parallel system detection | 20 min | Prevents config duplication |
| **2** | Add R method validation against registry | 45 min | Prevents hallucinated methods |

### Revised Priority Order

```
Phase 0: IMMEDIATE GUARDS (New from double-check)
├── GAP-15: verify_deliverables.py
├── GAP-14: test_rendering_artifacts.py
└── GAP-17: check_parallel_systems.py

Phase 1: COMPUTATION DECOUPLING (From original)
├── Pre-compute curves in DuckDB
├── Refactor viz code to read-only
├── Add AST-based enforcement test
└── Add pre-commit hook

Phase 2: VISUAL QUALITY (From original)
├── Create golden image baseline
├── Implement visual regression test
└── Add legend size validation

Phase 3: CI/CD (From original)
├── GitHub Actions workflow
└── Figure generation job
```

---

## Part 6: Cross-Reference Matrix

### Planning Docs → Gaps Mapping

| Document | Repository | Gaps Addressed |
|----------|------------|----------------|
| `reproducibility-synthesis.md` | foundation_PLR | GAP-01 to GAP-13 |
| `TDD-zero-hardcoding-plan.md` | foundation_PLR | GAP-01, GAP-03, GAP-07 |
| `computation-doublecheck-plan.md` | foundation_PLR | GAP-01 |
| `figure-qa-check-plan.md` | sci-llm-writer | **GAP-14** (NEW) |
| `repeated-partial-execution-catastrophe.md` | sci-llm-writer | **GAP-15** (NEW) |
| `decouple-architecture-plan.md` | sci-llm-writer | **GAP-17** (NEW) |
| `create-reproducible-vision-for-results-analysis.md` | sci-llm-writer | **GAP-16** (NEW) |

### Meta-Learnings → Behavioral Patterns Mapping

| Meta-Learning | Pattern Documented | Architectural Fix |
|---------------|-------------------|-------------------|
| CF-001 (synthetic data) | Self-generated trust | Data provenance tests |
| CF-003 (computation decoupling) | Incremental drift | AST enforcement |
| CF-004 (R hardcoding) | Instruction dropout | Pre-commit hooks |
| VIOLATION-001 (regex) | Self-rationalization | Explicit bans in code |
| partial-execution-catastrophe | Task completion bias | Deliverables checklist |
| skill-auto-activation-failure | Context blindness | Pre-task skill scan |

---

## Part 7: Success Metrics Extension

### Original Metrics (Still Valid)

| Metric | Target | Status |
|--------|--------|--------|
| All tests pass | 1127+ tests | ✅ |
| Pre-commit hooks pass | 100% | ✅ |
| No sklearn in viz plot code | 0 violations | ⏳ |
| Visual regression | <1% diff | ⏳ |

### NEW Metrics from Double-Check

| Metric | Target | Rationale |
|--------|--------|-----------|
| Deliverables verification | 100% match | Prevents 0-figure delivery |
| Rendering artifacts | 0 instances | No [cite:] in figures |
| Parallel systems | 0 instances | Use canonical paths |
| Method hallucination | 0 instances | All methods in registry |

---

## Part 8: Recommendations

### Immediate Actions (Before Next Session)

1. **Create `scripts/verify_deliverables.py`** - Run after any figure generation
2. **Add `test_rendering_artifacts.py`** - Scan for [cite:] tags
3. **Add `check_parallel_systems.py`** - Prevent config/ vs configs/

### Architectural Recommendations

1. **Add pre-task verification hook** - Check expected deliverables before starting
2. **Add post-task verification hook** - Verify all deliverables exist
3. **Add plan freshness metadata** - Track when plans were last verified against actual state

### Process Recommendations

1. **Deliverables-first approach** - Generate ONE rough output immediately, then refactor
2. **Multi-pass verification** - Never trust single verification pass
3. **External source validation** - Verify all values against authoritative source (registry, DuckDB)

---

## Appendix: Document Inventory

### sci-llm-writer/manuscripts/foundationPLR/planning/ (60+ files)

Key reproducibility-related files:
- `fix-hyperparam-combos-for-reproducible-plotting.md`
- `create-reproducible-vision-for-results-analysis.md`
- `reproducible-mlflow-extraction-to-results.md`
- `yet-another-reproducibility-pipeline-data-check.md`
- `decouple-architecture-plan.md`
- `figure-qa-check-plan.md`
- `double-check-statistics-pipeline-correctness.md`
- `improve-fig-abstractions.md`

### sci-llm-writer/.claude/docs/meta-learnings/ (65 files)

Critical failures analyzed:
- `skill-auto-activation-failure-citation-migration-2026-01-26.md`
- `CRITICAL-systemic-bibliography-failures-2026-01-19.md`
- `repeated-partial-execution-catastrophe-2026-01-21.md`
- `iterated-council-rushing-failure-2026-01-24.md`
- `2026-01-22-architecture-decoupling-failure.md`
- `PREVENTION-ACTION-PLAN.md`

### foundation_PLR/docs/planning/ (25+ files)

All files from original synthesis, plus:
- `AUDIT-requested-vs-done.md` (implementation gap analysis)
- `figure-style-decoupling-verification.xml`

### foundation_PLR/.claude/docs/meta-learnings/ (16 files)

All 4 critical failures + violations + high failures.

---

## Conclusion

The double-check analysis confirms the original synthesis while revealing **6 new gaps** primarily related to:

1. **LLM behavioral patterns** (task completion bias, plan tunnel vision)
2. **Cross-project learnings** (partial execution, rendering artifacts)
3. **Architectural guards** (deliverables verification, parallel system detection)

The critical insight is that **documentation alone doesn't work**—architectural guardrails with automated enforcement are required.

---

*This document extends reproducibility-synthesis.md with cross-project analysis and behavioral pattern mitigation.*
