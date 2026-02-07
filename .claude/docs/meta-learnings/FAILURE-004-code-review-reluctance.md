# FAILURE-004: Reluctance to Run Systematic Code Review

## Severity: HIGH

## The Failure

Despite MULTIPLE planning documents explicitly requesting:
- Decoupling of style from figure scripts
- No hardcoded values
- Consistent patterns across all files
- Systematic code review

Claude repeatedly:
1. Fixed files ONE BY ONE reactively instead of auditing ALL files upfront
2. Ignored planning docs that explicitly outlined requirements
3. Used different patterns in different files (`compose_figures()` vs `ggtitle()`)
4. Required user to repeatedly ask for the same fixes

## Planning Documents That Were IGNORED

At least **30+ planning documents** mention decoupling/hardcoding concerns:

### In `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/planning/`:
- `appendix-figure-creation-plan.md`
- `decouple-architecture-plan.md`
- `documentation-plan.md`
- `final-main-figure-polishing-plan.md`
- `improved-figure-coverage-plan.md`
- `improve-fig-abstractions.md`
- `latent-method-results-update.md`
- `manufacture-template-for-claude.md`
- `missing-figures-implementation-for-stratos.md`
- `reproduce-figures-after-yaml-update.md`
- `reproducible-mlflow-extraction-to-results.md`
- `uncertainty-visualization-plan.md`
- `final-final-figure-pipeline.polishing-plan.xml`
- `latent-methods-and-missing-figures-actionable.xml`
- `prefect-reproducibility-pipeline.xml`

### In `/home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/planning/`:
- `ACTION_PLAN.xml`
- `audit-2026-01-27-reproducibility-fixes.md`
- `AUDIT-requested-vs-done.md`
- `computation-doublecheck-plan-and-restriction-to-mlflow-duckdb-conversion.md`
- `double-check-reproducibility.md`
- `experiment-parameters-guardrail-improvements.md`
- `figure-batch-qa-update.md`
- `figure-data-flow-architecture.md`
- `figure-production-grade.md`
- `flexible-decomposable-ggplot2-figure-system.md`
- `hardcoding-guardrails-improvement.md`
- `implement-missing-ggplot2-figures.md`
- `improvement-from-openai-and-gemini.md`
- `ISSUE-test-documentation-improvements.md`
- `lookup-model-names-properly-double-check.md`

**ALL of these documents contain requirements that were repeatedly ignored.**

## Root Cause

### 1. Reactive Instead of Proactive
- Waited for user to point out each broken file
- Never ran systematic audit across ALL files

### 2. Different Patterns in Different Files
- Some files use `compose_figures()` with `panel_titles`
- Some files use `plot_annotation(tag_levels)`
- Some files now use `ggtitle()`
- **NO CONSISTENCY** despite planning docs requiring it

### 3. Failure to Read Planning Docs
- Planning docs exist explaining EXACTLY what to do
- Claude didn't read them before starting work
- Result: wasted effort, user frustration

## What Should Have Happened

1. **FIRST**: Read ALL planning docs (`docs/planning/*.md`)
2. **SECOND**: Grep ALL R files for inconsistent patterns
3. **THIRD**: Create comprehensive fix list
4. **FOURTH**: Fix ALL files in ONE systematic pass
5. **FIFTH**: Verify ALL outputs

## The Correct Pattern (ENFORCE THIS)

ALL multi-panel R figure scripts MUST use:

```r
# Add title to each panel
p_a <- create_panel_a() +
  ggtitle("A  Panel Title") +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0),
        plot.title.position = "plot")

p_b <- create_panel_b() +
  ggtitle("B  Panel Title") +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0),
        plot.title.position = "plot")

# Compose WITHOUT tag_levels
composed <- (p_a | p_b)  # NO plot_annotation(tag_levels)
```

## Files That Were Inconsistent (as of 2026-01-28)

| File | Was Using | Should Use |
|------|-----------|------------|
| fig_selective_classification.R | plot_annotation(tag_levels) | ggtitle() |
| fig_stratos_core.R | get_tag_levels(style) | ggtitle() |
| fig_prob_dist_by_outcome.R | compose_figures() | ggtitle() |
| fig_shap_importance.R | compose_figures() | ggtitle() |
| fig_vif_analysis.R | compose_figures() | ggtitle() |
| fig_calibration_dca_combined.R | plot_annotation(tag_levels) | ggtitle() |

## Prevention Rules

### 1. Read Planning Docs First
> Before ANY implementation work, grep planning docs for related requirements.
> `grep -r "keyword" docs/planning/`

### 2. Audit Before Fixing
> Before fixing ONE file, audit ALL files for the same pattern.
> `grep -l "pattern" src/r/figures/*.R`

### 3. One Pattern, Enforced Everywhere
> When establishing a pattern, grep for ALL violations and fix them together.
> Never "fix as you go" - that leads to inconsistency.

### 4. Systematic > Reactive
> PROACTIVE systematic review beats REACTIVE one-by-one fixes.
> The user should NEVER need to point out the same issue twice.

## Impact

- User frustration (asked "million times")
- Wasted session time on reactive fixes
- Inconsistent codebase despite existing planning docs
- Loss of trust in Claude's ability to follow documented requirements

## Status

- [x] All R figure scripts using consistent ggtitle() pattern (2026-01-28)
- [x] All compose_figures() usages in standalone scripts converted to ggtitle() (2026-01-28)
- [x] All plot_annotation(tag_levels) usages removed (2026-01-28)
- [x] All figures regenerated and verified (2026-01-28)
- [x] XML tracking plan updated with completion status (2026-01-28)

### Files Fixed
- fig_multi_metric_raincloud.R
- fig_roc_rc_combined.R
- fig_variance_decomposition.R
- fig_selective_classification.R
- fig_stratos_core.R
- fig_prob_dist_by_outcome.R
- fig_shap_importance.R
- fig_vif_analysis.R
- fig_calibration_dca_combined.R
- fig_subject_traces.R

All 10 multi-panel figures now use consistent `ggtitle("A  Title")` pattern.
