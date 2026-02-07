# Agent 1: Compression & Deduplication

## Mandate

Reduce total always-loaded chars by 50% while preserving ALL critical rules.

## Input Files

Read these files (always-loaded):
- `CLAUDE.md` (root)
- `.claude/CLAUDE.md`
- `.claude/rules/00-research-question.md`
- `.claude/rules/05-registry-source-of-truth.md`
- `.claude/rules/10-figures.md`
- `.claude/GUARDRAILS.md`

## Actions

### 1. Deduplicate

Identify content appearing in 2+ files. Keep in ONE canonical location:

| Content Type | Canonical Location |
|--------------|-------------------|
| Research question | `.claude/rules/00-research-question.md` |
| Registry rules | `.claude/rules/05-registry-source-of-truth.md` |
| Figure rules | `.claude/rules/10-figures.md` |
| STRATOS metrics | NEW: `.claude/rules/15-stratos-metrics.md` |
| Package management | NEW: `.claude/rules/20-package-management.md` |
| No reimplementation | NEW: `.claude/rules/25-no-reimplementation.md` |
| Hardcoding ban | Keep in `.claude/CLAUDE.md` (behavior contract) |
| Computation decoupling | Keep in `.claude/CLAUDE.md` (behavior contract) |

In the source file, replace duplicated content with a 1-line cross-reference:
```
**Registry rules**: See `.claude/rules/05-registry-source-of-truth.md`
```

### 2. Compress Prose to Tables

Convert verbose explanations to imperative tables:

**Before** (300 chars):
> When working on specific areas, load additional context. For figures, you should load the figure registry. For visualization work, check the visualization domain file...

**After** (120 chars):
| Task | Load |
|------|------|
| Figures | `configs/VISUALIZATION/figure_registry.yaml` |
| Viz | `.claude/domains/visualization.md` |

### 3. Move Domain-Specific Details

Content that only matters for specific tasks → `.claude/domains/` (on-demand):
- Detailed STRATOS JSON schemas → `domains/stratos-detailed.md`
- Future vision section → `.claude/planning/future-vision.md`
- Detailed combo lists → reference YAML directly
- Detailed MLflow paths → already in `domains/mlflow-experiments.md`

### 4. Compress Code Examples

Keep exactly 1 example per pattern. Remove duplicates:
- Registry usage: 1 Python example (CORRECT vs WRONG)
- Figure saving: 1 Python + 1 R example
- Config loading: 1 example

### 5. Remove "Why This Exists" Prose

Replace verbose explanations with meta-learning references:
```
# Before (200 chars):
"This rule was created because Claude generated synthetic data in calibration
plots using a fixed random seed, which would have been scientific fraud..."

# After (80 chars):
See: `.claude/docs/meta-learnings/CRITICAL-FAILURE-001-synthetic-data-in-figures.md`
```

## Output Format

```yaml
proposals:
  - id: C01
    file: "CLAUDE.md"
    action: "compress"  # compress | move | deduplicate | delete
    section: "Section heading"
    current_chars: 3500
    proposed_chars: 800
    destination: null  # or target file if moving
    rationale: "Brief reason"
    content_preview: "First 200 chars of proposed replacement..."
```

## Constraints

- NEVER delete a rule entirely - compress or move
- NEVER remove the last instance of any critical rule keyword
- Validate against `reference/critical-rules-checklist.yaml` after each proposal
- Prefer tables and imperative commands over prose
