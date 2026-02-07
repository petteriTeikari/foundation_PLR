# Subject Traces ggplot2 Generation Plan

> **Created:** 2026-01-31
> **Status:** Planning
> **Priority:** High (manuscript submission)

## Overview

This plan details the implementation of ggplot2-based subject trace figures to replace/complement the existing matplotlib implementation. The figures display raw and processed PLR (Pupillary Light Reflex) signals for demo subjects, demonstrating preprocessing effects.

### Current State

**Existing matplotlib implementation:**
- Script: `src/viz/individual_subject_traces.py`
- Outputs: `figures/generated/supplementary/subject_traces_control.png` and `subject_traces_glaucoma.png`
- Data source: Direct DuckDB queries to `SERI_PLR_GLAUCOMA.db`
- Layout: 6 subjects per figure (vertically stacked), 2 separate figures (control vs glaucoma)

**Existing R figure script (incomplete):**
- Script: `r/figures/fig_subject_traces.R`
- Currently expects pickle file: `data/private/demo_subjects_traces.pkl`
- Has graceful degradation with synthetic demo data
- Outputs to: `figures/generated/ggplot2/supplementary/`

### Target State

Two separate R-generated figures:
1. `fig_subject_traces_control.png` - 4 control subjects (2x2 grid)
2. `fig_subject_traces_glaucoma.png` - 4 glaucoma subjects (2x2 grid)

Both using:
- JSON data source (extracted from DuckDB)
- YAML-driven colors and layout
- `save_publication_figure()` (not `ggsave()`)
- `theme_foundation_plr()` for consistent styling

---

## A. DATA LAYER

### A.1 Data Source

**Primary source:** `/home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db`

**Relevant columns from `train`/`test` tables:**

| Column | Type | Description |
|--------|------|-------------|
| `time` | DOUBLE | Time in seconds (0 to ~66s) |
| `pupil_orig` | DOUBLE | Original pupil measurement (normalized) |
| `pupil_gt` | DOUBLE | Ground truth (human-annotated denoised signal) |
| `outlier_mask` | INTEGER | 1 = outlier, 0 = valid |
| `light_stimuli` | DOUBLE | Light stimulus intensity (for shading) |
| `subject_code` | VARCHAR | PLRxxxx format |
| `class_label` | VARCHAR | 'control', 'glaucoma', or None |
| `Red` | DOUBLE | Red light stimulus channel |
| `Blue` | DOUBLE | Blue light stimulus channel |

**Key statistics:**
- 1981 timepoints per subject
- 355 subjects in train table (106 control, 39 glaucoma with labels)
- Sampling rate: ~30 Hz

### A.2 Demo Subjects Configuration

**Source:** `configs/VISUALIZATION/demo_subjects.yaml`

The demo subjects are categorized by outlier percentage:

| Category | Control Subjects | Glaucoma Subjects |
|----------|------------------|-------------------|
| High outlier | PLR1071 (37.7%), PLR1033 (26.8%) | PLR4018 (26.5%), PLR4030 (22.9%) |
| Average outlier | PLR1063 (5.1%), PLR1088 (4.9%) | PLR4033 (12.7%), PLR4034 (11.1%) |
| Low outlier | PLR1042 (0.9%), PLR1002 (0.0%) | PLR4051 (3.3%), PLR4001 (2.2%) |

### A.3 Data Extraction Strategy

**Recommended approach:** Python extraction to JSON (consistent with other R figures)

**Rationale:**
1. All other R figures use JSON data from `data/r_data/`
2. R's `duckdb` package would work, but mixing data access patterns increases complexity
3. Python extraction allows validation and re-anonymization at extraction time
4. JSON is human-readable for debugging

### A.4 Extraction Script Design

**New script:** `scripts/export_subject_traces_for_r.py`

```python
# Pseudocode structure
def export_subject_traces():
    """Export demo subject trace data to JSON for R visualization."""

    # 1. Load demo_subjects config (get PLRxxxx codes)
    config = load_yaml("configs/VISUALIZATION/demo_subjects.yaml")

    # 2. For each subject, extract from DuckDB:
    #    - time, pupil_orig, pupil_gt, outlier_mask, light_stimuli
    #    - Red, Blue channels for stimulus bars

    # 3. Apply re-anonymization (PLRxxxx -> Hxxx/Gxxx)
    #    using data/private/subject_lookup.yaml

    # 4. Output to data/r_data/subject_traces.json (PUBLIC version)
```

**JSON structure:**
```json
{
    "metadata": {
        "created": "ISO timestamp",
        "generator": "scripts/export_subject_traces_for_r.py",
        "n_subjects": 12,
        "n_timepoints_per_subject": 1981
    },
    "subjects": [
        {
            "subject_id": "H001",
            "class_label": "control",
            "outlier_pct": 37.71,
            "note": "Highest outlier % in controls",
            "n_timepoints": 1981,
            "time": [...],
            "pupil_orig": [...],
            "pupil_gt": [...],
            "outlier_mask": [...],
            "light_stimuli": [...]
        }
    ]
}
```

### A.5 Graceful Degradation

When database or JSON is unavailable:
1. Check for `data/r_data/subject_traces.json`
2. If missing, warn and use synthetic demo data (already implemented in existing R script)
3. Output figure with clear "DEMO DATA" watermark

### A.6 DuckDB Data for Legacy Shiny Apps

**Note:** The legacy R Shiny apps (`inspect_outliers`, `inspect_EMD`) in `src/tools/ground-truth-creation/` could theoretically be updated to read from DuckDB instead of the original file-based data structure. This would require:

1. Modifying the apps to use R's `duckdb` package
2. Recomputing boolean outlier/inlier masks from the stored `outlier_mask` column
3. Loading per-subject time series from the database

This is documented here for future reference, but is **out of scope** for the current manuscript preparation.

---

## B. FIGURE DESIGN

### B.1 Match Existing Matplotlib Style

From the matplotlib output analysis:

1. **Layout**: Vertically stacked panels (one per subject) OR 2x2 grid
2. **X-axis**: Time (s), 0 to ~66s
3. **Y-axis**: "Pupil Size (normalized)"
4. **Signals shown:**
   - Ground Truth (green line, solid)
   - Raw signal (gray line, semi-transparent)
   - Outliers (red points)
5. **Light stimulus**: Yellow vertical bars indicating stimulus periods
6. **Legend**: Per-panel showing line types and outlier percentage

### B.2 Proposed ggplot2 Layout

**Two separate figures** (matching existing matplotlib):
1. `fig_subject_traces_control.png` - 4 control subjects
2. `fig_subject_traces_glaucoma.png` - 4 glaucoma subjects

**Panel layout**: 2x2 grid using `patchwork`
- Row 1: High outlier subjects (2)
- Row 2: Low outlier subjects (2)
- This shows range of signal quality

**Each panel includes:**
- Title: "A  H001 (Control) - 37.7% outliers" format
- Ground truth line (primary signal)
- Raw signal line (semi-transparent background)
- Outlier points (highlighted)
- Light stimulus bars at top (blue then red phases)
- Shared y-axis range for comparability

### B.3 Visual Elements

| Element | Color Source | Description |
|---------|--------------|-------------|
| Ground truth line | `--color-positive` (#379A8B teal) | Main processed signal |
| Raw signal | `--color-text-muted` (#888888) | Semi-transparent raw |
| Outlier points | `--color-negative` (#E3120B red) | Detected outliers |
| Blue stimulus bar | `--color-primary` (#006BA2) | 469nm stimulus |
| Red stimulus bar | `--color-negative` (#E3120B) | 640nm stimulus |
| Background | `--color-background` (#FBF9F3) | Economist off-white |

---

## C. CONFIG UPDATES

### C.1 Updates to `figure_layouts.yaml`

```yaml
fig_subject_traces_control:
  display_name: "Control Subject PLR Traces"
  description: "Shows raw and processed PLR signals for 4 control subjects"
  section: "supplementary"
  latex_label: "fig:subject-traces-control"
  filename: "fig_subject_traces_control"

  privacy_level: "private"  # Contains subject traces
  data_source: "subject_traces.json"

  layout: "2x2"
  tag_levels: "A"

  light_protocol:
    blue_color: "--color-primary"
    red_color: "--color-negative"
    show_stimulus: true

  y_axis:
    fixed: true
    range: [-100, 20]
    label: "Pupil Size (normalized)"

  display:
    show_raw: true
    show_processed: true
    show_outliers: true
    alpha_raw: 0.3
    alpha_processed: 1.0

  dimensions:
    width: 14
    height: 10
    units: "in"

fig_subject_traces_glaucoma:
  # Similar structure for glaucoma subjects
```

### C.2 Updates to `colors.yaml`

No new colors needed. Use existing semantic colors:
- `positive` for ground truth
- `negative` for outliers/red stimulus
- `neutral` for raw signal
- `primary` for blue stimulus

---

## D. IMPLEMENTATION STEPS

### Phase 1: Data Extraction (Python)

**Step 1.1: Create extraction script**
- File: `scripts/export_subject_traces_for_r.py`
- Read from `SERI_PLR_GLAUCOMA.db`
- Query demo subjects from `configs/VISUALIZATION/demo_subjects.yaml`
- Apply re-anonymization
- Output: `data/r_data/subject_traces.json`

**Step 1.2: Add to Makefile**
```makefile
export-subject-traces:
    uv run python scripts/export_subject_traces_for_r.py
```

### Phase 2: Update R Figure Script

**Step 2.1: Modify `r/figures/fig_subject_traces.R` to:**
- Load from JSON instead of pickle
- Accept `--class` argument (control/glaucoma)
- Generate two separate outputs

**Step 2.2: Update data loading function**
```r
load_subject_traces <- function(data_source = "subject_traces.json") {
  json_path <- file.path(PROJECT_ROOT, "data/r_data", data_source)

  if (!file.exists(json_path)) {
    warning("Subject traces JSON not found. Using demo data.")
    return(create_demo_data())
  }

  data <- jsonlite::fromJSON(json_path)
  # Convert to data frame format
  ...
}
```

**Step 2.3: Use color_defs from YAML**
```r
# CORRECT:
geom_line(aes(y = pupil_gt), color = color_defs[["--color-positive"]])

# WRONG (BANNED):
geom_line(aes(y = pupil_gt), color = "#009E73")
```

### Phase 3: Generate Both Figures

**Command-line interface:**
```bash
Rscript r/figures/fig_subject_traces.R --class control
Rscript r/figures/fig_subject_traces.R --class glaucoma
```

### Phase 4: Integrate with Generate Script

In `r/figures/generate_all_r_figures.R`:
```r
# Generate both subject trace figures
system2("Rscript", c("r/figures/fig_subject_traces.R", "--class", "control"))
system2("Rscript", c("r/figures/fig_subject_traces.R", "--class", "glaucoma"))
```

---

## E. TESTING

### E.1 Data Validation

```python
def test_json_schema():
    """Verify subject_traces.json has correct structure."""
    # Check required keys: metadata, subjects
    # Check each subject has required fields
```

### E.2 Hardcoding Tests

```python
def test_no_hardcoded_colors():
    """Verify R script uses color_defs, not hex colors."""
    # Parse R script for patterns like '#[0-9A-Fa-f]{6}'

def test_save_publication_figure_used():
    """Verify ggsave() is not used directly."""
```

### E.3 Data Privacy Tests

```python
def test_no_plr_codes_in_public_json():
    """Verify public JSON uses anonymized IDs only."""
    data = json.load(open("data/r_data/subject_traces.json"))
    for subject in data["subjects"]:
        assert not subject["subject_id"].startswith("PLR")
```

---

## F. DEPENDENCY GRAPH

```
SERI_PLR_GLAUCOMA.db
        │
        ▼
scripts/export_subject_traces_for_r.py
        │
        ├── configs/VISUALIZATION/demo_subjects.yaml
        ├── data/private/subject_lookup.yaml
        │
        ▼
data/r_data/subject_traces.json (anonymized)
        │
        ▼
r/figures/fig_subject_traces.R
        │
        ├── src/r/figure_system/config_loader.R
        ├── src/r/figure_system/save_figure.R
        ├── src/r/theme_foundation_plr.R
        │
        ▼
figures/generated/ggplot2/supplementary/
        ├── fig_subject_traces_control.png
        └── fig_subject_traces_glaucoma.png
```

---

## G. ESTIMATED EFFORT

| Phase | Estimated Time | Dependencies |
|-------|----------------|--------------|
| 1. Data Extraction | 2 hours | None |
| 2. Update R Script | 3 hours | Phase 1 |
| 3. Separate Control/Glaucoma | 1 hour | Phase 2 |
| 4. Testing | 2 hours | Phase 3 |
| 5. Documentation | 1 hour | Phase 4 |

**Total: ~9 hours**

---

## H. CRITICAL RULES (From CLAUDE.md)

1. **NO HARDCODED COLORS** - All colors from `color_defs <- load_color_definitions()`
2. **NO ggsave()** - Use `save_publication_figure()`
3. **NO hardcoded paths** - Use figure system routing
4. **All values from YAML** - Dimensions, colors, subjects all from config files
5. **Graceful degradation** - Demo data fallback when real data unavailable
