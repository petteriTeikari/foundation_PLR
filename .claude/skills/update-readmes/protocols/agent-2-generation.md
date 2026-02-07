# Agent 2: Generation & Update

## Mandate

For each issue from Agent 1, generate or update content. Group changes by category for batch approval.

## Generation Rules

### README Updates

#### Fixing Broken Links

```markdown
<!-- BEFORE (broken) -->
[calibration module](src/viz/calibration.py)

<!-- AFTER (fixed) -->
[calibration module](src/viz/calibration_plot.py)
```

- Resolve the correct target by searching for the closest matching filename
- If multiple candidates exist, prefer the one in the same directory
- If the target was truly deleted, remove the link and note what replaced it

#### Regenerating Config Tables

For config directories, generate tables from actual YAML:

```python
import yaml
from pathlib import Path

config_dir = Path("configs/CLS_HYPERPARAMS")
for yaml_file in sorted(config_dir.glob("*.yaml")):
    cfg = yaml.safe_load(yaml_file.read_text())
    # Extract parameter name, type, default, description
    # Format as markdown table
```

Table format:
```markdown
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | int | 6 | Tree depth |
```

- ALWAYS parse actual YAML, never write from memory
- Include the "last verified" timestamp at the bottom
- Link to the actual YAML file for full details

#### Adding Missing Module Entries

For new modules not in a directory's README:

```markdown
## Modules

| Module | Purpose |
|--------|---------|
| `existing_module.py` | Existing description |
| `new_module.py` | Brief description from module docstring or first class/function |
```

- Extract the module-level docstring for the description
- If no docstring, use the first class or function name as a hint
- Mark with `<!-- NEEDS REVIEW -->` if description is inferred

### Docstring Generation

Follow NumPy style (see `reference/docstring-standards.md`):

```python
def compute_retention_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    retention_rates: list[float] | None = None,
) -> pd.DataFrame:
    """Compute performance metrics at different retention rates.

    Evaluates model performance when only the most confident predictions
    are retained (selective classification).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    retention_rates : list[float] or None, optional
        Fraction of predictions to retain, from 0.0 to 1.0.
        If None, uses 20 evenly spaced points from 0.05 to 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: retention_rate, metric_name, metric_value.

    Raises
    ------
    ValueError
        If y_true and y_prob have different lengths.

    See Also
    --------
    src.viz.retained_metric : Visualization of retention curves.
    """
```

Rules:
- Match actual function signature EXACTLY (use AST, not guessing)
- Include type annotations from source code
- Write descriptions that explain WHAT and WHY, not HOW
- Include `See Also` for related functions in other modules
- Include `Raises` only if the function explicitly raises exceptions
- NEVER add docstrings to private functions (except `__init__`)
- NEVER add docstrings to test files

### Figure Plan Generation

For identified gaps, create plan files following `docs/repo-figures/CONTENT-TEMPLATE.md`:

1. Use sequential numbering from the highest existing plan number
2. Follow the `STYLE-GUIDE.md` visual consistency rules
3. Include:
   - Config file locations
   - Code paths
   - Extension guide (how to add a new variant)
4. NEVER include results, performance numbers, or comparisons
5. Add to the catalog table in `docs/repo-figures/README.md`

## Change Grouping

Group changes for human review:

```
Category 1: Broken Link Fixes (3 changes)
  - src/viz/README.md:15 - fix calibration link
  - configs/README.md:42 - fix dead config ref
  - README.md:28 - update CONTRIBUTING path

Category 2: Config Table Updates (5 changes)
  - configs/CLS_HYPERPARAMS/README.md - regenerate from YAML
  - configs/OUTLIER_MODELS/README.md - add new method entry
  ...

Category 3: Missing Module Documentation (8 changes)
  - src/data_io/README.md - add streaming_duckdb_export.py
  ...

Category 4: Docstring Additions (42 changes)
  - P0: src/data_io/registry.py - 3 functions
  - P0: src/viz/plot_config.py - 5 functions
  ...

Category 5: Figure Plans (4 new plans)
  - fig-repo-57: DuckDB table architecture
  ...
```

## Quality Checks Before Presenting

For each change:
1. Does the new content match actual code/config? (Parse, don't guess)
2. Are all internal links valid after the change?
3. Do docstrings match actual function signatures?
4. Do config tables match actual YAML contents?
5. Are figure plans code-architecture-only (no results)?

## Git Checkpoint Pattern

After human approval of each category:

```bash
git add <files in category>
git commit -m "docs(readmes): <category description>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

Use conventional commit prefixes:
- `docs(readmes):` for README updates
- `docs(docstrings):` for docstring additions
- `docs(figures):` for figure plan creation
