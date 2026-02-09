# Notebook Extensions

Community-contributed analysis notebooks for the Foundation PLR project.

## How to Contribute

1. **Copy the template**: `cp _template.qmd your_analysis.qmd`
2. **Write your logic** in `src/stats/` or `src/viz/` (not in the notebook)
3. **Write unit tests** in `tests/unit/`
4. **Import and use** your function in the `.qmd` notebook
5. **Render locally**: `quarto render your_analysis.qmd`
6. **Submit a PR** -- CI will validate the notebook renders without errors

## Rules

| Rule | Enforcement |
|------|-------------|
| `.qmd` files only | Pre-commit hook rejects `.ipynb` |
| No heavy logic in notebooks | Code review |
| Read from DuckDB only | Pre-commit bans `sklearn.metrics` imports |
| No hardcoded colors/paths | Pre-commit pattern check |
| Must render in CI | `quarto render` in GitHub Actions |

## Available Data

All notebooks should read from `../data/public/foundation_plr_results.db`:

| Table | Description | Rows |
|-------|-------------|------|
| `essential_metrics` | AUROC, calibration, Brier, net benefit per config | 406 |
| `predictions` | Per-subject predictions (y_true, y_prob) | 25,578 |
| `calibration_curves` | Pre-computed calibration curve bins | 4,060 |
| `dca_curves` | Decision curve analysis data | 20,300 |
| `distribution_stats` | Probability distribution summaries | 406 |
| `retention_metrics` | Selective classification curves | 29,792 |
