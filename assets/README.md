# Assets (`assets/`)

Static image files used in documentation and the main README.

## Contents

| File | Description |
|------|-------------|
| `mlflow_ui.png` | MLflow tracking UI screenshot |
| `mlflow_artifacts.png` | MLflow artifact store screenshot |
| `mlflow_classification.png` | Classification results in MLflow |
| `mlflow_featurization.png` | Featurization results in MLflow |
| `mlflow_imputation.png` | Imputation results in MLflow |
| `mlflow_outlier_detection.png` | Outlier detection results in MLflow |
| `prefect_flow.png` | Prefect flow diagram |
| `grid_search_init.png` | Hyperparameter grid search visualization |
| `PCA_exp.png` | PCA exploration plot |
| `tabfpn_v1.png` | TabPFN v1 architecture |
| `tabfpn_v1_vs_v2_below.png` | TabPFN version comparison |
| `units*.png` | UniTS model architecture diagrams |

## Usage

Reference images in Markdown:

```markdown
![MLflow UI](assets/mlflow_ui.png)
```

## Adding New Assets

1. Place image in `assets/`
2. Use descriptive filename (lowercase, underscores)
3. Optimize file size if large
4. Add to this README

## Image Guidelines

- **Format**: PNG for screenshots, SVG for diagrams
- **Size**: Keep under 500KB per image
- **Naming**: `category_description.png` (e.g., `mlflow_artifacts.png`)
- **Quality**: Ensure readable at documentation scale

## Note

For **generated figures**, use `figures/generated/` instead. This directory is for:
- Screenshots
- Architecture diagrams
- External references
- Documentation images
