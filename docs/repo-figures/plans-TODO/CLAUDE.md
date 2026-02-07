# Plans-TODO - Claude Instructions

## Purpose

This directory contains figure plans that document **config directories** and **system architecture**.

## Allowed Content

Unlike standard figure plans (which document results figures), plans in this directory MAY include:

### Code/Config Documentation

- File paths and directory structures
- Config composition patterns
- Hydra override examples
- YAML snippets showing structure
- How to extend/add new methods

### Methodological Context

- Brief theory WITHOUT results (e.g., "PCA finds orthogonal components" not "PCA achieved 0.85")
- Algorithm descriptions WITHOUT performance comparisons
- Mathematical formulations WITHOUT "which is best" claims
- Literature references for background

## STILL FORBIDDEN

Even in this subdirectory, these are BANNED:

| BANNED | Example |
|--------|---------|
| Performance metrics | "AUROC = 0.91", "F1 = 0.85" |
| Rankings | "Method X beats Y", "Top 3 methods" |
| Comparisons | "Foundation models outperform traditional" |
| Optimal values | "Best depth is 3", "Optimal learning rate" |

## Naming Convention

Plans here use the format: `fig-repo-{NN}-{descriptive-name}.md`

Current plans in this directory:
- fig-repo-51: Classifier Configuration Architecture
- fig-repo-52: Classifier Paradigms
- fig-repo-53: Outlier Detection Methods
- fig-repo-54: Imputation Model Landscape
- fig-repo-55: Registry as Single Source of Truth
- fig-repo-56: Experiment Configuration Hierarchy

## Relationship to READMEs

These figure plans create infographics that are LINKED FROM:
- `configs/CLS_HYPERPARAMS/README.md` → fig-repo-51
- `configs/CLS_MODELS/README.md` → fig-repo-51, fig-repo-52
- `configs/OUTLIER_MODELS/README.md` → fig-repo-53
- `configs/MODELS/README.md` → fig-repo-54
- `configs/mlflow_registry/README.md` → fig-repo-55
- `configs/experiment/README.md` → fig-repo-56

The infographic provides the 5-second overview; the README provides the 2-minute tutorial.
