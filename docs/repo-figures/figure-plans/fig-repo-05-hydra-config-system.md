# fig-repo-05: Hydra Configuration System

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-05 |
| **Title** | Hydra: Flexible Experiment Configuration |
| **Complexity Level** | L2-L3 (Process overview with code) |
| **Target Persona** | Research Scientist, ML Engineer |
| **Location** | configs/README.md, docs/ |
| **Priority** | P2 (High) |

## Purpose

Explain how Hydra allows running different experiment configurations without changing code, making factorial experiments easy.

## Key Message

"Change experiment settings in YAML files, not in code - and run all combinations automatically."

## Visual Concept

**Configuration composition diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    configs/                                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                   │
│  │defaults.  │  │VISUALIZATION│ │mlflow_    │                   │
│  │yaml       │  │/           │  │registry/  │                   │
│  │           │  │colors.yaml │  │parameters/│                   │
│  │ bootstrap:│  │            │  │           │                   │
│  │   n: 1000 │  │ primary:   │  │ outliers: │                   │
│  │   alpha:  │  │   #006BA2  │  │   - LOF   │                   │
│  │     0.95  │  │            │  │   - MOMENT│                   │
│  └───────────┘  └───────────┘  └───────────┘                   │
│         │              │              │                         │
│         └──────────────┼──────────────┘                         │
│                        ▼                                        │
│              ┌─────────────────┐                               │
│              │  HYDRA COMPOSES │                               │
│              │  into single    │                               │
│              │  config object  │                               │
│              └─────────────────┘                               │
│                        │                                        │
│                        ▼                                        │
│    ┌────────────────────────────────────────────────────────┐  │
│    │  python run.py outlier=LOF imputation=SAITS            │  │
│    │  python run.py outlier=MOMENT imputation=CSDI          │  │
│    │  python run.py --multirun outlier=LOF,MOMENT           │  │
│    └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. Multiple YAML file icons
2. Composition arrow showing merge
3. Command line examples
4. --multirun capability highlighted

### Optional Elements
1. Override syntax examples
2. Config groups explanation
3. Sweep visualization

## Text Content

### Title Text
"Hydra: One Codebase, Many Experiments"

### Labels/Annotations
- YAML files: "Settings live in YAML, not code"
- Composition: "Hydra combines them at runtime"
- Command line: "Override anything from command line"
- Multirun: "--multirun: Run all combinations automatically"

### Caption (for embedding)
Hydra configuration system allows experiment settings to be defined in YAML files and combined at runtime, enabling factorial experiments without code changes.

## Prompts for Nano Banana Pro

### Style Prompt
Technical diagram with developer aesthetic. Show file icons merging into a configuration object. Include code snippets in monospace. Blue/gray professional palette. Clean arrows showing data flow.

### Content Prompt
Create a diagram showing:
1. TOP: Multiple YAML files (configs/defaults.yaml, configs/VISUALIZATION/, configs/mlflow_registry/) as file icons
2. MIDDLE: These files flow into a "Hydra Composes" processor
3. BOTTOM: Command line examples showing how to override and run combinations

Highlight the "--multirun" feature as the key capability.

### Refinement Notes
- Show realistic YAML snippets (indentation matters)
- Make the composition concept visually clear
- The command line should look like a real terminal

## Alt Text

Diagram showing how Hydra combines multiple YAML configuration files into a single config object that can be overridden from the command line for factorial experiments.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
