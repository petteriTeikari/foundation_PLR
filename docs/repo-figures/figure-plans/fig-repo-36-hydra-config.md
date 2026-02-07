# fig-repo-36: Hydra Configuration System

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-36 |
| **Title** | Hydra Configuration System |
| **Complexity Level** | L2 (Technical) |
| **Target Persona** | ML Engineer, Software Engineer |
| **Location** | configs/README.md, ARCHITECTURE.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain how Hydra composes configurations from YAML files and enables parameter overrides.

## Key Message

"Hydra loads defaults.yaml and composes it with domain-specific configs. Override any parameter from command line: `python script.py +bootstrap.n_iterations=2000`"

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    HYDRA CONFIGURATION SYSTEM                                   │
│                    One source of truth for all parameters                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS HYDRA?                                                                 │
│  ══════════════                                                                 │
│                                                                                 │
│  A configuration framework that:                                                │
│  • Loads YAML files hierarchically                                              │
│  • Composes configs from multiple files                                         │
│  • Allows command-line overrides                                                │
│  • Prevents hardcoded values in code                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CONFIGURATION HIERARCHY                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  configs/                                                               │   │
│  │  ├── defaults.yaml              ← BASE configuration (loaded first)    │   │
│  │  │   ├── CLS_EVALUATION:                                               │   │
│  │  │   │   ├── glaucoma_params:                                          │   │
│  │  │   │   │   └── prevalence: 0.0354                                    │   │
│  │  │   │   └── BOOTSTRAP:                                                │   │
│  │  │   │       └── n_iterations: 1000                                    │   │
│  │  │   └── VISUALIZATION:                                                │   │
│  │  │       └── dpi: 100                                                  │   │
│  │  │                                                                     │   │
│  │  ├── VISUALIZATION/             ← Domain-specific configs              │   │
│  │  │   ├── figure_registry.yaml                                          │   │
│  │  │   └── plot_hyperparam_combos.yaml                                   │   │
│  │  │                                                                     │   │
│  │  └── mlflow_registry/           ← Method name registry                 │   │
│  │      └── parameters/                                                   │   │
│  │          └── classification.yaml                                       │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  HOW COMPOSITION WORKS                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                     │
│  │               │   │               │   │               │                     │
│  │  defaults.yaml│ + │  domain.yaml  │ + │  CLI override │  = Final Config    │
│  │  (base)       │   │  (extends)    │   │  (runtime)    │                     │
│  │               │   │               │   │               │                     │
│  └───────────────┘   └───────────────┘   └───────────────┘                     │
│                                                                                 │
│  Example:                                                                       │
│  defaults.yaml:     dpi: 100                                                    │
│  figure.yaml:       dpi: 150        (overrides default)                         │
│  CLI:               +dpi=300        (overrides everything)                      │
│  Final:             dpi: 300                                                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CODE PATTERN                                                                   │
│  ════════════                                                                   │
│                                                                                 │
│  # In Python script                                                             │
│  import hydra                                                                   │
│  from omegaconf import DictConfig                                               │
│                                                                                 │
│  @hydra.main(config_path="../configs", config_name="defaults")                  │
│  def main(cfg: DictConfig):                                                     │
│      # Access nested config values                                              │
│      n_bootstrap = cfg.CLS_EVALUATION.BOOTSTRAP.n_iterations  # 1000            │
│      prevalence = cfg.CLS_EVALUATION.glaucoma_params.prevalence  # 0.0354       │
│      dpi = cfg.VISUALIZATION.dpi  # 100                                         │
│                                                                                 │
│  # Command-line override                                                        │
│  $ python script.py CLS_EVALUATION.BOOTSTRAP.n_iterations=2000                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KEY CONFIG VALUES                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  │ Path                                        │ Value   │ Description         │ │
│  │ ─────────────────────────────────────────── │ ─────── │ ─────────────────── │ │
│  │ CLS_EVALUATION.glaucoma_params.prevalence   │ 0.0354  │ Disease prevalence  │ │
│  │ CLS_EVALUATION.BOOTSTRAP.n_iterations       │ 1000    │ Bootstrap samples   │ │
│  │ CLS_EVALUATION.BOOTSTRAP.alpha_CI           │ 0.95    │ CI level            │ │
│  │ VISUALIZATION.dpi                           │ 100     │ Figure resolution   │ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY NOT HARDCODE?                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  ❌ Hardcoded in script:                                                        │
│     n_iterations = 1000  # Where is this documented? How to change?             │
│                                                                                 │
│  ✅ Hydra config:                                                               │
│     n_iterations = cfg.CLS_EVALUATION.BOOTSTRAP.n_iterations                    │
│     # Documented in YAML, override via CLI, tracked in git                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **What is Hydra**: Brief explanation
2. **Hierarchy diagram**: configs/ folder structure
3. **Composition flow**: defaults + domain + CLI = final
4. **Code pattern**: Python @hydra.main example
5. **Key config values**: Table of important parameters
6. **Why not hardcode**: Comparison

## Text Content

### Title Text
"Hydra Configuration: Compose, Override, Track"

### Caption
Hydra loads configuration hierarchically: defaults.yaml provides base values, domain configs (VISUALIZATION/, mlflow_registry/) extend it, and CLI arguments override everything. Access config values via `cfg.SECTION.key`. Key parameters (prevalence, n_iterations, dpi) are documented in YAML, overridable at runtime, and tracked in git—never hardcoded in scripts.

## Prompts for Nano Banana Pro

### Style Prompt
Configuration hierarchy diagram with folder tree. Composition flow showing YAML + overrides = final. Code snippet example. Key values table. Clean, technical documentation aesthetic.

### Content Prompt
Create a Hydra config diagram:

**TOP - What is Hydra**:
- 4 bullet points

**MIDDLE LEFT - Hierarchy**:
- Folder tree: configs/ → defaults.yaml, VISUALIZATION/, mlflow_registry/

**MIDDLE RIGHT - Composition**:
- Flow: defaults + domain + CLI = Final

**BOTTOM LEFT - Code Pattern**:
- Python snippet with @hydra.main

**BOTTOM RIGHT - Key Values Table**:
- 4 important config paths and values

## Alt Text

Hydra configuration system diagram. Folder hierarchy: configs/defaults.yaml (base), VISUALIZATION/ (domain), mlflow_registry/ (method names). Composition: defaults.yaml + domain.yaml + CLI override = final config. Code pattern uses @hydra.main decorator to access cfg.SECTION.key. Key values table: prevalence 0.0354, n_iterations 1000, alpha_CI 0.95, dpi 100. Shows why configs are better than hardcoded values.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in configs/README.md
