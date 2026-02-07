# fig-repo-43: Hierarchical Experiment Configuration System

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-43 |
| **Title** | Hierarchical Experiment Configuration System |
| **Complexity Level** | L3 (Technical) |
| **Target Persona** | ML Engineer, Research Scientist |
| **Location** | configs/README.md, ARCHITECTURE.md |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:10 |

## Purpose

Show how the experiment configuration system uses Hydra composition to assemble complete experiment specifications from reusable sub-configs, enabling reproducible experiments with Pydantic validation.

## Key Message

"One experiment file composes from data, combos, subjects, figures, and MLflow configs. Pydantic validates at load time. Create new experiments with `make new-experiment`."

## Visual Concept

A hierarchical composition diagram showing:
1. Top: Single experiment file (paper_2026.yaml)
2. Middle: Hydra defaults pointing to 5 sub-config categories
3. Bottom: Each category resolving to specific YAML files
4. Side: Pydantic validation layer

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               HIERARCHICAL EXPERIMENT CONFIGURATION SYSTEM                       │
│               Composable, validated, reproducible experiment specs               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  THE COMPOSITION PATTERN                                                         │
│  ═══════════════════════                                                         │
│                                                                                  │
│                     ┌────────────────────────────────────────┐                   │
│                     │   configs/experiment/paper_2026.yaml   │                   │
│                     │   ════════════════════════════════════  │                   │
│                     │   defaults:                            │                   │
│                     │     - /data: seri_plr_2026            │                   │
│                     │     - /combos: paper_2026             │                   │
│                     │     - /subjects: demo_8_subjects      │                   │
│                     │     - /figures_config: publication    │                   │
│                     │     - /mlflow_config: production      │                   │
│                     │                                        │                   │
│                     │   experiment:                          │                   │
│                     │     name: "Foundation PLR Paper 2026"  │                   │
│                     │     version: "1.0.0"                   │                   │
│                     │     frozen: true                       │                   │
│                     └────────────────────────────────────────┘                   │
│                                      │                                           │
│            ┌─────────────┬──────────┬┴──────────┬─────────────┐                 │
│            ▼             ▼          ▼           ▼             ▼                 │
│     ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│     │   data/  │  │ combos/  │  │subjects/│  │figures_ │  │mlflow_  │          │
│     │          │  │          │  │         │  │ config/ │  │ config/ │          │
│     └────┬─────┘  └────┬─────┘  └────┬────┘  └────┬────┘  └────┬────┘          │
│          │             │             │            │            │                 │
│     ┌────▼─────┐  ┌────▼─────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐          │
│     │seri_plr_ │  │paper_    │  │demo_8_  │  │publica- │  │produc-  │          │
│     │2026.yaml │  │2026.yaml │  │subjects │  │tion.yaml│  │tion.yaml│          │
│     └──────────┘  └──────────┘  └─────────┘  └─────────┘  └─────────┘          │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  DIRECTORY STRUCTURE                                                             │
│  ═══════════════════                                                             │
│                                                                                  │
│  configs/                                                                        │
│  ├── experiment/              ← Entry point configs                              │
│  │   ├── paper_2026.yaml      (frozen: true)                                    │
│  │   └── synthetic.yaml       (for CI testing)                                  │
│  │                                                                               │
│  ├── data/                    ← Data source definitions                          │
│  │   ├── seri_plr_2026.yaml   (507 subjects)                                    │
│  │   └── synthetic_small.yaml (4 subjects)                                      │
│  │                                                                               │
│  ├── combos/                  ← Hyperparameter combinations                      │
│  │   └── paper_2026.yaml      (4 standard + 5 extended)                         │
│  │                                                                               │
│  ├── subjects/                ← Demo subject selections                          │
│  │   └── demo_8_subjects.yaml (4 control + 4 glaucoma)                          │
│  │                                                                               │
│  ├── figures_config/          ← Figure styling                                   │
│  │   ├── publication_ready.yaml (300 DPI, vector)                               │
│  │   └── draft_quality.yaml     (100 DPI, fast)                                 │
│  │                                                                               │
│  └── mlflow_config/           ← Tracking settings                                │
│      ├── production.yaml      (remote tracking)                                 │
│      └── local_testing.yaml   (local mlruns/)                                   │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PYDANTIC VALIDATION LAYER                                                       │
│  ═════════════════════════                                                       │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │  from src.config import load_experiment, ExperimentConfig             │       │
│  │                                                                       │       │
│  │  # Load with validation                                               │       │
│  │  cfg = load_experiment("paper_2026")                                  │       │
│  │                                                                       │       │
│  │  # Type-safe access                                                   │       │
│  │  cfg.name          # "Foundation PLR Paper 2026"                      │       │
│  │  cfg.version       # "1.0.0"                                          │       │
│  │  cfg.is_frozen     # True                                             │       │
│  │  cfg.data.source   # "SERI_PLR_GLAUCOMA.db"                           │       │
│  │                                                                       │       │
│  │  # Validation fails fast on invalid configs                           │       │
│  │  # - Missing required fields                                          │       │
│  │  # - Invalid types                                                    │       │
│  │  # - Unknown experiment names                                         │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CREATING NEW EXPERIMENTS                                                        │
│  ════════════════════════                                                        │
│                                                                                  │
│  # Interactive creation                                                          │
│  $ make new-experiment                                                           │
│                                                                                  │
│  # List available experiments                                                    │
│  $ make list-experiments                                                         │
│    paper_2026                                                                    │
│    synthetic                                                                     │
│                                                                                  │
│  # Validate all experiment configs                                               │
│  $ make validate-experiments                                                     │
│    Validating paper_2026.yaml... OK                                             │
│    Validating synthetic.yaml... OK                                              │
│                                                                                  │
│  # Run with specific experiment                                                  │
│  $ make run-experiment EXP=synthetic                                            │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FROZEN vs UNFROZEN EXPERIMENTS                                                  │
│  ══════════════════════════════                                                  │
│                                                                                  │
│  │ Experiment   │ frozen │ Purpose                                       │      │
│  │ ──────────── │ ────── │ ───────────────────────────────────────────── │      │
│  │ paper_2026   │ true   │ Publication results - DO NOT MODIFY           │      │
│  │ synthetic    │ false  │ CI testing - can be modified                  │      │
│                                                                                  │
│  Frozen experiments:                                                             │
│  • Generate warnings if modified                                                 │
│  • Ensure reproducibility                                                        │
│  • Linked to publication DOI                                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Top-to-bottom tree composition with side panels"
spatial_anchors:
  title:
    x: 0.5
    y: 0.05
    content: "HIERARCHICAL EXPERIMENT CONFIGURATION SYSTEM"
  composition_pattern:
    x: 0.5
    y: 0.25
    content: "Paper_2026.yaml with defaults pointing to sub-configs"
  directory_tree:
    x: 0.25
    y: 0.55
    content: "configs/ folder structure"
  pydantic_layer:
    x: 0.75
    y: 0.55
    content: "Python validation code example"
  make_commands:
    x: 0.25
    y: 0.85
    content: "make new-experiment, list-experiments"
  frozen_table:
    x: 0.75
    y: 0.85
    content: "Frozen vs unfrozen table"
```

## Content Elements

### Key Structures

| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Experiment file | `primary_pathway` | Entry point YAML (paper_2026.yaml) |
| Defaults list | `highlight_accent` | Hydra composition directives |
| Sub-config dirs | `secondary_pathway` | data/, combos/, subjects/ etc. |
| Specific files | `traditional_method` | Individual YAML files |
| Pydantic models | `foundation_model` | Python validation layer |
| Make commands | `callout_box` | CLI interface |

### Relationships/Connections

| From | To | Type | Label |
|------|-----|------|-------|
| Experiment file | Defaults | Contains | "defaults:" |
| Defaults | Sub-configs | Arrow | "references" |
| Sub-configs | YAML files | Arrow | "resolves to" |
| Load function | Pydantic | Arrow | "validates" |

### Callout Boxes

| Title | Content | Location |
|-------|---------|----------|
| "FROZEN" | Publication configs cannot be modified | Top right |
| "VALIDATED" | Pydantic catches errors at load time | Middle right |
| "COMPOSABLE" | Mix and match sub-configs freely | Bottom left |

## Text Content

### Labels (Max 30 chars each)

- Label 1: Experiment entry point
- Label 2: Hydra composition
- Label 3: Sub-config categories
- Label 4: Pydantic validation
- Label 5: Make commands
- Label 6: Frozen vs unfrozen

### Caption (for embedding)

The experiment configuration system uses Hydra composition to assemble complete experiment specifications from reusable sub-configs. A single experiment file (paper_2026.yaml) references data, combos, subjects, figures, and MLflow configs through Hydra defaults. Pydantic validates configurations at load time, catching errors early. Frozen experiments ensure reproducibility for published results.

## Prompts for Nano Banana Pro

### Style Prompt

Technical documentation infographic with clean folder tree hierarchy. Soft blue (#2E5B8C) for primary elements, gold (#D4A03C) for highlights. Warm off-white background (#FBF9F3). Clear typography placeholders. Mermaid-style flowing arrows connecting composition levels. Professional magazine quality.

### Content Prompt

Create a hierarchical configuration diagram:

**TOP - Title bar**: "HIERARCHICAL EXPERIMENT CONFIGURATION SYSTEM"

**UPPER MIDDLE - Composition Pattern**:
- Large YAML file box showing paper_2026.yaml
- defaults: list with 5 references
- Arrows flowing down to 5 category boxes

**LOWER LEFT - Directory Structure**:
- Folder tree: configs/ → experiment/, data/, combos/, subjects/, figures_config/, mlflow_config/
- Specific files under each

**LOWER RIGHT - Pydantic Layer**:
- Python code snippet showing load_experiment()
- Type-safe access examples

**BOTTOM LEFT - Make Commands**:
- Terminal commands: make new-experiment, list-experiments, validate-experiments

**BOTTOM RIGHT - Frozen Table**:
- Two-row table: paper_2026 (frozen: true), synthetic (frozen: false)

Economist off-white background (#FBF9F3). Clean sans-serif typography placeholders.

### Refinement Notes

- Ensure arrows clearly show composition flow (top-to-bottom)
- Directory tree should be readable at small sizes
- Code snippets should have syntax highlighting placeholders

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-43",
    "title": "Hierarchical Experiment Configuration System"
  },
  "content_architecture": {
    "primary_message": "One experiment file composes from reusable sub-configs with Pydantic validation",
    "layout_flow": "Top-to-bottom composition tree with side panels",
    "spatial_anchors": {
      "title": {"x": 0.5, "y": 0.05},
      "composition": {"x": 0.5, "y": 0.25},
      "directory": {"x": 0.25, "y": 0.55},
      "pydantic": {"x": 0.75, "y": 0.55},
      "commands": {"x": 0.25, "y": 0.85},
      "frozen": {"x": 0.75, "y": 0.85}
    },
    "key_structures": [
      {
        "name": "paper_2026.yaml",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Experiment entry point"]
      },
      {
        "name": "Hydra defaults",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["5 sub-config refs"]
      },
      {
        "name": "Sub-config categories",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["data/", "combos/", "subjects/", "figures_config/", "mlflow_config/"]
      },
      {
        "name": "Pydantic ExperimentConfig",
        "role": "foundation_model",
        "is_highlighted": true,
        "labels": ["Validates at load"]
      },
      {
        "name": "Make commands",
        "role": "callout_box",
        "is_highlighted": false,
        "labels": ["new-experiment", "list-experiments", "validate-experiments"]
      }
    ],
    "callout_boxes": [
      {"heading": "FROZEN", "body_text": "Publication configs locked for reproducibility."},
      {"heading": "VALIDATED", "body_text": "Pydantic catches config errors at load time."},
      {"heading": "COMPOSABLE", "body_text": "Mix and match sub-configs freely."}
    ]
  }
}
```

## Alt Text

Hierarchical experiment configuration system diagram. Top shows paper_2026.yaml with Hydra defaults referencing 5 categories: data, combos, subjects, figures_config, mlflow_config. Each category resolves to specific YAML files. Lower left shows configs/ directory tree. Lower right shows Pydantic validation code with load_experiment function. Bottom shows make commands for creating and validating experiments. Table shows frozen (paper_2026) vs unfrozen (synthetic) experiments.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in configs/README.md
