# fig-repo-25: Two Featurization Paths: Handcrafted Features vs MOMENT Embeddings

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-25 |
| **Title** | Two Featurization Paths |
| **Complexity Level** | L2-L3 (Architecture) |
| **Target Persona** | ML Engineer, Research Scientist |
| **Location** | docs/user-guide/, ARCHITECTURE.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Show the two alternative featurization paths in the repository: handcrafted physiological features (defined in YAML configs) vs MOMENT embeddings (using representation learning). Focus on CODE and CONFIG, not results.

## Key Message

"The repository supports two featurization methods: handcrafted features (YAML-configured time-window statistics) and MOMENT embeddings (768-d latent representations). Choose based on your use case."

## Code Architecture (Verified from Repository)

### Path 1: Handcrafted Features

**Config location**: `configs/PLR_FEATURIZATION/`

```yaml
# configs/PLR_FEATURIZATION/featuresSimple.yaml
FEATURES_METADATA:
  name: 'simple'
  feature_method: 'handcrafted_features'
FEATURES:
  MAX_CONSTRICTION:
    time_from: 'onset'
    time_start: 0
    time_end: 15
    measure: 'amplitude'
    stat: 'min'
  PHASIC:
    time_from: 'onset'
    time_start: 0
    time_end: 5
    measure: 'amplitude'
    stat: 'min'
  SUSTAINED:
    time_from: 'offset'
    time_start: -5
    time_end: 0
    measure: 'amplitude'
    stat: 'min'
  PIPR_AUC:
    time_from: 'offset'
    time_start: 0
    time_end: 12
    measure: 'amplitude'
    stat: 'AUC'
```

**Code path**:
```
src/featurization/
├── flow_featurization.py           # Prefect flow entry point
├── featurize_PLR.py                 # Main orchestration
│   └── featurize_subject()          # Per-subject extraction
├── featurizer_PLR_subject.py        # Feature computation
│   └── get_features_per_color()     # Per-color feature extraction
└── subflow_handcrafted_featurization.py
```

### Path 2: MOMENT Embeddings

**Config location**: `configs/PLR_EMBEDDING/MOMENT.yaml`

```yaml
# configs/PLR_EMBEDDING/MOMENT.yaml
MOMENT:
  MODEL:
    pretrained_model_name_or_path: 'AutonLab/MOMENT-1-large'
    model_kwargs:
      task_name: "reconstruction"  # or "embedding" for embeddings
  LINEAR_PROBING:
    task_name: "anomaly-detection"
    finetuning_mode: "linear-probing"
```

**Code path**:
```
src/featurization/embedding/
├── moment_embedding.py              # Main embedding extraction
│   └── import_moment_embedder()     # Load MOMENT for embeddings
│   └── get_embeddings_per_split()   # Compute 768-d vectors
└── dim_reduction.py                 # Optional PCA post-processing
```

**MOMENT Reference**: [Representation Learning Tutorial](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb)

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              TWO FEATURIZATION PATHS                                             │
│              configs/PLR_FEATURIZATION/ vs configs/PLR_EMBEDDING/                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                        ┌─────────────────────────────────┐                       │
│                        │  Imputed PLR Signal             │                       │
│                        │  (from Stage 2: Imputation)     │                       │
│                        └───────────────┬─────────────────┘                       │
│                                        │                                         │
│                          ┌─────────────┴─────────────┐                           │
│                          ▼                           ▼                           │
│  ╔═══════════════════════════════════╗  ╔═══════════════════════════════════╗   │
│  ║  PATH 1: HANDCRAFTED FEATURES     ║  ║  PATH 2: MOMENT EMBEDDINGS        ║   │
│  ╠═══════════════════════════════════╣  ╠═══════════════════════════════════╣   │
│  ║                                   ║  ║                                   ║   │
│  ║  📁 Config:                       ║  ║  📁 Config:                       ║   │
│  ║  configs/PLR_FEATURIZATION/       ║  ║  configs/PLR_EMBEDDING/MOMENT.yaml║   │
│  ║  ├── featuresSimple.yaml          ║  ║                                   ║   │
│  ║  └── featuresBaseline.yaml        ║  ║  MODEL:                           ║   │
│  ║                                   ║  ║    pretrained_model_name_or_path: ║   │
│  ║  FEATURES:                        ║  ║      'AutonLab/MOMENT-1-large'    ║   │
│  ║    MAX_CONSTRICTION:              ║  ║    model_kwargs:                  ║   │
│  ║      time_from: 'onset'           ║  ║      task_name: "embedding"       ║   │
│  ║      time_start: 0                ║  ║                                   ║   │
│  ║      time_end: 15                 ║  ║  See MOMENT tutorial:             ║   │
│  ║      measure: 'amplitude'         ║  ║  representation_learning.ipynb    ║   │
│  ║      stat: 'min'                  ║  ║                                   ║   │
│  ║    PIPR_AUC:                      ║  ╟───────────────────────────────────╢   │
│  ║      stat: 'AUC'                  ║  ║  📂 Code:                         ║   │
│  ║      ...                          ║  ║  src/featurization/embedding/     ║   │
│  ╟───────────────────────────────────╢  ║  └── moment_embedding.py          ║   │
│  ║  📂 Code:                         ║  ║      └── get_embeddings_per_split ║   │
│  ║  src/featurization/               ║  ║                                   ║   │
│  ║  └── featurize_PLR.py             ║  ║  Output: 768-d latent vector      ║   │
│  ║      └── featurizer_PLR_subject   ║  ║  (or reduced via PCA)             ║   │
│  ║                                   ║  ║                                   ║   │
│  ║  Output: N features               ║  ╚═══════════════════════════════════╝   │
│  ║  (configured in YAML)             ║                                          │
│  ╚═══════════════════════════════════╝                                          │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  HANDCRAFTED FEATURE TYPES                                                       │
│  ════════════════════════                                                        │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                          │   │
│  │  Feature         │ time_from │ time_start │ time_end │ stat            │   │
│  │  ─────────────────┼───────────┼────────────┼──────────┼─────────────── │   │
│  │  MAX_CONSTRICTION │ onset     │ 0s         │ 15s      │ min amplitude  │   │
│  │  PHASIC          │ onset     │ 0s         │ 5s       │ min amplitude  │   │
│  │  SUSTAINED       │ offset    │ -5s        │ 0s       │ min amplitude  │   │
│  │  PIPR            │ offset    │ 0s         │ 15s      │ min amplitude  │   │
│  │  PIPR_AUC        │ offset    │ 0s         │ 12s      │ AUC            │   │
│  │  BASELINE        │ onset     │ -5s        │ 0s       │ median         │   │
│  │                                                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Features are computed per light color (Blue, Red) → 2× the feature count       │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  MOMENT EMBEDDING MODES                                                          │
│  ══════════════════════                                                          │
│                                                                                  │
│  From https://github.com/moment-timeseries-foundation-model/moment:             │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                            │ │
│  │  task_name         │ Use case                    │ Output dimension       │ │
│  │  ──────────────────┼─────────────────────────────┼────────────────────── │ │
│  │  "embedding"       │ Representation learning     │ 768-d (large model)    │ │
│  │  "reconstruction"  │ Imputation/anomaly          │ Reconstructed signal   │ │
│  │  "forecasting"     │ Future prediction           │ Forecasted values      │ │
│  │                                                                            │ │
│  │  Our repo uses "embedding" mode for featurization via:                     │ │
│  │  import_moment_embedder() → model(x_enc=x) → outputs.embeddings            │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Optional: PCA dimensionality reduction (dim_reduction.py)                       │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  CHOOSING A PATH                                                                 │
│  ═══════════════                                                                 │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                            │ │
│  │  Choose HANDCRAFTED when:        │  Choose EMBEDDINGS when:               │ │
│  │  ─────────────────────────────────┼────────────────────────────────────── │ │
│  │  • Domain knowledge available     │  • Exploring new signals              │ │
│  │  • Interpretability required      │  • Transfer learning scenario         │ │
│  │  • Small sample size (N<500)      │  • Large-scale datasets               │ │
│  │  • Regulatory/clinical context    │  • Rapid prototyping                  │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Note: Results comparison is in the manuscript, not this repository.            │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Branching diagram**: PLR signal → two featurization paths
2. **Path 1 details**: Config location, YAML structure, code path
3. **Path 2 details**: MOMENT config, task_name modes, code path
4. **Feature types table**: Time-window based handcrafted features
5. **MOMENT modes table**: embedding vs reconstruction vs forecasting
6. **Decision guide**: When to choose each path

## Text Content

### Title Text
"Two Featurization Paths: Handcrafted Features vs MOMENT Embeddings"

### Caption
The repository supports two featurization paths. Path 1 (Handcrafted): Configure time-window statistics in `configs/PLR_FEATURIZATION/*.yaml` (MAX_CONSTRICTION, PHASIC, SUSTAINED, PIPR_AUC). Path 2 (Embeddings): Use MOMENT foundation model in "embedding" mode via `configs/PLR_EMBEDDING/MOMENT.yaml` to extract 768-dimensional latent representations. See MOMENT's [representation_learning.ipynb](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb) for details.

## Sources

- [MOMENT Repository](https://github.com/moment-timeseries-foundation-model/moment)
- [MOMENT Representation Learning Tutorial](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb)
- `configs/PLR_FEATURIZATION/featuresSimple.yaml`
- `configs/PLR_EMBEDDING/MOMENT.yaml`
- `src/featurization/featurize_PLR.py`
- `src/featurization/embedding/moment_embedding.py`

## Prompts for Nano Banana Pro

### Style Prompt
Architecture diagram with two parallel paths. Config files shown as code blocks. Directory tree structures. Tables for feature definitions and MOMENT modes. Decision matrix at bottom. Technical documentation style, no results/metrics.

### Content Prompt
Create a two-path featurization diagram:

**TOP - Branching Point**:
- Input: "Imputed PLR Signal"
- Two arrows leading to Path 1 and Path 2

**LEFT - Path 1 (Handcrafted)**:
- Config: configs/PLR_FEATURIZATION/
- YAML snippet showing feature definitions
- Code: src/featurization/featurize_PLR.py

**RIGHT - Path 2 (Embeddings)**:
- Config: configs/PLR_EMBEDDING/MOMENT.yaml
- MOMENT task_name: "embedding"
- Code: src/featurization/embedding/moment_embedding.py
- Link to MOMENT repo tutorial

**MIDDLE - Feature Types Table**:
- Handcrafted features: time_from, time_start, time_end, stat

**BOTTOM - MOMENT Modes Table**:
- embedding, reconstruction, forecasting

**FOOTER - Decision Guide**:
- When to choose each path (no performance metrics)

## Alt Text

Architecture diagram showing two featurization paths. Path 1 (Handcrafted): Config at configs/PLR_FEATURIZATION with YAML-defined time-window features (MAX_CONSTRICTION, PHASIC, SUSTAINED, PIPR_AUC), code at src/featurization/featurize_PLR.py. Path 2 (Embeddings): Config at configs/PLR_EMBEDDING/MOMENT.yaml using task_name "embedding", code at src/featurization/embedding/moment_embedding.py producing 768-d vectors. Tables show feature definitions and MOMENT modes. Decision guide at bottom explains when to use each path.

## Status

- [x] Draft created
- [x] Updated to focus on code/config, not results
- [ ] Generated
- [ ] Placed in docs/user-guide/
