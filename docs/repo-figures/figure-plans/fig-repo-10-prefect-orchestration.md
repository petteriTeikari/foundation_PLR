# fig-repo-10: Prefect Experiment Pipeline: 6 Subflows with Labor Division

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-10 |
| **Title** | Prefect Experiment Pipeline: 6 Subflows |
| **Complexity Level** | L3-L4 (Architecture) |
| **Target Persona** | ML Engineer, Team Lead |
| **Location** | docs/user-guide/prefect-blocks.md, ARCHITECTURE.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Show the 6 Prefect subflows that run the actual experiments, with labor division by professional persona. This is distinct from the 2-block post-experiment architecture (see fig-repo-18).

## Relationship to Other Figures

| Figure | Scope | Focus |
|--------|-------|-------|
| **fig-repo-10** (THIS) | 6 Experiment Subflows | Labor division, experiment pipeline |
| **fig-repo-18** | 2 Post-Experiment Blocks | Extraction vs Analysis |
| **fig-repo-37** | Prefect technical details | Retries, observability, code structure |

## Key Message

"The Prefect experiment pipeline enables labor division: domain experts define features, signal processing experts handle outlier detection, biostatisticians validate classification. Each subflow has 'MLflow as contract' - you can replace any implementation without breaking downstream."

## The 6 Experiment Subflows (Verified from Code)

| Subflow | File | Professional Persona | Input | Output |
|---------|------|---------------------|-------|--------|
| **Data Import** | `src/data_io/flow_data.py` | Data Engineer | Raw CSVs, SERI DB | Polars DataFrame |
| **Outlier Detection** | `src/anomaly_detection/flow_anomaly_detection.py` | Signal Processing Expert | DataFrame + config | MLflow runs + pickle |
| **Imputation** | `src/imputation/flow_imputation.py` | Signal Processing Expert | Outlier runs | MLflow runs + pickle |
| **Featurization** | `src/featurization/flow_featurization.py` | Domain Expert | Imputation runs | MLflow runs + DuckDB |
| **Classification** | `src/classification/flow_classification.py` | Biostatistician | Feature runs | MLflow runs + metrics |
| **Deployment** | `src/deploy/flow_deployment.py` | MLOps Engineer | Model artifacts | Model registry (placeholder) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│          PREFECT EXPERIMENT PIPELINE: 6 Subflows with Labor Division                │
│                                                                                     │
│  "MLflow as Contract" - Each subflow reads/writes to MLflow, enabling team members  │
│  with different expertise to work independently on their component.                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                 ││
│  │   🗄️ DATA IMPORT                    ⚡ PLR OUTLIER DETECTION                    ││
│  │   ────────────────                   ──────────────────────────                 ││
│  │   👤 Data Engineer                   👤 Signal Processing Expert               ││
│  │                                                                                 ││
│  │   ┌───────────┐      ┌──────┐       ┌───────────┐      ┌──────┐               ││
│  │   │ Raw CSVs  │  →   │ Load │   →   │ DataFrame │  →   │ Run  │               ││
│  │   │ SERI DB   │      │ Data │       │ + Config  │      │ 11   │               ││
│  │   └───────────┘      └──────┘       └───────────┘      │Methods│               ││
│  │                                                         └──┬───┘               ││
│  │   Tasks:                            Tasks:                 │                   ││
│  │   • load_from_csv()                 • run_LOF()            │ MLflow            ││
│  │   • load_from_duckdb()              • run_MOMENT()         ▼ Artifacts         ││
│  │   • validate_schema()               • run_ensemble()    ┌──────┐              ││
│  │                                                         │ .pkl │              ││
│  │                                                         └──────┘              ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                              │                                      │
│                                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                 ││
│  │   🔧 PLR IMPUTATION                  📊 PLR FEATURIZATION                       ││
│  │   ──────────────────                 ────────────────────                       ││
│  │   👤 Signal Processing Expert        👤 Domain Expert (Ophthalmologist)         ││
│  │                                                                                 ││
│  │   ┌───────────┐      ┌──────┐       ┌───────────┐      ┌──────┐               ││
│  │   │ Outlier   │  →   │ Run  │   →   │ Imputed   │  →   │ Extract│              ││
│  │   │ Runs      │      │ 8    │       │ Signals   │      │ 15     │              ││
│  │   │           │      │Methods│      │           │      │Features│              ││
│  │   └───────────┘      └──┬───┘       └───────────┘      └──┬───┘               ││
│  │                         │                                  │                    ││
│  │   Tasks:                │ MLflow     Tasks:                │ MLflow            ││
│  │   • run_SAITS()         ▼ Artifacts  • compute_amplitude() ▼ Artifacts         ││
│  │   • run_CSDI()       ┌──────┐        • compute_latency() ┌──────┐             ││
│  │   • run_MOMENT()     │ .pkl │        • compute_PIPR()    │ .db  │             ││
│  │   • run_linear()     └──────┘                            └──────┘             ││
│  │                                                                                 ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                              │                                      │
│                                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                 ││
│  │   📈 PLR CLASSIFICATION              🚀 MODEL DEPLOYMENT                         ││
│  │   ────────────────────               ──────────────────                         ││
│  │   👤 Biostatistician                 👤 MLOps Engineer                          ││
│  │                                                                                 ││
│  │   ┌───────────┐      ┌──────┐       ┌───────────┐      ┌──────┐               ││
│  │   │ Features  │  →   │ Train│   →   │ Best      │  →   │ Push │               ││
│  │   │ Runs      │      │ 5    │       │ Model     │      │ to   │               ││
│  │   │           │      │Classif│      │           │      │ Prod │               ││
│  │   └───────────┘      └──┬───┘       └───────────┘      └──────┘               ││
│  │                         │                                                       ││
│  │   Tasks:                │ MLflow     Tasks:                                     ││
│  │   • train_CatBoost()    ▼ Metrics    • select_best_model()                     ││
│  │   • train_XGBoost()  ┌──────┐        • register_model()  (placeholder)         ││
│  │   • run_bootstrap()  │ .pkl │        • validate_production()                   ││
│  │   • compute_STRATOS()│ +    │                                                  ││
│  │                      │scalar│                                                  ││
│  │                      └──────┘                                                  ││
│  │                                                                                 ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  WHY "MLflow AS CONTRACT"?                                                          │
│  ════════════════════════                                                           │
│                                                                                     │
│  Each subflow:                                                                      │
│  • READS from previous MLflow experiments (input contract)                          │
│  • WRITES to MLflow with standardized schema (output contract)                      │
│                                                                                     │
│  Benefits:                                                                          │
│  ✅ Team members work independently on their component                              │
│  ✅ Swap implementations without breaking downstream (e.g., LOF → MOMENT)           │
│  ✅ Track all experiments with metadata, artifacts, metrics                         │
│  ✅ Reproduce any configuration from MLflow run ID                                  │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  AFTER EXPERIMENTS: SEE fig-repo-18 (Two-Block Architecture)                        │
│  ═══════════════════════════════════════════════════════════                        │
│                                                                                     │
│  These 6 subflows output to MLflow (~20 GB of artifacts).                           │
│  Then TWO more flows process MLflow for publication:                                │
│                                                                                     │
│  [6 Experiment Subflows] → MLflow → [Block 1: Extraction] → DuckDB                  │
│                                            ↓                                        │
│                                    [Block 2: Analysis] → Figures                    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **6 Subflow boxes** with persona icons
2. **Data flow arrows** showing MLflow as intermediate storage
3. **Task lists** for each subflow
4. **Output formats** (pickle, DuckDB)
5. **"MLflow as Contract"** explanation
6. **Cross-reference** to fig-repo-18

## Text Content

### Title Text
"Prefect Experiment Pipeline: 6 Subflows with Labor Division"

### Caption
The preprocessing pipeline is organized as 6 Prefect subflows, each owned by a different professional persona: data engineers handle import, signal processing experts manage outlier detection and imputation, domain experts define features, biostatisticians validate classification. "MLflow as contract" design enables team members to work independently—swap any implementation (e.g., LOF → MOMENT) without breaking downstream flows. See fig-repo-18 for post-experiment processing.

## Prompts for Nano Banana Pro

### Style Prompt
Six connected flow boxes arranged in 2x3 grid. Each box has persona icon, tasks list, output format indicator. Data flow arrows between boxes show MLflow artifacts. Professional workflow diagram style with labor division emphasis.

### Content Prompt
Create a 6-subflow pipeline diagram:

**ROW 1**:
- Data Import (Data Engineer) → Polars DataFrame
- Outlier Detection (Signal Processing) → MLflow + pickle

**ROW 2**:
- Imputation (Signal Processing) → MLflow + pickle
- Featurization (Domain Expert) → MLflow + DuckDB

**ROW 3**:
- Classification (Biostatistician) → MLflow + metrics
- Deployment (MLOps) → Model registry

**BOTTOM - Why MLflow as Contract**:
- Team independence, swappable implementations, reproducibility

**FOOTER - Cross-reference**:
- Link to fig-repo-18 for post-experiment processing

## Alt Text

Six Prefect subflows for experiment pipeline. Data Import (Data Engineer) → Outlier Detection (Signal Processing Expert) → Imputation (Signal Processing Expert) → Featurization (Domain Expert) → Classification (Biostatistician) → Deployment (MLOps). Each subflow reads/writes MLflow artifacts enabling "MLflow as Contract" design where team members work independently. Footer links to fig-repo-18 for post-experiment processing.

## Status

- [x] Draft created
- [x] Updated with verified 6 subflows and labor division
- [ ] Generated
- [ ] Placed in ARCHITECTURE.md
