# fig-data-03: PLR Data Dictionary Structure

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-data-03 |
| **Title** | PLR Data Dictionary Structure |
| **Complexity Level** | L3-L4 (ML Engineer) |
| **Target Persona** | ML Engineer |
| **Location** | `src/data_io/README.md`, `data/README.md` |
| **Priority** | P1 (Critical) |
| **Aspect Ratio** | 16:10 |

## Purpose

Document the hierarchical `data_dict` structure that flows between pipeline stages. Developers need to understand the `{split}/{data|labels|metadata}/{column}` hierarchy to contribute to any module. This dict is the universal interface between data loading, preprocessing, and model training.

## Key Message

"The data dictionary is the universal interface -- understand its `{split}/{data|labels|metadata}/{column}` hierarchy to contribute to any module."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               PLR DATA DICTIONARY STRUCTURE                                      │
│               The universal interface between pipeline stages                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  data_dict                                                                       │
│  ═════════                                                                       │
│  │                                                                               │
│  ├── "train"                                                                     │
│  │   ├── "data"                                                                  │
│  │   │   ├── "pupil_raw"          float[N, T]  Raw pupil diameter               │
│  │   │   ├── "pupil_gt"           float[N, T]  Ground truth denoised            │
│  │   │   ├── "pupil_raw_imputed"  float[N, T]  After imputation                │
│  │   │   └── "time"               float[N, T]  Time axis                        │
│  │   │                                                                           │
│  │   ├── "labels"                                                                │
│  │   │   ├── "outlier_mask"       int[N, T]    0=clean, 1=outlier               │
│  │   │   └── "diagnosis"          int[N]       0=control, 1=glaucoma            │
│  │   │                                                                           │
│  │   └── "metadata"                                                              │
│  │       ├── "subject_id"         str[N]       Anonymized IDs (Hxxx/Gxxx)       │
│  │       └── "eye"                str[N]       L/R                              │
│  │                                                                               │
│  └── "test"                                                                      │
│      └── (same structure as train)                                               │
│                                                                                  │
│  WHO CONSUMES WHAT                                                               │
│  ═════════════════                                                               │
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Outlier Det.     │  │ Imputation       │  │ Classification   │                 │
│  │ (507 subjects)   │  │ (507 subjects)   │  │ (208 subjects)   │                 │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤                 │
│  │ data/pupil_raw  │  │ data/pupil_raw  │  │ data/*_imputed  │                 │
│  │ data/pupil_gt   │  │ data/pupil_gt   │  │ labels/diagnosis│                 │
│  │ labels/outlier  │  │ labels/outlier  │  │ metadata/*      │                 │
│  │  _mask          │  │  _mask          │  │                 │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                  │
│  Created by: src/data_io/data_wrangler.py → convert_df_to_dict()               │
│  Selected by: src/data_io/torch_data.py → PLRDataSelector                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| data_dict root | `primary_pathway` | Top-level dict with train/test splits |
| data branch | `raw_signal` | Pupil signals (raw, ground truth, imputed) |
| labels branch | `outlier_detection` | Outlier masks and diagnosis labels |
| metadata branch | `secondary_pathway` | Subject IDs and eye laterality |
| Outlier Detection consumer | `outlier_detection` | Uses data/pupil_raw, labels/outlier_mask |
| Imputation consumer | `imputation` | Uses data/pupil_raw, data/pupil_gt |
| Classification consumer | `classification` | Uses data/*_imputed, labels/diagnosis |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| data_wrangler.py | data_dict | Arrow | "convert_df_to_dict()" |
| data_dict | PLRDataSelector | Arrow | "select columns for task" |
| PLRDataSelector | Outlier Detection | Arrow | "pupil_raw + outlier_mask" |
| PLRDataSelector | Imputation | Arrow | "pupil_raw + pupil_gt" |
| PLRDataSelector | Classification | Arrow | "*_imputed + diagnosis" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "UNIVERSAL INTERFACE" | "All pipeline stages read from and write to this dict. Understand the hierarchy to contribute to any module." | Top right |

## Text Content

### Labels
- train/test: Split level
- data/labels/metadata: Category
- pupil_raw, pupil_gt: Columns
- N=507 (preprocess), N=208 (classify)

### Caption
The `data_dict` is the universal interface between pipeline stages. Created by `convert_df_to_dict()` in `data_wrangler.py`, it organizes data into `{split}/{category}/{column}` hierarchy. Outlier detection uses `data/pupil_raw` + `labels/outlier_mask` (507 subjects). Imputation uses `data/pupil_raw` + `data/pupil_gt`. Classification uses `data/*_imputed` + `labels/diagnosis` (208 labeled subjects).

## Prompts for Nano Banana Pro

### Style Prompt
Hierarchical tree diagram with color-coded branches. Top: data_dict root. Middle: tree showing train/test → data/labels/metadata → columns with types and shapes. Bottom: three consumer boxes showing which branches each task uses. Clean, architectural style.

### Content Prompt
Create "PLR Data Dictionary Structure" diagram:

**TOP**: data_dict tree hierarchy
- train/test splits
- data/labels/metadata categories
- Column names with types and shapes

**BOTTOM**: Three consumer cards
- Outlier Detection (507 subjects, reads pupil_raw + outlier_mask)
- Imputation (507 subjects, reads pupil_raw + pupil_gt)
- Classification (208 subjects, reads *_imputed + diagnosis)

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/data/` | Data split configuration |
| `configs/demo_subjects.yaml` | 8 demo subjects for development |

## Code Paths

| Module | Role |
|--------|------|
| `src/data_io/data_wrangler.py` | `convert_df_to_dict()`: Creates the data_dict from DataFrames |
| `src/data_io/torch_data.py` | `PLRDataSelector`: Selects columns for specific tasks |
| `src/data_io/data_utils.py` | `transform_data_for_momentfm()`: MOMENT-specific dict transforms |

## Extension Guide

To add a new column to the data dictionary:
1. Add the column in `src/data_io/data_wrangler.py` within `convert_df_to_dict()`
2. Update the selector in `src/data_io/torch_data.py` if the new column is task-specific
3. Add tests verifying the column exists in the expected shape
4. Update this figure plan to document the new column

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "data-03",
    "title": "PLR Data Dictionary Structure"
  },
  "content_architecture": {
    "primary_message": "The data_dict ({split}/{category}/{column}) is the universal interface between pipeline stages. Each task consumes different branches.",
    "layout_flow": "Top-down: dict hierarchy tree at top, three consumer cards below",
    "spatial_anchors": {
      "data_dict_root": {"x": 0.5, "y": 0.1},
      "tree_hierarchy": {"x": 0.5, "y": 0.35},
      "outlier_consumer": {"x": 0.2, "y": 0.75},
      "imputation_consumer": {"x": 0.5, "y": 0.75},
      "classification_consumer": {"x": 0.8, "y": 0.75}
    },
    "key_structures": [
      {
        "name": "data_dict",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["train/test", "data/labels/metadata"]
      },
      {
        "name": "Outlier Detection",
        "role": "outlier_detection",
        "is_highlighted": false,
        "labels": ["507 subjects", "pupil_raw + outlier_mask"]
      },
      {
        "name": "Classification",
        "role": "classification",
        "is_highlighted": false,
        "labels": ["208 subjects", "*_imputed + diagnosis"]
      }
    ],
    "callout_boxes": [
      {"heading": "UNIVERSAL INTERFACE", "body_text": "All pipeline stages read/write this dict. Created by convert_df_to_dict(), selected by PLRDataSelector."}
    ]
  }
}
```

## Alt Text

Tree diagram of PLR data dictionary hierarchy: data_dict splits into train/test, each containing data (pupil_raw, pupil_gt, pupil_raw_imputed), labels (outlier_mask, diagnosis), and metadata (subject_id, eye). Three consumer cards show which branches each task uses.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in data/README.md
