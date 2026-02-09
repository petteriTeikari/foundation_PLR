# fig-data-04: Data Versioning Decision Tree

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-data-04 |
| **Title** | Data Versioning Decision Tree |
| **Complexity Level** | L2 (Research Scientist / ML Engineer) |
| **Target Persona** | Research Scientist, ML Engineer |
| **Location** | `data/README.md` |
| **Priority** | P3 (Medium) |
| **Aspect Ratio** | 16:10 |

## Purpose

Document the rationale for the current data versioning approach (SHA256 checksums over a frozen dataset) and when to upgrade to DVC or Datalad. Helps researchers and engineers understand why "just checksums" is the right choice for publication freeze and what triggers would warrant a more sophisticated system.

## Key Message

"For frozen academic datasets, SHA256 checksums suffice. Upgrade to DVC when dataset changes, multi-version experiments, or external collaboration begins."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               DATA VERSIONING DECISION TREE                                      │
│               When checksums suffice vs when to upgrade                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  IS YOUR DATASET FROZEN FOR PUBLICATION?                                        │
│  ═══════════════════════════════════════                                         │
│                                                                                  │
│            ┌──── YES ────┐            ┌──── NO ────┐                            │
│            ▼              │            ▼             │                            │
│  ┌────────────────┐      │  ┌──────────────────┐   │                            │
│  │  SHA256         │      │  │  Does dataset     │   │                            │
│  │  CHECKSUMS      │      │  │  change between   │   │                            │
│  │  ✅ Sufficient  │      │  │  experiments?     │   │                            │
│  └────────────────┘      │  └──────┬───────────┘   │                            │
│  • Verify integrity      │         │                │                            │
│  • Detect corruption     │    YES  │  NO            │                            │
│  • Zero dependencies     │    ▼    ▼                │                            │
│  • Simple: sha256sum -c  │  ┌──────────┐  ┌──────────┐                          │
│                           │  │  DVC      │  │  Checksums│                          │
│  Foundation PLR: HERE ──►│  │  ✅ Use   │  │  ✅ Still │                          │
│                           │  └──────────┘  │  enough  │                          │
│                           │                └──────────┘                          │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMPARISON TABLE                                                                │
│  ════════════════                                                                │
│                                                                                  │
│  │ Feature               │ SHA256     │ DVC         │ Datalad     │             │
│  │ ───────────────────── │ ────────── │ ─────────── │ ─────────── │             │
│  │ Integrity check       │ ✅         │ ✅          │ ✅          │             │
│  │ Version history       │ ❌ (git)   │ ✅          │ ✅          │             │
│  │ Remote storage        │ ❌         │ ✅ (S3/GCS) │ ✅ (git-annex)│           │
│  │ Diff between versions │ ❌         │ ✅          │ ✅          │             │
│  │ Pipeline DAG          │ ❌         │ ✅ (dvc.yaml)│ ❌         │             │
│  │ Complexity            │ Zero       │ Low-Medium  │ Medium-High │             │
│  │ Dependencies          │ coreutils  │ pip install │ pip+git-annex│            │
│  │ Learning curve        │ None       │ Low         │ High        │             │
│  │ Frozen dataset fit    │ ✅ Perfect │ ⚠️ Overkill │ ⚠️ Overkill │             │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  UPGRADE TRIGGERS                                                                │
│  ════════════════                                                                │
│                                                                                  │
│  Adopt DVC when ANY of these occur:                                             │
│  • Dataset changes between experiments                                          │
│  • Multiple dataset versions need tracking                                      │
│  • External collaborators need data access                                      │
│  • Cloud storage (S3/GCS) for large files                                       │
│  • Pipeline DAG tracking (dvc.yaml)                                             │
│                                                                                  │
│  Current status: Publication freeze → SHA256 checksums ✅                       │
│  See: docs/planning/reproducibility-and-mlsecops-improvements.md               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Decision root | `primary_pathway` | "Is dataset frozen?" entry point |
| SHA256 path | `healthy_normal` | Current approach for frozen datasets |
| DVC path | `secondary_pathway` | Upgrade option for changing datasets |
| Datalad | `secondary_pathway` | Heavy-weight alternative |
| Comparison table | `callout_box` | 9-dimension feature comparison |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "CURRENT STATUS" | "Publication freeze → SHA256 checksums. Zero dependencies, simple verification." | Bottom left |
| "UPGRADE TRIGGERS" | "Dataset changes, multi-version experiments, external collaboration, cloud storage." | Bottom right |

## Text Content

### Labels
- SHA256: Zero-dep integrity
- DVC: Version + pipeline DAG
- Datalad: git-annex heavy

### Caption
Data versioning decision tree for academic projects. For frozen publication datasets, SHA256 checksums provide integrity verification with zero dependencies. DVC adds version history, remote storage, and pipeline DAGs but is overkill for static data. Foundation PLR uses checksums during publication freeze; upgrade triggers include dataset changes, multi-version experiments, or external collaboration needs.

## Prompts for Nano Banana Pro

### Style Prompt
Decision tree flowchart at top. Comparison table in middle. Upgrade triggers callout at bottom. Clean, structured, decision-support style. Use Mermaid-compatible layout.

### Content Prompt
Create "Data Versioning Decision Tree" diagram:

**TOP**: Decision flowchart
- "Dataset frozen?" → YES → SHA256 (current)
- "Dataset frozen?" → NO → "Changes between experiments?" → YES → DVC

**MIDDLE**: 3-column comparison table (SHA256 vs DVC vs Datalad)

**BOTTOM**: Upgrade triggers checklist + current status callout

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `data/_checksums.sha256` | Current SHA256 checksums for integrity verification |
| `data/public/DATA_MANIFEST.yaml` | Data manifest with provenance metadata |

## Code Paths

| Module | Role |
|--------|------|
| `Makefile` | `sha256sum -c data/_checksums.sha256` verification target |

## Extension Guide

To upgrade from SHA256 to DVC:
1. Install DVC: `uv add dvc`
2. Initialize: `dvc init`
3. Track data files: `dvc add data/public/foundation_plr_results.db`
4. Configure remote: `dvc remote add -d storage s3://bucket/path`
5. Commit `.dvc` files to git
6. Update `Makefile` to use `dvc pull` instead of `sha256sum -c`

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "data-04",
    "title": "Data Versioning Decision Tree"
  },
  "content_architecture": {
    "primary_message": "For frozen academic datasets, SHA256 checksums suffice. Upgrade to DVC when dataset changes or collaboration begins.",
    "layout_flow": "Decision tree at top, comparison table middle, triggers at bottom",
    "spatial_anchors": {
      "decision_root": {"x": 0.5, "y": 0.15},
      "sha256_leaf": {"x": 0.25, "y": 0.35},
      "dvc_leaf": {"x": 0.65, "y": 0.35},
      "comparison_table": {"x": 0.5, "y": 0.6},
      "upgrade_triggers": {"x": 0.5, "y": 0.85}
    },
    "key_structures": [
      {
        "name": "SHA256 Checksums",
        "role": "healthy_normal",
        "is_highlighted": true,
        "labels": ["Current approach", "Zero dependencies", "Frozen dataset"]
      },
      {
        "name": "DVC",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Version history", "Remote storage", "Pipeline DAG"]
      },
      {
        "name": "Datalad",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["git-annex", "High complexity"]
      }
    ],
    "callout_boxes": [
      {"heading": "CURRENT STATUS", "body_text": "Publication freeze → SHA256 checksums ✅"},
      {"heading": "UPGRADE TRIGGERS", "body_text": "Dataset changes, multi-version experiments, external collaboration, cloud storage needs."}
    ]
  }
}
```

## Alt Text

Decision tree for data versioning. Frozen dataset leads to SHA256 checksums (current approach). Changing dataset leads to DVC. Comparison table: SHA256 has zero complexity but no version history; DVC adds versions, remote storage, pipeline DAGs; Datalad adds git-annex but high complexity. Upgrade triggers: dataset changes, multi-version experiments, collaboration.

## Related Figures

- **fig-repro-24**: Git LFS vs DuckDB -- companion storage strategy decision
- **fig-repro-17**: Bitwise vs functional reproducibility -- what level we target
- **fig-repro-11**: Version pinning strategies -- dependency versioning (not data)
- **fig-repro-14**: Lockfiles as time machine -- code dependency parallel

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in data/README.md
