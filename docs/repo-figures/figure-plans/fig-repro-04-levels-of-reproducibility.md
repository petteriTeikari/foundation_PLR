# fig-repro-04: Levels of Reproducibility

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-04 |
| **Title** | Levels of Reproducibility |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | Biostatistician, ML Engineer, Grant Reviewer |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Define the reproducibility spectrum from "completely broken" to "bitwise identical" so readers understand what level their projects achieve.

## Key Message

"Reproducibility isn't binary. There are 6 levels from 'code missing' to 'bitwise identical'. Most scientific software achieves Level 1-2. Foundation PLR targets Level 4-5."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE REPRODUCIBILITY SPECTRUM                                 │
│                    Where does your project fall?                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                 IRREPRODUCIBLE ◄─────────────────────────► REPRODUCIBLE         │
│                                                                                 │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐                 │
│  │ Level 0 │ Level 1 │ Level 2 │ Level 3 │ Level 4 │ Level 5 │                 │
│  │ MISSING │ BROKEN  │ PARTIAL │ RUNS    │ SAME    │ BITWISE │                 │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘                 │
│      │         │         │         │         │         │                        │
│      │         │         │         │         │         │                        │
│      ▼         ▼         ▼         ▼         ▼         ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  LEVEL 0: CODE/DATA MISSING                                             │   │
│  │  "The code is on my old laptop"                                         │   │
│  │  Cannot even attempt reproduction                                       │   │
│  │  ████░░░░░░░░░░░░░░░░░░░░░░░░░░ 15% of papers                          │   │
│  │                                                                         │   │
│  │  LEVEL 1: BUILD/INSTALL FAILS                                           │   │
│  │  "ModuleNotFoundError: No module named 'pandas'"                        │   │
│  │  Code exists but won't run                                              │   │
│  │  ██████████░░░░░░░░░░░░░░░░░░░░ 35% of papers                          │   │
│  │                                                                         │   │
│  │  LEVEL 2: RUNS BUT CRASHES                                              │   │
│  │  "FileNotFoundError: /home/alice/data.csv"                              │   │
│  │  Some parts work, then fails                                            │   │
│  │  ████████████████░░░░░░░░░░░░░░ 24% of papers                          │   │
│  │                                                                         │   │
│  │  LEVEL 3: COMPLETES WITH DIFFERENT RESULTS                              │   │
│  │  "Output: 0.847 (paper said 0.912)"                                     │   │
│  │  Runs but numbers don't match                                           │   │
│  │  ██████████████████████░░░░░░░░ 20% of papers                          │   │
│  │                                                                         │   │
│  │  LEVEL 4: FUNCTIONALLY EQUIVALENT                                       │   │
│  │  Same conclusions, minor numerical differences                           │   │
│  │  ████████████████████████████░░ 5% of papers                           │   │
│  │                                                                         │   │
│  │  LEVEL 5: BITWISE IDENTICAL                                             │   │
│  │  SHA-256 of outputs matches exactly                                     │   │
│  │  ██████████████████████████████ <1% of papers                          │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  FOUNDATION PLR TARGET: Level 4-5                                               │
│  ✓ uv.lock ensures deterministic installs                                       │
│  ✓ DuckDB eliminates missing data                                               │
│  ✓ MLflow tracks exact experiment provenance                                    │
│  ✓ Random seeds documented in config                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Spectrum bar**: Visual gradient from red to green
2. **Six level boxes**: Each with name, example, description
3. **Percentage bars**: Estimated distribution of papers at each level
4. **Target callout**: Where Foundation PLR aims

## Text Content

### Title Text
"The Reproducibility Spectrum: 6 Levels from Broken to Perfect"

### Caption
Reproducibility exists on a spectrum from Level 0 (code missing) to Level 5 (bitwise identical). Studies suggest ~74% of papers fall at Levels 0-2 (R4R 2025), ~20% at Level 3, and <6% achieve Levels 4-5. Foundation PLR targets Level 4-5 through locked dependencies, consolidated data, and documented random seeds.

## Prompts for Nano Banana Pro

### Style Prompt
Horizontal spectrum from red (Level 0) to green (Level 5). Six stacked level descriptions with progress bars showing percentage of papers at each level. Clean, professional infographic. Target arrow pointing to Level 4-5.

### Content Prompt
Create "Reproducibility Spectrum" infographic:

**TOP - Gradient Bar**:
- 6 levels from MISSING to BITWISE
- Color gradient red → green

**MIDDLE - Level Descriptions** (stacked):
- Level 0: Missing (15%) - "code on old laptop"
- Level 1: Broken (35%) - ModuleNotFoundError
- Level 2: Crashes (24%) - FileNotFoundError
- Level 3: Different results (20%) - numbers don't match
- Level 4: Functional (5%) - same conclusions
- Level 5: Bitwise (<1%) - SHA-256 matches

**BOTTOM - Target**:
- Arrow pointing to Level 4-5
- Foundation PLR solutions listed

## Alt Text

Reproducibility spectrum infographic showing six levels. Level 0 Missing (15%): code unavailable. Level 1 Broken (35%): installation fails. Level 2 Crashes (24%): runs partially. Level 3 Different (20%): wrong results. Level 4 Functional (5%): same conclusions. Level 5 Bitwise (<1%): exact match. Color gradient from red (Level 0) to green (Level 5). Foundation PLR targets Level 4-5.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

