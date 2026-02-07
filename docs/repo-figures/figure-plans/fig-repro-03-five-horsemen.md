# fig-repro-03: The 5 Horsemen of Irreproducibility

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-03 |
| **Title** | The 5 Horsemen of Irreproducibility |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | All audiences |
| **Location** | README.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Create a memorable taxonomy of reproducibility failures that readers can use as a mental checklist for their own projects.

## Key Message

"Five factors doom reproducibility: unpinned dependencies, missing data, environment drift, random seeds, and hardware differences. Address all five or fail."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| R4R 2025 | ~40% dependency, ~20% data, ~25% environment failures | [10.1145/3736731.3746156](https://doi.org/10.1145/3736731.3746156) |
| Pimentel et al. 2023 | Error taxonomy from notebook analysis | [arXiv:2308.07333](https://arxiv.org/abs/2308.07333) |
| Malka et al. 2026 | Environment and container failures | [arXiv:2601.12811](https://arxiv.org/abs/2601.12811) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE 5 HORSEMEN OF IRREPRODUCIBILITY                          │
│                    Each one can doom your research                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  ⚔️ 1. DEPENDENCY DRIFT           (~40% of failures)                   │   │
│  │  ──────────────────────────────────────────────────                     │   │
│  │  "pip install pandas" today ≠ "pip install pandas" tomorrow            │   │
│  │                                                                         │   │
│  │  Foundation PLR: uv.lock pins EXACT versions                            │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  📁 2. MISSING DATA                (~20% of failures)                   │   │
│  │  ──────────────────────────────────────────────────                     │   │
│  │  "/home/alice/mydata.csv" doesn't exist on Bob's machine                │   │
│  │                                                                         │   │
│  │  Foundation PLR: DuckDB single-source database                          │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🖥️ 3. ENVIRONMENT DECAY           (~25% of failures)                   │   │
│  │  ──────────────────────────────────────────────────                     │   │
│  │  "Works on my machine" → different OS, libraries, Python version        │   │
│  │                                                                         │   │
│  │  Foundation PLR: Documented Ubuntu deps + .python-version               │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🎲 4. RANDOM SEEDS                (~10% of failures)                   │   │
│  │  ──────────────────────────────────────────────────                     │   │
│  │  np.random.rand() gives different results every run                     │   │
│  │                                                                         │   │
│  │  Foundation PLR: Seeds documented in configs/defaults.yaml              │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🔧 5. HARDWARE MISMATCH           (~5% of failures)                    │   │
│  │  ──────────────────────────────────────────────────                     │   │
│  │  GPU drivers, CPU architecture (x86 vs ARM), precision differences      │   │
│  │                                                                         │   │
│  │  Foundation PLR: Platform documented, CPU-only default                  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Sources: R4R 2025, Pimentel et al. 2023, Malka et al. 2026                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Five numbered cards**: Each "horseman" with icon
2. **Percentage breakdown**: How much each contributes to failures
3. **Simple example**: One-line illustration of the problem
4. **Foundation PLR solution**: How we address it
5. **Multi-source citations**: Three papers referenced

## Text Content

### Title Text
"The 5 Horsemen of Irreproducibility"

### Caption
Five factors doom scientific reproducibility: Dependency Drift (40%, solved by uv.lock), Missing Data (20%, solved by DuckDB), Environment Decay (25%, solved by documentation), Random Seeds (10%, solved by config files), and Hardware Mismatch (5%, solved by platform documentation). Foundation PLR addresses all five. Sources: R4R 2025, Pimentel et al. 2023, Malka et al. 2026.

## Prompts for Nano Banana Pro

### Style Prompt
Five stacked cards, each with distinct icon and color accent. Percentage badges on right side. Problem→Solution format within each card. Medieval-inspired icons but modern color palette. No fantasy elements—professional with personality.

### Content Prompt
Create "5 Horsemen of Irreproducibility" infographic:

**FIVE HORIZONTAL CARDS** stacked vertically:
1. Dependency Drift (40%) - swords icon - uv.lock solution
2. Missing Data (20%) - folder icon - DuckDB solution
3. Environment Decay (25%) - computer icon - documentation solution
4. Random Seeds (10%) - dice icon - config file solution
5. Hardware Mismatch (5%) - gear icon - platform docs solution

Each card: Icon | Name + % | One-line problem | One-line solution

**FOOTER**: Three paper citations

## Alt Text

The 5 Horsemen of Irreproducibility infographic showing five stacked cards. Dependency Drift (40%): unpinned packages changing between runs, solved by uv.lock. Missing Data (20%): hardcoded file paths, solved by DuckDB consolidation. Environment Decay (25%): OS and library differences, solved by documentation. Random Seeds (10%): unseeded randomness, solved by config files. Hardware Mismatch (5%): architecture differences, solved by platform documentation. Sources: R4R 2025, Pimentel 2023, Malka 2026.

## Related Figures

- **fig-repro-01**: Crisis statistics
- **fig-repro-08a/b**: Dependency hell deep dive
- **fig-repro-20**: DuckDB single source of truth

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md

