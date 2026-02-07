# fig-repro-01: The Reproducibility Crisis in Numbers

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-01 |
| **Title** | The Reproducibility Crisis in Numbers |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | Journal Editor, Grant Reviewer, PhD Student |
| **Location** | README.md, docs/reproducibility-guide.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Set the stage with shocking, peer-reviewed statistics about scientific computing reproducibility to motivate why our tooling choices matter.

## Key Message

"Only 3.2% of biomedical Jupyter notebooks reproduce identically. This is not a software problem—it's a science problem that costs $28 billion annually."

## Literature Sources (MANDATORY)

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Pimentel et al. 2023 | 3.2% of 27,271 Jupyter notebooks reproduce identically | [arXiv:2308.07333](https://arxiv.org/abs/2308.07333) |
| R4R (Donat-Bouillud et al. 2025) | Only 26% of R replication packages run successfully | [10.1145/3736731.3746156](https://doi.org/10.1145/3736731.3746156) |
| Freedman et al. 2015 | $28 billion annual waste in preclinical research | [10.1371/journal.pbio.1002165](https://doi.org/10.1371/journal.pbio.1002165) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE REPRODUCIBILITY CRISIS IN NUMBERS                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  JUPYTER NOTEBOOKS (PubMed)          R REPLICATION PACKAGES                    │
│  Pimentel et al. 2023                Donat-Bouillud et al. 2025                │
│                                                                                 │
│  27,271 notebooks examined           2,000 packages examined                   │
│           ↓                                   ↓                                 │
│                                                                                 │
│  ┌────────────────────┐              ┌────────────────────┐                    │
│  │░░░░░░░░░░░░░░░░░░░░│              │░░░░░░░░░░░░░░░░░░░░│                    │
│  │░░░░░░░░░░░░░░░░░░░░│              │░░░░░░░░░░░░░░░░░░░░│                    │
│  │░░░░░░░░░░░░░░░░░░░░│ 96.8%        │░░░░░░░░░░░░░░░░░░░░│ 74%               │
│  │░░░░░░░░░░░░░░░░░░░░│ FAIL         │░░░░░░░░░░░░░░░░░░░░│ FAIL              │
│  │░░░░░░░░░░░░░░░░░░░░│              │░░░░░░░░░░░░░░░░░░░░│                    │
│  │████████████████████│ 3.2%         │████████████████████│ 26%               │
│  └────────────────────┘ IDENTICAL    └────────────────────┘ RUN               │
│                                                                                 │
│  Only 879 of 27,271 notebooks        Only 520 of 2,000 packages                │
│  produced identical results          completed execution                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THIS IS NOT A SOFTWARE PROBLEM—IT'S A SCIENCE PROBLEM                          │
│                                                                                 │
│  $28 billion/year wasted in preclinical research (Freedman 2015)                │
│                                                                                 │
│  FOUNDATION PLR SOLUTION:                                                       │
│  ✓ uv.lock (locked dependencies)   → Addresses 40% of failures                  │
│  ✓ MLflow (experiment tracking)    → Full provenance chain                      │
│  ✓ DuckDB (single-source data)     → No missing files                           │
│  ✓ Documented environment          → Eliminates guesswork                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Stacked bar charts**: Jupyter (3.2% vs 96.8%) and R (26% vs 74%) success rates
2. **Sample sizes**: N=27,271 notebooks, N=2,000 packages
3. **Cost statistic**: $28 billion annotation
4. **Solution callout box**: How Foundation PLR addresses each issue
5. **Source citations**: All DOIs hyperlinked

## Text Content

### Title Text
"The Reproducibility Crisis: Only 3.2% of Scientific Notebooks Work"

### Caption
The reproducibility crisis is quantified: only 3.2% of biomedical Jupyter notebooks produce identical results when re-run (Pimentel et al. 2023, [arXiv:2308.07333](https://arxiv.org/abs/2308.07333)). R4R (2025) found only 26% of R replication packages execute successfully ([DOI](https://doi.org/10.1145/3736731.3746156)). This costs an estimated $28 billion annually in preclinical research alone (Freedman 2015). Foundation PLR addresses this through locked dependencies (uv.lock), tracked experiments (MLflow), and consolidated data (DuckDB).

## Prompts for Nano Banana Pro

### Style Prompt
Two stacked bar charts showing failure rates with stark contrast. Gray for failures, teal for successes. Dollar figure callout. Solution checkboxes. Clean infographic style with print-quality typography. No sci-fi effects.

### Content Prompt
Create a "Reproducibility Crisis in Numbers" infographic:

**TOP - Two Panels Side by Side**:
- Left: Jupyter study (27,271 notebooks, 3.2% succeed)
- Right: R study (2,000 packages, 26% succeed)
- Stacked bars showing dramatic failure proportions

**MIDDLE - Cost Callout**:
- "$28 billion/year wasted"
- Citation to Freedman 2015

**BOTTOM - Solutions**:
- Four checkboxes: uv.lock, MLflow, DuckDB, documented env
- Arrow pointing to "Foundation PLR's approach"

## Alt Text

Infographic showing the reproducibility crisis in scientific computing. Left panel: Jupyter notebooks study (Pimentel 2023) showing 27,271 notebooks examined, only 879 (3.2%) produced identical results, 96.8% failed. Right panel: R replication packages (R4R 2025) showing 2,000 packages examined, only 520 (26%) ran successfully. Center callout: $28 billion annual waste in preclinical research. Bottom: Foundation PLR solutions including uv.lock, MLflow, DuckDB, and documented environments.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md

