# fig-repro-02a: Why 96.8% of Notebooks Fail (ELI5)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-02a |
| **Title** | Why 96.8% of Notebooks Fail (ELI5) |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | Biologist, PI, PhD Student, Lab Manager |
| **Location** | README.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain the Jupyter reproducibility study findings using simple kitchen analogies that non-programmers can understand.

## Key Message

"Scientific notebooks fail for the same reasons recipes fail: missing ingredients, changed ingredient brands, wrong cupboard locations, and surprise random elements."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Pimentel et al. 2023 | Analyzed 27,271 notebooks, categorized failure types | [arXiv:2308.07333](https://arxiv.org/abs/2308.07333) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    WHY 96.8% OF NOTEBOOKS FAIL                                  │
│                    (Explained Simply)                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  IMAGINE A RECIPE...                                                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  "My notebook ran perfectly last year!"                                 │   │
│  │                                                                         │   │
│  │  But now...                                                             │   │
│  │                                                                         │   │
│  │  MISSING INGREDIENTS                            ~40% of failures        │   │
│  │     "Install pandas" ← But which version?                               │   │
│  │     The pandas from 2020 ≠ pandas from 2024                             │   │
│  │     [Icon: empty grocery bag]                                           │   │
│  │                                                                         │   │
│  │  INGREDIENTS IN WRONG CUPBOARD                  ~20% of failures        │   │
│  │     "/Users/jane/data/myfile.csv"                                       │   │
│  │     ↑ Only exists on Jane's computer!                                   │   │
│  │     [Icon: folder with question mark]                                   │   │
│  │                                                                         │   │
│  │  RECIPE CHANGED                                 ~25% of failures        │   │
│  │     "plt.plot()" worked differently in                                  │   │
│  │     matplotlib 2.0 vs 3.0                                               │   │
│  │     [Icon: two different version boxes]                                 │   │
│  │                                                                         │   │
│  │  SURPRISE INGREDIENTS                           ~10% of failures        │   │
│  │     Random numbers change every time                                    │   │
│  │     (no fixed seed)                                                     │   │
│  │     [Icon: dice]                                                        │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  THE RESULT:                                                                    │
│                                                                                 │
│  Out of 27,271 notebooks, researchers could only recreate                       │
│  identical results for 879 (3.2%)                                               │
│                                                                                 │
│  Source: Pimentel et al. 2023 (arXiv:2308.07333)                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Recipe metaphor**: Four failure types as cooking problems
2. **Percentage breakdowns**: ~40%, ~20%, ~25%, ~10%
3. **Simple icons**: Grocery bag, folder, version boxes, dice
4. **Headline statistic**: 27,271 → 879 (3.2%)
5. **Single citation**: arXiv:2308.07333

## Text Content

### Title Text
"Why 96.8% of Notebooks Fail (Explained Simply)"

### Caption
When researchers tried to re-run 27,271 biomedical Jupyter notebooks, 96.8% failed to produce identical results. The main culprits: missing dependencies (like a recipe missing ingredients), changed package versions (like a recipe with different ingredient brands), broken file paths (like ingredients stored in the wrong cupboard), and unseeded randomness (like surprise ingredients). Source: Pimentel et al. 2023, [arXiv:2308.07333](https://arxiv.org/abs/2308.07333)

## Prompts for Nano Banana Pro

### Style Prompt
Friendly infographic with kitchen/cooking metaphor. Four problems illustrated with simple icons. Warm colors for failures, cool teal for solution hints. No code visible. Approachable, not intimidating. Print-quality.

### Content Prompt
Create a "Why Notebooks Fail (ELI5)" infographic:

**HEADER**: Quote bubble "My notebook ran perfectly last year!"

**FOUR CARDS - each with icon + analogy**:
1. Missing Ingredients (empty bag icon) - dependencies
2. Wrong Cupboard (folder icon) - file paths
3. Recipe Changed (version boxes) - API changes
4. Surprise Ingredients (dice icon) - randomness

**BOTTOM**:
- Final statistic: 27,271 → 879 (3.2%)
- Single source citation

## Alt Text

ELI5 infographic explaining why 96.8% of Jupyter notebooks fail using a kitchen recipe analogy. Four failure types: Missing Ingredients (40% - unspecified dependencies), Ingredients in Wrong Cupboard (20% - hardcoded file paths), Recipe Changed (25% - API version changes), and Surprise Ingredients (10% - unseeded random numbers). Final statistic shows only 879 of 27,271 notebooks produced identical results. Source: Pimentel et al. 2023.

## Related Figures

- **fig-repro-02b**: Expert version with technical details
- **fig-repro-03**: The 5 Horsemen of Irreproducibility
- **fig-repo-14**: uv Package Manager (solution)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md

