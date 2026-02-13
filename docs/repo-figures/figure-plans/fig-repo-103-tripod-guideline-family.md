# fig-repo-103: TRIPOD Reporting Guideline Family

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-103 |
| **Title** | The TRIPOD Reporting Guideline Family: From Manuscripts to Code |
| **Complexity Level** | L2 (Concept explanation) |
| **Target Persona** | All |
| **Location** | README.md (For Reviewers > Reporting Standards), docs/ |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:9 |

## Purpose

Show how the TRIPOD family of reporting guidelines evolved over a decade and what each extension addresses. Contextualize TRIPOD-Code as the latest member, extending a single TRIPOD+AI sub-item (18f) into a complete checklist for code repository quality.

## Key Message

"TRIPOD-Code extends TRIPOD+AI Item 18f -- a single recommendation about code sharing -- into a complete reporting guideline for code repositories associated with prediction model studies."

## Visual Concept

Left-to-right timeline showing the TRIPOD guideline family as nodes on a branching tree. Each node shows the guideline name, year, scope (what it covers), and item count. A "zoom-in" callout connects Item 18f in TRIPOD+AI to the full TRIPOD-Code protocol.

```
                         THE TRIPOD GUIDELINE FAMILY
                         ============================

  2015                        2024                           2026
    |                           |                              |
    |                           |                              |
 ┌──────────┐  ────────>  ┌──────────────┐  ──────────>  ┌──────────────┐
 │  TRIPOD  │             │  TRIPOD+AI   │               │ TRIPOD-Code  │
 │  ------  │             │  ----------  │               │ -----------  │
 │ 22 items │             │ 49 items     │   ZOOM IN     │ Protocol     │
 │          │             │ (27 AI-new)  │   =======>    │ stage        │
 │ Scope:   │             │              │   Item 18f    │              │
 │ Predict. │             │ Scope:       │   expands     │ Scope:       │
 │ model    │             │ + AI/ML      │   into full   │ Code repos   │
 │ studies  │             │ methods      │   guideline   │ for pred.    │
 │ (manu-   │             │              │               │ models       │
 │ scripts) │             │              │               │              │
 └──────────┘             └──────────────┘               └──────────────┘
                                 |
                                 | branch
                                 v
                          ┌──────────────┐
                          │ TRIPOD-LLM   │
                          │ ----------   │
                          │ 2025         │
                          │ Scope:       │
                          │ LLM-based    │
                          │ prediction   │
                          └──────────────┘

 ┌────────────────────────────────────────────────────────────────────┐
 │  ITEM 18f ZOOM:                                                    │
 │  TRIPOD+AI says: "Share code for data preprocessing, model         │
 │  development, and evaluation"                                      │
 │  TRIPOD-Code asks: HOW should that code repository be structured?  │
 │  → Dependencies, license, testing, reproducibility, documentation  │
 └────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Left-to-right chronological with downward branch"
spatial_anchors:
  tripod_base:
    x: 0.15
    y: 0.45
    content: "TRIPOD (2015): 22-item checklist for prediction model studies"
  tripod_ai:
    x: 0.45
    y: 0.45
    content: "TRIPOD+AI (2024): 49 items total, adds AI/ML-specific guidance"
  tripod_code:
    x: 0.78
    y: 0.45
    content: "TRIPOD-Code (2026): Protocol for code repository reporting"
  tripod_llm:
    x: 0.45
    y: 0.80
    content: "TRIPOD-LLM (2025): Extension for LLM-based predictions"
  item_18f_callout:
    x: 0.62
    y: 0.15
    content: "Item 18f zoom: code sharing expands into full guideline"
  arrow_base_to_ai:
    x: 0.30
    y: 0.45
    content: "Evolution arrow"
  arrow_ai_to_code:
    x: 0.62
    y: 0.45
    content: "Zoom-in arrow from Item 18f"
  arrow_ai_to_llm:
    x: 0.45
    y: 0.62
    content: "Branch arrow downward"
```

## Content Elements

### Key Structures

| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| TRIPOD node | `secondary_pathway` | Base guideline (2015), 22 items |
| TRIPOD+AI node | `primary_pathway` | AI extension (2024), 49 items |
| TRIPOD-Code node | `highlight_accent` | Code repository guideline (2026), protocol |
| TRIPOD-LLM node | `secondary_pathway` | LLM extension (2025), sibling branch |
| Item 18f callout | `callout_box` | Explains the zoom-in from one item to full guideline |
| Timeline axis | `annotation` | Years: 2015, 2024, 2025, 2026 |

### Relationships/Connections

| From | To | Type | Label |
|------|-----|------|-------|
| TRIPOD | TRIPOD+AI | Arrow (thick) | "extends" |
| TRIPOD+AI | TRIPOD-Code | Arrow (thick, highlighted) | "Item 18f expands" |
| TRIPOD+AI | TRIPOD-LLM | Arrow (thin, branch) | "LLM branch" |

### Callout Boxes

| Title | Content | Location |
|-------|---------|----------|
| "Item 18f" | TRIPOD+AI recommends sharing analytical code. TRIPOD-Code defines HOW. | Between TRIPOD+AI and TRIPOD-Code nodes |
| "KEY INSIGHT" | One sub-item became an entire guideline -- code quality matters. | Top right |

### Numerical Annotations

| Value | Context |
|-------|---------|
| 22 | TRIPOD checklist items (2015) |
| 49 | TRIPOD+AI total items (27 AI-specific + 22 base) |
| 5-stage Delphi | TRIPOD-Code development process |
| 200+ | Expected Delphi participants |

## Text Content

### Labels (Max 30 chars each)

- Label 1: "TRIPOD (2015)"
- Label 2: "TRIPOD+AI (2024)"
- Label 3: "TRIPOD-Code (2026)"
- Label 4: "TRIPOD-LLM (2025)"
- Label 5: "22 checklist items"
- Label 6: "49 items (27 AI-new)"
- Label 7: "Protocol stage"
- Label 8: "Item 18f: Share code"
- Label 9: "Manuscripts"
- Label 10: "AI/ML methods"
- Label 11: "Code repositories"
- Label 12: "LLM predictions"

### Caption

The TRIPOD reporting guideline family evolved from manuscript-level recommendations (2015) through AI-specific extensions (2024) to code repository standards (2026). TRIPOD-Code extends a single TRIPOD+AI sub-item (18f: "share analytical code") into a complete reporting guideline covering dependencies, licensing, testing, and reproducibility.

## Prompts for Nano Banana Pro

### Style Prompt

Left-to-right timeline with four rounded nodes connected by arrows. Medical research aesthetic with matte colors. Clean white background. Each node is a card with title, year, and scope description. The rightmost node (TRIPOD-Code) is highlighted with gold accent. A zoom-in callout box between the middle and right nodes highlights Item 18f. The bottom node (TRIPOD-LLM) branches downward from the middle node. Subtle grid background. No decorative elements.

### Content Prompt

Create a "TRIPOD Guideline Family Tree" diagram:

**NODE 1 (left, blue-gray)**: "TRIPOD"
- Year: 2015
- "22 checklist items"
- Scope icon: manuscript/paper
- Citation: Collins et al., Ann Intern Med

**NODE 2 (center, deep blue)**: "TRIPOD+AI"
- Year: 2024
- "49 items (27 AI-specific)"
- Scope icon: AI/neural network
- Citation: Collins et al., BMJ

**NODE 3 (right, gold-highlighted)**: "TRIPOD-Code"
- Year: 2026
- "Protocol stage (5-stage Delphi)"
- Scope icon: code/repository
- Citation: Pollard et al., Diagn Progn Res

**NODE 4 (below center, light gray)**: "TRIPOD-LLM"
- Year: 2025
- Scope icon: language model
- Citation: Gallifant et al., Nat Med

**CALLOUT BOX** (between nodes 2 and 3):
- "Item 18f: Share analytical code"
- Arrow pointing to Node 3 with "expands into full guideline"

**ARROWS**:
- Thick arrow from Node 1 to Node 2
- Thick highlighted arrow from Node 2 to Node 3 (through Item 18f callout)
- Thin branching arrow from Node 2 downward to Node 4

### Refinement Notes

- Ensure TRIPOD-Code node stands out (gold border or background tint)
- The Item 18f callout is the key visual element -- make it prominent
- TRIPOD-LLM should be visually subordinate (lighter color, smaller)
- Timeline years should be clearly readable along the top or bottom

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-103",
    "title": "TRIPOD Reporting Guideline Family"
  },
  "content_architecture": {
    "primary_message": "TRIPOD-Code extends TRIPOD+AI Item 18f from one sub-item into a full code repository reporting guideline",
    "layout_flow": "Left-to-right chronological with downward branch for TRIPOD-LLM",
    "spatial_anchors": {
      "tripod": {"x": 0.15, "y": 0.45},
      "tripod_ai": {"x": 0.45, "y": 0.45},
      "tripod_code": {"x": 0.78, "y": 0.45},
      "tripod_llm": {"x": 0.45, "y": 0.80},
      "item_18f": {"x": 0.62, "y": 0.15}
    },
    "key_structures": [
      {
        "name": "TRIPOD",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["2015", "22 items", "Manuscripts"]
      },
      {
        "name": "TRIPOD+AI",
        "role": "primary_pathway",
        "is_highlighted": false,
        "labels": ["2024", "49 items", "AI/ML methods"]
      },
      {
        "name": "TRIPOD-Code",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["2026", "Protocol", "Code repositories"]
      },
      {
        "name": "TRIPOD-LLM",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["2025", "LLM predictions"]
      },
      {
        "name": "Item 18f Callout",
        "role": "callout_box",
        "is_highlighted": true,
        "labels": ["Share analytical code", "Expands into full guideline"]
      }
    ],
    "callout_boxes": [
      {"heading": "Item 18f", "body_text": "TRIPOD+AI recommends sharing code. TRIPOD-Code defines HOW the repository should be structured."},
      {"heading": "KEY INSIGHT", "body_text": "One sub-item became an entire guideline -- code quality matters for reproducible prediction models."}
    ]
  }
}
```

## Alt Text

Timeline showing TRIPOD guideline family: TRIPOD (2015, 22 items) to TRIPOD+AI (2024, 49 items) to TRIPOD-Code (2026, protocol), with TRIPOD-LLM (2025) branching from TRIPOD+AI. Item 18f callout shows how one code-sharing recommendation expanded into a full code repository guideline.

## References

- Collins GS, et al. (2015). Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD). *Ann Intern Med*, 162(1), 55-63.
- Collins GS, et al. (2024). TRIPOD+AI statement. *BMJ*, 385, e078378. [DOI: 10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
- Gallifant J, et al. (2025). Reporting guideline for the early-stage clinical evaluation of decision support systems driven by generative artificial intelligence (TRIPOD-LLM). *Nature Medicine*. [DOI: 10.1038/s41591-024-03507-2](https://doi.org/10.1038/s41591-024-03507-2)
- Pollard T, Sounack T, et al. (2026). Protocol for development of a reporting guideline (TRIPOD-Code). *Diagn Progn Res*, 10(4). [DOI: 10.1186/s41512-025-00217-4](https://doi.org/10.1186/s41512-025-00217-4)

Note: This figure documents the reporting guideline ecosystem, not research results. Performance comparisons are in the manuscript.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
