# fig-repo-104: TRIPOD-Code Development Process and Anticipated Areas

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-104 |
| **Title** | TRIPOD-Code: 5-Stage Development and Anticipated Checklist Areas |
| **Complexity Level** | L2 (Concept explanation) |
| **Target Persona** | All |
| **Location** | README.md (For Reviewers > Reporting Standards), docs/ |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:9 |

## Purpose

Show the TRIPOD-Code 5-stage Delphi consensus development process and the anticipated checklist domains for code repositories. Demonstrate that this repo proactively addresses anticipated requirements.

## Key Message

"TRIPOD-Code uses a rigorous 5-stage Delphi consensus process to develop minimum reporting requirements for prediction model code repositories, covering dependencies, licensing, testing, and reproducibility."

## Visual Concept

Two-panel layout. Left panel shows the 5-stage development pipeline as a vertical flow with key parameters annotated. Right panel shows the anticipated checklist areas derived from the protocol paper, with indicators showing which areas this repository already addresses.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TRIPOD-CODE: FROM PROTOCOL TO CHECKLIST                                        │
├────────────────────────────┬────────────────────────────────────────────────────┤
│                            │                                                    │
│  5-STAGE DEVELOPMENT       │  ANTICIPATED CHECKLIST AREAS                       │
│  =====================     │  ============================                      │
│                            │                                                    │
│  ┌──────────────────┐      │  ┌──────────────────────────────────────────┐      │
│  │ 1. META-REVIEW   │      │  │  A. Code Availability                    │      │
│  │    Current state  │      │  │     GitHub, long-term archival           │      │
│  │    of reporting   │      │  ├──────────────────────────────────────────┤      │
│  └────────┬─────────┘      │  │  B. Software Dependencies                │      │
│           │                │  │     Lockfiles, Docker, version pins       │      │
│  ┌────────▼─────────┐      │  ├──────────────────────────────────────────┤      │
│  │ 2. DELPHI        │      │  │  C. License Specification                │      │
│  │    200+ experts   │      │  │     MIT, CITATION.cff                    │      │
│  │    3 rounds max   │      │  ├──────────────────────────────────────────┤      │
│  │    70% threshold  │      │  │  D. Code Structure / Modularity          │      │
│  └────────┬─────────┘      │  │     ARCHITECTURE.md, module map           │      │
│           │                │  ├──────────────────────────────────────────┤      │
│  ┌────────▼─────────┐      │  │  E. Testing                              │      │
│  │ 3. CONSENSUS MTG │      │  │     2000+ tests, CI tiers, guardrails    │      │
│  │    15+ experts    │      │  ├──────────────────────────────────────────┤      │
│  │    Virtual        │      │  │  F. Reproducibility                      │      │
│  └────────┬─────────┘      │  │     Demo data, make commands, Docker     │      │
│           │                │  ├──────────────────────────────────────────┤      │
│  ┌────────▼─────────┐      │  │  G. Documentation by Pipeline Stage      │      │
│  │ 4. DEVELOP       │──────│──│     Preprocessing → Model → Evaluation   │      │
│  │    STATEMENT      │      │  ├──────────────────────────────────────────┤      │
│  │    Pilot + refine │      │  │  H. Long-term Archival                   │      │
│  └────────┬─────────┘      │  │     Zenodo DOI, CITATION.cff             │      │
│           │                │  └──────────────────────────────────────────┘      │
│  ┌────────▼─────────┐      │                                                    │
│  │ 5. DISSEMINATION │      │  NOTE: These areas are ANTICIPATED from the        │
│  │    Journals,      │      │  protocol paper. The actual checklist is pending   │
│  │    EQUATOR,       │      │  completion of the Delphi consensus process.       │
│  │    TRIPOD website │      │                                                    │
│  └──────────────────┘      │                                                    │
│                            │                                                    │
├────────────────────────────┴────────────────────────────────────────────────────┤
│  "produces"                                                                     │
│  Arrow from Stage 4 → Checklist Areas                                           │
│  The Delphi process determines WHICH areas and WHAT items make the final list   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Two-panel: left vertical pipeline, right stacked areas"
spatial_anchors:
  panel_divider:
    x: 0.42
    y: 0.5
    content: "Vertical panel divider"
  stage_1:
    x: 0.20
    y: 0.15
    content: "Meta-review: assess current reporting quality"
  stage_2:
    x: 0.20
    y: 0.32
    content: "Delphi exercise: 200+ stakeholders, 3 rounds, 70% threshold"
  stage_3:
    x: 0.20
    y: 0.49
    content: "Consensus meeting: 15+ experts, virtual"
  stage_4:
    x: 0.20
    y: 0.66
    content: "Develop statement: piloting and refinement"
  stage_5:
    x: 0.20
    y: 0.83
    content: "Dissemination: journals, EQUATOR, TRIPOD website"
  area_a:
    x: 0.70
    y: 0.12
    content: "Code Availability"
  area_b:
    x: 0.70
    y: 0.22
    content: "Software Dependencies"
  area_c:
    x: 0.70
    y: 0.32
    content: "License Specification"
  area_d:
    x: 0.70
    y: 0.42
    content: "Code Structure / Modularity"
  area_e:
    x: 0.70
    y: 0.52
    content: "Testing"
  area_f:
    x: 0.70
    y: 0.62
    content: "Reproducibility"
  area_g:
    x: 0.70
    y: 0.72
    content: "Documentation by Pipeline Stage"
  area_h:
    x: 0.70
    y: 0.82
    content: "Long-term Archival"
  produces_arrow:
    x: 0.42
    y: 0.66
    content: "Arrow from Stage 4 to checklist areas"
```

## Content Elements

### Key Structures

| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Stage 1: Meta-review | `secondary_pathway` | Systematic review of current code reporting |
| Stage 2: Delphi | `primary_pathway` | Main consensus-building phase |
| Stage 3: Consensus | `primary_pathway` | Expert meeting for item refinement |
| Stage 4: Statement | `highlight_accent` | Produces the actual checklist |
| Stage 5: Dissemination | `secondary_pathway` | Publication and adoption |
| Area A-H cards | `callout_box` | Anticipated checklist domains |
| "Produces" arrow | `annotation` | Links development to output |

### Relationships/Connections

| From | To | Type | Label |
|------|-----|------|-------|
| Stage 1 | Stage 2 | Arrow (vertical) | Sequential flow |
| Stage 2 | Stage 3 | Arrow (vertical) | Sequential flow |
| Stage 3 | Stage 4 | Arrow (vertical) | Sequential flow |
| Stage 4 | Stage 5 | Arrow (vertical) | Sequential flow |
| Stage 4 | Areas A-H | Arrow (horizontal, dashed) | "produces" |

### Callout Boxes

| Title | Content | Location |
|-------|---------|----------|
| "Delphi Parameters" | 200+ participants, 3 rounds max, 70% agreement threshold | Adjacent to Stage 2 |
| "SPECULATIVE" | Areas A-H are anticipated from the protocol. Final checklist pending Delphi consensus. | Bottom of right panel |

### Numerical Annotations

| Value | Context |
|-------|---------|
| 200+ | Delphi participant target |
| 70% | Agreement threshold for item inclusion |
| 3 | Maximum Delphi rounds |
| 5 | Development stages |
| 8 | Anticipated checklist areas (A-H) |

## Text Content

### Labels (Max 30 chars each)

- Label 1: "1. Meta-review"
- Label 2: "2. Delphi exercise"
- Label 3: "3. Consensus meeting"
- Label 4: "4. Develop statement"
- Label 5: "5. Dissemination"
- Label 6: "A. Code Availability"
- Label 7: "B. Software Dependencies"
- Label 8: "C. License Specification"
- Label 9: "D. Code Structure"
- Label 10: "E. Testing"
- Label 11: "F. Reproducibility"
- Label 12: "G. Pipeline Documentation"
- Label 13: "H. Long-term Archival"
- Label 14: "200+ experts"
- Label 15: "70% threshold"
- Label 16: "3 rounds max"
- Label 17: "produces"

### Caption

TRIPOD-Code (Pollard et al. 2026) follows a 5-stage Delphi consensus process to develop reporting requirements for prediction model code repositories. The anticipated checklist covers 8 areas (A-H): code availability, software dependencies, licensing, code structure, testing, reproducibility, pipeline-stage documentation, and long-term archival. This repository proactively addresses all 8 areas (see docs/TRIPOD-CODE-COMPLIANCE.md).

## Prompts for Nano Banana Pro

### Style Prompt

Two-panel layout with vertical divider. Left panel: five numbered stages flowing downward with connecting arrows. Right panel: eight stacked area cards. Medical research aesthetic with matte colors. Clean white background. Stage 4 highlighted with gold accent. Dashed arrow from Stage 4 crossing to right panel. Subtle parameter annotations next to Stage 2. Disclaimer note at bottom of right panel.

### Content Prompt

Create a "TRIPOD-Code Development Process" two-panel diagram:

**LEFT PANEL: "5-Stage Development"**

Five vertically stacked stages with downward arrows:

1. **META-REVIEW** (gray card)
   - "Assess current reporting quality"

2. **DELPHI EXERCISE** (blue card, larger)
   - Badge: "200+ experts"
   - Badge: "3 rounds max"
   - Badge: "70% threshold"

3. **CONSENSUS MEETING** (blue card)
   - "15+ experts, virtual"

4. **DEVELOP STATEMENT** (gold-highlighted card)
   - "Pilot, refine, finalize"
   - Dashed arrow crossing to right panel with label "produces"

5. **DISSEMINATION** (gray card)
   - "Journals, EQUATOR Network, TRIPOD website"

**RIGHT PANEL: "Anticipated Checklist Areas"**

Eight stacked cards (A through H):
- A: Code Availability (GitHub, archives)
- B: Software Dependencies (lockfiles, Docker)
- C: License Specification (MIT, CITATION.cff)
- D: Code Structure / Modularity (ARCHITECTURE.md)
- E: Testing (automated tests, CI)
- F: Reproducibility (demo data, make commands)
- G: Documentation by Pipeline Stage
- H: Long-term Archival (Zenodo, DOI)

**FOOTER NOTE** (italic, smaller text):
"Areas A-H are anticipated from the protocol paper. Final checklist pending Delphi consensus."

### Refinement Notes

- Stage 2 (Delphi) should be the largest stage card -- it's the core methodology
- Stage 4 should have gold accent to show it produces the checklist
- The "produces" arrow should be visually prominent (dashed, with label)
- Right panel cards should be uniform size, stacked evenly
- Footer disclaimer is important -- these are anticipated, not confirmed

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-104",
    "title": "TRIPOD-Code Development and Anticipated Areas"
  },
  "content_architecture": {
    "primary_message": "TRIPOD-Code uses 5-stage Delphi consensus to develop code repository reporting requirements across 8 anticipated areas",
    "layout_flow": "Two-panel: left vertical pipeline, right stacked area cards",
    "spatial_anchors": {
      "stage_1": {"x": 0.20, "y": 0.15},
      "stage_2": {"x": 0.20, "y": 0.32},
      "stage_3": {"x": 0.20, "y": 0.49},
      "stage_4": {"x": 0.20, "y": 0.66},
      "stage_5": {"x": 0.20, "y": 0.83},
      "areas_panel": {"x": 0.70, "y": 0.50}
    },
    "key_structures": [
      {
        "name": "Meta-review",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Stage 1", "Current reporting quality"]
      },
      {
        "name": "Delphi exercise",
        "role": "primary_pathway",
        "is_highlighted": false,
        "labels": ["Stage 2", "200+ experts", "3 rounds", "70% threshold"]
      },
      {
        "name": "Consensus meeting",
        "role": "primary_pathway",
        "is_highlighted": false,
        "labels": ["Stage 3", "15+ experts"]
      },
      {
        "name": "Develop statement",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["Stage 4", "Produces checklist"]
      },
      {
        "name": "Dissemination",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Stage 5", "EQUATOR, journals"]
      },
      {
        "name": "Checklist Areas A-H",
        "role": "callout_box",
        "is_highlighted": false,
        "labels": ["Code Availability", "Dependencies", "License", "Structure", "Testing", "Reproducibility", "Documentation", "Archival"]
      }
    ],
    "callout_boxes": [
      {"heading": "DELPHI PARAMETERS", "body_text": "200+ participants, 3 rounds max, 70% agreement threshold for item inclusion."},
      {"heading": "SPECULATIVE", "body_text": "Areas A-H anticipated from protocol. Final checklist pending Delphi consensus."}
    ]
  }
}
```

## Alt Text

Two-panel diagram: left panel shows TRIPOD-Code 5-stage development pipeline (meta-review, Delphi with 200+ experts and 70% threshold, consensus meeting, statement development, dissemination); right panel shows 8 anticipated checklist areas (code availability, dependencies, license, structure, testing, reproducibility, documentation, archival). Arrow from stage 4 to checklist areas shows the development produces the reporting requirements.

## References

- Pollard T, Sounack T, et al. (2026). Protocol for development of a reporting guideline (TRIPOD-Code) for code repositories associated with diagnostic and prognostic prediction model studies. *Diagn Progn Res*, 10(4). [DOI: 10.1186/s41512-025-00217-4](https://doi.org/10.1186/s41512-025-00217-4)
- Collins GS, et al. (2024). TRIPOD+AI statement. *BMJ*, 385, e078378. [DOI: 10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
- See also: `docs/TRIPOD-CODE-COMPLIANCE.md` for this repository's proactive compliance mapping.

Note: This figure documents the TRIPOD-Code development methodology, not research results.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
