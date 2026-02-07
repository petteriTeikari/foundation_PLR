# Figure Content Specification Template

**Version:** 2.3 (Aligned with Manuscript Template)
**See STYLE-GUIDE.md for all visual/aesthetic specifications.**

## Purpose

This template defines the CONTENT of a figure - WHAT to show, not HOW it looks.
Style/aesthetic is defined separately in STYLE-GUIDE.md.

**Key Principle**: Content specs use **semantic tags** (not colors/fonts). The STYLE-GUIDE.md maps semantic tags to visual properties.

---

## Figure Type Classification

**IMPORTANT:** Before filling out this template, determine if Nano Banana Pro is the right tool.

### Decision Matrix

| Your Figure Has... | Recommended Engine | This Template? |
|--------------------|-------------------|----------------|
| Eye anatomy, retinal cells, organic | **Nano Banana Pro** | YES |
| Conceptual pathways, flows | **Nano Banana Pro** | YES |
| Visual metaphors, 3D elements | **Nano Banana Pro** | YES |
| Pipeline diagrams, process flows | **Nano Banana Pro** | YES |
| Data plots with exact values | **R + ggplot2** | NO - use code |
| Complex flowcharts with text | **Mermaid** | NO - use diagram code |
| Matrices, structured tables | **LaTeX/TikZ** | NO |

### Hybrid Figures (Compositing)

For figures needing BOTH artistic rendering AND precise text:

```yaml
generation_strategy: "hybrid"
background_engine: "nano_banana"  # For visual elements
overlay_engine: "figma"           # For precise text
```

---

## Template

```markdown
# fig-repo-{NN}: {Title}

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-{NN} |
| **Title** | {Descriptive title} |
| **Complexity Level** | L1/L2/L3/L4 |
| **Target Persona** | PI / Biostatistician / Research Scientist / ML Engineer / All |
| **Location** | Where this figure will appear (README, docs/, etc.) |
| **Priority** | P1 (Critical) / P2 (High) / P3 (Medium) / P4 (Low) |

## Purpose

{1-2 sentences explaining WHY this figure exists and what question it answers}

## Key Message

{Single sentence: the one thing the viewer should understand after seeing this}

## Visual Concept

{Describe the visual approach: flowchart, comparison table, metaphor illustration, etc.}
{Include ASCII mockup of layout}

## Spatial Anchors

```yaml
layout_flow: "Describe flow direction"
spatial_anchors:
  element_1:
    x: 0.2
    y: 0.5
    content: "Description"
  element_2:
    x: 0.5
    y: 0.5
    content: "Description"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Element 1 | `primary_pathway` | Main component |
| Element 2 | `secondary_pathway` | Supporting component |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| Element A | Element B | Arrow | "causes" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "KEY INSIGHT" | Main takeaway | Top right |

## Text Content

### Labels (Max 30 chars each)
- Label 1: {text}
- Label 2: {text}

### Caption (for embedding)
{1-2 sentence caption for when figure is embedded in docs}

## Prompts for Nano Banana Pro

### Style Prompt
{Based on STYLE-GUIDE.md - include power keywords, background color, negative prompt reference}

### Content Prompt
{Describe what to show - use semantic tags translated to visual descriptions}

### Refinement Notes
{Any specific adjustments after initial generation}

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-NN",
    "title": "Figure Title"
  },
  "content_architecture": {
    "primary_message": "Single sentence",
    "layout_flow": "Flow description",
    "spatial_anchors": {
      "left": {"x": 0.2, "y": 0.5},
      "center": {"x": 0.5, "y": 0.5}
    },
    "key_structures": [
      {
        "name": "Structure Name",
        "role": "semantic_tag",
        "is_highlighted": true,
        "labels": ["Label 1"]
      }
    ],
    "callout_boxes": [
      {"heading": "TITLE", "body_text": "Explanation."}
    ]
  }
}
```

## Alt Text

{Accessible description for screen readers, 125 chars max}

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
```

---

## Semantic Tag Reference

Use these semantic tags instead of colors. STYLE-GUIDE.md maps tags to visual properties.

### ⚠️ CRITICAL: WORKFLOW USE ONLY

**Semantic tags are for INTERNAL workflow communication between Claude Code and Gemini.**

**They must NEVER appear as visible text in generated images:**
- When writing prompts, TRANSLATE tags to natural language
- Example: `primary_pathway` → "deep blue main pathway"
- Example: `melanopsin` → "gold-colored ipRGC elements"

**If Nano Banana Pro renders tag names as text, add to negative prompt:**
```
semantic tag names visible, "primary_pathway" text, "cone_L" text
```

### Structure Roles (Visual Neuroscience)
| Semantic Tag | When to Use |
|--------------|-------------|
| `primary_pathway` | Main/highlighted pathways, key elements |
| `secondary_pathway` | Supporting pathways, background |
| `foundation_model` | AI/FM elements (deep blue) |
| `traditional_method` | Traditional/baseline methods (gray) |
| `rod_photoreceptor` | Rod cells |
| `cone_L`, `cone_M`, `cone_S` | Cone subtypes |
| `melanopsin` | ipRGC/melanopsin elements (gold) |
| `healthy_normal` | Normal/success indicators |
| `abnormal_warning` | Warning/error indicators |
| `highlight_accent` | Key emphasis (gold) |

### Pipeline Stages
| Semantic Tag | When to Use |
|--------------|-------------|
| `raw_signal` | Input data, noisy signal |
| `outlier_detection` | First processing stage |
| `imputation` | Reconstruction stage |
| `features` | Extracted features |
| `classification` | Final prediction |

### Element Types
| Semantic Tag | When to Use |
|--------------|-------------|
| `callout_box` | Explanatory text boxes |
| `annotation` | Small labels |
| `title` | Main title |
| `section_heading` | Panel headings |

**DO NOT write**: `"color": "#2E5B8C"`
**DO write**: `"role": "primary_pathway"`

---

## Complexity Levels

| Level | Audience | Content Depth |
|-------|----------|---------------|
| **L1** | PI/Non-technical | Single concept, metaphor-based |
| **L2** | Biostatistician | Process overview, key steps |
| **L3** | Research Scientist | Technical details, code snippets |
| **L4** | ML Engineer | Architecture, implementation |

---

## Label Guidelines

- **Max length:** 30 characters (prevent wrapping)
- **Use singular:** "Outlier method" not "Outlier methods"
- **Abbreviate compounds:** "FM" not "Foundation Model" (expand in caption)
- **Split long labels:** Use multiple annotations

---

## Quality Checklist

Before finalizing:

- [ ] Primary message is clear in one sentence
- [ ] 10-20 content elements defined
- [ ] Semantic tags used (no hex codes)
- [ ] Spatial anchors specified
- [ ] Labels under 30 characters
- [ ] JSON export block included
- [ ] Alt text provided

---

*Aligned with manuscript CONTENT-TEMPLATE.md v2.3*
