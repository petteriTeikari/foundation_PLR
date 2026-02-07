# Master Prompting Instructions for Nano Banana Pro
## Gemini ↔ Claude Code Handshake Protocol

**Version:** 2.4 (Optimized with Coordinate Translation & Placeholder Protocol)
**Updated:** 2026-01-26

---

## Overview

This file contains the **generic workflow** for generating figures with Nano Banana Pro.

**What goes WHERE:**
| Content | Location |
|---------|----------|
| Workflow & Protocol | This file (PROMPTING-INSTRUCTIONS.md) |
| Semantic Tags & Colors | STYLE-GUIDE.md (manuscript-specific) |
| Power Keywords | STYLE-GUIDE.md |
| Negative Prompts | STYLE-GUIDE.md |
| Content Elements | fig-XX-YY.md files |

---

## Figure Type Decision Matrix (v2.4)

**Not all figures should use Nano Banana Pro.** Use this matrix to choose the right generation method:

| Figure Type | Recommended Engine | Output Format | Why |
|-------------|-------------------|---------------|-----|
| Anatomical illustrations | **Nano Banana Pro** | PNG (raster) | Organic textures, 3D depth |
| Biological pathways | **Nano Banana Pro** | PNG (raster) | Complex visual metaphors |
| Conceptual diagrams | **Nano Banana Pro** | PNG (raster) | Abstract relationships |
| Timelines with dates | **R + ggplot2** | PDF/SVG (vector) | Exact text, publication quality |
| Scientific data plots | **R + ggplot2** | PDF/SVG (vector) | Statistical graphics excellence |
| Data charts/graphs | **R + ggplot2** (or Matplotlib) | PDF/SVG (vector) | Precise axis labels |
| Flowcharts with text | **Mermaid/Graphviz** | SVG (vector) | Perfect text rendering |
| Matrices/tables | **LaTeX/TikZ** | PDF (vector) | Structured data |

### The Compositing Trick

For complex figures requiring BOTH artistic rendering AND precise text:

1. Generate **Background Art** using Nano Banana Pro (prompt: "leave labeled areas as solid placeholder blocks")
2. Generate **Data Overlay** using Python (transparent background)
3. Composite in Figma/PowerPoint

---

## The Master Reproducibility Prompt (v2.4 Optimized)

**Instructions:** Copy and paste this block into a new Gemini chat when generating any figure. Adapt the "System Role" line to your manuscript domain.

```
System Role: You are acting as a Scientific Illustrator and Visual Compiler for a high-impact academic manuscript.

Input Files:
- STYLE-GUIDE.md: Contains global aesthetic specifications, color palette, and semantic tag mappings.
- fig-XX-YY.md: Contains content architecture, spatial anchors, and specific data elements.

Your Task:
Synthesize these two files into a single, high-density prompt for Nano Banana Pro image generation.

Strict Compilation Rules:

1. Visual Priority (The "Look"):
   Start the prompt with the rendering keywords from STYLE-GUIDE (e.g., "Octane Render", "Subsurface scattering") to lock in the aesthetic engine state immediately.

2. Coordinate Translation (The "Layout"):
   Do NOT simply pass x/y coordinates. You MUST translate `spatial_anchors` into explicit natural language descriptions:
   - x < 0.35 → "Positioned prominently on the left side"
   - x > 0.65 → "Positioned prominently on the right side"
   - y < 0.35 → "Located in the upper region"
   - y > 0.65 → "Located in the lower region"
   - Center (0.35-0.65) → "Central focal point"
   ALWAYS combine these (e.g., "An element [Name] anchored in the top-left quadrant...").

3. Semantic Injection (The "Colors"):
   Map every semantic tag in the content file to its exact visual definition in the Style Guide.
   - Example: If content says `role: "primary_pathway"`, your prompt must write: "deep blue (#2E5B8C) primary pathway."
   - Do NOT use the tag name alone; use the color + description.

4. The Placeholder Protocol (The "Text"):
   The image generator struggles with legible text. For every label, callout, or numerical value defined in the content file:
   - REQUEST: A distinct, solid-colored rectangular container (light gray #E0E0E0) or clear hairline callout line indicating WHERE the text belongs.
   - INSTRUCT: "Render blank placeholder blocks or clean label backgrounds for post-processing. Do NOT attempt to render complex character glyphs."
   - EXCEPTION: Single large letters (A, B, C) for panel identification may be requested.

5. Anti-Hallucination (The "Negative"):
   Always append the negative prompt from STYLE-GUIDE. Additionally, explicitly forbid: "garbled text, illegible glyphs, blurred alphanumeric characters, pseudo-text, alien hieroglyphs."

Immediate Action: Acknowledge you understand this compiler logic. I will now upload the specific figure file.
```

---

## Spatial Anchor System

Use normalized coordinates (0.0 to 1.0) instead of percentages for deterministic layout.

### Generic Format
```json
{
  "spatial_anchors": {
    "panel_left": {"x": 0.2, "y": 0.5},
    "panel_center": {"x": 0.5, "y": 0.5},
    "panel_right": {"x": 0.8, "y": 0.5}
  }
}
```

### Standard Layout Presets
| Layout Type | Anchor 1 | Anchor 2 | Anchor 3 |
|-------------|----------|----------|----------|
| Three-column weighted | (0.2, 0.5) | (0.5, 0.5) | (0.8, 0.5) |
| Three-column equal | (0.17, 0.5) | (0.5, 0.5) | (0.83, 0.5) |
| Two-column | (0.25, 0.5) | (0.75, 0.5) | - |
| Central hub | (0.5, 0.5) | radial at r=0.35 | - |
| Two-row stacked | (0.5, 0.3) | (0.5, 0.7) | - |

**Why coordinates beat percentages**: Nano Banana Pro interprets `x=0.2` more deterministically than "left 40%". This reduces layout ambiguity.

---

## Priority of Truth (Conflict Resolution)

When content and style guide have conflicting specifications:

| Priority | Source | Dictates |
|----------|--------|----------|
| 1 (highest) | Content file | TEXT content, element names, numerical values |
| 2 | Style guide | HEX codes, font specs, rendering keywords |
| 3 | Spatial anchors | Layout arrangement |
| Fallback | Content wins | If unresolvable conflict, text accuracy > aesthetics |

---

## Semantic Tag System

**IMPORTANT:** Semantic tags are defined in STYLE-GUIDE.md, NOT here.

### How It Works
1. Content files use **semantic tags** (e.g., `role: "primary_pathway"`)
2. STYLE-GUIDE.md maps tags to visual properties (e.g., `primary_pathway → #2E5B8C`)
3. Gemini synthesizes both when generating

### Generic Tag Categories
Every STYLE-GUIDE.md should define mappings for these categories:

| Category | Example Tags | Purpose |
|----------|--------------|---------|
| **Structure Roles** | `primary_pathway`, `secondary_pathway`, `highlight_accent` | Element importance hierarchy |
| **Status Indicators** | `healthy_normal`, `abnormal_warning` | Good/bad states |
| **Element Types** | `callout_box`, `annotation`, `title` | UI component types |
| **Connection Types** | `direct_connection`, `indirect_connection`, `inhibitory` | Relationship types |

**See your manuscript's STYLE-GUIDE.md for the complete semantic tag reference.**

---

## Decoupling Examples

### ❌ Poor (Coupled) Content Description
```json
{
  "name": "Key Element",
  "color": "#2E5B8C",
  "font": "Helvetica Bold",
  "highlight": "blue glow"
}
```

### ✅ Good (Decoupled) Content Description
```json
{
  "name": "Key Element",
  "role": "primary_pathway",
  "is_highlighted": true,
  "description": "Main functional component"
}
```

**Why this works:** The STYLE-GUIDE.md defines that `primary_pathway` = specific color and `is_highlighted` = specific effect. Changing the style requires editing ONE file, not all content files.

---

## Workflow for Figure Generation

### Step 1: Prepare Content File
Claude Code creates/updates `fig-XX-YY.md` with:
- Semantic tags (NOT colors/fonts)
- Layout flow and spatial anchors
- Key structures and relationships
- Numerical annotations
- Callout box content

### Step 2: Upload to Gemini
1. Upload STYLE-GUIDE.md (sets aesthetic baseline)
2. Upload fig-XX-YY.md (provides content)
3. Paste Master Reproducibility Prompt
4. Gemini synthesizes and generates with Nano Banana Pro

### Step 3: Iterate if Needed
- Style issues → Update STYLE-GUIDE.md
- Content issues → Update fig-XX-YY.md
- Re-generate

### Step 4: Vision Validator (v2.4 - NEW)

Use Gemini's multimodal capability to QC the generated output:

1. Upload the generated PNG
2. Paste this validation prompt:

```
Compare this image against the requirements in fig-XX-YY.md.

Checklist:
1. Does it contain all [N] required elements from key_structures?
2. Are the semantic colors accurate to STYLE-GUIDE.md?
3. Are placeholder regions clearly visible for text post-processing?
4. Is the background #FBF9F3 (Economist off-white)?
5. Are spatial anchors respected (left elements on left, etc.)?

List specific missing elements or discrepancies.
```

3. If issues found → iterate or regenerate

### Step 5: Save Output
- Save generated image to `generated/fig-XX-YY.png`
- Log any prompt adjustments

---

## Power Keywords & Negative Prompts

**These are manuscript-specific. See STYLE-GUIDE.md for:**
- Rendering engine keywords (e.g., Octane Render, subsurface scattering)
- Texture keywords (e.g., matte lithograph, archival paper)
- Diagrammatic keywords (e.g., hairline callouts, dense layout)
- Negative prompt (styles to avoid)

---

## File Structure Template

```
manuscript/figures/
├── STYLE-GUIDE.md              ← Aesthetic + semantic tags (manuscript-specific)
├── PROMPTING-INSTRUCTIONS.md   ← This file (generic workflow)
├── CONTENT-TEMPLATE.md         ← Template for content files
├── section-XX/
│   ├── fig-XX-01-name.md       ← Content only
│   ├── fig-XX-02-name.md
│   └── ...
├── nano-banana-pro/
│   ├── GEMINI-OPTIMIZED-PROMPT.md
│   └── GEMINI-RESPONSE-*.md
├── visual-references/          ← Target aesthetic examples
└── generated/                  ← Output images
```

---

## Checklist Before Generating

- [ ] **Figure type assessed** - Is Nano Banana Pro the right tool? (see Decision Matrix)
- [ ] STYLE-GUIDE.md has complete semantic tag mappings
- [ ] STYLE-GUIDE.md has power keywords and negative prompt
- [ ] Content file uses semantic tags (no hex codes)
- [ ] Content file has spatial_anchors section
- [ ] Content file has JSON export block
- [ ] Master Reproducibility Prompt v2.4 is ready to paste
- [ ] Placeholder Protocol understood (request boxes, not text)

## Checklist After Generating

- [ ] **Run Vision Validator** (upload image back to Gemini)
- [ ] All key_structures elements present
- [ ] Semantic colors match STYLE-GUIDE.md
- [ ] Background is #FBF9F3
- [ ] Placeholder regions visible for post-processing
- [ ] No garbled text or pseudo-glyphs

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.4 | 2026-01-26 | Added Coordinate Translation, Placeholder Protocol, Vision Validator, Figure Type Decision Matrix |
| 2.3 | 2026-01-19 | Domain-agnostic refactor |
| 2.2 | 2026-01-18 | Added spatial anchors |
| 2.0 | 2026-01-15 | Content-style decoupling |
