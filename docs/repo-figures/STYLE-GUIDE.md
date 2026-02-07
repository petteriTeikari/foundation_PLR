# STYLE GUIDE v2.0 - Repository Documentation Infographics
## Target: Elegant Educational Infographic / Scientific Textbook Quality

**Version:** 2.0 (Aligned with Manuscript Visual Branding)
**Updated:** 2026-01-31
**Scope:** Repository documentation infographics (12 figures)
**Parent Style:** `sci-llm-writer/manuscripts/foundationPLR/.../STYLE-GUIDE.md`

---

## STYLE INTERPOLATION APPROACH

This guide inherits **75%** from the manuscript's established visual branding and **25%** from Economist ggplot2 aesthetics for GitHub-specific adjustments.

| Aspect | Manuscript (75%) | Economist Tweak (25%) |
|--------|------------------|----------------------|
| **Background** | #FBF9F3 (off-white) | ✓ Inherited |
| **3D Rendering** | Medical illustration quality | **NO GLOWING SCI-FI** - elegant, matte |
| **Typography** | Helvetica/Sans-serif | Slightly lighter weights for screen |
| **Density** | Dense infographic | Medium density (GitHub readability) |
| **Colors** | Deep semantic palette | Muted, professional |
| **Diagrams** | Octane/Ray-traced | **Mermaid-style elegance** (like fig-26-06) |

### CRITICAL: NO SCI-FI AESTHETICS

**BANNED for GitHub repo figures:**
- Glowing elements, neon highlights
- Futuristic UI elements
- Cyberpunk aesthetics
- Holographic effects
- Pure black backgrounds

**REQUIRED aesthetic:**
- Elegant scientific illustration (like attached examples)
- Warm, professional tones
- Matte finishes on 3D elements
- Natural lighting, no harsh glow

---

## DOMAIN FOCUS: VISUAL NEUROSCIENCE

This repository is about **pupillary light reflex (PLR), retina, time series foundation models, and clinical AI validation**.

### Anatomical Illustrations

**USE (Visual System):**
- Eye cross-sections
- Retinal layers
- Pupil/iris anatomy
- Photoreceptor cells (rods, cones, ipRGCs)
- Optic nerve (only as it relates to eye)

**DO NOT USE (Beyond Scope):**
- Brain tissue
- General neural anatomy
- Cortical structures
- Brain cross-sections
- Any non-ocular anatomy

### Domain Keywords for Prompts
```
pupillometry, pupillary light reflex, PLR signal, retinal ganglion cells,
melanopsin, photoreceptor, iris constriction, time series analysis,
biosignal preprocessing, foundation models, glaucoma screening,
clinical AI, STRATOS metrics, biostatistics, signal decomposition
```

---

## POWER KEYWORDS (For AI Prompts)

### Rendering Keywords (75% Manuscript)
```
Medical illustration quality, Ray-traced ambient occlusion,
Soft volumetric lighting, Subsurface scattering for membranes,
Subtle rim lighting on biological contours, Cinematic macro biology,
Scientific American infographic style
```

### Texture Keywords
```
High-grade matte lithograph, Archival paper texture,
Non-reflective eggshell finish, Economist off-white background,
Subtle parchment grain, ELEGANT not flashy
```

### Diagrammatic Keywords (25% Economist Influence)
```
Clean editorial layout, ggplot2 minimalist aesthetic,
Hairline vector callout lines, Professional data visualization,
Muted color accents, Clear information hierarchy,
Mermaid-style flow elegance (like error propagation example)
```

---

## COLOR PALETTE v2.0

### ⚠️ CRITICAL: INTERNAL REFERENCE ONLY

**The color hex codes and semantic tags below are for INTERNAL WORKFLOW USE ONLY.**

**NEVER let these appear as visible text in generated images:**
- NO hex codes (#FBF9F3, #2E5B8C, etc.) rendered as text
- NO semantic tag names (`primary_pathway`, `cone_L`, etc.) rendered as text
- NO "Hex:", "RGB:", "Semantic Tag:" labels in the figure

**These are mapping references for prompt construction, NOT content to display.**

When prompting Nano Banana Pro:
- **DO**: "Use deep blue for the main pathway"
- **DON'T**: "Use #2E5B8C for primary_pathway"

Add to negative prompt if this glitch occurs:
```
visible hex codes, color codes as text, "#" followed by numbers,
semantic tag names as labels, technical markup visible
```

### Background (MANDATORY)
| Element | Hex | RGB | Description |
|---------|-----|-----|-------------|
| **Primary Background** | #FBF9F3 | 251, 249, 243 | Economist off-white (NON-NEGOTIABLE) |
| Secondary Background | #F5F3EF | 245, 243, 239 | Slightly darker for panels |
| Callout Box Background | #FFFFFF | 255, 255, 255 | White for contrast |

### Typography Colors
| Element | Hex | Description |
|---------|-----|-------------|
| **Main Headings** | #1A1A1A | Near-black |
| Subheadings | #333333 | Dark charcoal |
| Body Text | #4A4A4A | Medium charcoal |
| Labels/Captions | #666666 | Lighter gray |

### Semantic Color Mapping (Visual Neuroscience Themed)

**⚠️ INTERNAL REFERENCE - DO NOT RENDER AS TEXT IN IMAGES**

| Semantic Tag | Hex | Usage |
|--------------|-----|-------|
| `primary_pathway` | #2E5B8C | Key pathways (deep blue) |
| `secondary_pathway` | #666666 | Supporting elements (gray) |
| `rod_photoreceptor` | #7D6B5D | Rod cells (brown) |
| `cone_L` | #C44536 | L-cones / Red pathway |
| `cone_M` | #5B8C3E | M-cones / Green pathway |
| `cone_S` | #4A7EAA | S-cones / Blue pathway |
| `melanopsin` | #D4A03C | ipRGC/melanopsin (gold) |
| `healthy_normal` | #5B8C3E | Normal/success indicators |
| `abnormal_warning` | #C44536 | Warning/error indicators |
| `highlight_accent` | #D4A03C | Key emphasis (gold) |
| `neural_tissue` | #D4C4B5 | Neural structures (beige) |
| `foundation_model` | #2E5B8C | AI/FM elements (deep blue) |
| `traditional_method` | #666666 | Traditional methods (gray) |

### Pipeline Stage Colors (for fig-repo-02)
| Stage | Hex | Description |
|-------|-----|-------------|
| Raw Signal | #C44536 | Red - problems to fix |
| Outlier Detection | #D4A03C | Gold - first processing |
| Imputation | #4A7EAA | Blue - reconstruction |
| Features | #5B8C3E | Green - extracted info |
| Classification | #2E5B8C | Deep blue - final prediction |

---

## TYPOGRAPHY

| Level | Font | Weight | Color | GitHub Note |
|-------|------|--------|-------|-------------|
| H1 - Title | Helvetica/Arial | Bold (700) | #1A1A1A | Large, clear |
| H2 - Section | Helvetica/Arial | Semibold (600) | #1A1A1A | Panel titles |
| H3 - Subsection | Sans-serif | Medium (500) | #333333 | Labels |
| Body | Sans-serif | Regular (400) | #4A4A4A | Descriptions |
| Labels | Sans-serif | Regular (400) | #666666 | Small annotations |
| Code | Monospace | Regular (400) | #333333 | Code snippets |

---

## FORMAT SPECIFICATIONS

### Repository Figures (GitHub-Optimized)
| Spec | Value | Notes |
|------|-------|-------|
| **Aspect Ratio** | 16:9 or 3:2 | Landscape for README embedding |
| **Resolution** | 300 DPI (export), 72 DPI display | High quality source |
| **Density** | 10-20 elements | Medium (readable at small sizes) |
| **Width** | 1200-1920px | Good for GitHub rendering |
| **Panels** | 2-6 panels | Organized layouts |
| **Max Label** | 30 characters | Prevent wrapping |

### Rendering Specifications
| Surface Type | Rendering |
|--------------|-----------|
| Eye anatomy | Matte with subsurface scattering |
| Cell bodies | Soft, organic, NON-GLOWING |
| Membranes | Semi-translucent, soft highlights |
| Flowchart elements | Clean vectors, matte fills |
| Text/Diagrams | Flat, clean, anti-aliased |

### Lighting Setup
- **Primary:** Top-left soft diffuse (natural daylight feel)
- **Rim light:** 10-15% intensity (subtle edge definition)
- **Ambient:** Soft ambient occlusion
- **Shadows:** Soft (10-20% opacity, NOT harsh)
- **NO:** Glowing edges, neon highlights, dramatic backlighting

---

## MERMAID-STYLE ELEGANCE

Reference: `fig-26-06-error-propagation-llm-pipeline.png` from manuscript figures.

### What Makes It Elegant
- Flowing organic arrows (not rigid right-angles)
- Muted color gradients (cream → gold → teal → blue → red)
- Rounded containers with soft shadows
- Clear visual hierarchy through size and color intensity
- Subtle decorative swooshes (error propagation arrow)
- Integrated callout boxes with clean borders

### How to Achieve in Prompts
```
Elegant flowchart in the style of high-end data visualization,
Mermaid-diagram aesthetic elevated to magazine quality,
Organic flowing arrows with subtle gradients,
Rounded rectangular nodes with soft drop shadows,
Color progression showing information flow,
Integrated callout boxes with hairline borders,
Clean sans-serif typography, NO GARBLED TEXT
```

---

## LAYOUT TEMPLATES

### Hero Figure (fig-repo-01)
```
┌─────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE: Large, Bold, Centered                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     →      ┌──────────────┐                │
│   │  LEFT PANEL  │   flow     │ RIGHT PANEL  │                │
│   │  (Before)    │   arrow    │  (After)     │                │
│   └──────────────┘            └──────────────┘                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  BOTTOM: Key takeaway / summary strip                           │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Panel Comparison (fig-repo-02)
```
┌─────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE                                                   │
├──────┬──────┬──────┬──────┬──────────────────────────────────┤
│ A    │ B    │ C    │ D    │ E                                 │
│      │      │      │      │                                   │
│Stage1│Stage2│Stage3│Stage4│Stage5                             │
│ box  │ box  │ box  │ box  │ box                               │
├──────┴──────┴──────┴──────┴──────────────────────────────────┤
│  ═══════════════════════════════════════════════════════►     │
│                    Error Propagation Arrow                     │
├─────────────────────────────────────────────────────────────────┤
│  Summary metrics / key numbers                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Flowchart/Decision (fig-repo-07)
```
┌─────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE                                                   │
├─────────────────────────────────────────────────────────────────┤
│                      ┌─────────┐                               │
│                      │  START  │                               │
│                      └────┬────┘                               │
│                           │                                     │
│                      ┌────▼────┐                               │
│                      │ Check?  │◇────────┐                     │
│                      └────┬────┘         │                     │
│                      [Yes]│          [No]│                     │
│                      ┌────▼────┐    ┌────▼────┐               │
│                      │ Action A│    │ Action B│               │
│                      └─────────┘    └─────────┘               │
│                                                                 │
│  ┌──────────────────┐            ┌──────────────────┐         │
│  │ CALLOUT: Detail  │            │ CALLOUT: Warning │         │
│  └──────────────────┘            └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## NEGATIVE PROMPT (MANDATORY)

### Standard Exclusions
```
cartoon, clip art, flat illustration, minimalist, abstract, artistic interpretation,
watercolor, sketch, hand-drawn, pencil, charcoal, vintage, retro, 1970s,
risograph, halftone, screen print, grainy texture, worn paper, collage,
neon colors, oversaturated, pure white background, harsh shadows,
sparse layout, corporate infographic, stock illustration, generic,
low detail, simplified, icon-based, emoji, cute, playful
```

### Anti-Sci-Fi (CRITICAL FOR GITHUB)
```
glowing effects, neon glow, sci-fi aesthetic, cyberpunk, futuristic,
holographic, plasma effects, energy beams, laser lights, tron-style,
matrix code, digital artifacts, glitch effects, dark mode extremes,
pure black backgrounds, dramatic rim lighting, lens flares
```

### Anti-Brain (Domain Scope)
```
brain tissue, brain anatomy, cortex, neural network biology (use diagrams instead),
brain scan, MRI brain, brain cross-section, cerebrum, cortical layers
```

### Anti-Text-Garbling
```
garbled text, illegible glyphs, blurred alphanumeric characters,
pseudo-text, alien hieroglyphs, corrupted letters, gibberish words,
misspelled labels, broken typography, scrambled characters
```

### Anti-Technical-Markup (Common Glitch Prevention)
```
visible hex codes, color codes as text, "#RRGGBB" visible,
semantic tag names as labels, "primary_pathway" visible,
"cone_L" visible, "melanopsin" as label text,
technical markup, internal reference text visible,
yaml syntax visible, json visible in image
```

### Anti-Prompt-Leakage (CRITICAL - Prompt Instructions as Text)

**PROBLEM**: Nano Banana Pro sometimes renders prompt keywords/instructions as visible text in the image.

**Example failure**: "(non-glowing, elegant)" appeared as text under "Foundation Models" label.

```
"non-glowing" as text, "elegant" as text, "matte" as text,
"(non-glowing, elegant)" visible, parentheses with style words,
prompt instructions as labels, style keywords visible,
"subsurface" text, "ray-traced" text, "volumetric" text,
"high-quality" text, "detailed" text, "professional" text,
rendering keywords as labels, aesthetic descriptors visible,
instruction text in parentheses, meta-instructions rendered,
"soft lighting" as label, "medical illustration" text
```

**PREVENTION**: When writing positive prompts:
1. Keep style keywords at the START of prompt (before content)
2. Never put style words near element labels
3. Use natural descriptions, not adjective lists
4. Avoid parenthetical style qualifiers near content

**DO**: "Foundation Models section showing MOMENT and UniTS network diagrams"
**DON'T**: "Foundation Models (non-glowing, elegant) MOMENT UniTS"

### Combined Negative Prompt (COPY THIS)
```
cartoon, clip art, flat illustration, minimalist, watercolor, sketch,
hand-drawn, vintage, retro, neon colors, oversaturated, pure white background,
harsh shadows, corporate infographic, stock illustration, generic,
glowing effects, neon glow, sci-fi aesthetic, cyberpunk, futuristic,
holographic, plasma effects, dark mode, pure black backgrounds,
brain tissue, brain anatomy, cortex, MRI brain,
garbled text, illegible glyphs, blurred characters, pseudo-text,
corrupted letters, gibberish words, broken typography,
visible hex codes, "#" followed by six characters, color codes as text,
semantic tag names visible, "primary_pathway" text, "cone_L" text,
technical markup visible, yaml syntax, json syntax,
"non-glowing" as text, "elegant" as text, "matte" as text,
parentheses with style words, prompt instructions as labels,
style keywords visible, rendering keywords as labels,
"(non-glowing, elegant)" visible, aesthetic descriptors as text
```

---

## QUALITY CHECKLIST

### Before Generation
- [ ] Background specified as #FBF9F3 (Economist off-white)
- [ ] NO glowing/sci-fi keywords in positive prompt
- [ ] Domain focus: eye/retina only (no brain)
- [ ] Negative prompt includes anti-sci-fi and anti-brain
- [ ] Semantic tags used (not hex codes in content)
- [ ] 10-20 content elements defined
- [ ] Panel layout sketched

### After Generation
- [ ] Background IS Economist off-white (#FBF9F3)
- [ ] NO glowing effects on any elements
- [ ] Eye/retina anatomy only (no brain)
- [ ] Matte, elegant finish on 3D elements
- [ ] Clean typography (or placeholder boxes)
- [ ] Looks like Scientific American / textbook quality
- [ ] Could appear in a journal without embarrassment

---

## EXAMPLE REFERENCE FIGURES

Located in manuscript repository:
- `fig-01-07-multi-paradigm-plr-protocol.png` - Multi-panel layout excellence
- `fig-01b-XX-underexplored-stimulus-paradigms.png` - Scientific data visualization
- `fig-XX-radiative-transfer.png` - Anatomical illustration with equations
- `fig-26-06-error-propagation-llm-pipeline.png` - **GOLD STANDARD** for flowcharts

### What These Examples Get Right
1. **Warm, professional color palette** (gold, teal, muted red)
2. **Organic flowing arrows** (not rigid)
3. **Elegant eye anatomy** (medical illustration quality)
4. **Dense but readable** (information-rich, clear hierarchy)
5. **Integrated callouts** (not floating randomly)
6. **NO SCI-FI** (no glow, no neon, no cyberpunk)

---

## APPENDIX: Economist ggplot2 Influence (25%)

The 25% Economist influence adds:

1. **Cleaner minimalism** for GitHub embedding
2. **Slightly lighter typography** for screen reading
3. **More whitespace** for small-scale viewing
4. **Muted accent colors** rather than saturated
5. **Grid-aligned layouts** for professional feel

This is NOT about making it look like an Economist chart - it's about professional polish suitable for a high-quality GitHub repository.

---

*Based on manuscript STYLE-GUIDE.md v1.1 with 75/25 interpolation for GitHub documentation*
*For comprehensive literature review figures, use the original STYLE-GUIDE.md*
