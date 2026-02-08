# fig-repo-101: PLR as the TSFM Proving Ground for Ophthalmic Biosignals

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-101 |
| **Title** | From PLR to Multimodal Ophthalmic AI: The Complexity Ladder |
| **Complexity Level** | L2-L3 (Technical vision) |
| **Target Persona** | Research Scientist, ML Engineer, PI |
| **Location** | Root README.md (Towards medical biosignal TSFMs section) |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:9 |

## Purpose

Position PLR as the simplest ophthalmic time series -- the ideal proving ground for TSFMs. If zero-shot preprocessing works for PLR, the approach transfers to more complex ophthalmic signals (ERG, VOG, ORG). This connects to the manuscript's discussion about multimodal ophthalmic AI and the Buergel late-fusion paradigm.

## Key Message

"PLR is the simplest ophthalmic time series. Proving TSFMs work here opens the door to automated preprocessing for ERG, VOG, and ORG -- and ultimately to multimodal ophthalmic foundation models."

## Visual Concept

A "complexity ladder" showing ophthalmic time series from simplest (PLR) to most complex (ORG), with TSFM applicability growing, feeding into a multimodal fusion architecture:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  FROM PLR TO MULTIMODAL OPHTHALMIC AI                                           │
│                                                                                 │
│  OPHTHALMIC TIME SERIES            SIGNAL                  TSFM                │
│  COMPLEXITY LADDER                 CHARACTERISTICS          STATUS              │
│  ═════════════════                                                              │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │                                                                │             │
│  │  SIMPLEST                                                      │             │
│  │  ┌──────────┐                                                  │             │
│  │  │   PLR    │  Univariate, 30 Hz, ~1800 samples, 1 min        │  ✅ THIS   │
│  │  │  ~~~~    │  Artifacts: blinks, tracking failures            │  REPO      │
│  │  │          │  MOMENT zero-shot competitive                    │             │
│  │  └──────────┘                                                  │             │
│  │       │                                                        │             │
│  │       ▼                                                        │             │
│  │  ┌──────────┐                                                  │             │
│  │  │   ERG    │  Multi-channel, 1-2 kHz, flash responses        │  🔮 NEXT   │
│  │  │  ╱╲_╱╲  │  Artifacts: baseline drift, noise, blinks       │  TARGET    │
│  │  │          │  a-wave, b-wave temporal dynamics                │             │
│  │  └──────────┘                                                  │             │
│  │       │                                                        │             │
│  │       ▼                                                        │             │
│  │  ┌──────────┐                                                  │             │
│  │  │   VOG    │  Multi-channel, 250-500 Hz, saccades+fixation  │  🔮 FUTURE │
│  │  │  ⌐¬_/\  │  Artifacts: blinks, saccade noise, head motion │             │
│  │  │          │  Same blink removal challenge as PLR            │             │
│  │  └──────────┘                                                  │             │
│  │       │                                                        │             │
│  │       ▼                                                        │             │
│  │  ┌──────────┐                                                  │             │
│  │  │   ORG    │  Noisy, requires averaging, emerging modality   │  🔮 FUTURE │
│  │  │  ___/‾‾  │  Individual traces barely visible above noise   │             │
│  │  │          │  Depolarization dip + sigmoid rise              │             │
│  │  └──────────┘                                                  │             │
│  │  MOST COMPLEX                                                  │             │
│  │                                                                │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                 │
│  MULTIMODAL FUSION (Buergel 2024 Paradigm)                                     │
│  ══════════════════════════════════════════                                     │
│                                                                                 │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    ┌──────────┐              │
│  │RETFound│  │ MOMENT │  │ Demographics│ │ VF    │    │  MLP     │              │
│  │ OCT/   │  │ PLR    │  │ Age, sex   │ │ data  │───→│  Head    │──→ Dx       │
│  │ Fundus │  │ embed. │  └────────┘  └────────┘    │  Network │              │
│  │ ~1024d │  │ ~96d   │                             └──────────┘              │
│  └───┬────┘  └───┬────┘                                                       │
│      └───────────┴────────────────────────────────────→ concat                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐            │
│  │  KEY INSIGHT: PLR is the proving ground. If TSFMs work for the │            │
│  │  simplest ophthalmic time series, they transfer upward. TSFM   │            │
│  │  preprocessing enables functional signal integration into       │            │
│  │  multimodal ophthalmic AI.                                      │            │
│  └─────────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Top-to-bottom complexity ladder (left half), multimodal fusion diagram (right half or bottom)"
spatial_anchors:
  complexity_ladder:
    x: 0.3
    y: 0.4
    content: "Four ophthalmic time series stacked by complexity"
  plr_entry:
    x: 0.3
    y: 0.18
    content: "PLR at top of ladder (simplest, current work)"
  erg_node:
    x: 0.3
    y: 0.35
    content: "ERG as next target"
  vog_node:
    x: 0.3
    y: 0.52
    content: "VOG as future target"
  org_node:
    x: 0.3
    y: 0.68
    content: "ORG as most complex (emerging)"
  fusion_diagram:
    x: 0.65
    y: 0.82
    content: "Late fusion: RETFound + MOMENT embeddings → MLP"
  insight_strip:
    x: 0.5
    y: 0.95
    content: "PLR as proving ground insight"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| PLR signal | `highlight_accent` | Simplest: univariate, 30 Hz, ~1800 samples, 1 min protocol |
| ERG signal | `primary_pathway` | Multi-channel, 1-2 kHz, flash-evoked a-wave + b-wave |
| VOG signal | `primary_pathway` | Multi-channel, 250-500 Hz, saccades + fixation + vergence |
| ORG signal | `secondary_pathway` | Emerging: noisy individual traces, requires averaging |
| RETFound encoder | `foundation_model` | Frozen structural imaging FM, ~1024-dim embeddings |
| MOMENT encoder | `foundation_model` | Frozen time series FM, ~96-dim embeddings after PCA |
| MLP head | `classification` | Lightweight trainable prediction network |
| Complexity arrow | `primary_pathway` | Increasing complexity from PLR to ORG |
| Concatenation | `primary_pathway` | Multimodal embedding fusion |
| Blink challenge | `abnormal_warning` | Shared across PLR, ERG, VOG |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| PLR | ERG | Complexity arrow | "More channels, higher Hz" |
| ERG | VOG | Complexity arrow | "Saccades + gaze events" |
| VOG | ORG | Complexity arrow | "Emerging, very noisy" |
| PLR | PLR "THIS REPO" | Highlight | "Proving ground" |
| RETFound + MOMENT | MLP | Concatenation | "Late fusion" |
| PLR success | All other signals | Implication arrow | "If here, then there" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "THIS REPO" | PLR preprocessing with TSFM: proven concept | Top of ladder |
| "NEXT TARGET" | ERG: same blink artifacts, multi-channel extension | Middle of ladder |
| "BUERGEL PARADIGM" | Frozen encoders + lightweight MLP = practical multimodal AI | Bottom fusion |
| "KEY INSIGHT" | PLR is the proving ground for ophthalmic TSFM preprocessing | Bottom strip |

## Text Content

### Labels (Max 30 chars each)
- Label 1: PLR (This Repo)
- Label 2: ERG (Next Target)
- Label 3: VOG (Future)
- Label 4: ORG (Emerging)
- Label 5: Univariate, 30 Hz
- Label 6: Multi-ch, 1-2 kHz
- Label 7: Multi-ch, 250-500 Hz
- Label 8: RETFound (~1024d)
- Label 9: MOMENT (~96d)
- Label 10: MLP Head
- Label 11: Late Fusion

### Caption (for embedding)
PLR is the simplest ophthalmic time series -- the ideal TSFM proving ground. Success here transfers to ERG (multi-channel), VOG (saccade events), and ORG (emerging). Robust TSFM preprocessing enables functional signal integration into multimodal ophthalmic AI via the Buergel late-fusion paradigm.

## Prompts for Nano Banana Pro

### Style Prompt
Medical illustration quality, ray-traced ambient occlusion, soft volumetric lighting, Economist off-white background (#FBF9F3), elegant scientific illustration, clean editorial layout, hairline vector callout lines, professional data visualization, Scientific American infographic style. Eye anatomy elements for the ophthalmic context: retinal layers, pupil, photoreceptors visible as small background elements.

### Content Prompt
Create a two-part infographic: a complexity ladder and a multimodal fusion diagram.

UPPER HALF - COMPLEXITY LADDER: Four ophthalmic time series stacked vertically from simplest (top) to most complex (bottom), connected by downward arrows showing increasing complexity:

1. PLR (top, highlighted in gold): A simple sinusoidal-like pupil diameter trace at 30 Hz. Label: "Univariate, 30 Hz, 1 min." Badge: "THIS REPO" in gold.
2. ERG (second): Multi-channel flash-evoked response showing characteristic a-wave dip and b-wave peak. Label: "Multi-channel, 1-2 kHz." Badge: "Next Target."
3. VOG (third): Multi-channel eye movement trace with saccadic jumps and smooth fixation periods. Label: "250-500 Hz, saccades." Badge: "Future."
4. ORG (bottom): Very noisy individual trace barely above noise floor, with averaged trace showing depolarization dip and sigmoid recovery. Label: "Emerging modality." Badge: "Future."

An arrow along the left side shows "increasing signal complexity" from top to bottom. A shared artifact note shows blink removal as a common challenge across PLR, ERG, and VOG.

LOWER HALF - MULTIMODAL FUSION: The Buergel late-fusion paradigm showing: RETFound (deep blue box, ~1024-dim, frozen snowflake icon) producing structural embeddings, MOMENT (teal box, ~96-dim, frozen snowflake) producing functional embeddings, concatenated with demographics and visual field data, feeding into a trainable MLP head (with flame icon) for diagnosis.

BOTTOM STRIP: Key insight callout: PLR proves TSFMs work for ophthalmic time series, enabling functional signal integration into multimodal AI.

### Refinement Notes
- PLR should be clearly the simplest and most prominent (gold accent)
- The complexity increase should be visually obvious through waveform complexity
- ORG should look genuinely noisy to convey the emerging nature
- The fusion diagram should feel like a natural consequence of the ladder
- Include subtle retinal anatomy background elements for ophthalmic context
- Frozen (snowflake) vs trainable (flame) icons for the Buergel paradigm

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-101",
    "title": "From PLR to Multimodal Ophthalmic AI: The Complexity Ladder"
  },
  "content_architecture": {
    "primary_message": "PLR is the simplest ophthalmic time series -- the proving ground for TSFM preprocessing that transfers to ERG, VOG, ORG and enables multimodal ophthalmic AI",
    "layout_flow": "Top-to-bottom complexity ladder feeding into multimodal fusion diagram",
    "spatial_anchors": {
      "plr": {"x": 0.3, "y": 0.18},
      "erg": {"x": 0.3, "y": 0.35},
      "vog": {"x": 0.3, "y": 0.52},
      "org": {"x": 0.3, "y": 0.68},
      "fusion": {"x": 0.65, "y": 0.82},
      "insight": {"x": 0.5, "y": 0.95}
    },
    "key_structures": [
      {"name": "PLR", "role": "highlight_accent", "is_highlighted": true, "labels": ["30 Hz", "Univariate", "THIS REPO"]},
      {"name": "ERG", "role": "primary_pathway", "is_highlighted": false, "labels": ["1-2 kHz", "Multi-channel", "Next Target"]},
      {"name": "VOG", "role": "primary_pathway", "is_highlighted": false, "labels": ["250-500 Hz", "Saccades", "Future"]},
      {"name": "ORG", "role": "secondary_pathway", "is_highlighted": false, "labels": ["Emerging", "Very noisy"]},
      {"name": "RETFound", "role": "foundation_model", "is_highlighted": true, "labels": ["~1024-dim", "Structural"]},
      {"name": "MOMENT", "role": "foundation_model", "is_highlighted": true, "labels": ["~96-dim", "Functional"]},
      {"name": "MLP Head", "role": "classification", "is_highlighted": false, "labels": ["Trainable", "Diagnosis"]}
    ],
    "callout_boxes": [
      {"heading": "THIS REPO", "body_text": "PLR preprocessing with TSFM: proven concept"},
      {"heading": "BUERGEL PARADIGM", "body_text": "Frozen encoders + lightweight MLP = practical multimodal AI"},
      {"heading": "KEY INSIGHT", "body_text": "PLR proves TSFMs work for ophthalmic time series, enabling functional signal integration"}
    ]
  }
}
```

## Alt Text

Complexity ladder showing four ophthalmic time series from simplest to most complex: PLR (30 Hz, univariate, highlighted as current work), ERG (1-2 kHz, multi-channel), VOG (250-500 Hz, saccade events), ORG (emerging, noisy). Below, the Buergel late-fusion paradigm shows RETFound structural embeddings and MOMENT functional embeddings concatenated into an MLP head for multimodal diagnosis. PLR is the proving ground for TSFM preprocessing across ophthalmic signals.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
