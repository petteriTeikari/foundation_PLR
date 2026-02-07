# fig-repo-03: Why Foundation Models?

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-03 |
| **Title** | Why Foundation Models for Biosignal Preprocessing? |
| **Complexity Level** | L1-L2 (Concept explanation) |
| **Target Persona** | PI, Biostatistician |
| **Location** | Root README.md |
| **Priority** | P2 (High) |

## Purpose

Explain to non-ML audiences why foundation models are interesting for this application and what advantages they might offer over traditional methods.

## Key Message

"Foundation models learned patterns from millions of time series, so they might recognize artifacts in pupil signals without being explicitly programmed to."

## Visual Concept

**Before/After comparison with training data visualization:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL METHODS                          │
│  ┌───────────┐                                                  │
│  │ Hand-coded│  →  "If value drops > 50% in 10ms, it's a blink"│
│  │   Rules   │                                                  │
│  └───────────┘      (Requires domain expert to write rules)     │
└─────────────────────────────────────────────────────────────────┘

                              VS

┌─────────────────────────────────────────────────────────────────┐
│                    FOUNDATION MODELS                            │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐              │
│  │ Pretrained│ →  │  See PLR  │ →  │ "This    │              │
│  │ on 1B+   │    │  signal   │    │ looks    │              │
│  │ datapoints│    │           │    │ like an  │              │
│  └───────────┘    └───────────┘    │ artifact"│              │
│                                     └───────────┘              │
│       (Learned patterns from ECG, stocks, sensors, etc.)       │
└─────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. Traditional approach: hand-coded rules
2. Foundation model approach: pretrained patterns
3. Scale indicator (1B+ datapoints pretraining)
4. Transfer learning concept (patterns from other domains)

### Optional Elements
1. Specific model names (MOMENT, UniTS, TimesNet)
2. Example artifact types they can detect
3. Limitation note (may not beat domain-specific approaches)

## Text Content

### Title Text
"Why Try Foundation Models?"

### Labels/Annotations
- Traditional: "Hand-coded rules require domain expertise"
- Foundation: "Pretrained on billions of time series datapoints"
- Transfer: "Patterns learned from ECG, sensors, stocks..."
- Application: "...applied to pupil signals"

### Caption (for embedding)
Foundation models are pretrained on massive time series datasets from diverse domains. They may recognize artifact patterns in pupil signals without domain-specific programming.

## Prompts for Nano Banana Pro

### Style Prompt
Educational infographic with friendly, accessible aesthetic. Two-panel comparison (top vs bottom). Use lightbulb or brain icon for the "learning" concept. Soft colors, not intimidating. Suitable for medical/clinical audience.

### Content Prompt
Create a comparison infographic:
TOP: "Traditional Methods" - show a rulebook/checklist icon leading to a simple decision
BOTTOM: "Foundation Models" - show a large cloud of diverse time series (ECG, stocks, sensors) leading to a brain/neural icon, which then examines a pupil signal

Include a scale indicator showing "1 Billion+ training datapoints"

### Refinement Notes
- Keep the visual simple - one concept per panel
- Avoid technical jargon in the figure itself
- The diversity of training data should be visually clear

## Alt Text

Comparison showing traditional rule-based methods versus foundation models that learn patterns from billions of diverse time series and apply them to pupil signals.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
