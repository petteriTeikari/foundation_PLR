# fig-trans-02: The TSFM Hype vs Reality

**Status**: 📋 PLANNED
**Tier**: 1 - Honest Limitations
**Target Persona**: Data scientists, ML engineers, skeptics

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-02 |
| Type | Quote collage + evaluation framework |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Present an academically honest assessment of Time Series Foundation Model claims vs published evidence from literature, helping practitioners evaluate TSFM claims critically.

---

## 3. Key Message

> "TSFMs show promise for specific tasks, but the 'revolutionize everything' claims don't hold up. Apply critical thinking: check baseline comparisons, metric normalization, and domain fit."

---

## 4. Literature Sources

| Source | Quote/Finding | Implication |
|--------|---------------|-------------|
| HackerNews on TimeGPT | "Difficult to take this paper seriously when they dismiss ARIMA" | Cherry-picked baselines |
| Jin et al. 2024 | "LLMs fail on datasets without clear periodicity" | Limited applicability |
| Schoenegger & Park 2023 | "GPT-4 forecasts ≈ 50% on real tournaments" | LLMs don't generalize |
| @predict_addict 2024 | "Normalized metrics hide bad forecasts" | Deceptive evaluation |
| Zeng et al. 2022 | "Are Transformers Effective? Linear models competitive" | Simpler often wins |
| Hewamalage et al. 2022 | "ML researchers adopt flawed evaluation practices" | Methodology problems |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  TSFM HYPE vs REALITY                                                      │
│  A Critical Evaluation Framework                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE HYPE                              THE REALITY                         │
│  ────────                              ──────────                          │
│                                                                            │
│  "Foundation models will              "LLMs fail on datasets without       │
│   revolutionize time series"           clear periodicity"                  │
│                                        — Jin et al. 2024                   │
│                                                                            │
│  "10x better than baselines"          "Excluding ARIMA makes it hard       │
│                                        to take this seriously"             │
│                                        — HackerNews                        │
│                                                                            │
│  "Zero-shot forecasting               "GPT-4 forecasts not significantly   │
│   across all domains"                  different from 50%"                 │
│                                        — Schoenegger & Park 2023           │
│                                                                            │
│  "State-of-the-art                    "Normalized metrics hide bad         │
│   performance"                         actual forecasts"                   │
│                                        — @predict_addict                   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHERE TSFMs CAN HELP                    WHERE THEY STRUGGLE               │
│  ────────────────────                    ──────────────────                │
│                                                                            │
│  ✓ Anomaly detection on dense signals    ✗ Forecasting irregular data     │
│  ✓ Imputation/reconstruction             ✗ Sparse time series (EHRs)      │
│  ✓ Zero-shot transfer (similar domains)  ✗ Event-driven patterns          │
│  ✓ When proper baselines ARE beaten      ✗ When ARIMA wasn't compared     │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  CRITICAL EVALUATION CHECKLIST                                             │
│  ────────────────────────────                                              │
│                                                                            │
│  Before accepting TSFM claims, ask:                                        │
│                                                                            │
│  □ Were simple baselines included? (ARIMA, linear, seasonal decomp)       │
│  □ Are metrics normalized or raw? (MSE vs MASE, scale-dependent?)         │
│  □ What's the domain fit? (Dense/sparse, periodic/aperiodic)              │
│  □ What's the sample size? (Large benchmark vs single dataset)            │
│  □ Is code/data available? (Reproducibility check)                        │
│                                                                            │
│  See: Hewamalage et al. 2022 "Forecast evaluation for data scientists"    │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE BALANCED VIEW                                                         │
│  ────────────────                                                          │
│                                                                            │
│  TSFMs are TOOLS, not magic. They're likely useful for:                    │
│  • Preprocessing (anomaly, imputation) on dense signals                   │
│  • Zero-shot exploration when you lack domain expertise                    │
│  • Transfer learning when you have similar source domains                  │
│                                                                            │
│  They're likely NOT useful for:                                            │
│  • Replacing domain knowledge in specialized applications                  │
│  • Sparse/irregular data without clear patterns                            │
│  • Production systems requiring interpretability                           │
│                                                                            │
│  💡 "The right tool for the right job"                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Content Elements

### Left Column: The Hype
- Marketing claims from TSFM papers and blogs
- "Revolutionary" language
- Broad applicability claims

### Right Column: The Reality
- Actual quotes from critics and honest evaluations
- Specific failure modes
- Proper citations

### Middle Section: Where TSFMs Help vs Struggle
- Two-column checklist
- Specific, actionable guidance

### Bottom Section: Critical Evaluation Checklist
- Questions to ask before accepting claims
- Framework for evaluating new TSFM papers

---

## 7. Text Content

### Title
"TSFM Hype vs Reality: A Critical Evaluation Framework"

### Caption
"Time Series Foundation Models show genuine promise for specific tasks on dense signals, but marketing claims often exceed evidence. Critical evaluation requires checking: Were simple baselines (ARIMA, linear) included? Are metrics normalized? Does the domain fit the model's training data? This framework helps practitioners navigate between hype and dismissal. TSFMs are tools, not magic—apply where appropriate."

---

## 8. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a professional "hype vs reality" comparison figure for Time Series Foundation Models.

LEFT SIDE - "THE HYPE" (red/orange theme):
- Marketing-style claims in quote bubbles
- "Foundation models will revolutionize time series"
- "10x better than baselines"
- "Zero-shot forecasting across all domains"

RIGHT SIDE - "THE REALITY" (blue/green theme):
- Academic quotes with citations
- "LLMs fail without clear periodicity" — Jin et al. 2024
- "GPT-4 ≈ 50% on real tournaments" — Schoenegger 2023
- "Normalized metrics hide bad forecasts" — @predict_addict

MIDDLE SECTION:
Two columns: "Where TSFMs Help" (checkmarks) vs "Where They Struggle" (X marks)

BOTTOM:
Critical evaluation checklist (5 questions to ask)

Style:
- Academic but accessible
- No sensationalism in either direction
- Balanced presentation
- Economist-style clarity
```

---

## 9. Alt Text

"A comparison figure contrasting TSFM marketing hype with research reality. Left side shows common claims like 'revolutionize time series' and '10x better'. Right side shows critical findings from literature: LLMs fail without periodicity, GPT-4 performs at chance level on real forecasting, normalized metrics hide poor predictions. Middle section lists where TSFMs can help (anomaly detection, imputation, zero-shot) vs where they struggle (sparse data, event-driven patterns). Bottom provides 5-point critical evaluation checklist for practitioners."

---

## 10. Status

- [x] Draft created
- [x] Revised to focus on evaluation framework, not results
- [ ] Generated
- [ ] Placed in documentation

## Note

Specific experimental results from this repository are in the manuscript, not this figure.
