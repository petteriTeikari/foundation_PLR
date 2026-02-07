# fig-repo-04: MLflow = Smart Lab Notebook

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-04 |
| **Title** | MLflow: Your Smart Digital Lab Notebook |
| **Complexity Level** | L1 (ELI5 for researchers) |
| **Target Persona** | PI, Biostatistician, Research Scientist |
| **Location** | docs/getting-started/, README.md |
| **Priority** | P2 (High) |

## Purpose

Explain MLflow to researchers who have never heard of it, using the lab notebook metaphor they understand.

## Key Message

"MLflow is like a lab notebook that automatically records every experiment you run, so you can always go back and see exactly what you did."

## Visual Concept

**Side-by-side: Physical lab notebook vs MLflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│   PHYSICAL LAB NOTEBOOK              MLFLOW                     │
│   ──────────────────────            ──────                      │
│                                                                 │
│   ┌─────────────┐                   ┌─────────────┐            │
│   │ 📓 Handwritten│                 │ 💻 Automatic │            │
│   │ notes        │                  │ logging      │            │
│   └─────────────┘                   └─────────────┘            │
│                                                                 │
│   • "Used LOF with k=5"             • Parameters saved          │
│   • "AUROC was ~0.85"               • Exact metrics stored      │
│   • "Forgot the seed..."            • Git hash recorded         │
│                                                                 │
│   ┌─────────────┐                   ┌─────────────┐            │
│   │ 😰 Can't    │                   │ 🔍 Search    │            │
│   │ reproduce   │                   │ & compare    │            │
│   └─────────────┘                   └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. Physical notebook icon (relatable starting point)
2. MLflow interface mockup
3. Key features: automatic logging, parameters, metrics
4. Reproducibility benefit highlighted

### Optional Elements
1. Comparison table format
2. Screenshot of actual MLflow UI
3. Link to MLflow documentation

## Text Content

### Title Text
"MLflow: Never Lose an Experiment Again"

### Labels/Annotations
- Notebook: "Manual notes, easy to forget details"
- MLflow: "Automatic tracking of everything"
- Parameters: "Every setting saved automatically"
- Metrics: "Exact numbers, not approximations"
- Search: "Find that experiment from 3 months ago"

### Caption (for embedding)
MLflow automatically records every experiment with exact parameters, metrics, and code versions - like a lab notebook that never forgets.

## Prompts for Nano Banana Pro

### Style Prompt
Friendly, non-intimidating educational graphic. Split comparison layout. Left side: traditional/manual (warm, sepia tones). Right side: digital/automated (cool, blue tones). Use familiar icons (notebook, computer, checkmark).

### Content Prompt
Create a side-by-side comparison:
LEFT: A handwritten lab notebook with scribbled notes, question marks, and a "Where did that result come from?" thought bubble
RIGHT: A clean MLflow dashboard showing organized parameters, metrics, and a search bar, with a "Found it!" thought bubble

Bottom: Show the key benefit - "Reproducibility"

### Refinement Notes
- The notebook side should look relatable, not mocked
- The MLflow side should look helpful, not intimidating
- Emphasize the search/retrieval capability

## Alt Text

Comparison of manual lab notebook with missing details versus MLflow's automatic experiment tracking with searchable parameters and metrics.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
