# fig-repo-07: How to Create a Figure

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-07 |
| **Title** | How to Create a Publication-Ready Figure |
| **Complexity Level** | L3 (Technical details) |
| **Target Persona** | Research Scientist, ML Engineer, LLM Agents |
| **Location** | src/r/README.md, CONTRIBUTING.md |
| **Priority** | P1 (Critical for contributors) |

## Purpose

Step-by-step guide showing the correct workflow for creating figures that pass pre-commit checks and comply with repository guidelines.

## Key Message

"Load colors from YAML, use the theme, call save_publication_figure() - never hardcode anything."

## Visual Concept

**Decision flowchart with code snippets:**

```
┌─────────────────────────────────────────────┐
│        START: Need to Create a Figure       │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  1. Check figure_registry.yaml              │
│     Does figure already exist?              │
└─────────────────────────────────────────────┘
          │                    │
        [YES]                [NO]
          │                    │
          ▼                    ▼
┌──────────────┐    ┌──────────────────────────┐
│Update script │    │ 2. Create fig-XXX.R      │
└──────────────┘    │    in src/r/figures/     │
                    └──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────┐
│  3. Load colors from YAML (NOT hardcoded!)  │
│     color_defs <- load_color_definitions()  │
└─────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────┐
│  4. Apply theme_foundation_plr()            │
│     p + theme_foundation_plr()              │
└─────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────┐
│  5. Save with save_publication_figure()     │
│     (NOT ggsave!)                           │
└─────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────┐
│  6. Run pre-commit check                    │
│     pre-commit run --all-files              │
└─────────────────────────────────────────────┘
          │                    │
       [PASS]               [FAIL]
          │                    │
          ▼                    ▼
┌──────────────┐    ┌──────────────────────────┐
│   COMMIT!    │    │ Fix hardcoded values     │
└──────────────┘    │ (colors, ggsave, dims)   │
                    └──────────────────────────┘
                               │
                               └──────→ (back to step 3)
```

## Content Elements

### Required Elements
1. Decision flowchart with YES/NO branches
2. Key code snippets in monospace boxes
3. Pre-commit check step with pass/fail outcomes
4. "BANNED" indicators for hardcoded values
5. Correct function names highlighted

### Optional Elements
1. File path references
2. Link to full documentation
3. Common error examples

## Text Content

### Title Text
"Figure Creation Workflow: The Anti-Hardcoding Guide"

### Labels/Annotations
- Step 1: "Check figure_registry.yaml"
- Step 2: "Create fig-XXX.R in src/r/figures/"
- Step 3: "load_color_definitions() - NOT #RRGGBB!"
- Step 4: "theme_foundation_plr()"
- Step 5: "save_publication_figure() - NOT ggsave()!"
- Step 6: "pre-commit run --all-files"
- BANNED box: "ggsave(), hardcoded colors, manual dimensions"

### Caption (for embedding)
The figure creation workflow ensures all figures comply with repository standards. Never hardcode colors or use ggsave() - always use the provided helper functions that load from YAML configs.

## Technical Notes

- **Data source**: src/r/figure_system/ codebase
- **Dependencies**: Requires understanding of pre-commit hooks
- **Updates needed**: If new enforcement checks are added

## Prompts for Nano Banana Pro

### Style Prompt
Technical flowchart with developer aesthetic. Dark mode friendly. Monospace font for code snippets. Green for correct actions, red for banned/wrong actions. Decision diamonds, process rectangles. Clean connecting arrows.

### Content Prompt
Create a vertical flowchart for the figure creation process:
1. Start node: "Need to Create Figure"
2. Decision: "Figure in registry?" with YES/NO branches
3. Process boxes for each step with R code snippets
4. Pre-commit check with PASS/FAIL branches
5. FAIL loops back to fix step
6. PASS leads to "Commit!"

Include a "BANNED" callout box listing: ggsave(), #RRGGBB colors, hardcoded dimensions

### Refinement Notes
- Make the code snippets readable but not overwhelming
- Emphasize the anti-hardcoding message with red X marks
- The loop-back from FAIL should be visually clear

## Alt Text

Flowchart showing 6 steps to create repository-compliant figures: check registry, create script, load colors from YAML, apply theme, save with helper function, and pass pre-commit checks.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
