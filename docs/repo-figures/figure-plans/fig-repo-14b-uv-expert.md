# fig-repo-14b: uv.lock: Full Dependency Trees (Expert)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-14b |
| **Title** | uv.lock: Capturing the Full Dependency Tree |
| **Complexity Level** | L3 (Expert - Technical deep-dive) |
| **Target Persona** | Software Engineers, DevOps, Reproducibility Engineers |
| **Location** | CONTRIBUTING.md, docs/development/ |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Show WHY uv provides reproducibility—the full transitive dependency tree is captured, not just top-level packages.

## Key Message

"requirements.txt lists what you asked for. uv.lock captures what you actually got—including every sub-dependency."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    uv.lock: FULL DEPENDENCY TREES                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  requirements.txt (TOP-LEVEL ONLY)        uv.lock (FULL TREE)                   │
│  ┌─────────────────────────────┐          ┌─────────────────────────────┐       │
│  │ numpy>=1.20                 │          │ numpy==1.24.3               │       │
│  │ pandas>=2.0                 │          │ pandas==2.1.4               │       │
│  │ scikit-learn>=1.0           │          │ scikit-learn==1.3.2         │       │
│  └─────────────────────────────┘          │ ├── scipy==1.11.4           │       │
│                                           │ ├── joblib==1.3.2           │       │
│  "What you wrote"                         │ ├── threadpoolctl==3.2.0    │       │
│                                           │ └── numpy==1.24.3 (pinned)  │       │
│                                           │ pytz==2023.3                │       │
│                                           │ python-dateutil==2.8.2      │       │
│                                           └─────────────────────────────┘       │
│                                           "What you actually get"               │
│                                                                                 │
│  DEPENDENCY TREE VISUALIZATION                                                  │
│  ─────────────────────────────                                                  │
│                                                                                 │
│  scikit-learn ─┬─► scipy ──────┬─► numpy                                        │
│                │               └─► lapack                                       │
│                ├─► joblib                                                       │
│                ├─► threadpoolctl                                                │
│                └─► numpy (shared!)                                              │
│                                                                                 │
│  pip: "scipy>=1.5" → Could be 1.5, 1.11, or 1.12 depending on when you install │
│  uv:  "scipy==1.11.4" → Always 1.11.4, guaranteed                              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMAND REFERENCE                                                              │
│  ─────────────────                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │ # OLD (BANNED)                    # NEW (REQUIRED)                         ││
│  │ pip install package         →     uv add package                           ││
│  │ pip install -r requirements.txt → uv sync                                  ││
│  │ pip freeze > requirements.txt →   (automatic via uv.lock)                  ││
│  │ conda install anything      →     uv add package  # CONDA BANNED           ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  PERFORMANCE                                                                    │
│  ───────────                                                                    │
│  pip install:  ████████████████████████████████████████░░░░░░░  60s            │
│  uv sync:      ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5s            │
│                10-100× faster (Rust-based resolver)                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Side-by-side file comparison**: requirements.txt vs uv.lock with visible sub-deps
2. **Dependency tree diagram**: Show transitive dependencies
3. **Version pinning explanation**: Why ==1.11.4 matters
4. **Command reference table**: pip → uv translations
5. **Speed comparison with technical context**: Rust-based resolver
6. **Conda ban notice**: Explicitly stated

## Text Content

### Title Text
"uv.lock: Capturing the Full Dependency Tree"

### Caption
requirements.txt only lists top-level packages with version ranges (numpy>=1.20). uv.lock captures the complete dependency tree with exact versions (numpy==1.24.3, plus scipy==1.11.4, joblib==1.3.2, etc.). This ensures reproducibility: every developer gets identical packages. Bonus: uv's Rust-based resolver is 10-100× faster than pip.

## Prompts for Nano Banana Pro

### Style Prompt
Technical documentation style with code blocks and dependency diagrams. Dark-mode code snippets. Dependency tree as flowchart with arrows. Side-by-side file comparison. Economist-style horizontal bars for performance. Matte, professional colors. Include "BANNED" callout for conda with red accent.

### Content Prompt
Create a technical comparison infographic:

**TOP - File Comparison**:
- LEFT: requirements.txt code block (3 lines, version ranges)
- RIGHT: uv.lock code block (10+ lines, exact versions, sub-deps indented)
- Highlight the sub-dependencies that pip doesn't capture

**MIDDLE - Dependency Tree**:
- Flowchart showing scikit-learn → scipy → numpy chain
- Arrows showing transitive dependencies
- "Shared dependency" notation

**BOTTOM LEFT - Command Reference**:
- Code block with pip → uv translations
- Red "BANNED" badge for conda

**BOTTOM RIGHT - Performance**:
- Horizontal bars: 60s vs 5s
- Note: "Rust-based resolver"

## Alt Text

Technical comparison of requirements.txt (3 top-level packages) versus uv.lock (full dependency tree with 10+ packages including sub-dependencies). Includes dependency tree flowchart showing transitive dependencies, command reference table for pip to uv migration, and performance comparison showing 10-100× speedup.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in CONTRIBUTING.md
