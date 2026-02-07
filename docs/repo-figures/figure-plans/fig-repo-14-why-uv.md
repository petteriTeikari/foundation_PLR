# fig-repo-14: Why uv? Package Management Done Right

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-14 |
| **Title** | Why uv? Package Management Done Right |
| **Complexity Level** | L1 (Concept explanation) |
| **Target Persona** | Research Scientists, Biostatisticians |
| **Location** | CONTRIBUTING.md, docs/getting-started/ |
| **Priority** | P0 (Critical - reproducibility foundation) |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain why this repository uses `uv` instead of pip/requirements.txt, emphasizing the reproducibility benefits that matter for scientific research.

## Key Message

"pip's requirements.txt lists what you asked for, but uv.lock ensures everyone gets the EXACT same packages—down to the smallest dependency."

## Visual Concept

**Side-by-side comparison with dependency trees:**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     WHY uv? PACKAGE MANAGEMENT DONE RIGHT                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ❌ pip + requirements.txt              ✅ uv + uv.lock                         │
│  ════════════════════════════           ══════════════════════                  │
│                                                                                 │
│  requirements.txt:                      uv.lock:                                │
│  ┌─────────────────────┐                ┌─────────────────────┐                 │
│  │ numpy>=1.20         │                │ numpy==1.24.3       │                 │
│  │ pandas>=2.0         │                │ pandas==2.1.4       │                 │
│  │ scikit-learn>=1.0   │                │ scikit-learn==1.3.2 │                 │
│  └─────────────────────┘                │ scipy==1.11.4       │ ← sub-deps!    │
│                                         │ joblib==1.3.2       │                 │
│  "What you asked for"                   │ threadpoolctl==3.2.0│                 │
│                                         └─────────────────────┘                 │
│                                         "What you actually get"                 │
│                                                                                 │
│  🎲 Different machines may get:         🔒 Every machine gets:                  │
│     • numpy 1.24.3 OR 1.26.0              • EXACT same versions                 │
│     • Different sub-dependencies           • Same sub-dependencies              │
│     • "Works on my machine!"               • "Works everywhere!"                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SPEED COMPARISON                                                               │
│  ───────────────                                                                │
│  pip install -r requirements.txt:  ████████████████████████████████░░  60s     │
│  uv sync:                          ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5s     │
│                                    └── 10-100x faster!                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  QUICK COMMAND REFERENCE                                                        │
│  ───────────────────────                                                        │
│  ❌ pip install package      →    ✅ uv add package                             │
│  ❌ pip install -r req.txt   →    ✅ uv sync                                    │
│  ❌ conda install package    →    ✅ uv add package (conda BANNED)              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. **Side-by-side comparison**: requirements.txt vs uv.lock
2. **Dependency tree visualization**: Show sub-dependencies only in uv.lock
3. **Speed comparison bar chart**: 60s vs 5s (10-100x faster)
4. **Command translation table**: pip → uv equivalents
5. **"Works on my machine" vs "Works everywhere"** callout

### Optional Elements
1. Story illustration: Two researchers with different environments
2. Lock file icon with padlock for uv.lock
3. Dice/randomness icon for pip's non-deterministic installs

## Text Content

### Title Text
"Why uv? Package Management Done Right"

### Labels/Annotations
- Header: "The Reproducibility Problem with pip"
- Section 1: "What you write" vs "What you get"
- Section 2: "Sub-dependencies are NOT locked by pip"
- Section 3: "uv is 10-100x faster than pip"
- Bottom: "Simple command replacements"

### Caption (for embedding)
pip's requirements.txt only lists the packages you explicitly requested, not their sub-dependencies. This means different machines can install different versions, leading to the infamous "works on my machine" problem. uv solves this with a lockfile (uv.lock) that captures EVERY dependency at exact versions. Bonus: uv is 10-100x faster than pip. We use uv exclusively in this repository—conda and pip are banned.

## Prompts for Nano Banana Pro

### Style Prompt
Technical comparison infographic with clean Economist-style layout. Two-column comparison (bad vs good). Use muted red for pip side, muted green/blue for uv side. Include small code snippets in monospace font on dark backgrounds. Speed comparison as horizontal bar chart. Matte, professional, no glowing effects. Medical research aesthetic.

### Content Prompt
Create a two-column comparison infographic:

**LEFT COLUMN (pip - red accent)**:
- Code block showing minimal requirements.txt
- Dice icon representing randomness
- "What you write" vs "What you get" disconnect
- Quote: "Works on my machine... maybe"

**RIGHT COLUMN (uv - blue/green accent)**:
- Code block showing comprehensive uv.lock with sub-deps
- Lock/padlock icon representing security
- "What you write" = "What you get" equality
- Quote: "Works everywhere, guaranteed"

**BOTTOM SECTION**:
- Horizontal bar chart: pip (60s full bar) vs uv (5s small bar)
- Command reference table: pip → uv translations

### Refinement Notes
- Make the sub-dependencies visually obvious (highlight them)
- The speed difference should be dramatic visually
- Include the "conda BANNED" note prominently
- This figure should make scientists feel SAFE about reproducibility

## Alt Text

Two-column comparison of pip vs uv package managers. Left side shows pip with incomplete requirements.txt lacking sub-dependencies, dice icon representing unpredictable installs, 60-second install time. Right side shows uv with comprehensive uv.lock including all sub-dependencies, lock icon representing reproducibility, 5-second install time. Bottom shows command translation: pip install → uv add.

## Technical Notes

### Web Search Sources
- [Real Python: uv vs pip](https://realpython.com/uv-vs-pip/)
- [DataCamp: Python UV Tutorial](https://www.datacamp.com/tutorial/python-uv)
- [Python Discourse: requirements.txt vs uv.lock](https://discuss.python.org/t/requirements-txt-or-uv-lock/78419)

### Verification in Codebase
- `pyproject.toml` exists at repo root
- `uv.lock` should exist (verify)
- No `requirements.txt` should be used for installation

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated (16:10 aspect ratio)
- [ ] Placed in CONTRIBUTING.md, docs/getting-started/
