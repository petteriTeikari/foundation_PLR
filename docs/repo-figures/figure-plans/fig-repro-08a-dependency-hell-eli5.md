# fig-repro-08a: Dependency Hell Visualized (ELI5)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-08a |
| **Title** | Dependency Hell Visualized (ELI5) |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | Biologist, PI, PhD Student |
| **Location** | README.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain dependency management using a Lego block analogy that non-programmers can instantly understand.

## Key Message

"Installing a package is like buying a Lego set—it needs specific other sets to work. Without exact instructions, you get the wrong pieces and nothing fits."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY HELL (Explained Simply)                           │
│                    Why "pip install" isn't enough                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE LEGO ANALOGY                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  You want to build: 🏰 CASTLE SET                                               │
│                                                                                 │
│  But it requires:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │      🏰 Castle                                                          │   │
│  │       ├── 🧱 Wall Pack (v2.0 exactly!)                                  │   │
│  │       │    ├── 🔲 Basic Bricks (v1.5+)                                  │   │
│  │       │    └── 🔗 Connectors (v3.0)                                     │   │
│  │       ├── 🚪 Door Set (any version)                                     │   │
│  │       └── 🏳️ Flag Pack (v1.0-2.0 only)                                  │   │
│  │            └── 🎨 Color Pack (must match Wall Pack version!)            │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  THE PROBLEM:                                                                   │
│  ─────────────                                                                  │
│  If you just say "give me a castle," the store might give you:                  │
│  • Wall Pack v3.0 (too new! doesn't fit)                                        │
│  • Connectors v2.0 (incompatible with Wall Pack v3.0)                           │
│  • Color Pack that doesn't match anything                                       │
│                                                                                 │
│  Result: ❌ Nothing fits together!                                              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE SOLUTION: A SHOPPING LIST                                                  │
│  ═══════════════════════════════                                                │
│                                                                                 │
│  "pip install pandas"           vs        "uv.lock file"                        │
│  ──────────────────                       ─────────────────                     │
│  "Just get me pandas"                     "Get me EXACTLY:                      │
│  (whatever version is                     pandas 2.1.3                          │
│   available today)                        numpy 1.24.0                          │
│                                           python-dateutil 2.8.2                 │
│  Tomorrow: different!                     pytz 2023.3                           │
│                                           ..."                                  │
│                                                                                 │
│  Foundation PLR uses uv.lock = exact shopping list for every package            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Lego tree diagram**: Package dependencies as building sets
2. **Version annotations**: Which versions are required
3. **Problem illustration**: Wrong pieces don't fit
4. **Side-by-side comparison**: pip vs uv.lock
5. **Shopping list metaphor**: Exact specifications

## Text Content

### Title Text
"Dependency Hell: Why 'pip install' Isn't Enough"

### Caption
Installing a Python package is like buying a Lego set—it needs other specific sets (dependencies) to work. Without exact version specifications, you get random versions that might not be compatible. A lockfile (like uv.lock) is a precise shopping list that ensures everyone gets exactly the same pieces, every time.

## Prompts for Nano Banana Pro

### Style Prompt
Colorful Lego-style building blocks forming a dependency tree. Friendly, approachable design. Version numbers on each block. Problem section shows mismatched pieces. Solution shows organized shopping list. No technical jargon visible.

### Content Prompt
Create "Dependency Hell ELI5" infographic:

**TOP - Lego Tree**:
- Castle at top
- Branches to Wall Pack, Door Set, Flag Pack
- Sub-branches with version requirements

**MIDDLE - Problem**:
- Mismatched blocks illustration
- "Nothing fits together!" label

**BOTTOM - Solution**:
- Two columns: pip (vague) vs uv.lock (exact)
- Shopping list metaphor

## Alt Text

Dependency hell explained using Lego analogy. A castle set requires Wall Pack v2.0 (which needs Basic Bricks v1.5+ and Connectors v3.0), Door Set (any version), and Flag Pack v1.0-2.0 (which needs matching Color Pack). Problem: without exact versions, you get incompatible pieces. Solution comparison: pip install (vague) vs uv.lock (exact shopping list with specific versions for every dependency).

## Related Figures

- **fig-repro-08b**: Technical details on dependency resolution (Expert version)
- **fig-repro-08c**: UMAP/t-SNE initialization trap (concrete example)
- **fig-repro-12**: Dependency explosion (technical details on 5→200+ problem)
- **fig-repro-14**: Lockfiles as time machine (solution concept)
- **fig-repo-14**: uv package manager deep dive (tool)

## Cross-References

Reader flow: **THIS FIGURE** (ELI5 concept) → **fig-repro-12** (technical problem) → **fig-repro-14** (solution) → **fig-repo-14** (tool)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md

