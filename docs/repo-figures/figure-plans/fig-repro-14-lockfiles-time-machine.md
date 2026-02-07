# fig-repro-14: Lockfiles: Your Time Machine

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-14 |
| **Title** | Lockfiles: Your Time Machine |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | PhD Student, Biologist, PI |
| **Location** | README.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Use a time machine metaphor to explain how lockfiles preserve exact dependency states for future reproduction.

## Key Message

"A lockfile is a photograph of your dependencies at a moment in time. Years later, you can recreate the exact same environment—like a time machine for your code."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LOCKFILES: YOUR TIME MACHINE                                 │
│                    Preserve your environment forever                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE TIME MACHINE ANALOGY                                                       │
│  ═════════════════════════                                                      │
│                                                                                 │
│  January 2024: Project works perfectly!                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  📸 uv lock                                                             │   │
│  │  ────────────────────────────────────────────────────────               │   │
│  │  Takes a "photograph" of your entire environment:                       │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  uv.lock                                                        │   │   │
│  │  │  ─────────────                                                  │   │   │
│  │  │  pandas = 2.1.3                                                 │   │   │
│  │  │  numpy = 1.24.0                                                 │   │   │
│  │  │  scikit-learn = 1.3.2                                           │   │   │
│  │  │  ... (200+ packages with exact versions)                        │   │   │
│  │  │                                                                 │   │   │
│  │  │  TIMESTAMP: 2024-01-15T10:32:45Z                                │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ────────────────────────── ⏰ 2 YEARS LATER ──────────────────────────────    │
│                                                                                 │
│  January 2026: Need to reproduce results for follow-up study                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  🔮 uv sync                                                             │   │
│  │  ────────────────────────────────────────────────────────               │   │
│  │  Recreates EXACTLY the same environment from the photograph:            │   │
│  │                                                                         │   │
│  │  $ uv sync                                                              │   │
│  │  Installing pandas 2.1.3...         (not 2.3.0, the "latest")          │   │
│  │  Installing numpy 1.24.0...         (not 1.26.4)                       │   │
│  │  Installing scikit-learn 1.3.2...   (not 1.4.0)                        │   │
│  │  ✓ Environment restored in 1.8 seconds!                                │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  YOUR CODE RUNS EXACTLY AS IT DID 2 YEARS AGO! ✨                               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  UNLIKE WITHOUT A LOCKFILE...                                                   │
│  ════════════════════════════                                                   │
│                                                                                 │
│  $ pip install -r requirements.txt                                              │
│  Installing pandas... (gets 2.3.0, not 2.1.3)                                   │
│  Installing numpy... (gets 1.26.4, not 1.24.0)                                  │
│                                                                                 │
│  ❌ ImportError: cannot import 'DataFrame' from 'pandas'                        │
│  ❌ DeprecationWarning: np.float is deprecated                                  │
│  ❌ ValueError: unknown parameter 'n_jobs' in version 1.4                       │
│                                                                                 │
│  YOUR CODE IS BROKEN. You're now debugging instead of researching.              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR: LOCKFILE COMMITTED                                             │
│  ═══════════════════════════════════                                            │
│                                                                                 │
│  uv.lock is in git → anyone can time-travel to our exact environment           │
│                                                                                 │
│  git clone ... && uv sync                                                       │
│  ───────────────────────────                                                    │
│  Same packages. Same versions. Same results. Always.                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Time machine metaphor**: Photograph analogy for lockfile creation
2. **Timeline**: 2024 → 2 years → 2026
3. **uv.lock snapshot**: Visual of locked versions with timestamp
4. **uv sync restoration**: Terminal showing exact version installs
5. **Failure scenario**: What happens without lockfile (errors)
6. **Foundation PLR callout**: Lockfile committed to git

## Text Content

### Title Text
"Lockfiles: A Time Machine for Your Dependencies"

### Caption
A lockfile is a photograph of your dependencies. uv.lock captures the exact version of every package at a moment in time. Years later, `uv sync` recreates that identical environment—like time travel for your code. Without a lockfile, pip install gets the latest versions, which may have breaking changes. Foundation PLR commits uv.lock to git so anyone can time-travel to our exact environment.

## Prompts for Nano Banana Pro

### Style Prompt
Time machine visual with "2024" and "2026" timeline. Camera/photograph icon for lockfile creation. Terminal mockups for commands. Error callouts for failure scenario. Warm, approachable design with a bit of whimsy (time machine theme).

### Content Prompt
Create "Lockfiles as Time Machine" infographic:

**TOP - Create Lockfile (2024)**:
- Camera icon + "uv lock"
- uv.lock file showing versions + timestamp

**MIDDLE - Time Jump**:
- "2 YEARS LATER" transition
- "uv sync" restoring exact versions
- Success message

**BOTTOM - Contrast**:
- Left: Without lockfile (pip errors)
- Right: With lockfile (works perfectly)
- "Foundation PLR: uv.lock in git"

## Alt Text

Lockfiles as time machine infographic. January 2024: uv lock takes a photograph of environment (uv.lock with exact versions like pandas 2.1.3, numpy 1.24.0). Two years later (January 2026): uv sync restores exact same environment in 1.8 seconds. Code runs exactly as it did. Contrast: without lockfile, pip install gets latest versions causing ImportError, DeprecationWarning, ValueError. Foundation PLR commits uv.lock to git so anyone can time-travel to the exact environment.

## Related Figures

- **fig-repro-08a**: Dependency hell ELI5 (Lego analogy) - why deps matter
- **fig-repro-08b**: pip vs uv technical comparison
- **fig-repro-12**: Dependency explosion - the problem this solves
- **fig-repo-14**: uv package manager - the tool that creates lockfiles

## Cross-References

Reader flow: **fig-repro-08a** (why deps matter) → **fig-repro-12** (the explosion problem) → **THIS FIGURE** (the solution concept) → **fig-repo-14** (the tool)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md

