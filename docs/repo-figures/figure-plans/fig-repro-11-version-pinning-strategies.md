# fig-repro-11: Version Pinning Strategies

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-11 |
| **Title** | Version Pinning Strategies |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | ML Engineer, Data Scientist |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Compare version specification strategies from loose (>=) to strict (==) and explain when each is appropriate.

## Key Message

"Development wants flexibility (>=). Production wants reproducibility (==). Use lockfiles to get both: flexible specs in pyproject.toml, exact pins in uv.lock."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VERSION PINNING STRATEGIES                                   │
│                    From flexible to frozen                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE FLEXIBILITY-REPRODUCIBILITY TRADEOFF                                       │
│  ═════════════════════════════════════════                                      │
│                                                                                 │
│  FLEXIBLE                                                    REPRODUCIBLE       │
│  (easy updates)                                              (guaranteed same)  │
│      ←─────────────────────────────────────────────────────────────→            │
│                                                                                 │
│  │ Spec          │ Example        │ Pros              │ Cons               │   │
│  │ ──────────────│ ───────────────│ ─────────────────  │ ────────────────── │   │
│  │ No version    │ pandas         │ Always latest     │ Chaos              │   │
│  │ Compatible    │ pandas~=2.0    │ Bug fixes auto    │ May break          │   │
│  │ Minimum       │ pandas>=2.0    │ New features      │ Very risky         │   │
│  │ Range         │ pandas>=2.0,<3 │ Bounded risk      │ Still varies       │   │
│  │ Exact         │ pandas==2.1.3  │ Reproducible      │ No updates         │   │
│  │ Lockfile      │ uv.lock        │ BOTH!             │ File to maintain   │   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SEMVER OPERATORS EXPLAINED                                                     │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  pandas==2.1.3       Exactly 2.1.3 (most strict)                                │
│  pandas~=2.1         ≥2.1.0, <2.2.0 (patch updates only)                        │
│  pandas^2.1          ≥2.1.0, <3.0.0 (minor updates OK - npm style)              │
│  pandas>=2.0,<3.0    Explicit range                                             │
│  pandas>=2.0         2.0 or any higher (danger!)                                │
│  pandas              Latest available (maximum danger!)                          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE LOCKFILE SOLUTION                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  pyproject.toml (FLEXIBLE - what you want)                              │   │
│  │  ─────────────────────────────────────────                              │   │
│  │  [project]                                                              │   │
│  │  dependencies = [                                                       │   │
│  │      "pandas>=2.0",        # Accept updates                             │   │
│  │      "numpy>=1.20",        # Accept updates                             │   │
│  │  ]                                                                      │   │
│  │                                                                         │   │
│  │  ──────────────────────── uv lock ─────────────────────────────────►    │   │
│  │                                                                         │   │
│  │  uv.lock (FROZEN - what you got)                                        │   │
│  │  ───────────────────────────────                                        │   │
│  │  [[package]]                                                            │   │
│  │  name = "pandas"                                                        │   │
│  │  version = "2.1.3"          # Exact version                             │   │
│  │  sha256 = "abc123..."       # Verified download                         │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Benefits:                                                                      │
│  • pyproject.toml stays readable (no 200 exact versions)                        │
│  • uv.lock guarantees reproducibility                                           │
│  • `uv lock --upgrade` updates when YOU decide                                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR POLICY                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  pyproject.toml: Flexible specifications (>=, ~=)                               │
│  uv.lock:        Committed to git (exact versions)                              │
│  Updates:        Monthly review + `uv lock --upgrade`                           │
│                                                                                 │
│  Never in pyproject.toml:                                                       │
│  • Bare package names without version                                           │
│  • Upper-unbounded ranges (>=X with no <Y)                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Flexibility spectrum**: Visual slider from flexible to reproducible
2. **Comparison table**: Six strategies with pros/cons
3. **SemVer operators**: Explanation of ==, ~=, ^, >=, etc.
4. **Lockfile flow**: pyproject.toml → uv lock → uv.lock
5. **Project policy**: Specific rules Foundation PLR follows

## Text Content

### Title Text
"Version Pinning: Balancing Flexibility and Reproducibility"

### Caption
Version specifications range from dangerously flexible (no version) to rigidly exact (==). Lockfiles provide both: use flexible specs in pyproject.toml for development, commit uv.lock for reproducibility. Foundation PLR uses >=, ~= in specs but always commits the lockfile.

## Prompts for Nano Banana Pro

### Style Prompt
Horizontal spectrum from "Flexible" to "Reproducible". Table showing six strategies. SemVer reference card. Flow diagram: pyproject.toml → uv lock → uv.lock. Policy callout box. Technical but accessible.

### Content Prompt
Create "Version Pinning Strategies" infographic:

**TOP - Spectrum**:
- Slider from FLEXIBLE to REPRODUCIBLE
- Table with 6 strategies and pros/cons

**MIDDLE - SemVer Reference**:
- Six operators with explanations

**BOTTOM - Solution**:
- Two-panel: pyproject.toml (flexible) → uv.lock (frozen)
- Arrow labeled "uv lock"
- Benefits list

## Alt Text

Version pinning strategies infographic. Spectrum from flexible to reproducible. Table compares strategies: no version (chaos), compatible ~= (may break), minimum >= (risky), range (varies), exact == (no updates), lockfile (both pros). SemVer operators explained. Solution flow: pyproject.toml with flexible specs, uv lock command, uv.lock with exact versions and SHA-256 hashes. Foundation PLR policy: flexible specs, committed lockfile, monthly upgrade reviews.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

