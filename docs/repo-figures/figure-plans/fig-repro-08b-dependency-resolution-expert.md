# fig-repro-08b: Dependency Resolution: pip vs uv

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-08b |
| **Title** | Dependency Resolution: pip vs uv |
| **Complexity Level** | L3 (Expert) |
| **Target Persona** | ML Engineer, Data Scientist |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Technical comparison of pip's non-deterministic resolution vs uv's lockfile-based deterministic approach.

## Key Message

"pip resolves dependencies at install time (non-deterministic). uv resolves once and locks (deterministic). Same pyproject.toml can produce different environments with pip; uv.lock guarantees identical environments."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY RESOLUTION: pip vs uv                             │
│                    Why lockfiles matter                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  pip (NO LOCKFILE)                     uv (WITH LOCKFILE)                       │
│  ════════════════                      ═══════════════════                      │
│                                                                                 │
│  pyproject.toml:                       pyproject.toml:                          │
│  ┌────────────────────┐                ┌────────────────────┐                   │
│  │ pandas>=2.0        │                │ pandas>=2.0        │                   │
│  │ numpy>=1.20        │                │ numpy>=1.20        │                   │
│  └────────────────────┘                └────────────────────┘                   │
│           │                                     │                               │
│           ▼                                     ▼                               │
│  ┌──────────────────────────┐          ┌──────────────────────────┐            │
│  │ pip install (Jan 2024)   │          │ uv lock (once)           │            │
│  │ → pandas 2.1.0           │          │ → pandas 2.1.3           │            │
│  │ → numpy 1.25.0           │          │ → numpy 1.24.0           │            │
│  └──────────────────────────┘          │ → ...200 more packages   │            │
│           │                            └──────────────────────────┘            │
│           ▼                                     │                               │
│  ┌──────────────────────────┐                   ▼                               │
│  │ pip install (Jan 2025)   │          ┌──────────────────────────┐            │
│  │ → pandas 2.2.1  ≠        │          │ uv sync (any time)       │            │
│  │ → numpy 1.26.4  ≠        │          │ → pandas 2.1.3  ✓        │            │
│  └──────────────────────────┘          │ → numpy 1.24.0  ✓        │            │
│           │                            └──────────────────────────┘            │
│           ▼                                     │                               │
│  ❌ DIFFERENT ENVIRONMENTS              ✅ IDENTICAL ENVIRONMENTS               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  HOW uv.lock WORKS                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  uv.lock (abbreviated):                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  [[package]]                                                            │   │
│  │  name = "pandas"                                                        │   │
│  │  version = "2.1.3"                                                      │   │
│  │  source = { registry = "https://pypi.org/simple" }                      │   │
│  │  sha256 = "abc123..."  ← Cryptographic verification!                    │   │
│  │  dependencies = [                                                       │   │
│  │    { name = "numpy", version = "1.24.0" },                              │   │
│  │    { name = "python-dateutil", version = "2.8.2" },                     │   │
│  │  ]                                                                      │   │
│  │                                                                         │   │
│  │  [[package]]                                                            │   │
│  │  name = "numpy"                                                         │   │
│  │  version = "1.24.0"                                                     │   │
│  │  ...                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Key features:                                                                  │
│  • Exact versions (not ranges)                                                  │
│  • SHA-256 hashes (verify packages weren't tampered with)                       │
│  • Transitive dependencies (all 200+ packages, not just direct)                 │
│  • Platform markers (linux/mac/windows compatibility)                           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PERFORMANCE COMPARISON                                                         │
│  ═════════════════════                                                          │
│                                                                                 │
│  │ Operation            │ pip        │ uv         │ Factor │                   │
│  │ ────────────────────  │ ────────── │ ────────── │ ────── │                   │
│  │ Fresh install        │ 45 sec     │ 3.2 sec    │ 14x    │                   │
│  │ Install from lock    │ N/A        │ 1.8 sec    │ -      │                   │
│  │ Dependency resolve   │ 12 sec     │ 0.8 sec    │ 15x    │                   │
│  │ Add one package      │ 8 sec      │ 0.5 sec    │ 16x    │                   │
│                                                                                 │
│  Source: uv benchmarks (astral.sh)                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Two-column flow diagram**: pip vs uv workflows
2. **Time progression**: Same input, different outputs over time
3. **uv.lock structure**: Annotated TOML example
4. **Feature list**: Exact versions, hashes, transitive deps
5. **Performance table**: pip vs uv benchmark comparison

## Text Content

### Title Text
"Dependency Resolution: pip vs uv Lockfiles"

### Caption
pip resolves dependencies at install time—the same pyproject.toml produces different environments on different days. uv resolves once and generates a lockfile (uv.lock) with exact versions, SHA-256 hashes, and all transitive dependencies. Foundation PLR's uv.lock specifies 200+ packages with cryptographic verification, ensuring identical environments regardless of when or where code runs.

## Prompts for Nano Banana Pro

### Style Prompt
Technical diagram with two parallel flows. Left (pip): diverging arrows showing different outcomes. Right (uv): single locked path. Code block showing lockfile structure. Benchmark table at bottom. Monospace fonts for code, professional colors.

### Content Prompt
Create "pip vs uv" technical infographic:

**TOP - Two Parallel Flows**:
- Left: pyproject.toml → pip install (Jan 2024) → pip install (Jan 2025) → DIFFERENT
- Right: pyproject.toml → uv lock → uv.lock → uv sync (any time) → IDENTICAL

**MIDDLE - Lockfile Structure**:
- Code block showing [[package]] entries
- Callouts: exact version, sha256 hash, dependencies

**BOTTOM - Benchmark Table**:
- Four operations comparing pip vs uv times
- 14-16x speedup factors

## Alt Text

Technical comparison of pip vs uv dependency resolution. Left flow shows pip install from pyproject.toml producing different environments over time (pandas 2.1.0 in Jan 2024, pandas 2.2.1 in Jan 2025). Right flow shows uv lock generating uv.lock once, then uv sync always producing identical environment. Lockfile structure shows exact versions, SHA-256 hashes, and dependency lists. Benchmark table: fresh install 14x faster with uv, dependency resolve 15x faster.

## Related Figures

- **fig-repro-08a**: ELI5 Lego analogy (simpler version)
- **fig-repro-08c**: UMAP/t-SNE initialization trap (concrete example)
- **fig-repro-12**: Dependency explosion (expanded technical details)
- **fig-repro-14**: Lockfiles as time machine (solution narrative)
- **fig-repo-14**: uv package manager overview (tool)

## Cross-References

Reader flow: **fig-repro-08a** (ELI5) → **THIS FIGURE** (technical lockfile comparison) → **fig-repro-12** (explosion details) → **fig-repro-14** (solution) → **fig-repo-14** (tool)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

