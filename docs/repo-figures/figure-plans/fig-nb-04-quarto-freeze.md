# fig-nb-04: Quarto Freeze -- Your CI Time Machine

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-04 |
| **Title** | Quarto Freeze -- Your CI Time Machine |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | PhD Student |
| **Location** | notebooks/README.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:9 |

## Purpose

Explain Quarto's freeze mechanism using a timeline metaphor (parallel to fig-repro-14 which uses the same metaphor for lockfiles). Freeze preserves COMPUTATION results, while lockfiles preserve DEPENDENCY versions. Together they form the full reproducibility story: same packages AND same outputs without re-executing expensive code.

## Key Message

"Quarto freeze is a time machine for computation -- it stores pre-computed results so CI only needs Pandoc, not your entire Python environment."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Quarto documentation | Freeze stores computational results for rendering without re-execution | [quarto.org/docs/projects/code-execution.html](https://quarto.org/docs/projects/code-execution.html) |
| Cetinkaya-Rundel & Lowndes 2022 | Quarto as open-source scientific publishing system | [DOI: 10.21105/joss.05313](https://doi.org/10.21105/joss.05313) |

## Visual Concept

```
+---------------------------------------------------------------------------+
|              QUARTO FREEZE: YOUR CI TIME MACHINE                           |
|              Pre-compute locally, render anywhere                          |
+---------------------------------------------------------------------------+
|                                                                           |
|  THE TIMELINE                                                             |
|  ============                                                             |
|                                                                           |
|  STEP 1: Developer renders locally                                        |
|  +---------------------------------------------------------------+       |
|  |  $ quarto render                                               |       |
|  |  ~~~~~~~~~~~~~~~                                               |       |
|  |  Needs: Python 3.11, DuckDB, pandas, matplotlib, src/...      |       |
|  |                                                                |       |
|  |  [Cell 1] import duckdb ... (runs Python)                      |       |
|  |  [Cell 2] df = conn.execute("SELECT ...") ... (queries DB)     |       |
|  |  [Cell 3] fig, ax = plt.subplots() ... (creates figure)        |       |
|  |  [Cell 4] Mermaid diagram (renders to SVG)                     |       |
|  |                                                                |       |
|  |  All cells execute. All outputs captured.                      |       |
|  +---------------------------------------------------------------+       |
|                                                                           |
|          |                                                                |
|          v  freeze: auto                                                  |
|                                                                           |
|  STEP 2: _freeze/ stores pre-computed results                             |
|  +---------------------------------------------------------------+       |
|  |  _freeze/                                                      |       |
|  |  +-- 01-pipeline-walkthrough/                                  |       |
|  |  |   +-- execute-results/                                      |       |
|  |  |       +-- html.json         <-- cell outputs as JSON        |       |
|  |  |       +-- figure-html/      <-- rendered figures as PNG     |       |
|  |  +-- 02-reproduce-and-extend/                                  |       |
|  |      +-- execute-results/                                      |       |
|  |          +-- html.json                                         |       |
|  |          +-- figure-html/                                      |       |
|  +---------------------------------------------------------------+       |
|                                                                           |
|          |                                                                |
|          v  git add _freeze/ && git commit                                |
|                                                                           |
|  STEP 3: _freeze/ committed to git                                        |
|  +---------------------------------------------------------------+       |
|  |  git log:                                                      |       |
|  |  abc1234 chore: update freeze after pipeline walkthrough edits |       |
|  |                                                                |       |
|  |  _freeze/ is versioned alongside source code.                  |       |
|  |  Anyone who clones gets the pre-computed results.              |       |
|  +---------------------------------------------------------------+       |
|                                                                           |
|          |                                                                |
|          v  CI pulls, runs quarto render                                  |
|                                                                           |
|  STEP 4: CI renders from frozen results                                   |
|  +---------------------------------------------------------------+       |
|  |  $ quarto render   (in CI)                                     |       |
|  |  ~~~~~~~~~~~~~~~                                               |       |
|  |  Needs: Pandoc only (no Python, no DuckDB, no packages)        |       |
|  |                                                                |       |
|  |  Reads _freeze/ JSON --> assembles HTML --> done.               |       |
|  |  No code execution. No kernel. No dependencies.                |       |
|  +---------------------------------------------------------------+       |
|                                                                           |
+---------------------------------------------------------------------------+
|                                                                           |
|  THE DIFFERENCE                                                           |
|  ==============                                                           |
|                                                                           |
|  WITHOUT FREEZE                          WITH FREEZE                      |
|  +---------------------------+           +---------------------------+    |
|  | CI needs:                 |           | CI needs:                 |    |
|  |   Python 3.11             |           |   Pandoc                  |    |
|  |   uv sync (200+ pkgs)    |           |   (already in quarto)     |    |
|  |   DuckDB + data files    |           |                           |    |
|  |   System R (optional)     |           | Build time: ~30 seconds   |    |
|  |                           |           |                           |    |
|  | Build time: 5+ minutes    |           | No secrets / data needed  |    |
|  | Data must be accessible   |           | No Python environment     |    |
|  | Any dep change = breakage |           | Deterministic output      |    |
|  +---------------------------+           +---------------------------+    |
|                                                                           |
+---------------------------------------------------------------------------+
|                                                                           |
|  FREEZE MODES                                                             |
|  ============                                                             |
|                                                                           |
|  freeze: auto    Re-execute ONLY when .qmd source changes.               |
|                  (Our choice -- balances freshness and speed)             |
|                                                                           |
|  freeze: true    NEVER re-execute. Always use frozen results.             |
|                  (Use for archival / publication snapshots)               |
|                                                                           |
|  freeze: false   ALWAYS re-execute. No caching.                          |
|                  (Default -- no freeze benefit)                           |
|                                                                           |
|  IMPORTANT: Freeze only applies to PROJECT renders (quarto render         |
|  in project root). Individual file renders (quarto render file.qmd)       |
|  always re-execute regardless of freeze setting.                          |
|                                                                           |
+---------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| Step 1 (local render) | `melanopsin` | Gold -- developer action |
| Step 2 (_freeze/ storage) | `primary_pathway` | Deep blue -- data preservation |
| Step 3 (git commit) | `primary_pathway` | Deep blue -- version control |
| Step 4 (CI render) | `healthy_normal` | Green -- fast, clean CI |
| "Without freeze" panel | `abnormal_warning` | Red -- slow, fragile |
| "With freeze" panel | `healthy_normal` | Green -- fast, reliable |
| freeze:auto label | `healthy_normal` | Green -- recommended mode |
| freeze:true label | `melanopsin` | Gold -- archival mode |
| freeze:false label | `secondary_pathway` | Gray -- default/no benefit |
| "IMPORTANT" nuance | `abnormal_warning` | Red -- critical caveat |
| Timeline arrows | `primary_pathway` | Deep blue -- flow direction |

## Content Elements

1. **Four-step timeline**: Local render --> _freeze/ storage --> git commit --> CI render
2. **Dependency contrast at each step**: Full Python env (Step 1) vs Pandoc only (Step 4)
3. **_freeze/ directory tree**: Actual structure with execute-results/ and figure-html/
4. **Side-by-side comparison**: "Without freeze" (5+ min, full env) vs "With freeze" (30 sec, Pandoc only)
5. **Three freeze modes**: auto (our choice), true (archival), false (default)
6. **Critical nuance callout**: Project renders vs individual file renders
7. **Git integration**: _freeze/ committed alongside source code

## Anti-Hallucination Rules

- `freeze: auto` re-executes when the SOURCE .qmd file changes, not when dependencies change. Do NOT claim it detects dependency changes.
- Freeze ONLY applies to project-level renders (`quarto render` in project root). Individual file renders (`quarto render file.qmd`) ALWAYS re-execute. This is a real and important nuance.
- The _freeze/ directory stores JSON output and rendered figures, NOT pickled Python objects. Do NOT claim it stores Python state.
- CI with freeze still needs Quarto installed (which bundles Pandoc). Do NOT claim only Pandoc is needed separately.
- The 30-second vs 5-minute comparison is illustrative. Actual times depend on project size and CI hardware. Do NOT present as measured benchmarks.
- This is analogous to fig-repro-14 (lockfiles as time machine) but for COMPUTATION not DEPENDENCIES. Do NOT conflate the two concepts.

## Text Content

### Title Text
"Quarto Freeze: Your CI Time Machine"

### Caption
Quarto freeze stores pre-computed notebook results so CI environments can render documentation without executing any code. Step 1: developer renders locally with full Python environment, executing all cells against DuckDB. Step 2: freeze captures outputs as JSON and PNG in _freeze/ directory. Step 3: _freeze/ is committed to git alongside source. Step 4: CI renders from frozen results using only Pandoc (bundled with Quarto), completing in seconds instead of minutes. Three modes available: auto (re-execute on source change, our default), true (never re-execute, for archival), false (always re-execute). Critical nuance: freeze only applies to project-level renders, not individual file renders.

## Prompts for Nano Banana Pro

### Style Prompt
Vertical timeline infographic on off-white background, echoing the time machine metaphor from fig-repro-14. Four numbered steps flowing downward with connecting arrows. Terminal mockups for commands. Side-by-side comparison panel at bottom. Clean sans-serif typography. Warm, educational tone with matte finish.

### Content Prompt
Create a "Quarto Freeze: CI Time Machine" infographic:

**MAIN FLOW - Vertical Timeline (4 steps)**:
1. Developer renders locally (gold) -- full Python env, all cells execute
2. _freeze/ stores results (blue) -- directory tree showing JSON + figures
3. git commit (blue) -- _freeze/ versioned in repo
4. CI renders from freeze (green) -- only Pandoc needed, 30 seconds

**COMPARISON PANEL**:
- Left: "Without freeze" (red tint) -- 5+ min, full env, fragile
- Right: "With freeze" (green tint) -- 30 sec, Pandoc only, deterministic

**BOTTOM STRIP**:
- Three freeze modes: auto (green, recommended), true (gold, archival), false (gray, default)
- Important caveat about project vs individual file renders

## Alt Text

Vertical timeline infographic showing Quarto freeze as a CI time machine in four steps. Step 1: developer renders locally needing Python 3.11, DuckDB, pandas, matplotlib, executing all notebook cells. Step 2: _freeze/ directory stores pre-computed results as JSON and PNG files in execute-results subdirectories. Step 3: _freeze/ committed to git so anyone who clones gets pre-computed results. Step 4: CI renders from frozen results needing only Pandoc, completing in approximately 30 seconds. Comparison panel contrasts without freeze (5+ minutes, full environment, any dependency change breaks) versus with freeze (30 seconds, Pandoc only, deterministic output). Bottom section explains three modes: auto (re-execute on source change), true (never re-execute), false (always re-execute), with caveat that freeze only applies to project-level renders.

## Related Figures

- **fig-repro-14**: Lockfiles as time machine (parallel metaphor for DEPENDENCIES)
- **fig-nb-03**: Our Quarto architecture (where freeze fits in the three enforcement layers)
- **fig-nb-01**: Notebook landscape (Quarto's freeze advantage over Jupyter)
- **fig-repo-68**: GitHub Actions CI pipeline (where freeze renders happen)

## Cross-References

Reader flow: **fig-nb-03** (our architecture, Layer 3 overview) --> **THIS FIGURE** (freeze deep dive)

Parallel to fig-repro-14: lockfiles are a time machine for DEPENDENCIES (same packages). Freeze is a time machine for COMPUTATION (same outputs). Together they guarantee: same packages + same outputs = full reproducibility.

## References

1. Quarto code execution and freeze documentation. https://quarto.org/docs/projects/code-execution.html
2. Cetinkaya-Rundel M, Lowndes JSS. "An Open-Source Scientific Publishing System." JOSS 8(89), 5313 (2022). DOI: 10.21105/joss.05313

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in notebooks/README.md
