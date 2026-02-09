# fig-nb-05: Notebook Testing Landscape

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-05 |
| **Title** | Notebook Testing Landscape |
| **Complexity Level** | L3 (Expert) |
| **Target Persona** | ML Engineer |
| **Location** | docs/notebooks-guide.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Map the notebook testing ecosystem as a layered pyramid, showing which tools operate at each layer and which ones this project uses (and why others are deliberately excluded).

## Key Message

"Notebook testing is a four-layer pyramid: static analysis catches bad patterns before execution, smoke tests catch runtime crashes, output regression catches numerical drift, and unit tests verify logic. Pick the layers that match your architecture."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| nbval | Notebook validation via stored output comparison | [arXiv:2001.04808](https://arxiv.org/abs/2001.04808) |
| nbmake (Treebeard) | Smoke testing via top-to-bottom execution | [github.com/treebeardtech/nbmake](https://github.com/treebeardtech/nbmake) |
| testbook (nteract) | Unit testing individual notebook cells | [github.com/nteract/testbook](https://github.com/nteract/testbook) |
| Kitware | Best practices for notebook CI | [blog.kitware.com](https://blog.kitware.com/notebooks-ci/) |

## Visual Concept

```
+---------------------------------------------------------------------------------+
|                    NOTEBOOK TESTING LANDSCAPE                                    |
|                    A four-layer testing pyramid                                  |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  TESTING PYRAMID                                                                |
|  ================                                                               |
|                                                                                 |
|                         /\                                                      |
|                        /  \          UNIT TESTING                                |
|                       / T4 \         testbook: inject values, assert results     |
|                      /      \        pytest: test src/ modules directly          |
|                     /--------\       Scope: individual functions                 |
|                    /          \                                                  |
|                   /    T3      \     OUTPUT REGRESSION                           |
|                  /              \    nbval: compare cell outputs to stored       |
|                 /                \   Catches: numerical drift, API changes       |
|                /------------------\  Scope: full notebook outputs                |
|               /                    \                                             |
|              /        T2            \   SMOKE TESTING                            |
|             /                        \  nbmake / quarto render                   |
|            /                          \ Catches: ImportError, FileNotFound       |
|           /----------------------------\Scope: top-to-bottom execution           |
|          /                              \                                        |
|         /            T1                  \  STATIC ANALYSIS                      |
|        /                                  \ AST parsing, linting                 |
|       /                                    \Catches: bad imports, hex colors,    |
|      /                                      \       savefig calls               |
|     /________________________________________\Scope: code patterns only         |
|                                                                                 |
|  MOST COMMON (run on every commit) ---> MOST SPECIFIC (run on demand)           |
|                                                                                 |
+---------------------------------------------------------------------------------+
|  TOOL MATRIX                                                                    |
|  ===========                                                                    |
|                                                                                 |
|  | Layer | Tool                  | What It Tests              | Our Status    | |
|  |-------|-----------------------|----------------------------|---------------| |
|  | T1    | check_notebook_format | AST imports, hex, savefig  | PRE-COMMIT+CI | |
|  | T1    | ruff / nbqa           | PEP8, unused imports       | AVAILABLE     | |
|  | T2    | quarto render         | Top-to-bottom execution    | CI            | |
|  | T2    | nbmake (pytest)       | Same, pytest integration   | NOT USED      | |
|  | T3    | nbval                 | Stored output comparison   | NOT USED      | |
|  | T4    | testbook              | Cell-level unit tests      | NOT USED      | |
|  | T4    | pytest (src/)         | Module-level unit tests    | 2042 TESTS    | |
|                                                                                 |
+---------------------------------------------------------------------------------+
|  WHY WE SKIP T3 (nbval)                                                        |
|  =======================                                                        |
|                                                                                 |
|  Our architecture makes output regression unnecessary:                          |
|                                                                                 |
|  [Thin Notebook] --reads--> [DuckDB] --tested by--> [pytest src/]              |
|                                                                                 |
|  - Notebooks are THIN: import from src/, no inline logic                        |
|  - All computation tested via 2042 pytest tests on src/ modules                 |
|  - quarto render --execute catches runtime crashes                              |
|  - Freeze output handles caching (no need for stored-output diffing)            |
|                                                                                 |
|  OUR STACK: check_notebook_format.py (T1) + quarto render (T2)                 |
|             + pytest src/ (T4) = sufficient coverage                            |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| T1 static layer | `healthy_normal` | Green - active, working |
| T2 smoke layer | `healthy_normal` | Green - active in CI |
| T3 regression layer | `secondary_pathway` | Gray - deliberately skipped |
| T4 unit test layer | `healthy_normal` | Green - 2042 tests via pytest |
| Pyramid outline | `primary_pathway` | Deep blue structure |
| "NOT USED" labels | `secondary_pathway` | Gray - excluded by design |

## Content Elements

1. **Testing pyramid**: Four layers from broad (static) to narrow (unit), with labels
2. **Tool matrix table**: Tool name, what it tests, our usage status
3. **Architecture rationale box**: Why we skip nbval (thin notebook + tested src/)
4. **Our stack summary**: The three tools we actually use and why they suffice
5. **Layer labels**: T1-T4 with scope descriptions
6. **Flow diagram**: Thin notebook -> DuckDB -> pytest src/ data flow

## Anti-Hallucination Rules

- DO NOT invent test counts. Our test suite has 2042 passed tests (from actual CI run).
- DO NOT claim we use nbval or testbook. We deliberately skip them.
- DO NOT claim nbmake is deprecated. It is maintained by Treebeard.
- DO NOT show performance numbers or AUROC. This is about testing infrastructure.
- The check_notebook_format.py script uses AST parsing (not regex) for detection.
- quarto render executes notebooks top-to-bottom; it is not a static check.

## Text Content

### Title Text
"Notebook Testing Landscape: A Four-Layer Pyramid"

### Caption
Notebook testing spans four layers: static analysis (AST-based pattern detection), smoke testing (top-to-bottom execution), output regression (stored output comparison), and unit testing (function-level assertions). Foundation PLR uses layers T1 (check_notebook_format.py pre-commit hook), T2 (quarto render in CI), and T4 (2042 pytest tests on src/ modules). We deliberately skip T3 (nbval) because our thin notebook architecture means all computation lives in tested src/ modules, not inline notebook code.

## Prompts for Nano Banana Pro

### Style Prompt
Testing pyramid infographic with four distinct layers. Bottom layer widest, top narrowest. Each layer labeled with tool names and scope. Professional muted palette on warm off-white background. Tool matrix table below pyramid. Rationale callout box at bottom explaining architectural decision.

### Content Prompt
Create "Notebook Testing Landscape" pyramid infographic:

**TOP - Testing Pyramid**:
- T4 (top, narrow): Unit testing - testbook, pytest src/
- T3: Output regression - nbval stored output comparison
- T2: Smoke testing - nbmake, quarto render
- T1 (bottom, wide): Static analysis - AST parsing, linting

**MIDDLE - Tool Matrix**:
- Table: Layer, Tool, What It Tests, Our Status (active/skipped)
- Green indicators for tools we use, gray for skipped

**BOTTOM - Architecture Rationale**:
- Flow: Thin Notebook -> DuckDB -> pytest src/
- "Why we skip nbval" explanation box

## Alt Text

Notebook testing pyramid infographic with four layers. Base layer T1: static analysis using check_notebook_format.py with AST-based detection, active in pre-commit and CI. Layer T2: smoke testing using quarto render for top-to-bottom execution, active in CI. Layer T3: output regression using nbval for stored output comparison, deliberately not used. Top layer T4: unit testing using pytest on src/ modules with 2042 tests. Tool matrix table showing each tool, what it tests, and usage status. Rationale box explaining thin notebook architecture makes output regression unnecessary.

## References

- nbval: Sherwood-Taylor et al. 2020 ([arXiv:2001.04808](https://arxiv.org/abs/2001.04808))
- nbmake: Treebeard ([github.com/treebeardtech/nbmake](https://github.com/treebeardtech/nbmake))
- testbook: nteract project ([github.com/nteract/testbook](https://github.com/nteract/testbook))
- Kitware: Notebook CI best practices ([blog.kitware.com](https://blog.kitware.com/notebooks-ci/))
- nbqa: Adapt linters to notebooks ([github.com/nbQA-dev/nbQA](https://github.com/nbQA-dev/nbQA))

## Related Figures

- **fig-nb-07**: From Notebook to Production (thin notebook pattern we rely on)
- **fig-repo-08**: Pre-commit Quality Gates (where check_notebook_format.py runs)
- **fig-repo-65**: Figure QA Categories (parallel testing concern for figures)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/notebooks-guide.md
