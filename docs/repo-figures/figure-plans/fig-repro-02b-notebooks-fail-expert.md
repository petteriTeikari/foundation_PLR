# fig-repro-02b: Why 96.8% of Notebooks Fail (Expert)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-02b |
| **Title** | Why 96.8% of Notebooks Fail (Expert) |
| **Complexity Level** | L3 (Expert) |
| **Target Persona** | ML Engineer, Biostatistician, Data Scientist |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Provide technical breakdown of Jupyter notebook failure modes from Pimentel et al. 2023 study with specific error categories and mitigation strategies.

## Key Message

"Reproducibility failures follow predictable patterns: import errors (38%), syntax/name errors (24%), file I/O (18%), and type/attribute errors (12%). Each has specific mitigations."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Pimentel et al. 2023 | Detailed error taxonomy from 27,271 notebooks | [arXiv:2308.07333](https://arxiv.org/abs/2308.07333) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    WHY 96.8% OF NOTEBOOKS FAIL                                  │
│                    Technical Analysis (Pimentel et al. 2023)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ERROR TAXONOMY (N=27,271 notebooks)                                            │
│  ═══════════════════════════════════                                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  ImportError / ModuleNotFoundError           38%  ████████████████████ │   │
│  │    • Missing package in environment                                     │   │
│  │    • Version-incompatible API                                           │   │
│  │    ─────────────────────────────────────────────────────────           │   │
│  │    Mitigation: uv.lock, pyproject.toml with pinned versions             │   │
│  │                                                                         │   │
│  │  SyntaxError / NameError                     24%  ████████████          │   │
│  │    • Python 2 vs 3 incompatibility                                      │   │
│  │    • Cell execution order dependencies                                  │   │
│  │    ─────────────────────────────────────────────────────────           │   │
│  │    Mitigation: Specify Python version, linear cell execution           │   │
│  │                                                                         │   │
│  │  FileNotFoundError / IOError                 18%  █████████             │   │
│  │    • Hardcoded absolute paths                                           │   │
│  │    • Missing data files                                                 │   │
│  │    ─────────────────────────────────────────────────────────           │   │
│  │    Mitigation: Relative paths, DuckDB consolidation                     │   │
│  │                                                                         │   │
│  │  TypeError / AttributeError                  12%  ██████                │   │
│  │    • API signature changes between versions                             │   │
│  │    • Deprecated function arguments                                      │   │
│  │    ─────────────────────────────────────────────────────────           │   │
│  │    Mitigation: Exact version pinning, deprecation awareness             │   │
│  │                                                                         │   │
│  │  Other (RuntimeError, ValueError, etc.)       8%  ████                  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  STUDY METHODOLOGY                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  Source:      PubMed Central (biomedical literature)                            │
│  Sample:      27,271 Jupyter notebooks from published papers                    │
│  Method:      Automated re-execution in clean environment                       │
│  Criterion:   Bitwise identical output                                          │
│  Result:      879 (3.2%) reproduced identically                                 │
│                                                                                 │
│  Note: "Functionally equivalent" (same results, different formatting)           │
│        was NOT counted as success—strict reproducibility standard               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR MITIGATIONS                                                     │
│  ═════════════════════════                                                      │
│                                                                                 │
│  │ Error Type    │ Our Solution                │ Implementation        │       │
│  │ ───────────── │ ─────────────────────────── │ ───────────────────── │       │
│  │ ImportError   │ uv.lock lockfile            │ pyproject.toml        │       │
│  │ SyntaxError   │ Python 3.11+ requirement    │ .python-version       │       │
│  │ FileNotFound  │ DuckDB single-source        │ SERI_PLR_GLAUCOMA.db  │       │
│  │ TypeError     │ Exact version pinning       │ No ~= or >= in deps   │       │
│  │ Random state  │ Documented seeds            │ configs/defaults.yaml │       │
│                                                                                 │
│  Source: arXiv:2308.07333                                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Error taxonomy bar chart**: Horizontal bars with percentages
2. **Error descriptions**: Specific error types and causes
3. **Mitigation strategies**: Per-error-type solutions
4. **Study methodology box**: Sample size, method, criterion
5. **Foundation PLR mapping table**: Error → solution mapping

## Text Content

### Title Text
"Why 96.8% of Notebooks Fail: Technical Analysis"

### Caption
Technical breakdown of Jupyter notebook failures from Pimentel et al. 2023 ([arXiv:2308.07333](https://arxiv.org/abs/2308.07333)). Of 27,271 biomedical notebooks, import/module errors account for 38% of failures, syntax/name errors 24%, file I/O errors 18%, and type/attribute errors 12%. Foundation PLR addresses each category: uv.lock for dependencies, Python version pinning, DuckDB data consolidation, and exact version specifications.

## Prompts for Nano Banana Pro

### Style Prompt
Technical infographic with horizontal bar chart for error types. Monospace code font for error names. Mitigation callouts with arrows. Muted professional palette. Table at bottom mapping errors to solutions.

### Content Prompt
Create an expert-level "Why Notebooks Fail" infographic:

**TOP - Error Taxonomy Bar Chart**:
- ImportError 38%
- SyntaxError 24%
- FileNotFoundError 18%
- TypeError 12%
- Other 8%
- Each with short description and mitigation

**MIDDLE - Methodology Box**:
- Source: PubMed Central
- Sample: 27,271 notebooks
- Criterion: Bitwise identical

**BOTTOM - Solutions Table**:
- Error type → Our solution → Implementation file

## Alt Text

Technical analysis infographic showing Jupyter notebook failure taxonomy. Horizontal bar chart shows: ImportError/ModuleNotFoundError (38%) mitigated by uv.lock, SyntaxError/NameError (24%) mitigated by Python version pinning, FileNotFoundError/IOError (18%) mitigated by DuckDB consolidation, TypeError/AttributeError (12%) mitigated by exact version pinning, Other errors (8%). Study methodology: 27,271 PubMed notebooks, bitwise identical criterion, 3.2% success rate. Source: arXiv:2308.07333.

## Related Figures

- **fig-repro-02a**: ELI5 version with kitchen analogy
- **fig-repo-14**: uv Package Manager deep dive
- **fig-repro-08b**: Dependency resolution technical details

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

