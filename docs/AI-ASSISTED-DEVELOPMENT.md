# AI-Assisted Development Practices

> **TRIPOD-Code Area**: T11 -- Transparency of AI tooling in scientific code development.
>
> This document describes the role, scope, and guardrails governing AI-assisted
> development in the foundation_PLR repository. It is intended for reviewers,
> collaborators, and readers who wish to understand how AI code generation was
> controlled and validated throughout the project.

---

## 1. Overview

This repository was developed with the assistance of **Claude Code** (Anthropic),
an AI coding agent that operates within a terminal environment with access to the
full project codebase. Claude Code reads files, executes commands, and proposes
edits, but all changes are subject to human review, automated testing, and
pre-commit validation before entering the main branch.

The AI agent operates under a structured instruction hierarchy committed to the
repository itself (see [Section 6: Transparency](#6-transparency)). These
instructions define what the agent may and may not do, which statistical methods
it must defer to verified implementations, and how it must handle clinical data.

---

## 2. What AI Assisted With

The following categories of work received AI assistance:

| Category | Description |
|----------|-------------|
| **Code generation** | Python and R scripts for data extraction, visualization, and orchestration |
| **Test writing** | Over 2,200 automated tests across unit, integration, and figure QA categories |
| **Configuration management** | Hydra configs, YAML registries, pre-commit hook setup |
| **Refactoring** | Computation decoupling (extraction vs. visualization), anti-hardcoding enforcement |
| **Documentation** | Docstrings, architecture documents, CLAUDE.md instruction files |
| **Data pipeline plumbing** | MLflow-to-DuckDB extraction, re-anonymization, streaming inserts |
| **Figure infrastructure** | Style system, color palettes, save utilities, JSON data export for reproducibility |
| **CI/CD** | GitHub Actions workflows, pre-commit hooks, linting configuration |

In all cases, the AI agent proposed changes that were reviewed by the human
developer before being committed. The agent did not have autonomous push access
to the repository.

---

## 3. What AI Did NOT Do

The following decisions were made entirely by the human research team, without
AI-generated recommendations being treated as authoritative:

| Category | Rationale |
|----------|-----------|
| **Statistical analysis design** | The STRATOS-compliant metric framework (Van Calster et al. 2024) was selected by the research team. The AI was instructed to implement, not to choose. |
| **Clinical decision thresholds** | Net benefit thresholds (5%, 10%, 15%, 20%) reflect clinical judgment about glaucoma screening cost-benefit tradeoffs. |
| **Evaluation metric selection** | The five STRATOS domains (discrimination, calibration, overall performance, clinical utility, probability distributions) were mandated by the research design. |
| **Research question framing** | The decision to fix the classifier (CatBoost) and vary preprocessing was a methodological choice, not an AI recommendation. |
| **Data collection and labeling** | All clinical data originate from Najjar et al. (2023, Br J Ophthalmol). Ground truth outlier masks and denoised signals were created by domain experts. |
| **Method selection** | The 11 outlier detection methods, 8 imputation methods, and 5 classifiers in the registry were chosen based on the research design. |
| **Manuscript writing** | The scientific narrative, interpretation of results, and conclusions are authored by the research team. |

---

## 4. Guardrails

### 4.1 Pre-Commit Hooks

The repository enforces 10 pre-commit hooks that run before every commit. These
prevent classes of errors that were identified during development, several of
which originated from AI-generated code (see [Section 5](#5-meta-learning-system)).

| Hook | Purpose | Prevents |
|------|---------|----------|
| `ruff` | Python linting with auto-fix | Style violations, unused imports |
| `ruff-format` | Deterministic Python formatting | Formatting inconsistency |
| `registry-integrity` | Verifies registry sources agree (canary, YAML, module, tests) | Method count drift |
| `registry-validation` | Runs registry tests (exactly 11/8/5 methods) | Invalid method inclusion |
| `r-hardcoding-check` | Detects hardcoded hex colors, `ggsave()`, custom themes in R | CRITICAL-FAILURE-004 recurrence |
| `computation-decoupling` | Blocks metric computation imports in `src/viz/` | CRITICAL-FAILURE-003 recurrence |
| `renv-sync-check` | Verifies R lockfile is in sync with dependencies | R reproducibility drift |
| `extraction-isolation-check` | Ensures synthetic data never contaminates production artifacts | Data provenance contamination |
| `figure-isolation-check` | Ensures synthetic data never appears in `figures/generated/` | CRITICAL-FAILURE-001 recurrence |
| `notebook-format-check` | Enforces Quarto-only policy, validates `.qmd` headers | Format inconsistency |

Hook definitions: [`.pre-commit-config.yaml`](../.pre-commit-config.yaml)

### 4.2 Automated Test Suite

The repository contains **2,200+ tests** organized by domain:

- **Unit tests**: Individual functions in `src/stats/`, `src/data_io/`, `src/extraction/`
- **Integration tests**: End-to-end extraction and analysis pipeline validation
- **Registry tests**: Method counts are exactly 11 outlier, 8 imputation, 5 classifier
- **Figure QA tests** (`tests/test_figure_qa/`): Multi-priority validation
  - P0: Synthetic/fake data detection (scientific integrity)
  - P1: Invalid metrics, overlapping visual elements
  - P2: DPI, dimensions, font sizes (publication standards)
  - P3: Accessibility (color contrast, distinguishable series)
- **Extraction guardrail tests**: Memory monitoring, stall detection, streaming insert verification

### 4.3 Architectural Enforcement

**Computation decoupling**: All metric computation occurs in the extraction
layer (`src/extraction/`). Visualization code (`src/viz/`) is restricted to
reading pre-computed results from DuckDB. This separation is enforced by both
a pre-commit hook and import-level checks.

**Registry as single source of truth**: The file
`configs/mlflow_registry/parameters/classification.yaml` defines the complete
set of valid methods. All code that references methods must validate against
this registry. No parsing of MLflow run names is permitted.

**Anti-hardcoding**: Hex colors, literal file paths, method names, and figure
dimensions must come from configuration files or the style system, never from
inline literals. This is enforced for R code by a dedicated pre-commit hook
and for Python code by the style system API (`setup_style()`, `COLORS`,
`save_figure()`).

---

## 5. Meta-Learning System

A distinctive feature of this project's AI-assisted development is the
**meta-learning system**: a collection of structured failure reports that
document incidents where AI-generated code introduced errors. Each report
follows a consistent format (summary, root cause, impact, fix, prevention)
and is committed to the repository at
[`.claude/docs/meta-learnings/`](../.claude/docs/meta-learnings/).

### 5.1 Critical Failures Documented

| ID | Incident | Automated Prevention Added |
|----|----------|---------------------------|
| CRITICAL-FAILURE-001 | AI generated calibration plots using **synthetic data** instead of real experimental predictions. All four model curves were identical due to a shared random seed. | Figure QA tests (P0), extraction isolation hook, figure isolation hook |
| CRITICAL-FAILURE-002 | Hardcoded values (paths, colors, method names) persisted despite explicit instructions and existing configuration systems. | R hardcoding pre-commit hook, anti-hardcoding self-check protocol |
| CRITICAL-FAILURE-003 | Visualization code computed metrics on the fly instead of reading from DuckDB, violating the two-block architecture. | Computation decoupling pre-commit hook, banned-import list |
| CRITICAL-FAILURE-004 | R figure scripts used `ggsave()` with hardcoded paths instead of `save_publication_figure()`. | R hardcoding pre-commit hook |
| CRITICAL-FAILURE-005 | An extraction script accumulated all data in memory before writing, causing 21 hours of swap thrashing (18 TB disk writes) without detection. | Streaming insert pattern, memory monitoring (`src/extraction/guardrails.py`), stall detection |
| VIOLATION-001 | AI used regex to parse Python source code despite an explicit ban; the correct approach is AST parsing. | Documentation reinforcement, code review protocol |

### 5.2 The Feedback Loop

The meta-learning system implements a closed feedback loop:

```
Incident detected (human review)
    --> Failure report written (.claude/docs/meta-learnings/)
    --> Root cause identified
    --> Automated prevention added (pre-commit hook, test, or CI check)
    --> AI instruction files updated (CLAUDE.md)
    --> Same class of error is now blocked automatically
```

This means that each failure made the system more robust. The 10 pre-commit
hooks and the figure QA test suite are direct products of this process. No
failure class identified during development lacks an automated prevention
mechanism.

---

## 6. Transparency

### 6.1 Committed Instruction Files

The AI instruction files are committed to the repository and are publicly
visible. They are not hidden internal tooling; they are a deliberate
transparency feature that allows reviewers to inspect exactly what constraints
governed AI-assisted development.

| File | Purpose |
|------|---------|
| [`CLAUDE.md`](../CLAUDE.md) | Top-level project context, pipeline description, data provenance |
| [`.claude/CLAUDE.md`](../.claude/CLAUDE.md) | Behavioral contract: what the AI must and must not do |
| [`.claude/rules/`](../.claude/rules/) | Numbered rules files (research question, registry, figures, STRATOS, packages, no-reimplementation) |
| [`.claude/docs/meta-learnings/`](../.claude/docs/meta-learnings/) | 18 failure and violation reports |
| [`.claude/skills/`](../.claude/skills/) | Reusable multi-step protocols (code QA, documentation optimization) |
| [`.claude/domains/`](../.claude/domains/) | Domain-specific context (MLflow experiments, visualization, testing) |

### 6.2 What the Instruction Files Encode

The instruction hierarchy encodes the following categories of constraints:

- **Research design invariants**: The classifier is fixed; only preprocessing varies.
- **Statistical reporting requirements**: All five STRATOS domains must be reported; AUROC alone is insufficient.
- **Data integrity rules**: The registry defines exactly 11 outlier methods, 8 imputation methods, and 5 classifiers.
- **Architectural boundaries**: Visualization code may only read from DuckDB.
- **Banned patterns**: No reimplementation of verified statistical methods, no regex for structured data parsing, no `pip`/`conda`.
- **Privacy constraints**: Subject-level data uses anonymized codes; original identifiers are gitignored.

### 6.3 Co-Authorship Attribution

Commits that received substantial AI assistance include a `Co-Authored-By`
trailer in the commit message, providing traceability at the version control
level.

---

## 7. Limitations

AI-assisted development in a scientific context carries inherent risks that
must be acknowledged:

1. **Domain knowledge gaps**. The AI agent does not understand the clinical
   significance of pupillary light reflex measurements or the ophthalmological
   context of glaucoma screening. All clinical decisions require domain expert
   review.

2. **Plausible but incorrect code**. AI-generated code can appear syntactically
   correct and pass basic tests while containing subtle statistical or logical
   errors. The synthetic data incident (CRITICAL-FAILURE-001) demonstrates that
   AI can produce scientifically meaningless output that looks visually
   reasonable.

3. **Instruction adherence is imperfect**. Despite explicit bans documented in
   CLAUDE.md, the AI agent violated rules on multiple occasions (regex for code
   parsing, hardcoded values, computation in visualization code). Automated
   enforcement via pre-commit hooks and CI is more reliable than instruction
   files alone.

4. **Context window limitations**. In long development sessions, the AI agent
   can lose track of previously established constraints, leading to repeated
   errors. The meta-learning system (Section 5) was created specifically to
   address this failure mode.

5. **Testing coverage is necessary but not sufficient**. The 2,200+ test suite
   catches many classes of errors, but cannot verify scientific validity of
   results. Human expert review of statistical outputs, figure interpretation,
   and clinical conclusions remains essential.

6. **Reproducibility depends on the full toolchain**. While the code is
   reproducible, the AI-assisted development process itself is not fully
   deterministic. Different AI sessions may produce different (but functionally
   equivalent) implementations.

---

## 8. Recommendations for Reviewers

- Inspect the [`.claude/docs/meta-learnings/`](../.claude/docs/meta-learnings/)
  directory to understand what went wrong during development and how it was
  fixed.
- Verify that pre-commit hooks are active by running `pre-commit run --all-files`.
- Run `pytest tests/test_figure_qa/ -v` to confirm figure QA passes.
- Review [`CLAUDE.md`](../CLAUDE.md) and [`.claude/CLAUDE.md`](../.claude/CLAUDE.md)
  to understand the constraints under which AI-generated code was produced.
- Note that all statistical methods (calibration via pmcalibration, model
  stability via pminternal, decision curve analysis via dcurves) use canonical
  R package implementations, not AI-generated reimplementations.

---

## References

- Collins GS, et al. (2024). TRIPOD+AI statement. *BMJ*.
- Pollard TJ, Sounack M, et al. (2026). TRIPOD-Code protocol. *Diagn Progn Res*, 10, 3.
- Van Calster B, Collins GS, Vickers AJ, et al. (2024). Performance evaluation of predictive AI models. STRATOS Initiative TG6.
- Najjar RP, et al. (2023). Pupillary light reflex for glaucoma screening. *Br J Ophthalmol*.

---

*This document is part of the repository's TRIPOD-Code compliance mapping.
See also: [TRIPOD-CODE-COMPLIANCE.md](TRIPOD-CODE-COMPLIANCE.md).*
