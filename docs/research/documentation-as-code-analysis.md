# Documentation-as-Code Research & Decision Analysis

> **Research Date:** 2026-01-23
> **Context:** Foundation PLR scientific research repository
> **Goal:** Determine optimal automated documentation strategy for publication and long-term maintenance
> **Analyst:** Claude Opus 4.5 (LLM-as-Judge synthesis)

---

## User Prompt (Verbatim)

```
So could we do a mini research on comments-as-code, and any automated documentation frameworks!
How would you now create automatically documentation from our codebase with the help of Claude Code
as it is now unrealistic to assume that someone wants to manually be updating the documentation.
The documentation should occur automatically with as little developer effort as possible with
excellent DevEx! What do the cool kids use for documentation-as-code especially suitable for
scientific labs in which people do not necessarily have such software skills, but would enjoy
having that automated SWE as a service there? Think of this repo now and how to get it as
documented as possible for publication, AND keep it updated if/when someone wants to keep on
working on it and add new methods and maybe update the whole study like in 2028 when we foundation
models dominating everything, but there should not be any need to write from scratch the glue code
necessarily? Hydra and MLflow can still be relevant?
```

---

## Part 1: Input Synthesis - LLM Recommendations Summary

### OpenAI Recommendations
- **Tools:** MkDocs + Material, Docusaurus, Antora, Sphinx
- **Key insight:** "Treat Claude as a drafting engine, not author of record"
- **Framework:** Diátaxis (Tutorials/How-to/Reference/Explanation)
- **CI essentials:** Link checker, Markdown lint, Vale prose lint, PR previews
- **Policy:** Define what Claude can do vs. humans must verify

### Gemini Recommendations
- **Tools:** MkDocs, Docusaurus, Astro Starlight, Mermaid.js
- **Key insight:** "Sidecar Pattern" - docs in same repo as code
- **Templates:** API endpoints, ADRs, User Guides, READMEs
- **CI:** Markdownlint, Lychee (link checker), Mermaid CLI validation
- **Prompt engineering:** "Living Document" prompt for Claude

### Grok Recommendations
- **Tools:** MkDocs, Docusaurus, Sphinx, GitBook, Jekyll
- **Key insight:** For Python-heavy projects, Sphinx excels at auto-extracted docs
- **Workflow:** PR-based reviews, semantic versioning, linters (Vale, Markdownlint)
- **AI handling:** Treat AI output as draft, add traceability metadata

---

## Part 2: Repository Context Analysis

### Current State of foundation_PLR

| Aspect | Current State | Implication for Doc Strategy |
|--------|---------------|------------------------------|
| **Language** | Python 3.11+ | Sphinx or mkdocstrings natural fit |
| **Package manager** | UV (conda banned) | Need UV-compatible tooling |
| **Config** | Hydra YAML | Can auto-generate config docs |
| **Experiment tracking** | MLflow | Can link to experiment artifacts |
| **Docstrings** | NumPy-style (600+ just added) | **Ready for autodoc extraction** |
| **Diagrams** | Mermaid in .claude/ | Already using Mermaid syntax |
| **Team size** | Individual/small | Minimal overhead critical |
| **Audience** | Scientific reviewers, future researchers | Need clear, citable docs |
| **Longevity** | 2+ years (2028 updates mentioned) | Must be maintainable long-term |
| **Current docs** | READMEs, ARCHITECTURE.md, KNOWLEDGE_GRAPH.md | Foundation exists |

### Unique Requirements for Scientific Research

1. **Reproducibility:** Documentation must enable exact replication of experiments
2. **Citation:** Methods sections must be citable/linkable for publications
3. **Versioning:** Track which doc version corresponds to which code/experiment version
4. **Low maintenance:** Researchers aren't dedicated SWEs
5. **Publication-ready:** Figures, methods must be extractable for papers
6. **Config documentation:** Hydra configs need human-readable explanations

---

## Part 3: Web Research Findings (2026 Landscape)

### 🚨 CRITICAL FINDING: MkDocs Ecosystem Transition

**This is the most important discovery affecting our decision.**

From [Material for MkDocs Blog (Nov 2025)](https://squidfunk.github.io/mkdocs-material/blog/2025/11/05/zensical/):

> "The underlying MkDocs project that powers Material for MkDocs has been **unmaintained since August 2024**... With MkDocs unmaintained and facing fundamental supply chain concerns, we cannot guarantee Material for MkDocs will continue working reliably in the future."

**Timeline:**
- **August 2024:** MkDocs core became unmaintained
- **November 5, 2025:** Material for MkDocs entered maintenance mode
- **May 1, 2026:** Insiders repository will be deleted
- **November 2026:** Material for MkDocs support ends (12-month commitment)
- **Late 2026:** Zensical expected to reach feature parity

**Zensical** is the replacement framework:
- Same creators as Material for MkDocs + mkdocstrings
- MIT licensed (fully open source)
- Backwards compatible with mkdocs.yml configs
- Currently v0.0.11 (alpha, missing multi-version docs)
- 4-5x faster builds, better search ("Disco")

### mkdocstrings - The Key to Automatic Python Docs

From [mkdocstrings documentation](https://mkdocstrings.github.io/):

> "mkdocstrings provides automatic documentation from sources... It is able to visit the Abstract Syntax Tree (AST) of the source code to extract useful information."

**Key features:**
- Supports NumPy-style, Google-style, Sphinx-style docstrings
- Cross-references with intersphinx-like functionality
- Used by: FastAPI, Pydantic, Prefect, Textual, NVIDIA, Google, Microsoft
- Latest: v1.0.1 (January 19, 2026)
- **Works with Zensical** (same creators)

### Claude Code GitHub Actions - Official Automation

From [Claude Code Docs](https://code.claude.com/docs/en/github-actions):

> "Claude Code GitHub Actions brings AI-powered automation to your GitHub workflow. With a simple @claude mention in any PR or issue, Claude can analyze your code, create pull requests, implement features, and fix bugs."

**Documentation-specific features:**
- PR-Triggered Docs Updater: Auto-updates docs when code changes
- Scheduled Docs Maintainer: Daily/monthly sync to ensure docs align with code
- Automatic PR Documentation Generator: Creates changelog-style docs for merged PRs

### Scientific Python Community Position

From [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/docs/):

> "Sphinx is a popular documentation framework for scientific libraries with a history of close usage with scientific tools like LaTeX."

But also notes MkDocs is used by modern projects like Polars, Pydantic, FastAPI.

### Vale Prose Linter

From [Vale.sh](https://vale.sh):
- Enforces style guides (Microsoft, Google, custom)
- CI-integrated quality checks
- VS Code integration for real-time feedback

---

## Part 4: Critical Assessment

### MkDocs vs Sphinx - Head-to-Head for foundation_PLR

| Criterion | MkDocs + Material | Sphinx | Winner |
|-----------|-------------------|--------|--------|
| **Setup time** | ~15 minutes | ~1 hour | MkDocs |
| **Markdown native** | ✅ Yes | ❌ reST (MyST addon) | MkDocs |
| **NumPy docstring extraction** | ✅ mkdocstrings | ✅ autodoc | Tie |
| **Live preview** | ✅ Auto-refresh | ❌ Manual rebuild | MkDocs |
| **PDF output** | ❌ Limited | ✅ Native | Sphinx |
| **Scientific credibility** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Sphinx |
| **Long-term maintenance (2028+)** | ⚠️ Uncertain (Zensical) | ✅ Stable | Sphinx |
| **Claude compatibility** | ✅ Direct Markdown | ⚠️ Needs conversion | MkDocs |
| **Mermaid diagrams** | ✅ Plugin | ⚠️ Extension | MkDocs |
| **Hydra config docs** | ⚠️ Manual | ⚠️ Manual | Tie |

### The Zensical Wildcard

**Risk assessment:**
- Starting with Material for MkDocs in Jan 2026 gives ~10 months before support ends
- Zensical migration should be "seamless" (same creators claim)
- mkdocstrings will continue working with Zensical
- But: Zensical is alpha, missing features

**Mitigation:** Design docs to be framework-agnostic (pure Markdown + docstrings)

### Why NOT Sphinx for foundation_PLR

Despite Sphinx's scientific credibility:
1. **Learning curve:** reST syntax adds friction for quick updates
2. **Claude friction:** Claude outputs Markdown naturally; reST needs conversion
3. **Developer experience:** No live reload, slower iteration
4. **Overkill:** We don't need PDF output or LaTeX integration for web docs

### Why NOT Docusaurus

1. **JavaScript ecosystem:** Adds Node.js dependency to Python-only project
2. **No Python autodoc:** Would need to manually write all API docs
3. **Ecosystem mismatch:** Not used in scientific Python community

---

## Part 5: Multi-Hypothesis Decision Matrix

### Scoring Criteria (1-5 scale)

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Setup complexity** | 15% | Time/effort to get running |
| **Python ecosystem fit** | 20% | Autodoc, community, tooling |
| **Scientific credibility** | 15% | Reviewer/community acceptance |
| **Claude integration** | 20% | AI-assisted documentation ease |
| **Long-term maintenance** | 20% | Viability through 2028+ |
| **DevEx quality** | 10% | Developer happiness |

### Hypothesis 1: MkDocs + Material + mkdocstrings (Current Best)

| Criterion | Score | Weighted | Rationale |
|-----------|-------|----------|-----------|
| Setup complexity | 5 | 0.75 | Single YAML config, UV compatible |
| Python ecosystem fit | 5 | 1.00 | mkdocstrings extracts NumPy docstrings |
| Scientific credibility | 4 | 0.60 | Used by Polars, Pydantic, FastAPI |
| Claude integration | 5 | 1.00 | Direct Markdown, Mermaid native |
| Long-term maintenance | 3 | 0.60 | ⚠️ MkDocs sunset, Zensical transition |
| DevEx quality | 5 | 0.50 | Live reload, clean UX |
| **TOTAL** | | **4.45/5** | |

### Hypothesis 2: Sphinx + MyST + autodoc (Traditional Scientific)

| Criterion | Score | Weighted | Rationale |
|-----------|-------|----------|-----------|
| Setup complexity | 2 | 0.30 | More config, conf.py complexity |
| Python ecosystem fit | 5 | 1.00 | Built for Python, ReadTheDocs |
| Scientific credibility | 5 | 0.75 | Gold standard for science |
| Claude integration | 3 | 0.60 | MyST helps but still friction |
| Long-term maintenance | 5 | 1.00 | Stable, mature ecosystem |
| DevEx quality | 2 | 0.20 | No live reload, slow builds |
| **TOTAL** | | **3.85/5** | |

### Hypothesis 3: Zensical (Early Adoption)

| Criterion | Score | Weighted | Rationale |
|-----------|-------|----------|-----------|
| Setup complexity | 4 | 0.60 | Slightly more than MkDocs |
| Python ecosystem fit | 5 | 1.00 | mkdocstrings works |
| Scientific credibility | 3 | 0.45 | Too new, unproven |
| Claude integration | 5 | 1.00 | Markdown native |
| Long-term maintenance | 4 | 0.80 | Future-proof but alpha risk |
| DevEx quality | 5 | 0.50 | 4-5x faster, modern |
| **TOTAL** | | **4.35/5** | |

### Hypothesis 4: Hybrid - MkDocs now → Zensical later (Recommended)

| Criterion | Score | Weighted | Rationale |
|-----------|-------|----------|-----------|
| Setup complexity | 5 | 0.75 | Start simple, migrate later |
| Python ecosystem fit | 5 | 1.00 | mkdocstrings throughout |
| Scientific credibility | 4 | 0.60 | Modern but professional |
| Claude integration | 5 | 1.00 | Markdown throughout |
| Long-term maintenance | 5 | 1.00 | Planned transition path |
| DevEx quality | 5 | 0.50 | Best of both worlds |
| **TOTAL** | | **4.85/5** | ⭐ WINNER |

---

## Part 6: Final Recommendation

### The Winning Strategy: Phased Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DOCUMENTATION STRATEGY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1 (Now - Oct 2026): MkDocs + Material + mkdocstrings             │
│  ├── Immediate value: Auto-extract 600+ docstrings                      │
│  ├── Claude integration: Native Markdown                                │
│  └── GitHub Pages hosting: Free, automated                              │
│                                                                          │
│  PHASE 2 (Oct 2026): Zensical Migration                                 │
│  ├── Same mkdocstrings integration                                      │
│  ├── Same Markdown content                                              │
│  └── 4-5x faster builds, better search                                  │
│                                                                          │
│  ONGOING: Claude Code GitHub Actions                                    │
│  ├── Auto-update docs on code changes                                   │
│  ├── Monthly sync validation                                            │
│  └── PR documentation generation                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Approach Wins

1. **Immediate ROI:** 600+ docstrings become web documentation in < 1 day
2. **Future-proof:** Same content works with Zensical (backwards compatible)
3. **Claude-native:** Markdown input/output, no conversion needed
4. **Low friction:** Researchers can edit docs without learning new syntax
5. **Automated:** GitHub Actions + Claude Code = minimal manual work
6. **Scientific credibility:** mkdocstrings used by NVIDIA, Google, Microsoft

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MkDocs sunset | Planned Zensical migration; content is framework-agnostic |
| Zensical alpha bugs | Stay on MkDocs until Zensical 1.0 stable |
| AI hallucinations | Vale linting + human review for factual claims |
| Link rot | Lychee link checker in CI |

---

## Part 7: Implementation Plan

### Phase 1A: Foundation (Day 1)

```bash
# Install documentation stack
uv pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-mermaid2-plugin

# Create minimal mkdocs.yml
# Create docs/ folder structure
# Generate initial API reference from docstrings
```

### Phase 1B: CI/CD (Day 2)

```yaml
# .github/workflows/docs.yml
- Markdownlint for style consistency
- Lychee for broken links
- MkDocs build validation
- GitHub Pages deployment
```

### Phase 1C: Claude Integration (Day 3)

```yaml
# .github/workflows/claude-docs.yml
- Trigger on code changes
- Claude reviews and suggests doc updates
- Creates PR for review
```

### Folder Structure

```
docs/
├── index.md                    # Landing page
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── configuration.md        # Hydra config guide
├── user-guide/
│   ├── pipeline-overview.md
│   ├── outlier-detection.md
│   ├── imputation.md
│   ├── featurization.md
│   └── classification.md
├── api-reference/
│   ├── anomaly_detection.md    # Auto-generated from docstrings
│   ├── classification.md
│   ├── data_io.md
│   ├── featurization.md
│   ├── imputation.md
│   └── ...
├── tutorials/
│   ├── running-experiments.md
│   └── adding-new-methods.md
├── explanation/
│   ├── stratos-metrics.md
│   └── research-question.md
└── research/
    └── documentation-as-code-analysis.md  # This document
```

---

## Part 8: Quality Enforcement

### CI Pipeline Components

```yaml
name: Documentation Quality

on:
  push:
    paths: ['docs/**', '**/*.md', 'src/**/*.py']
  pull_request:
    paths: ['docs/**', '**/*.md', 'src/**/*.py']

jobs:
  lint-and-build:
    runs-on: ubuntu-latest
    steps:
      # 1. Markdown formatting
      - uses: DavidAnson/markdownlint-cli2-action@v16
        with:
          globs: '**/*.md'

      # 2. Link validation
      - uses: lycheeverse/lychee-action@v1
        with:
          args: --verbose --no-progress './**/*.md'

      # 3. Build validation
      - run: uv pip install mkdocs mkdocs-material mkdocstrings[python]
      - run: mkdocs build --strict

      # 4. Deploy to GitHub Pages (on main only)
      - run: mkdocs gh-deploy --force
        if: github.ref == 'refs/heads/main'
```

### Vale Configuration (Optional Enhancement)

```yaml
# .vale.ini
StylesPath = .github/styles
MinAlertLevel = suggestion

[*.md]
BasedOnStyles = Vale, write-good
```

---

## Part 9: Addressing User's Specific Questions

### Q: "Hydra and MLflow can still be relevant?"

**Yes, absolutely.** The documentation strategy enhances them:

1. **Hydra configs:** Document all YAML options with explanations
2. **MLflow artifacts:** Link documentation to experiment results
3. **Reproducibility:** Document exact configs needed to reproduce results

### Q: "Update the whole study like in 2028?"

The strategy handles this:

1. **API docs auto-update:** mkdocstrings extracts docstrings on every build
2. **Claude Code Actions:** Suggests doc updates when code changes
3. **Version tags:** mike plugin (or Zensical) handles versioned docs
4. **Framework-agnostic content:** Pure Markdown works across tools

### Q: "Automated SWE as a service for scientific labs?"

This stack provides exactly that:

| Manual Effort | Automated Alternative |
|---------------|----------------------|
| Writing API docs | mkdocstrings extracts from docstrings |
| Updating docs | Claude Code GitHub Actions |
| Checking links | Lychee in CI |
| Style consistency | Markdownlint + Vale |
| Deployment | GitHub Pages auto-deploy |

---

## Part 10: Sources

### Primary Sources

- [mkdocstrings Documentation](https://mkdocstrings.github.io/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Zensical Announcement](https://squidfunk.github.io/mkdocs-material/blog/2025/11/05/zensical/)
- [Claude Code GitHub Actions](https://code.claude.com/docs/en/github-actions)
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/docs/)

### Tool Comparisons

- [MkDocs vs Sphinx - Towards Data Science](https://towardsdatascience.com/switching-from-sphinx-to-mkdocs-documentation-what-did-i-gain-and-lose-04080338ad38/)
- [Python Documentation: MkDocs vs Sphinx](https://www.pythonsnacks.com/p/python-documentation-generator)
- [Scientific Python Cookie](https://scientific-python-cookie.readthedocs.io/en/latest/guides/docs/)

### CI/Quality Tools

- [Vale Prose Linter](https://vale.sh)
- [Lychee Link Checker](https://github.com/lycheeverse/lychee)
- [Markdownlint](https://github.com/DavidAnson/markdownlint)

### Automation

- [Claude Code Automatic PR Documentation Generator](https://github.com/marketplace/actions/claude-code-automatic-pr-documentation-generator)
- [Automate Your Documentation with Claude Code](https://medium.com/@fra.bernhardt/automate-your-documentation-with-claude-code-github-actions-a-step-by-step-guide-2be2d315ed45)

---

## Part 11: Decision Rationale Summary

### Why I Recommend MkDocs + mkdocstrings → Zensical

1. **Immediate value:** We have 600+ NumPy-style docstrings ready to extract
2. **Claude-native:** Markdown in/out, no conversion friction
3. **Scientific adoption:** Used by FastAPI, Pydantic, Polars, NVIDIA
4. **Future-proof:** Zensical is backwards compatible, same ecosystem
5. **DevEx excellence:** Live reload, fast builds, clean output
6. **Low maintenance:** GitHub Actions automates everything

### Why NOT Sphinx (Despite Scientific Credibility)

1. **Developer friction:** reST syntax, no live reload
2. **Claude friction:** Would need Markdown→reST conversion
3. **Overkill features:** Don't need PDF/LaTeX for web docs
4. **Modern alternatives exist:** mkdocstrings matches Sphinx autodoc quality

### Why NOT Wait for Zensical

1. **Alpha status:** v0.0.11, missing multi-version docs
2. **Risk:** New framework bugs in production
3. **Migration easy:** Same content, same mkdocstrings, just change CLI

---

## Next Steps

1. **Approve this plan** (user decision)
2. **Create mkdocs.yml** and folder structure
3. **Set up GitHub Actions** for automated deployment
4. **Generate initial API reference** from existing docstrings
5. **Add Claude Code workflow** for ongoing maintenance

---

*Document generated by Claude Opus 4.5 | Research synthesis date: 2026-01-23*
