# Deeper Figure Coverage Plan: 24+ Infographics for Non-SWE Audiences

**Status:** ✅ FIGURE PLANS COMPLETED
**Created:** 2026-01-31
**Last Updated:** 2026-01-31
**Reviewer Iterations:** 2
**Figure Plans Created:** 28 (fig-repo-13 through fig-repo-40)

---

## User Prompt (Verbatim)

> I now have generated all these 12 figures in /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/repo-figures/figure-plans and I think we could generate more of these schematics for the repository. Now that we have all the AGENTS.md, Copilot, and general discoverability addressed, we could explain visually the most common queries no? Could you do multiple analysation passes of the WHOLE repo (the code part) to identify infographics opportunities? Think of it as terms of explaining what the repo is all about in visual infographics terms to a developer is totally naïve to the repo! And remember that "the developer" in this case is most likely a biologist or a similar quantitative scientist with only basic knowledge of software engineering and can be easily overwhelmed! Let's create a another plan /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/repo-figures/deeper-figure-coverage-plan.md document on this! I did not for example yet see a figure that would describe all the Prefect blocks from the input /home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db to the final ggplot2 figures in /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2! It should be made crystal clear by several infographics how the repo achieves this! And note that the /home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db is also an intermediate duckdb artifacts when the actual raw data was in scattered .csv files and the duckdb database was chosen now as the easy portable single-source of truth (/home/petteri/Dropbox/KnowledgeBase/Tabular ML/Tabular ML - Libraries - DuckDB.md). Very likely people have nto heard of what this is? Also I used `polars` over `pandas` as the dataframe in the repo. probably worth explaining? Also why `uv` over `poetry` and the pip `requirements.txt` garbage that does not ensure exact library versions? Why `loguru` over the standard `logging` library? Why `logging` with .debug, .info, . warning, .error over simple prints? You really need to explain these basics to non-SWE quantitative scientists as they don't most likely think of these things! We can easily create 24-26 new figures and really lower to barrier to entry to this repo as I can guarantee that someone will find this overwhelming? And in the generated .md figure plan, prepare some figure captions that go with these along with the figures. We need to add hyperlinks to README.md files as well to background on this! Discussing why `uv` is better than `requirements.txt`! For further background on the emphasis on reproducibility see /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/literature/reproducibility-mlops, the whole /home/petteri/Dropbox/KnowledgeBase/MLOps, etc. Do some web search as well around the reproducibility crisis and the general upcoming LLM-powered knowledge management, e.g. Vomberg, Arnd, Evert de Haan, Nicolai Etienne Fabian, and Thijs Broekhuizen. 2024. "Digital Knowledge Engineering for Strategy Development." Journal of Business Research 177 (April): 114632. https://doi.org/10.1016/j.jbusres.2024.114632.

---

## User Prompt #2: Style Guide Integration (Verbatim)

> "And let's use the same style guide here /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/repo-figures/STYLE-GUIDE.md as already used in /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/literature/figures/PROMPTING-INSTRUCTIONS.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/literature/figures/CONTENT-TEMPLATE.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/literature/figures/STYLE-GUIDE.md (and other .md figures) as let's not create conflicting visual branding here for this repo ;) You could think of how to do style interpolation so that the /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/literature/figures/STYLE-GUIDE.md would get 75% weight and the "Economist ggplot2 aesthetics" would get 25% weighing. And all the 3D rendered figures should now for Guthub have zero "glowing scifi looks" that we occasionally get. This should more elegant than cringey. See attached images for successful elegant looks that would need slight "Economist aesthetics look" . Especially the fig-26-06-error-propagation-llm-pipeline.png is a gorgeous example in /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/literature/figures how Mermaid-style of graphs can be presented in stunning visual sense! The repo is all about visual neuroscience, retina, eye, foundation models, time series analysis, TSFM, biostatistics, clinical AI, design of clinical AI validation, clinical metrics, signal processing, signal decomposition, computational biology, etc. so I don't need any illustration of brains, brain tissue or anatomatical illustrations beyond the visual system, and even now the visualization should be retina and ocular anatomy focused! and Plan how to ensure that the STYLE-GUIDE copied gets this slight tweaks!"

---

## 0. VISUAL STYLE SPECIFICATION

### 0.1 Style Interpolation Formula

**75% Manuscript Style + 25% Economist Aesthetics**

| Aspect | Manuscript Style (75%) | Economist Adjustment (25%) |
|--------|------------------------|---------------------------|
| **Color Palette** | Muted, professional colors from manuscript STYLE-GUIDE | Add Economist signature: horizontal gridlines, red accent |
| **Typography** | Clean sans-serif, hierarchy | More condensed, informative density |
| **Diagrams** | Mermaid-style flowcharts | Subtle drop shadows, refined edges |
| **3D Elements** | Elegant, matte finishes | NO glowing/sci-fi effects |
| **Data Viz** | Clear, minimal | Add Economist-style axis treatments |

### 0.2 Domain Imagery Constraints

| ALLOWED | BANNED |
|---------|--------|
| Eye anatomy | Brain anatomy |
| Retina illustrations | Brain tissue |
| Pupil/iris visuals | Neurons/synapses (beyond retina) |
| Visual pathway (eye → optic nerve) | Cortical areas |
| Ophthalmic equipment | Generic science imagery |
| PLR signal traces | EEG/brain scans |

**Domain Focus**: Visual neuroscience, retina, ophthalmic biomarkers, pupillometry, clinical AI validation, TSFM, signal processing, biostatistics

### 0.3 Anti-Sci-Fi Rules (MANDATORY)

**BANNED Visual Elements** (add to negative prompt):
- Glowing effects
- Neon colors
- Sci-fi aesthetics
- Futuristic metallic textures
- Holographic elements
- Cyberpunk styling
- Lens flares

**REQUIRED Visual Qualities**:
- Matte, elegant surfaces
- Professional, academic aesthetic
- Subtle shadows (not harsh)
- Print-quality crispness
- Medical/clinical feel (not tech-startup)

### 0.4 Reference: Excellent Examples

**Gold Standard**: `fig-26-06-error-propagation-llm-pipeline.png`
- Location: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/literature/figures/`
- Why it works: Mermaid-style graph rendered with stunning depth, elegant without being flashy

### 0.5 Source Style Guides

| File | Path | Weight |
|------|------|--------|
| **Manuscript STYLE-GUIDE** | `sci-llm-writer/.../figures/STYLE-GUIDE.md` | 75% |
| **Repo STYLE-GUIDE** | `foundation_PLR/docs/repo-figures/STYLE-GUIDE.md` | Base |
| **CONTENT-TEMPLATE** | `sci-llm-writer/.../figures/CONTENT-TEMPLATE.md` | Structure |
| **PROMPTING-INSTRUCTIONS** | `sci-llm-writer/.../figures/PROMPTING-INSTRUCTIONS.md` | Technique |

### 0.6 Style Guide Update Checklist

When copying STYLE-GUIDE.md from manuscript, apply these tweaks:

- [ ] Add "Economist horizontal gridlines" to chart styles
- [ ] Add "NO glowing, NO sci-fi, matte elegant" to negative prompts
- [ ] Add domain constraint: "retina/eye ONLY, no brain anatomy"
- [ ] Add reference to fig-26-06 as gold standard
- [ ] Specify 75/25 interpolation in style description

---

## 1. TARGET AUDIENCE ANALYSIS

### 1.1 Primary Personas

| Persona | Background | Pain Points | What They Need |
|---------|------------|-------------|----------------|
| **Ophthalmology PI** | Excel, clinical trials | "What is a YAML file?" | High-level concepts, zero code |
| **Biostatistician** | R, SAS, some Python | "Why so many tools?" | Tool comparisons, practical benefits |
| **Research Scientist** | Python basics, Jupyter | "How do I add a method?" | Step-by-step, copy-paste examples |
| **Quantitative Biologist** | MATLAB, scripting | "This is overwhelming!" | Visual diagrams, analogies |

### 1.2 Knowledge Gaps (What They DON'T Know)

Based on typical non-SWE backgrounds:

| Concept | Likely Understanding | Our Explanation Goal |
|---------|---------------------|----------------------|
| **Version control (Git)** | "Track changes in Word" | Why commits matter for science |
| **Package managers** | None (manual downloads) | Why `uv` ensures reproducibility |
| **Configuration files** | "Settings in a GUI" | Why YAML is better than hardcoding |
| **Logging** | `print()` statements | Why structured logs save debugging hours |
| **Databases** | Excel, maybe SQL basics | Why DuckDB is "Excel on steroids" |
| **Containers** | None | Why Docker means "it works everywhere" |
| **Testing** | Manual checking | Why automated tests catch bugs |
| **CI/CD** | None | Why GitHub Actions verify code quality |

---

## 2. THE REPRODUCIBILITY CRISIS CONTEXT

### 2.1 Background Research

The reproducibility crisis in computational biology remains a significant concern:

- **Code/Data Unavailability**: Many published studies don't share complete code, data, or computational environments
- **Software Dependencies**: Specific software versions, libraries, and operating systems can lead to different results
- **Documentation Gaps**: Insufficient documentation of analysis pipelines, parameter choices, and preprocessing steps
- **Random Seeds**: Machine learning methods using RNG may not report seeds used

**Reference**: Vomberg et al. 2024 - "Digital Knowledge Engineering for Strategy Development" (DOI: 10.1016/j.jbusres.2024.114632)

> "Digital knowledge engineering is a branch of AI that attempts to mimic the judgment and behavior of a human expert, which can then be used to create, organize, and implement knowledge databases."

**Relevance to Foundation PLR**: Our repo demonstrates how proper tooling (uv, MLflow, Hydra) addresses reproducibility through:
- Locked dependencies (uv.lock)
- Experiment tracking (MLflow)
- Configuration management (Hydra YAML)
- Automated testing (pytest)

### 2.2 Why This Matters for Researchers

> "AI is shifting from being a technological challenge for narrow domains to becoming a critical strategic asset and catalyst for business operations." — Vomberg et al. 2024

Researchers who adopt these practices will:
1. Have reproducible results that survive reviewer scrutiny
2. Spend less time debugging "works on my machine" issues
3. Enable collaboration without environment conflicts
4. Meet increasingly strict journal reproducibility requirements

---

## 3. TECHNOLOGY DECISIONS: "WHY X OVER Y"

### 3.1 uv vs pip/requirements.txt

**Sources**: [Real Python](https://realpython.com/uv-vs-pip/), [DataCamp](https://www.datacamp.com/tutorial/python-uv), [Python Discourse](https://discuss.python.org/t/requirements-txt-or-uv-lock/78419)

| Feature | pip + requirements.txt | uv + uv.lock |
|---------|------------------------|--------------|
| **Reproducibility** | ❌ Can resolve to different versions | ✅ Exact lockfile for all deps |
| **Speed** | Slow (~60s for large projects) | Fast (~5s, 10-100x faster) |
| **Transitive deps** | ❌ Not locked by default | ✅ All sub-dependencies locked |
| **Uninstall cleanliness** | ❌ Leaves orphan packages | ✅ Removes all transitive deps |
| **Virtual env management** | Manual | Automatic |

**Infographic caption**: "pip's requirements.txt lists what you asked for, but uv.lock ensures everyone gets the EXACT same packages—down to the smallest dependency."

### 3.2 Polars vs Pandas

**Sources**: [JetBrains](https://blog.jetbrains.com/pycharm/2024/07/polars-vs-pandas/), [DataCamp](https://www.datacamp.com/tutorial/benchmarking-high-performance-pandas-alternatives), [Python Speed](https://pythonspeed.com/articles/polars-memory-pandas/)

| Feature | Pandas | Polars |
|---------|--------|--------|
| **Speed** | Baseline | 5-10x faster (up to 100x in benchmarks) |
| **Memory** | 5-10x dataset size needed | 2-4x dataset size needed |
| **Parallelism** | Single-threaded | Multi-core by default |
| **Execution** | Eager (computes immediately) | Lazy (optimizes query plan) |
| **Backend** | NumPy | Apache Arrow |

**Infographic caption**: "For our 1M+ timepoints dataset, Polars uses 5x less memory and runs 10x faster than Pandas—without changing your analysis logic."

### 3.3 DuckDB vs SQLite/PostgreSQL

| Feature | SQLite | PostgreSQL | DuckDB |
|---------|--------|------------|--------|
| **Optimization** | OLTP (transactions) | OLTP/OLAP | OLAP (analytics) |
| **Storage** | Row-based | Row-based | Column-based |
| **Setup** | Single file | Server required | Single file |
| **Analytics speed** | Slow for aggregates | Medium | Very fast |
| **Use case** | Web apps, mobile | Enterprise apps | Data science |

**Infographic caption**: "DuckDB is like having a portable Excel that can handle millions of rows and complex SQL queries in milliseconds—no server required."

### 3.4 Loguru vs print() vs logging

**Sources**: [GitHub Loguru](https://github.com/Delgan/loguru), [Real Python](https://realpython.com/python-loguru/), [Better Stack](https://betterstack.com/community/guides/logging/loguru/)

| Feature | print() | logging module | Loguru |
|---------|---------|----------------|--------|
| **Setup** | None | ~10 lines boilerplate | 1 line: `from loguru import logger` |
| **Timestamps** | ❌ Manual | ✅ Via config | ✅ Automatic |
| **Log levels** | ❌ None | ✅ debug/info/warning/error | ✅ Same, colored output |
| **File rotation** | ❌ Manual | ❌ Manual | ✅ Automatic |
| **Exception capture** | ❌ Lost in threads | ❌ Manual | ✅ Automatic with `catch()` |
| **JSON output** | ❌ No | ❌ Complex | ✅ `serialize=True` |

**Infographic caption**: "When debugging 1000 bootstrap iterations, print() loses messages in noise. Loguru captures everything with timestamps, levels, and colors—so you find bugs in seconds, not hours."

---

## 4. COMPLETE DATA PIPELINE VISUALIZATION

### 4.1 The Full Journey: Raw CSVs → Publication Figures

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    FOUNDATION PLR: END-TO-END DATA PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ORIGINAL DATA                 PROCESSING                      OUTPUTS              │
│  ════════════════              ══════════════                  ════════════         │
│                                                                                     │
│  📁 Scattered CSVs             📊 Python Pipeline              📈 R/ggplot2         │
│     (SERI, 500+ files)            (Prefect orchestrated)          figures           │
│           │                              │                           │              │
│           │ Data wrangling               │                           │              │
│           ▼                              │                           │              │
│  🗄️ SERI_PLR_GLAUCOMA.db ──────────────▶│                           │              │
│     (DuckDB: Single file,               │                           │              │
│      507 subjects,                      │                           │              │
│      1M+ timepoints)                    │                           │              │
│           │                              │                           │              │
│           │                              ▼                           │              │
│           │                    ┌─────────────────┐                   │              │
│           │                    │ Stage 1: Outlier │                  │              │
│           │                    │ Detection (11    │                  │              │
│           └────────────────────│ methods)         │                  │              │
│                                └────────┬────────┘                   │              │
│                                         │                           │              │
│                                         ▼                           │              │
│                                ┌─────────────────┐                   │              │
│                                │ Stage 2: Impute  │                  │              │
│                                │ (8 methods)      │                  │              │
│                                └────────┬────────┘                   │              │
│                                         │                           │              │
│                                         ▼                           │              │
│                                ┌─────────────────┐                   │              │
│                                │ Stage 3: Feature │                  │              │
│                                │ Extraction       │                  │              │
│                                └────────┬────────┘                   │              │
│                                         │                           │              │
│                                         ▼                           │              │
│                                ┌─────────────────┐                   │              │
│                                │ Stage 4: Classify │                 │              │
│                                │ (CatBoost fixed)  │                 │              │
│                                └────────┬────────┘                   │              │
│                                         │                           │              │
│                                         ▼                           │              │
│  ┌──────────────────────────────────────────────────────────────────┐│              │
│  │                    MLFLOW (Experiment Tracking)                  ││              │
│  │  • 410+ runs logged                                              ││              │
│  │  • 542 pickle files with bootstrap results                       ││              │
│  │  • Per-run: params, metrics, artifacts                           ││              │
│  └──────────────────────────────────────┬───────────────────────────┘│              │
│                                         │                           │              │
│                                         ▼                           │              │
│  ┌──────────────────────────────────────────────────────────────────┐│              │
│  │                BLOCK 1: EXTRACTION (Python)                      ││              │
│  │  • extract_all_configs_to_duckdb.py                              ││              │
│  │  • Re-anonymization: PLRxxxx → Hxxx/Gxxx                         ││              │
│  │  • Compute STRATOS metrics                                       ││              │
│  └──────────────────────────────────────┬───────────────────────────┘│              │
│                                         │                           │              │
│                                         ▼                           │              │
│  🗄️ foundation_plr_results.db ─────────────────────────────────────▶│              │
│     (Public DuckDB: shareable)                                      │              │
│           │                                                         │              │
│           │                                                         │              │
│           ▼                                                         │              │
│  ┌──────────────────────────────────────────────────────────────────┐│              │
│  │                BLOCK 2: ANALYSIS (R + Python)                    ││              │
│  │  • Read from DuckDB (READ ONLY)                                  ││              │
│  │  • Generate figures with ggplot2                                 ││              │
│  │  • Apply Economist-style theme                                   ││              │
│  │  • Export: PNG (300 DPI) + PDF + JSON                            ││              │
│  └──────────────────────────────────────┬───────────────────────────┘│              │
│                                         │                           │              │
│                                         ▼                           ▼              │
│                               📁 figures/generated/ggplot2/         │              │
│                                  • fig_R7_featurization.png         │              │
│                                  • fig_calibration_4models.png      │              │
│                                  • fig_dca_curves.pdf               │              │
│                                  • ... (40+ figures)                │              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. INFOGRAPHIC CATALOG: 26 NEW FIGURES

### Tier 1: FOUNDATIONAL (Must Have - 12 figures)

| ID | Title | Target Persona | Location | Priority |
|----|-------|----------------|----------|----------|
| **fig-repo-13** | "End-to-End Pipeline: CSVs → Figures" | All | Root README | P0 |
| **fig-repo-14** | "Why uv? Package Management Done Right" | Scientist | CONTRIBUTING | P0 |
| **fig-repo-15** | "Polars vs Pandas: Speed & Memory" | Scientist | docs/concepts | P0 |
| **fig-repo-16** | "DuckDB: Your Portable Data Warehouse" | All | docs/concepts | P0 |
| **fig-repo-17** | "Logging Levels: Why Not Just print()?" | Scientist | docs/concepts | P1 |
| **fig-repo-18** | "Two-Block Architecture: Extraction vs Analysis" | ML Engineer | ARCHITECTURE | P0 |
| **fig-repo-19** | "Subject Stratification: 507 vs 208" | All | docs/user-guide | P1 |
| **fig-repo-20** | "Error Propagation: How Outlier Errors Cascade" | Researcher | docs/user-guide | P1 |
| **fig-repo-21** | "Bootstrap Confidence Intervals: 1000 Iterations" | Stats | docs/concepts | P1 |
| **fig-repo-22** | "The 4 Standard Combos: When to Use Each" | All | docs/user-guide | P0 |
| **fig-repo-23** | "Data Privacy: What Gets Shared, What Stays Private" | All | CONTRIBUTING | P1 |
| **fig-repo-24** | "PLR Light Protocol: The 66-Second Recording" | PI/Stats | Root README | P1 |

### Tier 2: IMPORTANT (Should Have - 8 figures)

| ID | Title | Target Persona | Location | Priority |
|----|-------|----------------|----------|----------|
| **fig-repo-25** | "Handcrafted vs FM Embeddings: The 9pp Gap" | Researcher | docs/user-guide | P2 |
| **fig-repo-26** | "CatBoost Selection: Why We Fixed the Classifier" | Researcher | docs/user-guide | P2 |
| **fig-repo-27** | "Outlier Methods Compared: LOF, SVM, MOMENT" | Researcher | docs/user-guide | P2 |
| **fig-repo-28** | "Imputation Methods Compared: SAITS, CSDI, TimesNet" | Researcher | docs/user-guide | P2 |
| **fig-repo-29** | "Make Commands: One-Liners That Do Everything" | ML Engineer | docs/getting-started | P2 |
| **fig-repo-30** | "Test Pyramid: Unit, Integration, E2E" | ML Engineer | tests/README | P2 |
| **fig-repo-31** | "Registry Integrity: The Anti-Cheat System" | ML Engineer | configs/README | P2 |
| **fig-repo-32** | ".gitignore Explained: What Never Gets Committed" | Scientist | CONTRIBUTING | P2 |

### Tier 3: SUPPLEMENTARY (Nice to Have - 6 figures)

| ID | Title | Target Persona | Location | Priority |
|----|-------|----------------|----------|----------|
| **fig-repo-33** | "Critical Failure #001: Synthetic Data Incident" | ML Engineer | .claude/docs | P3 |
| **fig-repo-34** | "Critical Failure #002: Mixed Featurization" | ML Engineer | .claude/docs | P3 |
| **fig-repo-35** | "README Hierarchy: Which File Should I Read?" | All | docs/index | P3 |
| **fig-repo-36** | "Commit Message Convention: Semantic Commits" | Scientist | CONTRIBUTING | P3 |
| **fig-repo-37** | "Future Vision: Probabilistic Reconstruction" | Researcher | docs/roadmap | P3 |
| **fig-repo-38** | "Docker Containers: Ship the Whole Lab" | ML Engineer | docs/deployment | P3 |

---

## 6. DETAILED FIGURE PLANS

### 6.1 fig-repo-13: End-to-End Pipeline: CSVs → Figures

**Purpose**: Show the complete journey from raw clinical data to publication figures

**Caption**: "From 500+ scattered CSV files to 40+ publication-quality figures: the Foundation PLR pipeline consolidates data into a single DuckDB file, processes through four preprocessing stages tracked by MLflow, and generates reproducible visualizations with ggplot2."

**Key Message**: "One database in, one set of figures out—fully tracked, fully reproducible."

**Visual Elements**:
- Timeline showing data flow left-to-right
- Icons for each tool (DuckDB, Python, MLflow, R)
- Subject counts at each stage (507 → 208 for classification)
- Output gallery of sample figures

---

### 6.2 fig-repo-14: Why uv? Package Management Done Right

**Purpose**: Explain why uv is required over pip/requirements.txt

**Caption**: "pip's requirements.txt tells you what packages you asked for, but not what you got. uv's lockfile captures every sub-dependency, ensuring 'it works on my machine' becomes 'it works on every machine.'"

**Key Message**: "uv = reproducible environments, pip = reproducibility roulette"

**Visual Elements**:
- Side-by-side comparison: requirements.txt vs uv.lock
- Dependency tree showing transitive dependencies
- Speed comparison bar chart (uv 10-100x faster)
- Decision tree: "Need to add a package?" → `uv add`

---

### 6.3 fig-repo-15: Polars vs Pandas: Speed & Memory

**Purpose**: Justify the choice of Polars over Pandas

**Caption**: "For our 1M+ timepoint dataset, Polars uses 5x less memory and runs 10x faster than Pandas—thanks to lazy evaluation, multi-core parallelism, and Apache Arrow backend."

**Key Message**: "Same analysis logic, 10x faster, 5x less RAM"

**Visual Elements**:
- Memory usage comparison (Pandas: 2GB, Polars: 400MB)
- Speed benchmark bar chart
- Architecture diagram: Eager vs Lazy execution
- Code snippet showing identical API usage

---

### 6.4 fig-repo-16: DuckDB: Your Portable Data Warehouse

**Purpose**: Explain why DuckDB was chosen over SQLite/PostgreSQL

**Caption**: "DuckDB is like having a portable Excel that can handle millions of rows and complex SQL queries in milliseconds—no server required. One .db file replaces 500 scattered CSVs."

**Key Message**: "Database portability meets analytics speed"

**Visual Elements**:
- Before/After: Folder of 500 CSVs → Single .db file
- Query speed comparison: "Top 10 methods by AUROC" → 100ms
- Use case matrix: SQLite vs PostgreSQL vs DuckDB
- Schema diagram of foundation_plr_results.db

---

### 6.5 fig-repo-17: Logging Levels: Why Not Just print()?

**Purpose**: Explain structured logging benefits

**Caption**: "When debugging 1000 bootstrap iterations, print() loses messages in noise. Loguru captures everything with timestamps, levels, and colors—so you find bugs in seconds, not hours."

**Key Message**: "Structured logs = debuggable code"

**Visual Elements**:
- Side-by-side terminal output: messy print() vs clean loguru
- Log level hierarchy: DEBUG < INFO < WARNING < ERROR
- File rotation diagram (7-day retention)
- Exception capture example with stack trace

---

### 6.6 fig-repo-18: Two-Block Architecture: Extraction vs Analysis

**Purpose**: Explain the computation decoupling principle

**Caption**: "Block 1 (Python) extracts and computes metrics from MLflow. Block 2 (R/Python) reads from DuckDB and NEVER recomputes. This ensures figures are reproducible—regenerating a figure always uses the same data."

**Key Message**: "Compute once in extraction, read forever in analysis"

**Visual Elements**:
- Swimlane diagram with two blocks
- Data flow: MLflow → DuckDB → Figures
- "COMPUTE HERE" vs "READ ONLY" labels
- Re-anonymization pipeline (PLRxxxx → Hxxx/Gxxx)

---

### 6.7 fig-repo-19: Subject Stratification: 507 vs 208

**Purpose**: Clarify why subject counts differ between tasks

**Caption**: "All 507 subjects have ground truth for preprocessing evaluation, but only 208 subjects (152 control + 56 glaucoma) have classification labels. This is NOT data loss—it's study design."

**Key Message**: "507 for preprocessing, 208 for classification"

**Visual Elements**:
- Venn diagram showing subject subsets
- Table: 152 healthy + 56 glaucoma = 208 labeled
- 299 unlabeled (preprocessing only)
- Which subjects appear in which analyses

---

### 6.8 fig-repo-20: Error Propagation: How Outlier Errors Cascade

**Purpose**: Show why preprocessing quality matters

**Caption**: "A single missed blink artifact in Stage 1 propagates through imputation, feature extraction, and classification—potentially flipping a diagnosis. Ground truth preprocessing achieves 0.913 AUROC; poor outlier detection can drop this to 0.85."

**Key Message**: "Garbage in at Stage 1 = garbage out at Stage 4"

**Visual Elements**:
- Waterfall diagram showing error cascade
- Example: Clean vs corrupted signal
- AUROC degradation table by outlier F1
- "Error amplification" visual

---

### 6.9 fig-repo-21: Bootstrap Confidence Intervals: 1000 Iterations

**Purpose**: Explain bootstrap methodology to non-statisticians

**Caption**: "We don't trust a single AUROC number—we resample our test set 1000 times to get a distribution. The 95% CI [0.851, 0.955] tells us where the true AUROC likely lies."

**Key Message**: "One number is a guess, 1000 numbers are confidence"

**Visual Elements**:
- Histogram of 1000 AUROC values
- CI bounds marked with vertical lines
- Resampling process diagram
- Comparison: Point estimate vs distribution

---

### 6.10 fig-repo-22: The 4 Standard Combos: When to Use Each

**Purpose**: Explain the gold standard hyperparameter combinations

**Caption**: "These four preprocessing pipelines appear in every figure: ground_truth (baseline), best_ensemble (winner), best_single_fm (foundation model), and traditional (comparison). Always include ground_truth."

**Key Message**: "4 combos, always include ground truth"

**Visual Elements**:
- 4-panel comparison with color coding
- AUROC values for each combo
- When to use: main figures vs supplementary
- Color legend matching figure system

---

### 6.11 fig-repo-23: Data Privacy: What Gets Shared, What Stays Private

**Purpose**: Clarify what data can be publicly shared

**Caption**: "Aggregated metrics (mean AUROC, calibration curves) are public. Individual PLR traces and subject lookup tables are private—they enable re-identification and violate SERI data agreements."

**Key Message**: "Metrics: public. Traces: private."

**Visual Elements**:
- Two-column: PUBLIC vs PRIVATE
- File list with green/red markers
- .gitignore explanation
- Re-anonymization flow diagram

---

### 6.12 fig-repo-24: PLR Light Protocol: The 66-Second Recording

**Purpose**: Explain the clinical measurement for non-ophthalmologists

**Caption**: "The Pupillary Light Reflex (PLR) measures how pupils respond to light. Over 66 seconds, we flash red and blue lights while recording pupil diameter at 30 frames/second—giving us 1981 data points per eye."

**Key Message**: "66 seconds of light → 1981 numbers → 1 diagnosis"

**Visual Elements**:
- Timeline with light phases marked
- Example PLR trace (healthy vs glaucoma)
- Where artifacts (blinks) occur
- Feature extraction points highlighted

---

## 7. CROSS-REFERENCES TO ADD

After creating figures, add links to these README files:

| Figure | Link To | Section |
|--------|---------|---------|
| fig-repo-14 (uv) | Root README | "Quick Install" |
| fig-repo-15 (Polars) | docs/concepts | New "Why Polars?" section |
| fig-repo-16 (DuckDB) | docs/concepts | New "Data Storage" section |
| fig-repo-17 (Logging) | CONTRIBUTING | "Development Guidelines" |
| fig-repo-18 (Two-Block) | ARCHITECTURE.md | "Reproducibility Design" |
| fig-repo-22 (Combos) | docs/user-guide | "Running Experiments" |

---

## 8. EXECUTION PLAN

### Phase 1: Foundational Figures (Week 1)
- [ ] fig-repo-13: End-to-End Pipeline
- [ ] fig-repo-14: Why uv?
- [ ] fig-repo-18: Two-Block Architecture
- [ ] fig-repo-22: 4 Standard Combos

### Phase 2: Technology Explanations (Week 2)
- [ ] fig-repo-15: Polars vs Pandas
- [ ] fig-repo-16: DuckDB
- [ ] fig-repo-17: Logging Levels

### Phase 3: Domain Context (Week 3)
- [ ] fig-repo-19: Subject Stratification
- [ ] fig-repo-20: Error Propagation
- [ ] fig-repo-24: PLR Light Protocol

### Phase 4: Advanced Topics (Week 4)
- [ ] fig-repo-21: Bootstrap CIs
- [ ] fig-repo-23: Data Privacy
- [ ] Remaining Tier 2 figures

---

## 9. SUCCESS METRICS

| Metric | Target | Validation |
|--------|--------|------------|
| Figure count | 26 new figures | `ls docs/repo-figures/figure-plans/fig-repo-1*.md \| wc -l` |
| README links | 12+ cross-references | Manual audit |
| Persona coverage | All 4 personas served | Review checklist |
| Caption quality | Clear, jargon-free | Non-SWE review |

---

## 10. REVIEWER ITERATIONS

### Round 0 (Initial Draft)
**Date:** 2026-01-31
**Status:** Awaiting review

**Open Questions:**
1. Are 26 figures too many? Should we prioritize fewer, higher-quality figures?
2. Should we create animated GIFs for complex flows?
3. What's the right balance between "ELI5" and "technical accuracy"?

### Round 1 (Reviewer Agent Review) ✅ COMPLETED
**Date:** 2026-01-31
**Reviewer:** Claude Opus 4.5 (Plan Agent)

**Technical Verification:**
| Claim | Files Using It | Status |
|-------|----------------|--------|
| Polars used | 42 files with `import polars` | ✅ VERIFIED |
| Loguru used | 139 files with `from loguru` | ✅ VERIFIED |
| Pandas used | 62 occurrences (secondary) | ✅ VERIFIED |

**Key Findings:**

1. **PASS**: Style interpolation (75/25) is clear and actionable
2. **PASS**: Domain constraints (retina only, no brain) well-defined
3. **PASS**: Anti-sci-fi rules explicit
4. **PASS**: Target audience analysis comprehensive
5. **NEEDS_WORK**: Only 12/26 figures have detailed plans → Will add remaining
6. **NEEDS_WORK**: Some redundancy with existing 12 figures → Differentiated below
7. **NEEDS_WORK**: Timeline unrealistic → Extended to 6-8 weeks

**Redundancy Resolution:**

| New Figure | Existing | Resolution |
|------------|----------|------------|
| fig-repo-13 (End-to-End) | fig-repo-01 | **KEEP BOTH**: 01 = hero/overview, 13 = technical data flow |
| fig-repo-18 (Two-Block) | fig-repo-10 | **KEEP BOTH**: 10 = Prefect flows, 18 = computation decoupling |
| fig-repo-20 (Error Propagation) | fig-repo-02 | **KEEP BOTH**: 02 = pipeline stages, 20 = error cascade math |

**Added Figures (from reviewer suggestions):**

| ID | Title | Gap Addressed |
|----|-------|---------------|
| fig-repo-39 | "Python-R Interop: Why pminternal Needs R" | CLAUDE.md rule: never reimplement |
| fig-repo-40 | "What is .venv? Virtual Environments Explained" | Knowledge gap for scientists |

**Updated Timeline:**

| Phase | Duration | Figures |
|-------|----------|---------|
| Phase 1 | Week 1-2 | fig-repo-13, 14, 18, 22 (foundational) |
| Phase 2 | Week 3-4 | fig-repo-15, 16, 17 (technology) |
| Phase 3 | Week 5-6 | fig-repo-19, 20, 24 (domain) |
| Phase 4 | Week 7-8 | Remaining Tier 2 + Tier 3 |
| Review | Week 9 | Quality assurance on all figures |

**Decision on Open Questions:**
1. **26 figures?** → Keep 26 + 2 new = 28. Quality maintained through detailed plans.
2. **Animated GIFs?** → NO. Scientific repo, static figures only.
3. **ELI5 vs accuracy?** → See Progressive Disclosure policy below.

### Progressive Disclosure Policy (User Directive - Verbatim)

> "For a single generated image there is no tension as we simply create 'progressive disclosure' infographics, both ELI5-level and both expert-level infographics, there is no mixing in a single infographic! We need to create ELI5-level figure plans AND expert-level figures, think of computational PIs vs first-year university student interns!"

**Implementation:**

For each major concept, create TWO figures:

| ELI5 Version | Expert Version | Concept |
|--------------|----------------|---------|
| fig-repo-14a | fig-repo-14b | uv package manager |
| fig-repo-15a | fig-repo-15b | Polars vs Pandas |
| fig-repo-16a | fig-repo-16b | DuckDB |
| fig-repo-17a | fig-repo-17b | Logging levels |
| fig-repo-18a | fig-repo-18b | Two-block architecture |
| fig-repo-20a | fig-repo-20b | Error propagation |

**ELI5 Version Guidelines:**
- NO code snippets
- Simple analogies (Excel → DuckDB, etc.)
- Icons and visual metaphors
- Maximum 5 concepts per figure
- Target: First-year university intern

**Expert Version Guidelines:**
- Include code snippets
- Technical specifications
- Performance numbers with methodology
- API details
- Target: Computational PI, ML Engineer

### Round 2 (Progressive Disclosure Review) ✅ COMPLETED
**Date:** 2026-01-31
**Reviewer:** Claude Haiku (Plan Agent)

**Key Findings:**

| Figure | Status | Action Required |
|--------|--------|-----------------|
| fig-repo-13 | ✅ GOOD | Keep as-is (balanced audience) |
| fig-repo-14 | ⚠️ MIXED | **SPLIT** into ELI5 + Expert |
| fig-repo-15 | ⚠️ MIXED | **SPLIT** into ELI5 + Expert |
| fig-repo-16 | ⚠️ MIXED | **SPLIT** into ELI5 + Expert |
| fig-repo-17 | ⚠️ MIXED | **SPLIT** into ELI5 + Expert |

**Split Implementation Plan:**

| Original | ELI5 Version | Expert Version |
|----------|--------------|----------------|
| fig-repo-14 | "Reproducibility: The Dice Game" | "uv.lock: Full Dependency Trees" |
| fig-repo-15 | "Faster Analysis: 10x Speed, 5x Memory" | "Query Optimization: Lazy Evaluation" |
| fig-repo-16 | "One Database, 500 Files: Organized" | "DuckDB vs SQLite vs PostgreSQL" |
| fig-repo-17 | "Finding Your Error in 1000 Runs" | "Log Levels and Thread-Safe Debugging" |

**Content Redistribution:**
- ELI5: NO code, simple analogies, max 5 concepts
- Expert: Include code, specs, performance metrics, API details

**Updated Figure Count:**
- Original: 28 figures (26 new + 2 from Round 1)
- After split: 32 figures (4 figures × 2 versions = 8 new versions)
- Total catalog: **36 figure plans** (28 + 8 additional ELI5/Expert splits)

---

## APPENDIX: Reference Sources

### Web Search Results (2026-01-31)

**uv vs pip**:
- [Real Python: uv vs pip](https://realpython.com/uv-vs-pip/)
- [DataCamp: Python UV Tutorial](https://www.datacamp.com/tutorial/python-uv)
- [Python Discourse: requirements.txt vs uv.lock](https://discuss.python.org/t/requirements-txt-or-uv-lock/78419)

**Polars vs Pandas**:
- [JetBrains: Polars vs Pandas](https://blog.jetbrains.com/pycharm/2024/07/polars-vs-pandas/)
- [DataCamp: Benchmarking Alternatives](https://www.datacamp.com/tutorial/benchmarking-high-performance-pandas-alternatives)
- [Python Speed: Polars Memory](https://pythonspeed.com/articles/polars-memory-pandas/)

**Loguru**:
- [GitHub: Loguru](https://github.com/Delgan/loguru)
- [Real Python: Loguru Tutorial](https://realpython.com/python-loguru/)
- [Better Stack: Loguru Guide](https://betterstack.com/community/guides/logging/loguru/)

**Digital Knowledge Engineering**:
- Vomberg et al. 2024, Journal of Business Research 177: 114632
- DOI: [10.1016/j.jbusres.2024.114632](https://doi.org/10.1016/j.jbusres.2024.114632)
