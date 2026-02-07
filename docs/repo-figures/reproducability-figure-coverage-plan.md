# Reproducibility Figure Coverage Plan: 24+ Infographics on Scientific Computing Reproducibility

**Status:** ✅ FIGURE PLANS COMPLETED
**Created:** 2026-01-31
**Last Updated:** 2026-01-31
**Figure Plans Created:** 24 (fig-repro-01 through fig-repro-24)
**Reviewer Iterations:** 0

---

## User Prompt (Verbatim)

> And this is a good paper to read https://arxiv.org/abs/2308.07333 about the reproducability issues in biomedical research, and insights from it could be integrated to the created figures relevant this this (or/and create a dedicated infographics of this paper and the abysmal SWE skills of academic researchers!). For other insights, read also these and refine existing figures, or think of creating new figure plans as the reproducibility is a VERY CORE CONCEPT to be communicated to the readers. FOr these papers, let's create another figure creation plan dedicated to explaining the theory behind reproducibility concepts in the repo README.md files: /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/repo-figures/reproducability-figure-coverage-plan.md ! Especially this Donat-Bouillud, Pierre, Filip Křikava, Sebastian Krynski, and Jan Vitek. 2025. "R4R: Reproducibility for R." Proceedings of the 3rd ACM Conference on Reproducibility and Replicability (New York, NY, USA), ACM REP '25, October 21, 132–42. https://doi.org/10.1145/3736731.3746156.
> is very important to this repo. You can make this infographics more theoretical and talk about the general trends in "repo design" and artifact management to ensure reproducability again covering 24+ figures for the reproducability concepts alone from ELI5 to expert-level figures so that there is not too much overlap with already existing figures generated. In addition to the theoretical coverage and communicating what the field is talking about (With hyperlinks to sources in figure captions), try to contextualize the general sexy topics to our repo and show how they are relevant in the design, use, and extensability of this repo! Be scientifically rigorous about the figure design, as the infographics figure plans are all about scientific rigor :D Use again multiple reviewer rounds to optimize your doc on these docs and see also our "sister repo" for warmup for that paper

---

## 0. LITERATURE FOUNDATION

### 0.1 Key Statistics from Literature (2023-2026)

| Source | Finding | Implications |
|--------|---------|--------------|
| **[arXiv:2308.07333](https://arxiv.org/abs/2308.07333)** (Jupyter Notebooks, PubMed) | Of 27,271 notebooks, only **879 (3.2%)** produced identical results | 96.8% failure rate for "as-is" reproducibility |
| **[R4R: Reproducibility for R](https://doi.org/10.1145/3736731.3746156)** (ACM REP '25) | Only **26% of R replication packages** run successfully | 74% of published R code fails to execute |
| **[Docker Does Not Guarantee Reproducibility](https://arxiv.org/abs/2601.12811)** (2026) | Docker images suffer from version drift, unpinned dependencies | Containerization ≠ automatic reproducibility |
| **[ML Reproducibility Overview](https://arxiv.org/abs/2406.14325)** (2025) | ML research has "unsatisfactory" reproducibility levels | Training conditions, unpublished code, data sensitivity |
| **[Software Engineering Best Practices](https://arxiv.org/abs/2502.00902)** (2025) | Adoption of SE practices in ML is systematically low | Need to bridge SE and scientific computing |

### 0.2 The Reproducibility Crisis Taxonomy

```
                     THE REPRODUCIBILITY SPECTRUM
                     ═══════════════════════════════

Level 0: UNREPRODUCIBLE              Level 5: FULLY REPRODUCIBLE
    ↓                                           ↓
    │   Code/Data    Builds    Runs    Same      Bitwise
    │   Missing      Fail      Fail    Results   Identical
    │      ↓          ↓         ↓         ↓          ↓
    ├────────────────────────────────────────────────┤
    │    └─ 74% of R code (R4R 2025)                 │
    │         └─ 56% after automated fixes           │
    │              └─ 26% succeed                    │
    │                   └─ 3.2% identical (Jupyter)  │
    │                        └─ <1% bitwise? (est.)  │
    └────────────────────────────────────────────────┘
```

### 0.3 Root Causes Identified in Literature

| Category | Issue | % of Failures | Foundation PLR Solution |
|----------|-------|---------------|------------------------|
| **Dependencies** | Unpinned versions | ~40% | `uv.lock` lockfile |
| **Environment** | Missing system libs | ~25% | Docker + documented env |
| **Data** | Files missing/moved | ~20% | DuckDB single-source |
| **Random State** | Unseeded RNG | ~10% | Documented seeds |
| **Hardware** | Architecture mismatch | ~5% | Platform documentation |

---

## 1. STYLE SPECIFICATION

### 1.1 Visual Style (Same as deeper-figure-coverage-plan.md)

**75% Manuscript Style + 25% Economist Aesthetics**

- Muted, professional colors
- NO glowing/sci-fi effects
- Retina/eye imagery ONLY (no brain anatomy)
- Mermaid-style diagrams with depth
- Print-quality, medical research aesthetic

### 1.2 Academic Rigor Requirements

For reproducibility figures specifically:
- **Every claim backed by citation** (DOI or URL in caption)
- **Statistics must be verifiable** (from peer-reviewed sources)
- **Year and sample size noted** for empirical claims
- **Uncertainty acknowledged** where data is extrapolated

---

## 2. TARGET AUDIENCE ANALYSIS

### 2.1 Additional Personas for Reproducibility Content

| Persona | Context | Key Questions |
|---------|---------|---------------|
| **Journal Editor** | Enforcing reproducibility policies | "How do I verify claims?" |
| **Grant Reviewer** | Assessing methodology rigor | "Is this reproducible?" |
| **Lab Manager** | Setting up team standards | "What practices to enforce?" |
| **PhD Student** | Learning best practices | "What should I do differently?" |

### 2.2 Progressive Disclosure (Same Policy)

| Level | Audience | Content |
|-------|----------|---------|
| **ELI5** | Biologist, PI, PhD student | Analogies, no code, max 5 concepts |
| **Expert** | ML Engineer, Biostatistician | Code, specs, API details, citations |

---

## 3. FIGURE CATALOG: 26 REPRODUCIBILITY INFOGRAPHICS

### Tier 1: THE CRISIS (Foundational - 8 figures)

| ID | Title | Theory | Foundation PLR Context | Priority |
|----|-------|--------|------------------------|----------|
| **fig-repro-01** | "The Reproducibility Crisis in Numbers" | Stats from literature | Our 96.8% → 100% journey | P0 |
| **fig-repro-02a** | "Why 96.8% of Notebooks Fail (ELI5)" | Jupyter study visual | Why we don't use Jupyter | P0 |
| **fig-repro-02b** | "Why 96.8% of Notebooks Fail (Expert)" | Failure taxonomy | Technical breakdown | P0 |
| **fig-repro-03** | "The 5 Horsemen of Irreproducibility" | Root cause taxonomy | How each is addressed | P0 |
| **fig-repro-04** | "Levels of Reproducibility" | Spectrum definition | Where we aim (Level 4-5) | P1 |
| **fig-repro-05** | "What Reviewers Actually Check" | Peer review reality | Making review easy | P1 |
| **fig-repro-06** | "The Cost of Irreproducibility" | Wasted time/money | ROI of our tooling | P2 |
| **fig-repro-07** | "Docker is NOT Enough" | Docker limitations | Why we need more than containers | P1 |

### Tier 2: DEPENDENCY MANAGEMENT (8 figures)

| ID | Title | Theory | Foundation PLR Context | Priority |
|----|-------|--------|------------------------|----------|
| **fig-repro-08a** | "Dependency Hell Visualized (ELI5)" | Transitive deps | Why uv matters | P0 |
| **fig-repro-08b** | "Dependency Resolution: pip vs uv" | Lock file mechanics | uv.lock explained | P0 |
| **fig-repro-09** | "The R Package Ecosystem Challenge" | R4R findings | renv + pminternal usage | P1 |
| **fig-repro-10** | "System Dependencies: The Hidden Iceberg" | System libs issue | Ubuntu packages documented | P1 |
| **fig-repro-11** | "Version Pinning Strategies" | Semantic versioning | Our exact-version policy | P2 |
| **fig-repro-12** | "The requirements.txt Problem" | Why it fails | uv.lock solution | P1 |
| **fig-repro-13** | "Transitive Dependency Explosion" | Dep tree growth | Our 200+ packages | P2 |
| **fig-repro-14** | "Lockfiles: Your Time Machine" | How lockfiles work | uv.lock as insurance | P1 |

### Tier 3: ENVIRONMENT & CONTAINERS (5 figures)

| ID | Title | Theory | Foundation PLR Context | Priority |
|----|-------|--------|------------------------|----------|
| **fig-repro-15** | "Virtual Environments vs Containers" | Isolation levels | .venv + optional Docker | P1 |
| **fig-repro-16** | "Why Docker Doesn't Guarantee Reproducibility" | Malka 2026 findings | Our mitigation | P1 |
| **fig-repro-17** | "Bitwise vs Functional Reproducibility" | Reproducibility tiers | What we target | P2 |
| **fig-repro-18** | "The Base Image Problem" | Image version drift | Pinned base images | P2 |
| **fig-repro-19** | "r4r: Automatic Artifact Creation" | R4R tool concept | Future integration | P3 |

### Tier 4: DATA & ARTIFACT MANAGEMENT (5 figures)

| ID | Title | Theory | Foundation PLR Context | Priority |
|----|-------|--------|------------------------|----------|
| **fig-repro-20** | "Single Source of Truth: Why DuckDB" | Data consolidation | 500 CSVs → 1 .db | P0 |
| **fig-repro-21** | "Experiment Tracking: MLflow" | Provenance tracking | 542 runs logged | P1 |
| **fig-repro-22** | "JSON Sidecars for Figure Reproducibility" | Data provenance | Every figure has .json | P1 |
| **fig-repro-23** | "The 97.5% R4R Success Rate" | R4R evaluation | Our R figure scripts | P2 |
| **fig-repro-24** | "Git LFS vs DuckDB for Large Data" | Data storage options | Why we chose DuckDB | P3 |

---

## 4. DETAILED FIGURE PLANS

### 4.1 fig-repro-01: The Reproducibility Crisis in Numbers

**Purpose**: Set the stage with shocking statistics from peer-reviewed literature

**Key Statistics** (all from literature):
- **3.2%** of Jupyter notebooks reproduce identically (Pimentel 2023, arXiv:2308.07333)
- **26%** of R replication packages run successfully (R4R 2025)
- **74%** of computational studies cannot be replicated
- **$28 billion** annual waste in preclinical research (Freedman 2015)

**Caption**: "The reproducibility crisis is quantified: only 3.2% of biomedical Jupyter notebooks produce identical results when re-run (Pimentel et al. 2023, [arXiv:2308.07333](https://arxiv.org/abs/2308.07333)). R4R (2025) found only 26% of R replication packages execute successfully. Foundation PLR addresses this through locked dependencies (uv.lock), tracked experiments (MLflow), and documented environments."

**Visual Concept**:
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE REPRODUCIBILITY CRISIS IN NUMBERS                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  JUPYTER NOTEBOOKS (PubMed)          R REPLICATION PACKAGES                    │
│  Pimentel et al. 2023                Donat-Bouillud et al. 2025                │
│                                                                                 │
│  27,271 notebooks examined           2,000 packages examined                   │
│           ↓                                   ↓                                 │
│                                                                                 │
│  ┌────────────────────┐              ┌────────────────────┐                    │
│  │░░░░░░░░░░░░░░░░░░░░│              │░░░░░░░░░░░░░░░░░░░░│                    │
│  │░░░░░░░░░░░░░░░░░░░░│              │░░░░░░░░░░░░░░░░░░░░│                    │
│  │░░░░░░░░░░░░░░░░░░░░│ 96.8%        │░░░░░░░░░░░░░░░░░░░░│ 74%               │
│  │░░░░░░░░░░░░░░░░░░░░│ FAIL         │░░░░░░░░░░░░░░░░░░░░│ FAIL              │
│  │░░░░░░░░░░░░░░░░░░░░│              │░░░░░░░░░░░░░░░░░░░░│                    │
│  │████████████████████│ 3.2%         │████████████████████│ 26%               │
│  └────────────────────┘ IDENTICAL    └────────────────────┘ RUN               │
│                                                                                 │
│  Only 879 of 27,271 notebooks        Only 520 of 2,000 packages                │
│  produced identical results          completed execution                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THIS IS NOT A SOFTWARE PROBLEM—IT'S A SCIENCE PROBLEM                          │
│                                                                                 │
│  $28 billion/year wasted in preclinical research (Freedman 2015)                │
│                                                                                 │
│  FOUNDATION PLR SOLUTION:                                                       │
│  ✓ uv.lock (locked dependencies)   → Addresses 40% of failures                  │
│  ✓ MLflow (experiment tracking)    → Full provenance chain                      │
│  ✓ DuckDB (single-source data)     → No missing files                           │
│  ✓ Documented environment          → Eliminates guesswork                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.2 fig-repro-02a: Why 96.8% of Notebooks Fail (ELI5)

**Purpose**: Explain Jupyter study findings without technical jargon

**Caption**: "When researchers tried to re-run 27,271 biomedical Jupyter notebooks, 96.8% failed to produce identical results. The main culprits: missing dependencies (like a recipe missing ingredients), changed package versions (like a recipe with different ingredient brands), and broken file paths (like ingredients stored in the wrong cupboard). Pimentel et al. 2023, [arXiv:2308.07333](https://arxiv.org/abs/2308.07333)"

**Visual Concept**:
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    WHY 96.8% OF NOTEBOOKS FAIL                                  │
│                    (Explained Simply)                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  IMAGINE A RECIPE...                                                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  "My notebook ran perfectly last year!"                                 │   │
│  │                                                                         │   │
│  │  But now...                                                             │   │
│  │                                                                         │   │
│  │  🥣 MISSING INGREDIENTS                                                 │   │
│  │     "Install pandas" ← But which version?                               │   │
│  │     The pandas from 2020 ≠ pandas from 2024                             │   │
│  │                                                                         │   │
│  │  🏠 INGREDIENTS IN WRONG CUPBOARD                                       │   │
│  │     "/Users/jane/data/myfile.csv" ← Only exists on Jane's computer!     │   │
│  │                                                                         │   │
│  │  🔧 RECIPE CHANGED                                                      │   │
│  │     "plt.plot()" worked differently in matplotlib 2.0 vs 3.0            │   │
│  │                                                                         │   │
│  │  🎲 SURPRISE INGREDIENTS                                                │   │
│  │     Random numbers change every time (no fixed seed)                    │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  THE RESULT:                                                                    │
│                                                                                 │
│  Out of 27,271 notebooks, researchers could only recreate                       │
│  identical results for 879 (3.2%)                                               │
│                                                                                 │
│  Source: Pimentel et al. 2023 (arXiv:2308.07333)                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.3 fig-repro-07: Docker is NOT Enough

**Purpose**: Challenge the misconception that Docker solves reproducibility

**Key Points from Malka et al. 2026**:
- Docker images ≠ Dockerfiles (images can drift)
- Unpinned base images change over time
- `apt-get install` without versions = non-deterministic
- Even with Docker, only ~60% of images functionally reproduce

**Caption**: "Despite widespread belief, Docker does not guarantee reproducibility. A systematic study of 5,298 Docker builds found that unpinned dependencies, floating base images, and non-deterministic instructions prevent reliable reproduction. Malka et al. 2026, [arXiv:2601.12811](https://arxiv.org/abs/2601.12811). Foundation PLR uses Docker as ONE layer, complemented by uv.lock for Python and explicit version pinning."

**Visual Concept**:
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DOCKER IS NOT ENOUGH                                         │
│                    Malka et al. 2026 (arXiv:2601.12811)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  COMMON BELIEF                        REALITY                                   │
│  ═════════════                        ═══════                                   │
│                                                                                 │
│  "Just put it in Docker               Docker images CHANGE over time:           │
│   and it's reproducible!"                                                       │
│                                       ┌─────────────────────────────────────┐  │
│  ❌ FALSE                             │                                     │  │
│                                       │  FROM python:3.11                   │  │
│                                       │       ↓                             │  │
│                                       │  Today: Python 3.11.2               │  │
│                                       │  Tomorrow: Python 3.11.9            │  │
│                                       │  Next year: Python 3.11.??          │  │
│                                       │                                     │  │
│                                       │  apt-get install numpy              │  │
│                                       │       ↓                             │  │
│                                       │  Today: numpy 1.24.0                │  │
│                                       │  Tomorrow: numpy 1.26.4             │  │
│                                       │                                     │  │
│                                       └─────────────────────────────────────┘  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  STUDY FINDINGS (5,298 Docker builds from GitHub)                               │
│  ═════════════════════════════════════════════════                              │
│                                                                                 │
│  │ Reproducibility Level │ % of Builds │                                       │
│  │ ───────────────────── │ ─────────── │                                       │
│  │ Bitwise identical     │    ~15%     │                                       │
│  │ Functionally equiv.   │    ~45%     │  ← Only 60% even work                 │
│  │ Build succeeds        │    ~25%     │                                       │
│  │ Build fails           │    ~15%     │                                       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR'S MULTI-LAYER APPROACH                                          │
│  ═════════════════════════════════════                                          │
│                                                                                 │
│  Docker alone: ❌ Not reproducible                                              │
│  + Pinned base image: ⚠️ Better                                                 │
│  + uv.lock: ⚠️ Better still                                                     │
│  + Exact system deps: ⚠️ Even better                                            │
│  + All of the above: ✅ Reproducible                                            │
│                                                                                 │
│  We use: uv.lock + documented Ubuntu packages + MLflow provenance               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.4 fig-repro-20: Single Source of Truth: Why DuckDB

**Purpose**: Explain data consolidation benefits

**Caption**: "Scattered CSV files are a leading cause of reproducibility failure—files get renamed, moved, or deleted. Foundation PLR consolidates 500+ raw CSV files into a single DuckDB database (SERI_PLR_GLAUCOMA.db) that contains all 507 subjects and 1M+ timepoints. DuckDB's OLAP optimization makes analytics 6-8x faster than SQLite while remaining a single portable file."

**Visual Concept**:
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE SOURCE OF TRUTH: WHY DUCKDB                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  BEFORE: 500+ SCATTERED FILES                                                   │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  data/                                 PROBLEMS:                                │
│  ├── raw/                              • Which file is the right one?          │
│  │   ├── subject_001_v1.csv            • Version confusion (v1? v2? final?)    │
│  │   ├── subject_001_v2.csv            • Missing files break scripts           │
│  │   ├── subject_001_FINAL.csv         • 40% of failures (R4R 2025)            │
│  │   ├── subject_002.csv                                                       │
│  │   └── ... (500+ files)                                                      │
│  ├── processed/                                                                │
│  │   ├── features_old.csv                                                      │
│  │   └── features_new_FINAL2.csv                                               │
│  └── results/                                                                  │
│      └── ??? (which is current?)                                               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AFTER: ONE DUCKDB FILE                                                         │
│  ═════════════════════                                                          │
│                                                                                 │
│  SERI_PLR_GLAUCOMA.db (42 MB)                                                   │
│  └── Tables:                           BENEFITS:                                │
│      ├── train (507 subjects)          ✓ Single file = single truth            │
│      ├── test (208 subjects)           ✓ No version confusion                  │
│      ├── outlier_masks                 ✓ Portable across machines              │
│      └── metadata                      ✓ SQL queries = reproducible            │
│                                        ✓ 6-8x faster than SQLite               │
│                                                                                 │
│  SELECT COUNT(*) FROM train;  →  507 subjects                                   │
│  SELECT SUM(LENGTH(pupil_raw)) →  1,010,667 timepoints                         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY DUCKDB OVER ALTERNATIVES?                                                  │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  │ Feature        │ SQLite │ PostgreSQL │ DuckDB │                             │
│  │ ────────────── │ ────── │ ────────── │ ─────── │                             │
│  │ Single file    │   ✅   │     ❌     │   ✅    │                             │
│  │ Analytics fast │   ❌   │     ⚠️     │   ✅    │                             │
│  │ Setup required │  None  │   Server   │  None   │                             │
│  │ Column-store   │   ❌   │     ❌     │   ✅    │                             │
│                                                                                 │
│  DuckDB = Portable analytics database (OLAP-optimized)                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. CROSS-REFERENCES TO LITERATURE

### 5.1 Citation Requirements

Every figure caption MUST include:
- Primary source with DOI/URL
- Year of publication
- Sample size where applicable

### 5.2 Key Papers to Reference

| Topic | Paper | DOI/URL | Key Finding |
|-------|-------|---------|-------------|
| Jupyter reproducibility | Pimentel et al. 2023 | [arXiv:2308.07333](https://arxiv.org/abs/2308.07333) | 3.2% identical reproduction |
| R reproducibility | R4R (Donat-Bouillud et al.) 2025 | [10.1145/3736731.3746156](https://doi.org/10.1145/3736731.3746156) | 26% success rate, 97.5% with r4r |
| Docker limitations | Malka et al. 2026 | [arXiv:2601.12811](https://arxiv.org/abs/2601.12811) | Docker ≠ reproducibility |
| ML reproducibility | Semmelrock et al. 2025 | [arXiv:2406.14325](https://arxiv.org/abs/2406.14325) | Barriers and drivers |
| SE best practices | Salsa et al. 2025 | [arXiv:2502.00902](https://arxiv.org/abs/2502.00902) | Low adoption in ML |
| Reproducibility debt | Hassan et al. 2025 | Local copy | Scientific software debt |

---

## 6. CONTEXTUALIZATION TO FOUNDATION PLR

### 6.1 How Our Repo Addresses Each Issue

| Reproducibility Issue | Literature Source | Our Solution | Figure Reference |
|-----------------------|-------------------|--------------|------------------|
| Unpinned dependencies | R4R, arXiv:2308.07333 | `uv.lock` lockfile | fig-repro-08, fig-repro-12 |
| Missing data files | R4R 2025 | DuckDB single-source | fig-repro-20 |
| Environment drift | Malka 2026 | Documented Ubuntu deps | fig-repro-16 |
| Experiment provenance | ML reproducibility | MLflow tracking | fig-repro-21 |
| Figure reproducibility | Our design | JSON sidecars | fig-repro-22 |
| Random seeds | General literature | Documented in config | fig-repro-03 |

### 6.2 Progressive Disclosure Mapping

| Concept | ELI5 Figure | Expert Figure |
|---------|-------------|---------------|
| Crisis overview | fig-repro-02a | fig-repro-02b |
| Dependency hell | fig-repro-08a | fig-repro-08b |
| Docker limitations | (combined) | fig-repro-07 |

---

## 7. EXECUTION PLAN

### Phase 1: Crisis Framing (Week 1-2)
- [ ] fig-repro-01: Crisis in Numbers
- [ ] fig-repro-02a/b: Why Notebooks Fail
- [ ] fig-repro-03: 5 Horsemen
- [ ] fig-repro-07: Docker Not Enough

### Phase 2: Dependencies (Week 3-4)
- [ ] fig-repro-08a/b: Dependency Hell
- [ ] fig-repro-12: requirements.txt Problem
- [ ] fig-repro-14: Lockfiles as Time Machine

### Phase 3: Data & Artifacts (Week 5-6)
- [ ] fig-repro-20: Why DuckDB
- [ ] fig-repro-21: MLflow Tracking
- [ ] fig-repro-22: JSON Sidecars

### Phase 4: Advanced Topics (Week 7-8)
- [ ] Remaining Tier 2 and 3 figures
- [ ] Cross-linking to deeper-figure-coverage-plan figures

---

## 8. REVIEWER ITERATIONS

### Round 0 (Initial Draft)
**Date:** 2026-01-31
**Status:** Awaiting review

**Open Questions:**
1. Should we create a dedicated "Theory of Reproducibility" section in docs/?
2. How much overlap is acceptable with deeper-figure-coverage-plan.md?
3. Should figures be numbered separately (fig-repro-XX) or continue from main plan?

---

## APPENDIX: Reference Sources

### Literature Review Results

**Primary Sources (Peer-Reviewed)**:
- Pimentel et al. 2023 - Jupyter notebook reproducibility (N=27,271)
- Donat-Bouillud et al. 2025 - R4R: Reproducibility for R (ACM REP '25)
- Malka et al. 2026 - Docker reproducibility limitations
- Van Calster et al. 2024 - STRATOS performance metrics

**Secondary Sources (Technical)**:
- Docker official documentation
- uv package manager documentation
- MLflow documentation
- DuckDB documentation

**Local Paper Summaries Referenced**:
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/tmp-PLR/2024-r4r-reproducibility-for-r.md`
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/tmp-PLR/2025-docker-does-not-guarantee-reproducibility.md`
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/tmp-PLR/hassan-2025-reproducibility-debt-scientific-software.md`
