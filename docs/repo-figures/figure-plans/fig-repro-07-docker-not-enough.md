# fig-repro-07: Docker is NOT Enough

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-07 |
| **Title** | Docker is NOT Enough |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | ML Engineer, DevOps, Biostatistician |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Challenge the widespread misconception that containerization automatically solves reproducibility, using evidence from Malka et al. 2026 and explaining the nuances of Docker's failure modes.

## Key Message

"'Just put it in Docker' is not a solution. Only ~60% of Docker builds produce functionally equivalent results due to unpinned dependencies, floating base images, and non-deterministic instructions. Container registries help, but they're not a silver bullet."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Malka et al. 2026 | Docker images ≠ reproducibility; systematic study of 5,298 builds | [arXiv:2601.12811](https://arxiv.org/abs/2601.12811) |
| Zacchiroli 2021 | Randomness in neural network training from tooling | [arXiv:2106.11872](https://arxiv.org/abs/2106.11872) |
| Testing Research Software Survey 2025 | Research developers unfamiliar with testing tools | [arXiv:2501.17739](https://arxiv.org/abs/2501.17739) |
| CORE-Bench 2024 | Computational reproducibility benchmark | [arXiv:2409.11363](https://arxiv.org/html/2409.11363v1) |
| Boettiger 2015 | Introduction to Docker for reproducible research | [arXiv:1410.0846](https://arxiv.org/abs/1410.0846) |

## Visual Concept

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
│                                       │  Jan 2024: Python 3.11.2            │  │
│                                       │  Jan 2025: Python 3.11.9            │  │
│                                       │  Jan 2026: Python 3.11.??           │  │
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
│  ═══════════════════════════════════════════════                                │
│                                                                                 │
│  │ Reproducibility Level     │ % of Builds │ Cumulative │                      │
│  │ ─────────────────────────  │ ─────────── │ ────────── │                      │
│  │ Bitwise identical         │    ~15%     │    15%     │                      │
│  │ Functionally equivalent   │    ~45%     │    60%     │ ← Only 60% work!     │
│  │ Build succeeds (diff out) │    ~25%     │    85%     │                      │
│  │ Build fails               │    ~15%     │   100%     │                      │
│                                                                                 │
│  WHY DOCKER FAILS:                                                              │
│  • FROM python:3.11 → Version changes over time (rolling tags)                  │
│  • apt-get install → Gets latest, not specific version                          │
│  • pip install → Without lockfile, versions vary                                │
│  • COPY . . → Depends on local file state                                       │
│  • RUN commands with network → External resources change                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  "BUT WHAT ABOUT CONTAINER REGISTRIES?"                                         │
│  ══════════════════════════════════════                                         │
│                                                                                 │
│  ✅ YES: Push built image to registry = immutable artifact                      │
│  ✅ YES: Tag images with versions (myapp:v1.2.3) not :latest                    │
│  ✅ YES: Use digest pinning (sha256:abc123...) for guaranteed immutability      │
│                                                                                 │
│  ❌ BUT: Who maintains the registry? (cost, security, retention policies)       │
│  ❌ BUT: Image bloat (1-5GB per image × versions × projects)                    │
│  ❌ BUT: Security vulnerabilities accumulate in frozen images                   │
│  ❌ BUT: Base image CVEs require rebuilding (breaking "reproducibility")        │
│  ❌ BUT: Most researchers don't use registries properly                         │
│                                                                                 │
│  THE PRODUCTION FANTASY:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │   main branch → prod registry → production environment                  │   │
│  │   staging    → staging registry → QA environment                        │   │
│  │   dev branch → dev registry → development sandbox                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Reality for research code: No branches, no CI/CD, no registries,               │
│  just "docker build" on laptop and hope it works on the cluster.                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CROSS-PLATFORM GOTCHAS                                                         │
│  ══════════════════════                                                         │
│                                                                                 │
│  Build on Ubuntu 22.04          Run on Windows (WSL2)                           │
│  ─────────────────────          ────────────────────                            │
│  • Native Linux kernel          • Emulated Linux in Hyper-V                     │
│  • glibc 2.35                   • Different glibc in WSL2 distro                │
│  • ext4 filesystem              • 9P filesystem (performance hit)               │
│  • Normal permissions           • UID/GID mapping headaches                     │
│                                                                                 │
│  Build on x86_64 Linux          Run on Apple Silicon (M1/M2/M3)                 │
│  ──────────────────────         ─────────────────────────────────               │
│  • AMD64 architecture           • ARM64 via Rosetta emulation                   │
│  • Native speed                 • 2-10x slower for compute                      │
│  • All packages work            • Some packages lack ARM builds                 │
│                                                                                 │
│  "Containers share the host kernel" - you cannot run a linux/amd64              │
│  container natively on an arm64 host without emulation overhead.                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  TESTING IN SCIENTIFIC CODE: THE ELEPHANT IN THE ROOM                           │
│  ════════════════════════════════════════════════════                           │
│                                                                                 │
│  Testing Research Software Survey (arXiv:2501.17739):                           │
│                                                                                 │
│  "Research software developers employ widely divergent testing approaches,      │
│   suggesting NO STANDARDIZED METHODOLOGY exists across the research community." │
│                                                                                 │
│  Key findings:                                                                  │
│  • 68% of sampled DL projects have NO unit tests at all                         │
│  • Researchers unfamiliar with existing testing tools                           │
│  • Test case design is a major struggle                                         │
│  • "Evaluating correctness of test outputs" is difficult                        │
│                                                                                 │
│  Why this matters for Docker:                                                   │
│  Without tests, you don't know if your container ACTUALLY works!                │
│  "Build succeeded" ≠ "Code produces correct results"                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR'S MULTI-LAYER APPROACH                                          │
│  ═════════════════════════════════════                                          │
│                                                                                 │
│  Layer 1: Python Dependencies                                                   │
│           uv.lock → EXACT versions, every package, SHA-256 hashes               │
│                                                                                 │
│  Layer 2: System Dependencies                                                   │
│           docs/environment.md → Ubuntu packages with versions                   │
│                                                                                 │
│  Layer 3: Data                                                                  │
│           DuckDB → Single source, portable, versioned                           │
│                                                                                 │
│  Layer 4: Experiment State                                                      │
│           MLflow → Tracks every run with parameters and artifacts               │
│                                                                                 │
│  Layer 5: Tests                                                                 │
│           pytest → Verifies code actually produces correct output               │
│                                                                                 │
│  Result: Docker is ONE layer, not the whole solution                            │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ Docker alone:       ❌ ~60% reproducible (Malka et al.)               │    │
│  │ + Pinned base:      ⚠️ Better but still vulnerable to apt             │    │
│  │ + uv.lock:          ⚠️ Better still, Python deps locked               │    │
│  │ + System deps:      ⚠️ Even better                                    │    │
│  │ + Tests:            ✅ Now you KNOW it works                          │    │
│  │ + ALL of the above: ✅ Actually reproducible                          │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Deep Dive: Why Registries Don't Solve Everything

### The Ideal DevOps Workflow (Industry)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Developer   │     │   CI/CD      │     │  Registry    │
│  commits to  │ ──▶ │   Pipeline   │ ──▶ │  Stores      │
│  feature/x   │     │   builds     │     │  image:sha   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                    │
                            ▼                    ▼
                     Tests pass?           Deploy to:
                     Lint pass?            - dev (on PR)
                     Security scan?        - staging (on merge)
                                          - prod (on tag)
```

**What this requires:**
- CI/CD infrastructure (GitHub Actions, GitLab CI, Jenkins)
- Container registry (ECR, GCR, Docker Hub, Harbor)
- Proper tagging strategy (semver + git SHA)
- Retention policies and cost management
- Security scanning and CVE patching workflow

### Reality for Research Code

```
┌──────────────┐     ┌──────────────┐
│  Researcher  │     │   Laptop     │
│  writes      │ ──▶ │   docker     │ ──▶ "Hope it works on the cluster"
│  code        │     │   build      │
└──────────────┘     └──────────────┘
```

**What researchers typically have:**
- One branch (main or master)
- No CI/CD (maybe a broken GitHub Actions workflow)
- No registry (images built locally, maybe pushed to Docker Hub :latest)
- No tests (or broken tests that nobody runs)
- No versioning strategy

### The Notebook Problem

Most research computing happens in Jupyter notebooks:

> "Just 4% of notebooks on GitHub reproduce."
> — [marimo blog](https://marimo.io/blog/slac-marimo)

Notebooks + Docker = Double trouble:
- Notebooks have hidden state (cell execution order matters)
- `requirements.txt` from notebooks often incomplete
- Docker images with Jupyter add another layer of complexity

## Testing: The Missing Layer

### Test Pyramid for Research Software

```
                    ┌─────────────────────┐
                    │   End-to-End Tests  │  ← "Does the whole pipeline
                    │   (rare in research) │    produce correct output?"
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
               │     Integration Tests          │  ← "Do components work
               │     (occasionally present)     │    together?"
               └───────────────┬───────────────┘
                               │
    ┌──────────────────────────┴──────────────────────────┐
    │                    Unit Tests                        │  ← 68% of DL projects
    │              (often missing in research)             │    have NONE
    └──────────────────────────────────────────────────────┘
```

### What Tests Would Catch

| Issue | Test Type | Example |
|-------|-----------|---------|
| Dependency version change breaks API | Unit test | `test_numpy_array_creation()` |
| Different results on different platforms | Integration | `test_pipeline_output_matches_reference()` |
| Docker build produces wrong results | E2E | `test_docker_run_produces_expected_auroc()` |
| Model training is non-deterministic | Reproducibility | `test_training_with_seed_matches_baseline()` |

## Content Elements

1. **Myth vs Reality split**: Common belief vs evidence
2. **Version drift example**: FROM python:3.11 changing over time
3. **Study results table**: 5,298 builds, percentage breakdown
4. **Container registry section**: Benefits and limitations
5. **Cross-platform issues**: x86 vs ARM, Linux vs Windows
6. **Testing gap**: Research software testing statistics
7. **Multi-layer solution**: Five layers with their contributions
8. **Cumulative improvement bar**: Docker alone → Docker + all layers

## Text Content

### Title Text
"Docker is NOT Enough: Why Containers Don't Guarantee Reproducibility"

### Caption
A systematic study of 5,298 Docker builds found only ~60% produced functionally equivalent results (Malka et al. 2026, [arXiv:2601.12811](https://arxiv.org/abs/2601.12811)). Floating base images (`FROM python:3.11`), unpinned apt packages, and version drift cause silent changes. Container registries help with immutability but add infrastructure burden. Cross-platform issues (x86 vs ARM, Linux vs Windows) add complexity. Foundation PLR uses a multi-layer approach: uv.lock for Python, documented system deps, DuckDB for data, MLflow for provenance, and tests for verification—Docker is one layer, not the solution.

## Prompts for Nano Banana Pro

### Style Prompt
Split panel: myth (speech bubble) vs reality (code drift diagram). Study results table with cumulative bar. Multi-layer stack diagram showing Docker as just one component. Warning colors for the problem, solution colors for the fix. Include cross-platform comparison and testing gap statistics.

### Content Prompt
Create "Docker is NOT Enough" infographic:

**TOP - Myth vs Reality**:
- Left: Speech bubble "Just put it in Docker!"
- Right: Code box showing version drift over time

**UPPER-MIDDLE - Study Results**:
- Table: Bitwise 15%, Functional 45%, Different 25%, Fail 15%
- Bullet list: Why Docker fails

**LOWER-MIDDLE - Registry Paradox**:
- Benefits: Immutability, versioning, digest pinning
- Limitations: Cost, security updates, researcher adoption

**BOTTOM - Multi-Layer Solution**:
- Five stacked layers: uv.lock, system deps, DuckDB, MLflow, tests
- Progression bar: Docker alone (60%) → All layers (high %)

## Alt Text

Docker reproducibility infographic challenging the myth that containers guarantee reproducibility. Left panel shows common belief "Just put it in Docker" marked false. Right panel shows version drift: FROM python:3.11 resolving to different versions over time. Study of 5,298 builds (Malka 2026): 15% bitwise identical, 45% functionally equivalent (60% total working), 25% different output, 15% fail. Container registries provide immutability but add infrastructure burden and don't address testing gaps. Cross-platform issues between x86/ARM and Linux/Windows add complexity. Testing statistics: 68% of DL projects have no unit tests. Foundation PLR multi-layer solution: uv.lock (Python), documented system deps, DuckDB (data), MLflow (provenance), pytest (verification). Docker is one layer, not the complete solution.

## Related Figures

- **fig-repro-08a/b/c**: Dependency hell deep dive with UMAP example
- **fig-repro-15**: Virtual environments AND containers (complementary) - for isolation layer details
- **fig-repro-12**: Dependency explosion - why lockfiles matter inside Docker
- **fig-repo-14**: uv package manager

## Cross-References

This figure subsumes **fig-repro-16** (archived) which covered the same Malka et al. study with less depth.

For detailed isolation layer comparison (venv vs renv vs Docker), see **fig-repro-15**.
For why lockfiles are needed even inside Docker, see **fig-repro-12**.

## Status

- [x] Draft created
- [x] Deep analysis added (registry, cross-platform, testing)
- [x] Subsumes fig-repro-16 (archived)
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md
