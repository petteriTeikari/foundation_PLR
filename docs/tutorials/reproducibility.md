# Reproducibility in Machine Learning Research

> **Start Here** | Want to reproduce our results? Jump to [Quick Reproducibility Checklist](#quick-reproducibility-checklist).

> **The uncomfortable truth**: You find a paper claiming AUROC = 0.95 for your disease. You email the authors for code. Two weeks later: "Sorry, the postdoc who wrote it left. We can't find the code."
>
> This happens more often than anyone wants to admit.

---

## The Reproducibility Crisis

![Statistics on the reproducibility crisis: percentage of studies that fail to reproduce across fields, estimated costs of irreproducible research, and the gap between 'code available on request' claims and actual code availability.](../repo-figures/assets/fig-repro-01-crisis-in-numbers.jpg)

**The Numbers Don't Lie**

In Nature's 2016 survey of 1,576 scientists, **70% had tried and failed to reproduce another scientist's experiments** (Baker, 2016). For computational research specifically:

- **Gundersen & Kjensmo (2018)**: Only 6% of AI conference papers (IJCAI, AAAI) share code
- **Pineau et al. (2021)**: Code sharing varies from 10% (neuroscience) to 70% (computer vision)
- **Raff (2019)**: Only 63.5% of ML papers were reproducible with significant effort

The common causes include:
- Missing code
- Missing data
- Incomplete documentation
- Environment differences
- Random seed issues

---

## Why Notebooks Fail

### ELI5 Version

![Why Jupyter notebooks fail at reproducibility: hidden state from out-of-order cell execution, missing dependency tracking, and the inability to run notebooks end-to-end without manual intervention.](../repo-figures/assets/fig-repro-02a-notebooks-fail-eli5.jpg)

**The Hidden State Problem**

Jupyter notebooks allow out-of-order execution. You might run cell 5, then cell 2, then cell 7. The notebook "works" for you, but when someone runs it top-to-bottom, it breaks.

### Expert Version

![Technical analysis of notebook reproducibility failures: non-deterministic cell execution order creates hidden state dependencies, pip freeze captures environment but not execution sequence, notebook diffs are unreadable JSON.](../repo-figures/assets/fig-repro-02b-notebooks-fail-expert.jpg)

**Why This Repository Uses Scripts**

Scripts enforce linear execution:
```
python script.py  # Always runs the same way
```

Notebooks don't:
```
[Cell 5 modified 10:32am]
[Cell 2 modified 10:35am]  # Which order did they run?
[Cell 5 modified 10:40am]  # Was this before or after cell 2?
```

---

## The Five Horsemen of Irreproducibility

![Five categories of reproducibility failures: environment drift (packages change), hidden state (execution order matters), data provenance (unclear origins), random seeds (non-deterministic results), and manual steps (undocumented interventions).](../repo-figures/assets/fig-repro-03-five-horsemen.jpg)

**What Breaks Reproducibility**

| Horseman | Problem | Our Solution |
|----------|---------|--------------|
| **Missing Code** | "Email me for code" | GitHub repository |
| **Missing Data** | "Data available upon request" | DuckDB database |
| **Dependency Hell** | Version conflicts | UV + lock file |
| **Environment Drift** | "Works on my machine" | Docker + explicit versions |
| **Random Seeds** | Non-deterministic results | Seeded randomness + bootstrap |

---

## Levels of Reproducibility

![Reproducibility levels from weakest to strongest: Level 1 code available (may not run), Level 2 code runs (environment differs), Level 3 environment pinned (Docker + lockfiles), Level 4 results match (deterministic), Level 5 bitwise identical.](../repo-figures/assets/fig-repro-04-levels-of-reproducibility.jpg)

**Not All Reproducibility Is Equal**

| Level | Definition | Difficulty |
|-------|------------|------------|
| **Level 0** | Doesn't run | Most published code |
| **Level 1** | Runs but different results | Common |
| **Level 2** | Same conclusions | Good |
| **Level 3** | Same numbers (± tolerance) | Better |
| **Level 4** | Bitwise identical | Hard |

**This repository targets Level 3** - same numbers within statistical tolerance across runs and machines.

---

## What Reviewers Check

![Reviewer checklist: dependency lockfiles, data availability statements, containerized environments, test suites, continuous integration, and documented random seeds.](../repo-figures/assets/fig-repro-05-what-reviewers-check.jpg)

**Reproducibility Checklist**

- [ ] Code is available
- [ ] Data is available (or synthetic data provided)
- [ ] Dependencies are specified with versions
- [ ] Random seeds are set
- [ ] Instructions to run are clear
- [ ] Expected outputs are documented
- [ ] Compute requirements are stated

---

## The Cost of Irreproducibility

![Estimated costs of irreproducible research: wasted researcher time, retracted papers, delayed clinical translation, and duplicated effort across labs.](../repo-figures/assets/fig-repro-06-cost-of-irreproducibility.jpg)

**Why This Matters**

Irreproducible research:
- Wastes research funding
- Delays scientific progress
- Erodes public trust
- Can harm patients (medical ML)

---

## Dependency Management

### The Problem

![Dependency conflicts: package A needs version 1.x of a library, package B needs version 2.x, and both cannot coexist. Dependency resolution either fails or makes an arbitrary choice.](../repo-figures/assets/fig-repro-08a-dependency-hell-eli5.jpg)

**Dependency Conflicts**

```
Package A requires: numpy>=1.20,<1.25
Package B requires: numpy>=1.24,<2.0
Package C requires: numpy>=1.26

# No numpy version satisfies all three!
```

### The Solution: Lock Files {#lockfiles}

![Lockfiles (uv.lock, renv.lock) as a time machine: they record the exact versions of every transitive dependency at a point in time, allowing exact environment reconstruction months or years later.](../repo-figures/assets/fig-repro-14-lockfiles-time-machine.jpg)

**Lock Files Are Time Machines**

A lock file (`uv.lock`, `poetry.lock`) records:
- Exact version of every package
- Exact version of every dependency
- Hash of each package file

```bash
# Create environment from lock file
uv sync  # Installs EXACTLY what was recorded
```

### Version Pinning Strategies

![Comparison of pinning strategies: no pinning (packages float), requirements.txt (direct deps only), lockfiles (all transitive deps pinned), Docker images (frozen OS + packages), and Nix (bit-reproducible). Trade-off between flexibility and reproducibility.](../repo-figures/assets/fig-repro-11-version-pinning-strategies.jpg)

| Strategy | Example | Trade-off |
|----------|---------|-----------|
| Loose | `numpy>=1.20` | May break with updates |
| Bounded | `numpy>=1.20,<2.0` | Some protection |
| Exact | `numpy==1.24.3` | Reproducible but rigid |
| Lock file | `uv.lock` | **Best of both worlds** |

---

## Docker: Necessary but Not Sufficient

![Diagram showing why Docker alone doesn't guarantee reproducibility: base images change over time (ubuntu:22.04 today differs from ubuntu:22.04 last month due to apt updates), network dependencies can disappear, and building from Dockerfile is not the same as running from a frozen image.](../repo-figures/assets/fig-repro-07-docker-not-enough.jpg)

**Docker Misconceptions**

❌ "I put it in Docker, so it's reproducible"

Docker helps, but:
- Base images change (`python:3.11` today ≠ `python:3.11` next year)
- `apt-get install` pulls latest versions
- Random seeds still matter

### Virtual Environments vs Containers

![Comparison of virtual environments and Docker containers for dependency isolation: venvs isolate Python packages with minimal overhead and instant startup, while containers isolate the entire OS stack including system libraries, compilers, and Python interpreter. This repository provides both: .venv/ for development and Dockerfile for full reproducibility.](../repo-figures/assets/fig-repro-15-venv-vs-containers.jpg)

| Aspect | Virtual Environment | Docker Container |
|--------|---------------------|------------------|
| Isolation level | Python packages | Entire OS |
| Overhead | Minimal | Moderate |
| Startup time | Instant | Seconds |
| Use case | Development | Deployment |

**This repository provides both:**
- `.venv/` for development
- `Dockerfile` for full reproducibility

---

## The Base Image Problem

![Diagram showing how Docker base images change over time as upstream packages are updated. Even with the same tag, apt-get install today produces different packages than last month. Solution: multi-stage builds with explicit version pinning and digest hashes.](../repo-figures/assets/fig-repro-18-base-image-problem.jpg)

**Pin Your Base Images**

```dockerfile
# BAD - changes over time
FROM python:3.11

# GOOD - specific version
FROM python:3.11.7-slim-bookworm@sha256:abc123...
```

---

## How We Achieve Reproducibility

### 1. Single Source of Truth

![Diagram showing DuckDB as the single archival artifact: MLflow runs (ephemeral, large) are extracted into one DuckDB file (permanent, portable). All figures, statistics, and LaTeX tables are generated from this single database, ensuring consistency across analyses.](../repo-figures/assets/fig-repro-20-duckdb-single-source.jpg)

All results are stored in DuckDB, not scattered CSVs.

### 2. Figure Reproducibility

![JSON sidecar pattern for figure reproducibility: each generated figure is accompanied by a JSON file containing the exact data plotted, source database hash, generation parameters, and timestamp. Given the JSON, the figure can be exactly reproduced without re-running the pipeline.](../repo-figures/assets/fig-repro-22-json-sidecars-figure-reproducibility.jpg)

Every figure has a JSON sidecar with the exact data used.

### 3. Automatic Artifacts

![R4R (Ready for Review) automatic artifact generation: when the pipeline completes, it produces all artifacts needed for review -- figures with JSON sidecars, DuckDB database, LaTeX tables, and a manifest file listing what was generated and from which data.](../repo-figures/assets/fig-repro-19-r4r-automatic-artifacts.jpg)

The pipeline automatically captures:
- Git commit hash
- Environment snapshot
- Input data checksums
- Random seeds used

---

## Bitwise vs Functional Reproducibility

![Comparison of bitwise reproducibility (byte-identical outputs, extremely difficult to achieve due to floating-point non-determinism across hardware) versus functional reproducibility (same scientific conclusions, tolerable numerical differences). This project targets functional reproducibility.](../repo-figures/assets/fig-repro-17-bitwise-vs-functional.jpg)

**Do You Need Bitwise Identical Results?**

| Type | Definition | When Needed |
|------|------------|-------------|
| **Bitwise** | Exact same bytes | Rarely (security, legal) |
| **Functional** | Same scientific conclusions | Usually sufficient |

For this repository:
- AUROC might vary by ±0.001 across runs (floating point)
- Conclusions remain the same
- Bootstrap CIs capture the uncertainty

---

## R Package Ecosystem

![Diagram of R's package ecosystem for reproducibility: CRAN (official repository), Bioconductor (bioinformatics packages), and GitHub (development versions) as sources, with renv providing project-local snapshots and renv.lock recording exact versions of all R packages.](../repo-figures/assets/fig-repro-09-r-package-ecosystem.jpg)

**R Reproducibility with renv**

This repository uses `renv` for R packages:

```r
# Restore exact package versions
renv::restore()
```

The `renv.lock` file pins all R package versions.

---

## System Dependencies

![Diagram showing the dependency stack beyond Python packages: system libraries (libblas, liblapack), compiler toolchains (gcc), Python interpreter version, OS-level packages (libcurl, libssl), and hardware-specific optimizations (CUDA, MKL) that all affect reproducibility.](../repo-figures/assets/fig-repro-10-system-dependencies.jpg)

**The Hidden Layer**

Python packages often need system libraries:
- `libpng` for image processing
- `libblas` for linear algebra
- `libcurl` for downloads

Our `Dockerfile` installs all required system dependencies.

---

## Quick Reproducibility Checklist

### For This Repository

```bash
# 1. Clone
git clone https://github.com/petteriTeikari/foundation_PLR.git
cd foundation_PLR

# 2. Environment
uv sync  # Creates venv from lock file

# 3. R packages (if needed)
Rscript -e "renv::restore()"

# 4. Run
make reproduce  # Full pipeline
make reproduce-from-checkpoint  # From pre-computed data
```

### Verify Reproduction

```bash
# Compare outputs to expected
make test-reproducibility
```

---

## References

### Reproducibility Crisis
- Baker M. (2016). 1,500 scientists lift the lid on reproducibility. **Nature**.
- Freedman LP, et al. (2015). The economics of reproducibility in preclinical research. **PLOS Biology**.

### Best Practices
- Wilson G, et al. (2017). Good enough practices in scientific computing. **PLOS Computational Biology**.
- Sandve GK, et al. (2013). Ten simple rules for reproducible computational research. **PLOS Computational Biology**.

### Tools
- [UV Documentation](https://github.com/astral-sh/uv)
- [renv Documentation](https://rstudio.github.io/renv/)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

---

## See Also

- [Dependencies Guide](dependencies.md) - UV, Polars, DuckDB details
- [How to Read the Plots](reading-plots.md) - Understanding our visualizations
- [Makefile Reference](https://github.com/petteriTeikari/foundation_PLR/blob/main/Makefile) - Available commands
