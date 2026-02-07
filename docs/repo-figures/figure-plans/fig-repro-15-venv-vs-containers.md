# fig-repro-15: Virtual Environments AND Containers (Complementary, Not Competing)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-15 |
| **Title** | Virtual Environments AND Containers: Complementary Technologies |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | ML Engineer, Data Scientist, Research Software Developer |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Clarify the common misconception that virtual environments and containers are competing technologies. Show how they work together as complementary isolation layers—venvs isolate language-specific packages while containers isolate the entire system. Foundation PLR uses BOTH.

## Key Message

"Virtual environments (venv/renv/npm) and Docker are NOT competing technologies—they're complementary layers. venv isolates Python packages, renv isolates R packages, and Docker wraps everything including system libraries and the OS. Use venvs inside Docker for the best of both worlds."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│         VIRTUAL ENVIRONMENTS AND CONTAINERS: COMPLEMENTARY, NOT COMPETING       │
│                                                                                 │
│         "They solve different problems at different layers"                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  COMMON MISCONCEPTION                                                           │
│  ════════════════════                                                           │
│                                                                                 │
│  ❌ "Should I use venv OR Docker?"                                              │
│  ❌ "Docker replaces virtual environments"                                       │
│  ❌ "venv is enough, I don't need containers"                                    │
│                                                                                 │
│  ✅ CORRECT: Use venv INSIDE Docker for maximum reproducibility                 │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHAT EACH TECHNOLOGY ISOLATES                                                  │
│  ═════════════════════════════                                                  │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │                         THE ISOLATION STACK                                ││
│  │                                                                            ││
│  │    ┌─────────────────────────────────────────────────────────────────┐    ││
│  │    │                    YOUR APPLICATION CODE                         │    ││
│  │    └─────────────────────────────────────────────────────────────────┘    ││
│  │                               │                                            ││
│  │    ┌─────────────────────────────────────────────────────────────────┐    ││
│  │    │  LANGUAGE PACKAGE MANAGERS (isolate packages per language)      │    ││
│  │    │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │    ││
│  │    │  │ Python venv  │ │   R renv     │ │   Node npm   │             │    ││
│  │    │  │ + uv.lock    │ │ + renv.lock  │ │ + package-   │             │    ││
│  │    │  │              │ │              │ │   lock.json  │             │    ││
│  │    │  │ numpy 1.24   │ │ ggplot2 3.4  │ │ react 18.2   │             │    ││
│  │    │  │ pandas 2.0   │ │ dplyr 1.1    │ │ d3 7.8       │             │    ││
│  │    │  └──────────────┘ └──────────────┘ └──────────────┘             │    ││
│  │    └─────────────────────────────────────────────────────────────────┘    ││
│  │                               │                                            ││
│  │    ┌─────────────────────────────────────────────────────────────────┐    ││
│  │    │  SYSTEM LIBRARIES (shared by all languages, need container)     │    ││
│  │    │                                                                 │    ││
│  │    │  libcurl, openssl, glibc, libpng, libffi, zlib, HDF5           │    ││
│  │    │  These CANNOT be isolated by venv/renv/npm!                     │    ││
│  │    └─────────────────────────────────────────────────────────────────┘    ││
│  │                               │                                            ││
│  │    ┌─────────────────────────────────────────────────────────────────┐    ││
│  │    │  LANGUAGE RUNTIMES (shared or containerized)                    │    ││
│  │    │                                                                 │    ││
│  │    │  Python 3.11.2, R 4.4.0, Node 20.11.0                          │    ││
│  │    └─────────────────────────────────────────────────────────────────┘    ││
│  │                               │                                            ││
│  │    ┌─────────────────────────────────────────────────────────────────┐    ││
│  │    │  OPERATING SYSTEM (shared or containerized)                     │    ││
│  │    │                                                                 │    ││
│  │    │  Ubuntu 22.04, Debian 12, Alpine 3.18                           │    ││
│  │    └─────────────────────────────────────────────────────────────────┘    ││
│  │                               │                                            ││
│  │    ┌─────────────────────────────────────────────────────────────────┐    ││
│  │    │  HOST KERNEL (always shared, even with containers)              │    ││
│  │    │                                                                 │    ││
│  │    │  Linux 6.x kernel - containers share this!                      │    ││
│  │    └─────────────────────────────────────────────────────────────────┘    ││
│  │                                                                            ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │                      ISOLATION BOUNDARIES                                  ││
│  │                                                                            ││
│  │    Layer              │ venv │ renv │ npm │ Docker │ VM   │               ││
│  │    ───────────────────│──────│──────│─────│────────│──────│               ││
│  │    Python packages    │  ✅  │  ❌  │  ❌ │   ✅   │  ✅  │               ││
│  │    R packages         │  ❌  │  ✅  │  ❌ │   ✅   │  ✅  │               ││
│  │    Node packages      │  ❌  │  ❌  │  ✅ │   ✅   │  ✅  │               ││
│  │    System libraries   │  ❌  │  ❌  │  ❌ │   ✅   │  ✅  │               ││
│  │    Language runtime   │  ❌  │  ❌  │  ❌ │   ✅   │  ✅  │               ││
│  │    Operating system   │  ❌  │  ❌  │  ❌ │   ✅   │  ✅  │               ││
│  │    Kernel             │  ❌  │  ❌  │  ❌ │   ❌   │  ✅  │               ││
│  │                                                                            ││
│  │    Speed              │ Fast │ Fast │ Fast│ Medium │ Slow │               ││
│  │    Size               │ ~MB  │ ~MB  │ ~MB │  ~GB   │ ~GB  │               ││
│  │    Setup time         │ <1s  │ <1s  │ <1s │  mins  │ mins │               ││
│  │                                                                            ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR: THE COMBINED APPROACH                                          │
│  ═════════════════════════════════════                                          │
│                                                                                 │
│  We use BOTH venv and Docker, serving different purposes:                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   DEVELOPMENT WORKFLOW (venv + renv for speed)                          │   │
│  │   ─────────────────────────────────────────────                         │   │
│  │                                                                         │   │
│  │   $ uv sync                    # Python: 3 seconds                      │   │
│  │   $ Rscript -e "renv::restore()" # R: 30 seconds                        │   │
│  │   $ npm install                 # JS: 10 seconds                        │   │
│  │                                                                         │   │
│  │   → Fast iteration, IDE integration, breakpoints work                   │   │
│  │   → uv.lock ensures Python is reproducible                              │   │
│  │   → renv.lock ensures R is reproducible                                 │   │
│  │                                                                         │   │
│  │   ⚠️ BUT: System libs (glibc, HDF5) from host machine                   │   │
│  │   ⚠️ Different machines may have different system libs!                 │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   REPRODUCIBILITY WORKFLOW (Docker wrapping venv + renv)                │   │
│  │   ───────────────────────────────────────────────────────               │   │
│  │                                                                         │   │
│  │   Dockerfile:                                                           │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │   │ FROM python:3.11.2-slim-bookworm   # Pin Python + Debian        │  │   │
│  │   │                                                                 │  │   │
│  │   │ # System dependencies (pinned)                                  │  │   │
│  │   │ RUN apt-get update && apt-get install -y \                      │  │   │
│  │   │     libhdf5-dev=1.10.8+repack1-1 \                              │  │   │
│  │   │     r-base=4.4.0-1                                              │  │   │
│  │   │                                                                 │  │   │
│  │   │ # Python packages (from lockfile)                               │  │   │
│  │   │ COPY uv.lock pyproject.toml ./                                  │  │   │
│  │   │ RUN pip install uv && uv sync --frozen                          │  │   │
│  │   │                                                                 │  │   │
│  │   │ # R packages (from lockfile)                                    │  │   │
│  │   │ COPY renv.lock ./                                               │  │   │
│  │   │ RUN R -e "renv::restore()"                                      │  │   │
│  │   │                                                                 │  │   │
│  │   │ # Application code                                              │  │   │
│  │   │ COPY . .                                                        │  │   │
│  │   └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                         │   │
│  │   $ docker build -t foundation-plr:v1.0.0 .  # 5-10 minutes            │   │
│  │   $ docker run foundation-plr:v1.0.0 make reproduce                    │   │
│  │                                                                         │   │
│  │   → Full isolation: Python + R + system libs + OS                       │   │
│  │   → Same result on any machine with Docker                              │   │
│  │   → Lockfiles STILL used inside container                               │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   CI/CD WORKFLOW (GitHub Actions uses both)                             │   │
│  │   ─────────────────────────────────────────                             │   │
│  │                                                                         │   │
│  │   - name: Test with venv (fast feedback)                                │   │
│  │     run: |                                                              │   │
│  │       uv sync                                                           │   │
│  │       pytest tests/unit/                                                │   │
│  │                                                                         │   │
│  │   - name: Test in Docker (full reproducibility check)                   │   │
│  │     run: |                                                              │   │
│  │       docker build -t test .                                            │   │
│  │       docker run test make test-all                                     │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY VENV INSIDE DOCKER (NOT JUST DOCKER)                                       │
│  ═════════════════════════════════════════                                      │
│                                                                                 │
│  Q: "If Docker isolates everything, why bother with venv inside?"               │
│                                                                                 │
│  A: Because Docker WITHOUT a lockfile is not reproducible!                      │
│                                                                                 │
│  ❌ BAD Dockerfile:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ FROM python:3.11                  # Floating tag!                       │   │
│  │ RUN pip install pandas numpy      # No versions!                        │   │
│  │ # Result: Different every time you rebuild                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ✅ GOOD Dockerfile:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ FROM python:3.11.2-slim-bookworm  # Pinned tag!                         │   │
│  │ COPY uv.lock pyproject.toml ./                                          │   │
│  │ RUN pip install uv && uv sync --frozen  # From lockfile!                │   │
│  │ # Result: Identical every time                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  The lockfile provides the WHAT (exact versions)                                │
│  Docker provides the WHERE (isolated environment)                               │
│  Together they provide HOW (reproducible execution)                             │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  MULTI-LANGUAGE PROJECTS: THE REAL VALUE                                        │
│  ═══════════════════════════════════════                                        │
│                                                                                 │
│  Foundation PLR uses Python, R, AND JavaScript:                                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Docker Container                                │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │   │                      System Libraries                           │  │   │
│  │   │              (libcurl, openssl, HDF5, ...)                      │  │   │
│  │   └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                │                                        │   │
│  │   ┌────────────┐  ┌────────────┐  ┌────────────┐                       │   │
│  │   │  Python    │  │     R      │  │   Node.js  │                       │   │
│  │   │  3.11.2    │  │   4.4.0    │  │   20.11.0  │                       │   │
│  │   ├────────────┤  ├────────────┤  ├────────────┤                       │   │
│  │   │   .venv    │  │   renv     │  │node_modules│                       │   │
│  │   │ + uv.lock  │  │ + renv.lock│  │+package-   │                       │   │
│  │   │            │  │            │  │ lock.json  │                       │   │
│  │   │ pandas     │  │ ggplot2    │  │ react      │                       │   │
│  │   │ numpy      │  │ pminternal │  │ d3         │                       │   │
│  │   │ mlflow     │  │ dcurves    │  │ vite       │                       │   │
│  │   └────────────┘  └────────────┘  └────────────┘                       │   │
│  │         │                │                │                             │   │
│  │         └────────────────┼────────────────┘                             │   │
│  │                          │                                              │   │
│  │              R calls Python via reticulate                              │   │
│  │              Python calls R via rpy2                                    │   │
│  │              JS serves Python/R outputs via API                         │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Each language has its own lockfile. Docker unifies them.                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Comparison Table

| Use Case | Just venv | Just Docker | venv + Docker |
|----------|-----------|-------------|---------------|
| Fast development iteration | ✅ | ❌ Slow rebuild | ✅ Use venv directly |
| IDE debugging | ✅ | ⚠️ Remote attach | ✅ Use venv directly |
| Cross-team reproducibility | ⚠️ System libs differ | ⚠️ Need lockfile inside | ✅ Full isolation |
| Multi-language projects | ❌ Python only | ✅ | ✅ Each language isolated |
| CI/CD testing | ✅ Fast | ⚠️ Slow | ✅ venv for unit, Docker for integration |
| Publication reproducibility | ❌ System libs vary | ⚠️ Need lockfiles | ✅ Complete solution |

## Content Elements

1. **Misconception callout**: "venv OR Docker?" → "venv AND Docker"
2. **Layer diagram**: Shows isolation at each level
3. **Comparison table**: What each technology isolates
4. **Three workflows**: Development, Reproducibility, CI/CD
5. **Dockerfile examples**: Bad vs good practices
6. **Multi-language diagram**: Python + R + JS in one container
7. **Why both**: Lockfiles provide WHAT, Docker provides WHERE

## Text Content

### Title Text
"Virtual Environments AND Containers: Complementary, Not Competing"

### Caption
Virtual environments (venv, renv, npm) and Docker containers are complementary technologies, not alternatives. venv isolates Python packages but shares system libraries with the host. renv isolates R packages. Docker wraps everything—including system libs, language runtimes, and the OS—but still needs lockfiles inside for package reproducibility. Foundation PLR uses all three: uv.lock for Python (200+ packages), renv.lock for R (pminternal, dcurves), and Docker for full system isolation. Use venvs for fast development; Docker for reproducible publication.

## Prompts for Nano Banana Pro

### Style Prompt
Layer cake diagram showing isolation levels from code to kernel. Comparison table with checkmarks for what each technology isolates. Two workflow boxes: development (fast) and reproducibility (complete). Multi-language diagram showing Python, R, and Node.js with their respective lockfiles inside a Docker container. Professional, educational style.

### Content Prompt
Create "venv AND Docker" infographic:

**TOP - Misconception**:
- Cross out "venv OR Docker"
- Highlight "venv AND Docker"

**UPPER - Layer Diagram**:
- Stack: Code → Packages → System libs → Runtime → OS → Kernel
- Mark which layers each tool isolates

**MIDDLE - Comparison Table**:
- Rows: Package types, system libs, runtime, OS, kernel
- Columns: venv, renv, npm, Docker, VM

**LOWER - Two Workflows**:
- Development: uv sync, fast, IDE works
- Reproducibility: Docker + lockfiles, complete isolation

**BOTTOM - Multi-Language**:
- Docker container containing Python venv, R renv, Node modules
- All with their lockfiles
- Arrows showing interop (reticulate, rpy2)

## Alt Text

Virtual environments and Docker as complementary technologies diagram. Shows common misconception "venv OR Docker" crossed out, replaced with "venv AND Docker." Layer diagram shows code at top, then language packages (isolated by venv/renv/npm), then system libraries and OS (isolated by Docker), with kernel shared by all. Comparison table shows venv isolates only Python, renv only R, npm only Node, Docker isolates all including system libs and OS, only VMs isolate kernel. Two workflows: development uses venv directly for speed; reproducibility uses Docker containing venvs and lockfiles. Multi-language diagram shows Foundation PLR's Docker container with Python venv (uv.lock), R renv (renv.lock), and Node (package-lock.json) all communicating via reticulate and rpy2.

## Related Figures

- **fig-repro-07**: Docker is NOT enough (why Docker alone fails) - for multi-layer solution
- **fig-repro-08a/b/c**: Dependency hell deep dive with UMAP example
- **fig-repro-12**: Dependency explosion - why lockfiles matter
- **fig-repo-14**: uv package manager (Python lockfiles)
- **fig-repo-30**: Python-R interop

## Cross-References

For why Docker alone is insufficient, see **fig-repro-07**.
For why lockfiles are needed inside Docker, see **fig-repro-12**.
This figure focuses on the complementary nature of isolation layers, while fig-repro-07 focuses on failure modes.

## Status

- [x] Draft created
- [x] Updated to emphasize complementary nature
- [x] Added multi-language support (Python + R + JS)
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md
