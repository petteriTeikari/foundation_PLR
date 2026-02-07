# fig-repro-10: System Dependencies: The Hidden Iceberg

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-10 |
| **Title** | System Dependencies: The Hidden Iceberg |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | ML Engineer, DevOps, Lab Manager |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Reveal the hidden layer of system-level dependencies that cause reproducibility failures even when Python/R packages are correctly specified.

## Key Message

"Your Python lockfile is useless if the system doesn't have libssl-dev, gcc, or the right CUDA drivers. Document system dependencies or fail silently."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SYSTEM DEPENDENCIES: THE HIDDEN ICEBERG                      │
│                    The 25% of failures you don't see coming                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE ICEBERG MODEL                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│                        ┌───────────────────┐                                    │
│         VISIBLE:       │  Your Python Code │                                    │
│                        └───────────────────┘                                    │
│  ──────────────────────────────────────────────────── water line ────────────  │
│                        ┌───────────────────┐                                    │
│                        │  pip packages     │                                    │
│                        │  (pandas, numpy)  │                                    │
│                        └───────────────────┘                                    │
│                        ┌───────────────────┐                                    │
│         HIDDEN:        │  System libraries │ ← 25% of failures                 │
│                        │  (libssl, libblas)│                                    │
│                        └───────────────────┘                                    │
│                        ┌───────────────────┐                                    │
│                        │  C/C++ compilers  │                                    │
│                        │  (gcc, cmake)     │                                    │
│                        └───────────────────┘                                    │
│                        ┌───────────────────┐                                    │
│                        │  Python/R runtime │                                    │
│                        │  (version, build) │                                    │
│                        └───────────────────┘                                    │
│                        ┌───────────────────┐                                    │
│                        │  OS + kernel      │                                    │
│                        │  (Ubuntu 22.04)   │                                    │
│                        └───────────────────┘                                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON FAILURES                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  ERROR: Could not build wheels for scipy                                        │
│         → Missing: gfortran, libblas-dev, liblapack-dev                         │
│                                                                                 │
│  ERROR: ImportError: libssl.so.1.1: cannot open shared object file              │
│         → Missing: libssl-dev                                                   │
│                                                                                 │
│  ERROR: Could not find CUDA drivers                                             │
│         → Missing: nvidia-driver-XXX, cuda-toolkit                              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR'S SOLUTION                                                      │
│  ═════════════════════════                                                      │
│                                                                                 │
│  docs/environment.md specifies:                                                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  # Ubuntu 22.04 LTS (Jammy Jellyfish)                                   │   │
│  │                                                                         │   │
│  │  ## Required System Packages                                            │   │
│  │  sudo apt-get install -y \                                              │   │
│  │      build-essential \           # gcc, g++, make                       │   │
│  │      python3.11-dev \            # Python headers                       │   │
│  │      libssl-dev \                # OpenSSL for requests                 │   │
│  │      libffi-dev \                # Foreign function interface           │   │
│  │      r-base r-base-dev \         # R 4.4+                               │   │
│  │      libcurl4-openssl-dev \      # For R packages                       │   │
│  │      libxml2-dev                 # For R packages                       │   │
│  │                                                                         │   │
│  │  ## Verified On                                                         │   │
│  │  - Ubuntu 22.04.3 LTS (kernel 5.15.0-91)                                │   │
│  │  - Python 3.11.7                                                        │   │
│  │  - R 4.4.0                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Why document:                                                                  │
│  • Reviewers can match your environment                                         │
│  • Docker builds are deterministic                                              │
│  • CI/CD pipelines don't fail mysteriously                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Iceberg diagram**: Layers from visible code to hidden OS
2. **Error examples**: Three common failure messages
3. **Solution code block**: apt-get install with comments
4. **Version specifications**: OS, kernel, Python, R versions

## Text Content

### Title Text
"System Dependencies: The 25% of Failures You Don't See Coming"

### Caption
Python/R lockfiles only capture part of the dependency tree. Below the waterline: system libraries (libssl, libblas), compilers (gcc, cmake), runtime versions, and OS. These cause 25% of reproducibility failures (R4R 2025). Foundation PLR documents all system dependencies in docs/environment.md with specific package names and verified platform information.

## Prompts for Nano Banana Pro

### Style Prompt
Iceberg visualization with waterline. Above: code visible. Below: progressively deeper layers (packages, libs, compilers, runtime, OS). Error message callouts. Terminal-style code block for apt-get. Professional blues and whites.

### Content Prompt
Create "System Dependencies Iceberg" infographic:

**TOP - Iceberg**:
- Above water: "Your Python Code"
- Below water (layers): pip packages → system libs → compilers → runtime → OS
- 25% failure annotation

**MIDDLE - Error Examples**:
- Three common error messages with missing package

**BOTTOM - Solution**:
- apt-get install code block with comments
- "Verified on" section with versions

## Alt Text

System dependencies iceberg diagram. Above waterline: visible Python code. Below waterline (hidden layers): pip packages (pandas, numpy), system libraries (libssl, libblas), C/C++ compilers (gcc, cmake), Python/R runtime, OS + kernel. These hidden layers cause 25% of failures. Common errors shown: scipy wheels fail (needs gfortran), libssl not found, CUDA drivers missing. Foundation PLR solution: docs/environment.md with apt-get install commands and verified versions.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

