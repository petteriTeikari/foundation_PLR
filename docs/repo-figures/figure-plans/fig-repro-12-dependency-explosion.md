# fig-repro-12: The Dependency Explosion

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-12 |
| **Title** | The Dependency Explosion: Why requirements.txt Fails |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | Data Scientist, ML Engineer, PhD Student |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |
| **Merged From** | fig-repro-12 (requirements.txt problem) + fig-repro-13 (transitive explosion) |

## Purpose

Explain why requirements.txt is insufficient for reproducibility by showing how a small number of direct dependencies explodes into hundreds of transitive dependencies, and why lockfiles are necessary.

## Key Message

"Your 5 listed packages actually install 200+ packages. Foundation PLR has 12 direct dependencies that expand to 200+ transitive (17x factor). Each unlocked package is a reproducibility risk. requirements.txt captures WHAT you want, uv.lock captures WHAT you GET."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              THE DEPENDENCY EXPLOSION: WHY requirements.txt FAILS               │
│              From simple lists to 200+ packages                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  EXAMPLE 1: SIMPLE PROJECT                                                      │
│  ═════════════════════════                                                      │
│                                                                                 │
│  requirements.txt:                     pip install output:                      │
│  ┌────────────────────┐                ┌────────────────────────────────────┐  │
│  │ pandas             │                │ Collecting pandas                   │  │
│  │ scikit-learn       │                │   Downloading pandas-2.1.3.whl      │  │
│  │ matplotlib         │                │ Collecting numpy>=1.23.2            │  │
│  │ seaborn            │                │   Downloading numpy-1.26.0.whl      │  │
│  │ jupyter            │                │ Collecting python-dateutil>=2.8.2   │  │
│  └────────────────────┘                │ Collecting pytz>=2020.1             │  │
│       5 packages                       │ Collecting tzdata>=2022.1           │  │
│                                        │ Collecting joblib>=1.1.1            │  │
│                                        │ Collecting threadpoolctl>=2.0.0     │  │
│                                        │ Collecting scipy>=1.5.0             │  │
│                                        │ ... (200+ more packages)            │  │
│                                        └────────────────────────────────────┘  │
│                                             200+ packages installed!            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXAMPLE 2: FOUNDATION PLR                                                      │
│  ═════════════════════════                                                      │
│                                                                                 │
│  pyproject.toml:                                                                │
│  dependencies = [                          12 packages                          │
│      "pandas", "numpy", "polars",          ─────────────                        │
│      "duckdb", "mlflow", "catboost",                                            │
│      "matplotlib", "seaborn", "hydra-core",                                     │
│      "loguru", "prefect", "pydantic"                                            │
│  ]                                                                              │
│                                                                                 │
│  THE EXPLOSION (partial tree):                                                  │
│                                                                                 │
│  pandas ────┬── numpy                                                           │
│             ├── python-dateutil ── six                                          │
│             ├── pytz                                                            │
│             └── tzdata                                                          │
│                                                                                 │
│  mlflow ────┬── flask ─────┬── werkzeug                                         │
│             │              ├── jinja2 ── markupsafe                             │
│             │              └── click                                            │
│             ├── sqlalchemy ├── greenlet                                         │
│             │              └── typing-extensions                                │
│             ├── requests ──┬── urllib3                                          │
│             │              ├── certifi                                          │
│             │              ├── charset-normalizer                               │
│             │              └── idna                                             │
│             ├── protobuf                                                        │
│             ├── cloudpickle                                                     │
│             └── ... (50+ more!)                                                 │
│                                                                                 │
│  catboost ──┬── numpy (shared with pandas!)                                     │
│             ├── scipy ─────┬── numpy (again!)                                   │
│             │              └── ...                                              │
│             ├── plotly                                                          │
│             └── graphviz                                                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE NUMBERS                                                                    │
│  ═══════════                                                                    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Simple project:   ████ 5 direct  →  ██████████████████████ 200+ total  │   │
│  │                    (40x expansion)                                      │   │
│  │                                                                         │   │
│  │  Foundation PLR:   ████████ 12 direct  →  ████████████████████ 200+     │   │
│  │                    (17x expansion)                                      │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Each of those 200+ packages:                                                   │
│  • Has a specific version that must match                                       │
│  • Might have different behavior across versions                                │
│  • Could introduce breaking changes in updates                                  │
│  • Could be silently replaced by a compromised version                          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  "BUT I USED pip freeze!"                                                       │
│  ════════════════════════                                                       │
│                                                                                 │
│  pip freeze > requirements.txt                                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  certifi==2023.7.22                                                     │   │
│  │  charset-normalizer==3.2.0                                              │   │
│  │  contourpy==1.1.1                                                       │   │
│  │  ... (200 packages)                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  STILL BROKEN because:                                                          │
│  ❌ No SHA-256 hashes (packages could be tampered)                              │
│  ❌ No source URLs (might get different source)                                 │
│  ❌ No platform markers (linux vs mac vs windows)                               │
│  ❌ Flat list (no dependency relationships shown)                               │
│  ❌ No distinction between direct and transitive deps                           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHAT uv.lock PROVIDES                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  [[package]]                                                                    │
│  name = "pandas"                                                                │
│  version = "2.1.3"                                                              │
│  source = { registry = "https://pypi.org/simple" }  ← Source URL               │
│  sha256 = "abc123..."                                ← Tamper-proof            │
│  dependencies = [                                    ← Relationship tree       │
│    { name = "numpy", version = "1.26.0" },                                      │
│    { name = "python-dateutil", version = "2.8.2" },                             │
│  ]                                                                              │
│  [package.markers]                                   ← Platform compatibility  │
│  python_version = ">=3.11"                                                      │
│                                                                                 │
│  ✅ Exact versions    ✅ Hashes    ✅ Sources    ✅ Markers    ✅ Dep tree       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMPARISON SUMMARY                                                             │
│  ══════════════════                                                             │
│                                                                                 │
│  │ Feature              │ requirements.txt │ pip freeze │ uv.lock    │         │
│  │ ──────────────────── │ ──────────────── │ ────────── │ ────────── │         │
│  │ Direct deps          │ ✅ Yes           │ ✅ Yes     │ ✅ Yes     │         │
│  │ Transitive deps      │ ❌ No            │ ✅ Yes     │ ✅ Yes     │         │
│  │ Exact versions       │ ⚠️ Optional      │ ✅ Yes     │ ✅ Yes     │         │
│  │ SHA-256 hashes       │ ❌ No            │ ❌ No      │ ✅ Yes     │         │
│  │ Source URLs          │ ❌ No            │ ❌ No      │ ✅ Yes     │         │
│  │ Platform markers     │ ❌ No            │ ❌ No      │ ✅ Yes     │         │
│  │ Dependency tree      │ ❌ No            │ ❌ No      │ ✅ Yes     │         │
│  │ Fast install         │ ❌ Slow          │ ❌ Slow    │ ✅ 10-100x │         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Simple project example**: 5 packages → 200+ installed
2. **Foundation PLR example**: 12 direct → 200+ transitive
3. **Dependency trees**: Three examples showing expansion (pandas, mlflow, catboost)
4. **Bar chart comparison**: Direct vs total with expansion factors
5. **pip freeze limitations**: Five problems with flat freeze
6. **uv.lock structure**: What a proper lockfile contains
7. **Comparison table**: requirements.txt vs pip freeze vs uv.lock

## Text Content

### Title Text
"The Dependency Explosion: Why requirements.txt Fails"

### Caption
requirements.txt lists direct dependencies, but pip silently installs 200+ transitive dependencies (17-40x expansion). Even pip freeze is insufficient—it lacks cryptographic hashes, source URLs, platform markers, and dependency relationships. uv.lock provides all five: exact versions, SHA-256 verification, source registry, platform compatibility, and the full dependency tree. Foundation PLR has 12 direct dependencies expanding to 200+ transitive—all locked with uv.lock.

## Prompts for Nano Banana Pro

### Style Prompt
Two-example comparison showing simple and complex projects. Dependency tree diagrams. Bar charts with expansion factors. pip freeze vs uv.lock code comparison with annotations. Technical but accessible with clear problem→solution flow.

### Content Prompt
Create "Dependency Explosion" infographic:

**TOP - Two Examples**:
- Simple: 5 → 200+ (40x)
- Foundation PLR: 12 → 200+ (17x)

**UPPER-MIDDLE - Tree Diagrams**:
- pandas, mlflow, catboost trees showing expansion

**LOWER-MIDDLE - pip freeze Problems**:
- Five X marks showing what's missing

**BOTTOM - uv.lock Solution**:
- Code block with annotations
- Comparison table: requirements.txt vs pip freeze vs uv.lock

## Alt Text

Dependency explosion infographic. Two examples: simple project (5 packages → 200+ installed, 40x expansion) and Foundation PLR (12 direct deps: pandas, numpy, mlflow, catboost, etc. → 200+ transitive, 17x expansion). Dependency trees show pandas needs numpy, dateutil, pytz, tzdata; mlflow needs flask, sqlalchemy, requests with 50+ subdeps; catboost shares numpy with pandas. pip freeze problems: no hashes, no sources, no platform markers, flat list, no direct/transitive distinction. uv.lock solution provides version, source URL, sha256 hash, dependencies list, platform markers. Comparison table shows uv.lock has all features requirements.txt and pip freeze lack.

## Related Figures

- **fig-repro-08a**: Dependency hell ELI5 (Lego analogy) ← prerequisite concept
- **fig-repro-08b**: pip vs uv (technical lockfile comparison) ← complementary
- **fig-repro-08c**: UMAP initialization trap (concrete example)
- **fig-repro-14**: Lockfiles as time machine ← solution concept
- **fig-repo-14**: uv package manager ← tool that creates lockfiles

## Cross-References

This figure consolidates:
- `fig-repro-12` (archived): The requirements.txt problem
- `fig-repro-13` (archived): Transitive dependency explosion

Reader flow: **fig-repro-08a** (ELI5 why deps matter) → **THIS FIGURE** (technical problem) → **fig-repro-14** (lockfile solution) → **fig-repo-14** (uv tool)

## Status

- [x] Draft created
- [x] Merged from fig-repro-12 + fig-repro-13
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md
