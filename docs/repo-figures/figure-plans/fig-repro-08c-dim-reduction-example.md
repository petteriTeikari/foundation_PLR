# fig-repro-08c: The UMAP/t-SNE Initialization Trap

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-08c |
| **Title** | The UMAP/t-SNE Initialization Trap |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | Data Scientist, ML Engineer, Bioinformatician |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Provide a concrete, widely-recognized example of how changing default parameters in a popular library can silently break reproducibility—even when the algorithm is "the same." This is the canonical example of why lockfiles matter.

## Key Message

"In 2019, researchers claimed UMAP was superior to t-SNE for preserving global structure. In 2021, Kobak & Linderman showed this 'superiority' was entirely due to different default initialization settings—not the algorithms themselves. If you didn't pin your UMAP version, your visualizations might have changed without you knowing."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Becht et al. 2019 | UMAP better than t-SNE for single-cell visualization | [Nature Biotech 37:38-44](https://www.nature.com/articles/nbt.4314) |
| Kobak & Linderman 2021 | "Superiority" due to initialization, not algorithm | [Nature Biotech 39:156-157](https://www.nature.com/articles/s41587-020-00809-z) |
| UMAP Documentation | Reproducibility warnings | [umap-learn.readthedocs.io](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) |
| Meister 2021 (Twitter) | "Neither t-SNE nor UMAP should be called analysis tools" | [Twitter thread](https://x.com/hippopedoid/status/1356286119621636098) |

## The Scientific Controversy

### Timeline

```
2018: UMAP released, gains popularity in single-cell RNA-seq community
      └── Default: init='spectral' (Laplacian eigenmaps)

2019: Becht et al. Nature Biotech: "UMAP is better than t-SNE"
      └── Compared UMAP (spectral init) vs t-SNE (random init)
      └── Concluded UMAP "better preserves global structure"
      └── Paper cited 3000+ times

2021: Kobak & Linderman Nature Biotech: "Wait, it's the initialization"
      └── Showed t-SNE + PCA init ≈ UMAP + spectral init
      └── Showed UMAP + random init ≈ t-SNE + random init
      └── The "superiority" was an artifact of different defaults!

2023: UMAP changes n_jobs default behavior
      └── random_state now disables parallelism... but with a bug
      └── Warning given, but n_jobs not actually set to 1
```

### The Core Issue

```python
# What researchers thought they were comparing (2019):
#
# "UMAP algorithm" vs "t-SNE algorithm"
#
# What they were ACTUALLY comparing:
#
# "UMAP + Laplacian eigenmap init" vs "t-SNE + random init"
#
# These are NOT equivalent comparisons!
```

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE UMAP/t-SNE INITIALIZATION TRAP                           │
│                    A Cautionary Tale About Default Parameters                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE CLAIM (Becht et al. 2019, Nature Biotechnology)                            │
│  ═══════════════════════════════════════════════════                            │
│                                                                                 │
│  "UMAP provides... the most meaningful organization of cell clusters"           │
│  "UMAP better preserves global structure than t-SNE"                            │
│                                                                                 │
│  Cited 3000+ times. Influenced countless bioinformatics workflows.              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE REALITY (Kobak & Linderman 2021, Nature Biotechnology)                     │
│  ═════════════════════════════════════════════════════════                      │
│                                                                                 │
│  What was compared:                                                             │
│  ┌────────────────────────────┐    ┌────────────────────────────┐              │
│  │        t-SNE               │    │         UMAP               │              │
│  │  (sklearn default 2019)    │    │  (umap-learn default 2019) │              │
│  │                            │    │                            │              │
│  │  init = "random"           │    │  init = "spectral"         │              │
│  │       ↓                    │    │       ↓                    │              │
│  │  Random starting           │    │  Laplacian eigenmap        │              │
│  │  positions                 │    │  starting positions        │              │
│  │       ↓                    │    │       ↓                    │              │
│  │  Poor global structure     │    │  Good global structure     │              │
│  └────────────────────────────┘    └────────────────────────────┘              │
│                                                                                 │
│  The "superiority" of UMAP was entirely due to initialization!                  │
│                                                                                 │
│  When using SAME initialization:                                                │
│  • t-SNE + PCA init    ≈ UMAP + spectral init   (both good)                    │
│  • t-SNE + random init ≈ UMAP + random init     (both poor)                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY THIS MATTERS FOR REPRODUCIBILITY                                           │
│  ════════════════════════════════════                                           │
│                                                                                 │
│  Scenario 1: You wrote your analysis in 2020                                    │
│  ─────────────────────────────────────────                                      │
│  requirements.txt:          Result:                                             │
│  umap-learn>=0.4            Got umap-learn 0.4.6                                │
│                             init='spectral' (default)                           │
│                             → Visualization shows clear clusters                 │
│                                                                                 │
│  Scenario 2: Reviewer tries to reproduce in 2024                                │
│  ───────────────────────────────────────────────                                │
│  requirements.txt:          Result:                                             │
│  umap-learn>=0.4            Got umap-learn 0.5.6                                │
│                             init='spectral' still... BUT                        │
│                             n_jobs default changed!                             │
│                             random_state behavior changed!                      │
│                             → Different visualization!                          │
│                                                                                 │
│  Scenario 3: With a lockfile                                                    │
│  ────────────────────────────                                                   │
│  uv.lock:                   Result:                                             │
│  umap-learn==0.4.6          Got umap-learn 0.4.6                                │
│  numpy==1.19.5              Exact same dependencies                             │
│  scikit-learn==0.24.2       → Identical visualization ✓                         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE DEEPER PROBLEM: STOCHASTIC ALGORITHMS                                      │
│  ═════════════════════════════════════════                                      │
│                                                                                 │
│  From UMAP documentation:                                                       │
│                                                                                 │
│  "UMAP is a stochastic algorithm – it makes use of randomness both to           │
│   speed up approximation steps, and to aid in solving hard optimization         │
│   problems. This means that different runs of UMAP can produce different        │
│   results."                                                                     │
│                                                                                 │
│  Even with random_state set:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  • Multi-threading introduces race conditions                           │   │
│  │  • Different machines → different results (even with same seed!)        │   │
│  │  • Library version changes → different random number generation         │   │
│  │  • NumPy version changes → different numerical precision                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  As Markus Meister noted on Twitter:                                            │
│  "Perhaps we can agree that neither t-SNE nor UMAP should be called             │
│   'analysis tools'. They are mostly automated paint brushes that will           │
│   draw the data according to whatever interpretation you want                   │
│   (aka 'initialization')."                                                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  OTHER EXAMPLES OF "DEFAULT CHANGED" BREAKING REPRODUCIBILITY                   │
│  ════════════════════════════════════════════════════════════                   │
│                                                                                 │
│  │ Library        │ Version │ Change                      │ Impact             │
│  │ ────────────── │ ─────── │ ─────────────────────────── │ ────────────────── │
│  │ sklearn        │ 0.22→1.0│ many default params changed │ Models differ      │
│  │ pandas         │ 1.x→2.x │ copy-on-write behavior      │ Memory issues      │
│  │ numpy          │ 1.x→2.x │ dtype promotion rules       │ Numeric differences│
│  │ matplotlib     │ 3.5→3.7 │ default colormap (viridis)  │ Plots look different│
│  │ torch          │ 1.x→2.x │ deterministic algorithms    │ Training differs   │
│                                                                                 │
│  "Randomness In Neural Network Training" (arXiv:2106.11872):                    │
│  "The impact of non-determinism is nuanced—while top-line metrics such as       │
│   top-1 accuracy are not noticeably impacted, model performance on certain      │
│   parts of the data distribution is far more sensitive to randomness."          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE FIX: EXPLICIT IS BETTER THAN IMPLICIT                                      │
│  ═════════════════════════════════════════                                      │
│                                                                                 │
│  ❌ BAD: Rely on defaults                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  import umap                                                            │   │
│  │  reducer = umap.UMAP()  # Whatever the current default is...            │   │
│  │  embedding = reducer.fit_transform(data)                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ✅ GOOD: Explicit parameters + lockfile                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  import umap                                                            │   │
│  │  reducer = umap.UMAP(                                                   │   │
│  │      n_neighbors=15,                                                    │   │
│  │      min_dist=0.1,                                                      │   │
│  │      metric='euclidean',                                                │   │
│  │      init='spectral',       # ← Explicit initialization                 │   │
│  │      random_state=42,       # ← Explicit seed                           │   │
│  │      n_jobs=1,              # ← Disable parallelism for determinism     │   │
│  │  )                                                                      │   │
│  │  embedding = reducer.fit_transform(data)                                │   │
│  │                                                                         │   │
│  │  # AND use uv.lock to pin umap-learn==0.5.6, numpy==1.24.0, etc.       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ✅ BEST: Save the embedding, not just the parameters                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  # Save the actual embedding to DuckDB/parquet                          │   │
│  │  df['umap_x'] = embedding[:, 0]                                         │   │
│  │  df['umap_y'] = embedding[:, 1]                                         │   │
│  │  df.to_parquet('embeddings.parquet')                                    │   │
│  │                                                                         │   │
│  │  # Now reviewers can use the EXACT same visualization                   │   │
│  │  # without re-running the stochastic algorithm                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Code Examples

### The Problem in Action

```python
# analysis_2020.py - Written with umap-learn 0.4
import umap
import numpy as np

np.random.seed(42)
data = np.random.randn(1000, 50)

# Researcher thinks this is reproducible because they set random_state
reducer = umap.UMAP(random_state=42)
embedding_2020 = reducer.fit_transform(data)
# Got beautiful clusters!

# analysis_2024.py - Same code, umap-learn 0.5.6
# PROBLEM: n_jobs default behavior changed
# PROBLEM: internal random number generation changed
# RESULT: Different embedding, even with same random_state!
```

### The Fix

```python
# analysis_reproducible.py - Explicit everything
import umap
import numpy as np
import json

# 1. Document the environment
with open('environment.json', 'w') as f:
    json.dump({
        'umap_version': umap.__version__,
        'numpy_version': np.__version__,
    }, f)

# 2. Use explicit parameters (never rely on defaults)
reducer = umap.UMAP(
    n_neighbors=15,
    n_components=2,
    metric='euclidean',
    min_dist=0.1,
    spread=1.0,
    init='spectral',
    random_state=42,
    n_jobs=1,  # Critical for determinism
    low_memory=False,
)

# 3. Pin dependencies in uv.lock
# umap-learn==0.5.6
# numpy==1.24.0
# scikit-learn==1.3.0

# 4. Save the actual embedding (not just code to generate it)
embedding = reducer.fit_transform(data)
np.save('embedding.npy', embedding)
```

## Content Elements

1. **Timeline**: History of the UMAP vs t-SNE controversy
2. **Side-by-side comparison**: What was actually compared
3. **Three scenarios**: With version range, with lockfile, reviewer fails
4. **Stochastic algorithm warnings**: Direct quotes from UMAP docs
5. **Table of other breaking changes**: sklearn, pandas, numpy, torch
6. **Code examples**: Bad vs good practices
7. **The fix**: Explicit parameters + lockfile + save outputs

## Text Content

### Title Text
"The UMAP/t-SNE Initialization Trap: Why Default Parameters Break Science"

### Caption
In 2019, Becht et al. claimed UMAP preserved global structure better than t-SNE, influencing thousands of bioinformatics workflows. In 2021, Kobak & Linderman showed this "superiority" was entirely due to different default initialization settings—UMAP used Laplacian eigenmaps while t-SNE used random initialization. When using the same initialization, both algorithms performed similarly. This is a canonical example of why lockfiles matter: without pinning `umap-learn==0.4.6`, your visualizations might silently change when the library updates. Foundation PLR uses explicit parameters AND lockfiles AND saves computed embeddings to ensure true reproducibility.

## Alt Text

UMAP/t-SNE initialization trap infographic. Shows 2019 Becht et al. claim that UMAP better preserves global structure, then 2021 Kobak & Linderman rebuttal showing the difference was due to initialization defaults, not algorithms. Side-by-side comparison: t-SNE with random init vs UMAP with spectral init—not equivalent comparisons. Three reproducibility scenarios show how requirements.txt with version ranges fails while lockfiles succeed. Code examples demonstrate bad practice (relying on defaults) versus good practice (explicit parameters, lockfile, saved embeddings). Table lists other library changes that broke reproducibility: sklearn, pandas, numpy, matplotlib, torch.

## Related Figures

- **fig-repro-08a**: Dependency hell (ELI5 Lego analogy)
- **fig-repro-08b**: pip vs uv lockfiles (technical)
- **fig-repro-12**: requirements.txt problem
- **fig-repro-14**: Lockfiles as time machines

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md
