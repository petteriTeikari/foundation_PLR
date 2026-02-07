# Figure Test Harness - Comprehensive QA Planning

## User Request (Verbatim)

> Well based on this failure, could we plan in /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/planning/figure-test-harness.md how to create a proper test harness that would check for EVERYTHING that you can think for the automated QA of generated figures to minimize the need for human supervision/proofreading needed for low-level QA! Do a multiple-hypothesis analysis of all possible failure points with user reviewer agents helping you to refine the analysis complete for all types of failures. After convergence of an excellent in-detail analysis capturing all imaginable failure cases with ggplot2 figure creation, let's continue then with /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/planning/figure-test-harness-TDD-execution.xml, that is a crash-resistant plan with progress tracking implementing all the automated checks for R (or we can obviously call some non-R library for the verification as well). Do multi-hypothesis planning for all the possible implementations options to /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/planning/figure-test-harness.md and keep the .xml action plan compact and as easy to follow the instructions as possible! Does this sound clear? Massive use of reviewer agents familiar with QA, test engineering, ggplot2 plots, science visualization, reproducible ML experiments, etc. Be scientifically rigorous as this is now scientific work, not software work done by a junior engineer printing the "hello world" to figures.

## Context: Critical Failure That Prompted This

See: `.claude/docs/meta-learnings/CRITICAL-FAILURE-001-synthetic-data-in-figures.md`

**Summary**: Claude generated calibration plots using SYNTHETIC data instead of REAL experimental predictions. All 4 models showed identical curves because of a fixed random seed. This would have been scientific fraud if published.

---

## Multi-Hypothesis Failure Analysis

### Domain 1: Data Provenance Failures

**Reviewer Persona**: Data Engineering / Reproducibility Expert

#### 1.1 Synthetic vs Real Data Detection

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Synthetic data substitution** | Metadata flag `"data_source": "real"` required | JSON schema validation |
| **Fixed random seed patterns** | Entropy analysis on prediction arrays | `scipy.stats.entropy()` check |
| **Cross-model identical data** | Pairwise correlation between model predictions | Fail if `corr > 0.99` for different models |
| **Template data not replaced** | SHA-256 hash comparison with known synthetic hashes | Blacklist of synthetic data hashes |

#### 1.2 Data Lineage Validation

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Missing provenance chain** | Require `source_file`, `extraction_timestamp`, `model_path` | JSON schema enforcement |
| **Stale cached data** | Compare JSON timestamp vs source `.pkl` mtime | `os.path.getmtime()` comparison |
| **Wrong experiment loaded** | MLflow run_id verification | Cross-reference with MLflow API |
| **Truncated/incomplete data** | Row count validation against expected N | `len(y_true) == expected_n` |

#### 1.3 Checksum Validation

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Corrupted data transfer** | SHA-256 of source data stored in JSON | Hash verification on load |
| **Version mismatch** | Git commit hash of extraction script | `git rev-parse HEAD` embedding |
| **Silent data modification** | Immutable data directories | File permission checks |

#### 1.4 Cross-Model Uniqueness (CRITICAL)

**This would have caught CRITICAL-FAILURE-001:**

```python
def validate_cross_model_uniqueness(predictions_dict: dict, threshold: float = 0.99):
    """
    Fail if any two models have suspiciously identical predictions.

    The original failure: all 4 models had correlation = 1.0 due to
    shared random seed generating identical synthetic data.
    """
    model_names = list(predictions_dict.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            corr = np.corrcoef(
                predictions_dict[m1]['y_prob'],
                predictions_dict[m2]['y_prob']
            )[0, 1]
            if corr > threshold:
                raise ValidationError(
                    f"Models {m1} and {m2} have correlation {corr:.4f} > {threshold}. "
                    f"Are they using the same synthetic data?"
                )
```

---

### Domain 2: Statistical/Scientific Validity Failures

**Reviewer Persona**: Biostatistician / STRATOS Compliance Expert

#### 2.1 STRATOS Metric Validation

| Metric | Valid Range | Failure Mode | Detection |
|--------|-------------|--------------|-----------|
| **AUROC** | [0.5, 1.0] | Below chance performance | `auroc < 0.5` check |
| **Calibration slope** | [0.0, 2.0] typical | Perfect calibration artifact | `slope == 1.0 exactly` is suspicious |
| **O:E ratio** | [0.5, 2.0] typical | Extreme miscalibration | Range check with warning |
| **Brier score** | [0.0, 0.25] | Impossible values | `brier > 0.25` or `brier < 0` |
| **IPA (scaled Brier)** | [-∞, 1.0] | Above 1.0 impossible | Range validation |
| **Net Benefit** | [-prevalence, 1-prevalence] | Outside theoretical bounds | Bound checking |

#### 2.2 Calibration Curve Validation

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Empty bins** | All bin counts = 0 | `if all(counts == 0)` |
| **Single-point calibration** | Only one non-null observed value | `sum(~np.isnan(observed)) < 3` |
| **Inverted calibration** | Negative slope | `calibration_slope < 0` with warning |
| **Perfect calibration artifact** | slope=1.0, intercept=0.0 exactly | Equality check with tolerance |

#### 2.3 Bootstrap CI Validation

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **CI narrower than point estimate variation** | CI width < SD of means | Statistical sanity check |
| **Inverted CIs** | `CI_lo > CI_hi` | Ordering check |
| **Zero-width CI** | `CI_lo == CI_hi` | Likely computation error |
| **CI outside valid range** | `CI_lo < 0` for positive-only metrics | Range check |

#### 2.4 DCA Validation

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Net benefit above "treat all"** | Impossible by definition | `nb_model > nb_all + epsilon` |
| **Missing reference strategies** | "Treat All" and "Treat None" required | Key existence check |
| **Non-monotonic threshold sweep** | Thresholds must be sorted | `np.all(np.diff(thresholds) > 0)` |
| **Negative net benefit at low threshold** | Unusual but possible | Warning, not error |

#### 2.5 Sample Size Validation

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Too few events** | `n_events < 10` insufficient for calibration | Minimum event check |
| **Extreme class imbalance** | `prevalence < 0.01` or `> 0.99` | Range warning |
| **N mismatch across figures** | Same experiment should have same N | Cross-figure N comparison |

---

### Domain 3: Visual Rendering Failures

**Reviewer Persona**: ggplot2 / Data Visualization Expert

#### 3.1 Multi-Series Visibility (CRITICAL)

**This domain directly addresses CRITICAL-FAILURE-001:**

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **All curves overlapping** | Image entropy analysis | `imagehash` perceptual hashing |
| **Missing legend entries** | Legend count vs data series count | Parse SVG/extract legend |
| **Invisible lines** | Line width = 0 or alpha = 0 | ggplot2 theme inspection |
| **Color collision** | Multiple series with same color | Palette uniqueness check |

```python
def validate_multi_series_visibility(image_path: str, expected_series: int):
    """
    Use perceptual hashing to detect if multiple series are actually visible.

    Algorithm:
    1. Compute perceptual hash of full image
    2. Mask out each series color and re-hash
    3. If removing a color doesn't change the hash, that series is invisible
    """
    import imagehash
    from PIL import Image

    img = Image.open(image_path)
    base_hash = imagehash.phash(img)

    # This requires knowing the expected colors from the theme
    # Or using color clustering to find distinct line colors
    pass  # Implementation details in TDD plan
```

#### 3.2 ggplot2 S7/4.0+ Compatibility

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Deprecated function warnings** | Capture stderr during render | `tryCatch` with warning handler |
| **S7 object coercion errors** | Class type validation | `inherits(obj, "ggplot")` |
| **Theme element type mismatch** | Post-4.0 theme validation | `validate_ggplot_theme()` |
| **Scale function deprecation** | `scale_*_continuous()` vs new syntax | Package version check |

#### 3.3 Text Rendering

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Text truncation** | Bounding box overflow | PDF text extraction + bbox analysis |
| **Font substitution** | Missing fonts replaced | `extrafont` package validation |
| **Overlapping labels** | Text collision detection | `ggrepel` usage or bbox check |
| **Unicode rendering issues** | Special characters (μ, ², α) | Regex search in output |

#### 3.4 Axis and Scale Issues

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Axis limits clipping data** | Points outside visible range | Compare data range to axis limits |
| **Log scale with zeros** | Infinite values | `any(data <= 0)` for log scales |
| **Reversed axes** | X or Y axis flipped | Scale direction check |
| **Missing gridlines** | Visual parsing | Theme element presence |

#### 3.5 Facet and Panel Issues

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Empty facet panels** | Facet with no data | `nrow(facet_data) == 0` |
| **Misaligned facets** | Panel spacing inconsistent | Layout geometry check |
| **Missing facet labels** | Strip text empty | Theme element validation |

---

### Domain 4: Reproducibility Failures

**Reviewer Persona**: Reproducible Research / MLOps Expert

#### 4.1 Environment Reproducibility

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **R version mismatch** | Different behavior across versions | `R.version.string` embedding |
| **Package version drift** | Functions deprecated/changed | `sessionInfo()` capture |
| **Python-R interop issues** | rpy2 version sensitivity | Version lock in requirements |
| **OS-specific rendering** | Font, locale differences | Platform string embedding |

#### 4.2 Path and Location Failures

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Hardcoded absolute paths** | `/home/petteri/...` in code | Regex scan for absolute paths |
| **Missing relative path resolution** | `./data` vs `data/` inconsistency | Path normalization |
| **Working directory dependency** | Different results from different cwd | `here::here()` usage |
| **Temp file cleanup failure** | Orphaned temp files | `on.exit()` cleanup patterns |

#### 4.3 Random Seed Management

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Unseeded randomness** | Non-reproducible bootstrap | `set.seed()` at entry points |
| **Seed collision** | Same seed reused (CRITICAL-FAILURE-001!) | Unique seed per operation |
| **Seed not logged** | Cannot reproduce exact run | Seed value in metadata |
| **Global seed pollution** | One function affects another | Local RNG state isolation |

```r
# Anti-pattern (caused CRITICAL-FAILURE-001):
set.seed(42)  # Global, reused for ALL models!
for (model in models) {
    generate_data()  # All get same "random" data
}

# Correct pattern:
for (i in seq_along(models)) {
    set.seed(42 + i)  # Unique seed per model
    # Or better: use model hash as seed component
    set.seed(digest::digest2int(models[[i]]$name))
}
```

#### 4.4 Caching and Memoization

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Stale cache served** | Cache invalidation failure | Hash-based cache keys |
| **Cache key collision** | Different inputs, same key | Include all inputs in key |
| **Memory cache vs disk mismatch** | Inconsistent state | Single source of truth |

---

### Domain 5: Publication Standards Failures

**Reviewer Persona**: Scientific Publishing / Journal Standards Expert

#### 5.1 Resolution and Dimensions

| Standard | Requirement | Detection |
|----------|-------------|-----------|
| **Print DPI** | ≥300 DPI for raster | `identify -verbose` or PIL |
| **Vector preferred** | PDF/SVG for line art | File extension check |
| **Figure width** | Single column: 3.5", double: 7" | PDF mediabox extraction |
| **Aspect ratio** | Journal-specific (often 4:3 or 16:9) | Width/height ratio |

#### 5.2 Color Standards

| Standard | Requirement | Detection |
|----------|-------------|-----------|
| **CMYK for print** | RGB may shift colors | Color profile check |
| **Colorblind safe** | Distinguishable palette | Simulate deuteranopia |
| **Black & white fallback** | Grayscale distinguishable | Desaturation test |
| **Color consistency** | Same model = same color across figures | Cross-figure color audit |

#### 5.3 Text Standards

| Standard | Requirement | Detection |
|----------|-------------|-----------|
| **Minimum font size** | ≥6pt after scaling | Font size extraction |
| **Sans-serif for figures** | Arial/Helvetica typical | Font family check |
| **Embedded fonts** | No font substitution | PDF font listing |
| **No bitmap text** | Vector text only | Text extraction test |

#### 5.4 Statistical Notation

| Standard | Requirement | Detection |
|----------|-------------|-----------|
| **CI notation** | "95% CI [lo, hi]" not "(lo-hi)" | Regex pattern check |
| **P-value formatting** | "P < 0.001" not "P = 0.000" | Decimal place validation |
| **Metric precision** | AUROC: 2-3 decimals | Significant figure count |
| **Uncertainty always shown** | Point estimates need CI/SE | CI presence check |

#### 5.5 Journal-Specific Requirements

| Journal | Key Requirements |
|---------|-----------------|
| **BMJ Open** | 300 DPI, 170mm max width, EPS/TIFF preferred |
| **PLOS ONE** | 300-600 DPI, TIFF preferred, 3.5" single column |
| **Nature** | 300 DPI, vector preferred, specific color palette |
| **Br J Ophthalmol** | 300 DPI, 84mm (single) or 174mm (double) |

---

### Domain 6: Accessibility & Test Engineering Failures

**Reviewer Persona**: Accessibility Expert + Test Engineering Specialist

#### 6.1 Colorblind Accessibility

| Failure Mode | Detection Strategy | Implementation |
|--------------|-------------------|----------------|
| **Red-green confusion** | Deuteranopia simulation | `colorspace::deutan()` |
| **Low contrast pairs** | WCAG contrast ratio | `colorspace::contrast_ratio()` |
| **Color-only encoding** | No shape/pattern backup | Visual audit flag |
| **Legend color matching** | Legend colors match plot | Color extraction comparison |

#### 6.2 Test Coverage Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| **No visual regression tests** | Silent rendering changes | `vdiffr` or `imagehash` |
| **No property-based tests** | Edge cases missed | `hypothesis` (Python) |
| **No integration tests** | Python→R→PDF pipeline | End-to-end smoke tests |
| **No mutation tests** | Tests don't catch bugs | `mutmut` (Python) |

#### 6.3 Test Oracle Problem

| Challenge | Solution |
|-----------|----------|
| **What is "correct" rendering?** | Golden file comparison with tolerance |
| **Floating point comparison** | Epsilon-based equality |
| **Platform-dependent rendering** | Normalize before comparison |
| **Evolving "correct" output** | Explicit golden file updates with review |

#### 6.4 Visual Regression Testing

```python
# Using imagehash for perceptual comparison
import imagehash
from PIL import Image

def compare_figures(generated: str, golden: str, threshold: int = 5):
    """
    Compare generated figure to golden reference.

    Args:
        threshold: Maximum Hamming distance (0 = identical, higher = more different)
    """
    gen_hash = imagehash.phash(Image.open(generated))
    gold_hash = imagehash.phash(Image.open(golden))

    distance = gen_hash - gold_hash
    if distance > threshold:
        raise AssertionError(
            f"Figure {generated} differs from golden by {distance} "
            f"(threshold: {threshold})"
        )
```

---

## Implementation Options Analysis

### Option A: Python-Native Validation (Recommended)

**Pros:**
- Full control over validation logic
- Easy integration with existing Python codebase
- Rich ecosystem (imagehash, PIL, hypothesis)
- Better error messages and debugging

**Cons:**
- Must call R for ggplot2-specific checks
- Some duplication with R testing tools

**Architecture:**
```
tests/
├── conftest.py                    # pytest fixtures
├── test_figure_qa/
│   ├── test_data_provenance.py    # Domain 1
│   ├── test_statistical_validity.py # Domain 2
│   ├── test_visual_rendering.py   # Domain 3
│   ├── test_reproducibility.py    # Domain 4
│   ├── test_publication_standards.py # Domain 5
│   └── test_accessibility.py      # Domain 6
├── golden_files/                  # Reference images
└── fixtures/
    └── sample_data/               # Test data
```

### Option B: R-Native Validation (testthat + vdiffr)

**Pros:**
- Native ggplot2 introspection
- vdiffr designed for ggplot2
- Stays in R ecosystem

**Cons:**
- Harder Python integration
- Less flexible validation logic
- Weaker property-based testing

### Option C: Hybrid Approach (Selected)

**Best of both worlds:**
1. **R side**: vdiffr for visual regression, ggplot2 theme validation
2. **Python side**: Data validation, statistical checks, orchestration
3. **Shared**: JSON schema validation, cross-language assertions

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Orchestration Layer                    │
│  - pytest runner                                                 │
│  - JSON schema validation                                        │
│  - Data provenance checks                                        │
│  - Statistical validity                                          │
│  - Cross-model uniqueness                                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    R Validation Layer (via rpy2)                 │
│  - vdiffr visual regression                                      │
│  - ggplot2 theme introspection                                   │
│  - Calibration computation (pmcalibration)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Artifacts                              │
│  - JUnit XML test report                                         │
│  - HTML visual diff report                                       │
│  - JSON validation manifest                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Priority Matrix

Based on failure severity and detection difficulty:

| Priority | Check | Domain | Catches Failure Type |
|----------|-------|--------|---------------------|
| **P0 CRITICAL** | Cross-model uniqueness | 1 | CRITICAL-FAILURE-001 |
| **P0 CRITICAL** | Data source = "real" flag | 1 | Synthetic data substitution |
| **P0 CRITICAL** | Source file hash validation | 1 | Wrong data loaded |
| **P1 HIGH** | STRATOS metric ranges | 2 | Impossible statistics |
| **P1 HIGH** | Bootstrap CI validity | 2 | Computation errors |
| **P1 HIGH** | Multi-series visibility | 3 | Overlapping curves |
| **P1 HIGH** | Calibration curve validity | 2 | Empty/single-point curves |
| **P2 MEDIUM** | DPI ≥ 300 | 5 | Print quality |
| **P2 MEDIUM** | Colorblind simulation | 6 | Accessibility |
| **P2 MEDIUM** | Visual regression | 3 | Silent rendering changes |
| **P3 LOW** | Font embedding | 5 | Substitution on other systems |
| **P3 LOW** | Session info capture | 4 | Future reproducibility |

---

## Convergence Status

- [x] Domain 1 analysis complete (Data Provenance)
- [x] Domain 2 analysis complete (Statistical Validity)
- [x] Domain 3 analysis complete (Visual Rendering)
- [x] Domain 4 analysis complete (Reproducibility)
- [x] Domain 5 analysis complete (Publication Standards)
- [x] Domain 6 analysis complete (Accessibility + Test Engineering)
- [x] Cross-domain synthesis complete
- [x] Implementation options ranked
- [x] Ready for TDD execution plan

---

## Next Step

Proceed to `planning/figure-test-harness-TDD-execution.xml` for the implementation plan.
