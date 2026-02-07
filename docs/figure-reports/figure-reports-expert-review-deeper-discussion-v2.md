# Expert Review Addendum v2: Sample Size, Pilot Study Framing, and STRATOS Compliance

> **Generated**: 2026-01-29 (v2 with Petteri annotations integrated)
> **Previous version**: 2026-01-28
> **Purpose**: Guidance for Discussion section regarding sample size limitations and honest reporting
> **Related Document**: `figure-reports-expert-review-v2.md`

---

## Executive Summary

The Foundation PLR study operates under **severe sample size constraints** that fundamentally limit what claims can be made:

| Task | N | Events | Practical Implications |
|------|---|--------|------------------------|
| Preprocessing evaluation | 507 | N/A | Adequate for signal reconstruction benchmarking |
| Classification (development) | 208 | 56 glaucoma | EPV=7.0 for handcrafted; **INADEQUATE for embeddings** |
| Classification (validation) | 0 external | - | **No external validation possible** |

**CRITICAL FRAMING** (author emphasis):
> This is NOT a classifier study. TSFM preprocessing is the CORE message. Classification is a downstream proxy to demonstrate preprocessing impact. Calibration issues at small N are expected and do not diminish the preprocessing contribution.

---

## Section 1: Events Per Variable (EPV) Analysis

### 1.1 The EPV Problem with 56 Events

Events per variable (EPV) quantifies overfitting risk. With only 56 glaucoma events:

| Feature Set | Dimensions | EPV | Risk Assessment |
|-------------|-----------|-----|-----------------|
| **Amplitude bins (featuresSimple.yaml)** | **8** | **7.0** | Borderline acceptable |
| Full handcrafted suite | ~25 | 2.2 | High overfitting risk |
| MOMENT embeddings | 96 | 0.58 | **UNACCEPTABLE - Exploratory only** |

**AUTHOR CORRECTION**: We used exactly 8 features from `featuresSimple.yaml` (4 amplitude bins per color). The EPV is therefore **exactly 7.0**, not a range of 4.7-14.

**Classic guidance** (Riley 2020, Peduzzi 1996):
- EPV < 10: High risk of overfitting
- EPV < 5: Model likely unreliable
- EPV < 1: Model development inappropriate

### 1.2 Why Embedding Features Cannot Be Trusted

The ~12.6pp gap (handcrafted 0.830 vs embeddings 0.704 mean AUROC) may be **partially or entirely artifactual**:

1. **Overfitting masking**: With EPV = 0.58, the model memorizes noise rather than learning generalizable patterns
2. **Regularization cannot save you**: Even CatBoost's built-in regularization cannot compensate for 96 parameters with 56 events
3. **Bootstrap CI interpretation**: Wide confidence intervals reflect parameter instability, not just sampling variance

**Key insight from Riley 2023**: "The smaller the sample size, the more different the models in the multiverse will be and, crucially, the more varied their predicted values for the same individual."

### 1.3 Future Research: Dimensionality Reduction

**AUTHOR DIRECTION**: The initial MOMENT embedding approach was quick prototyping. Proper exploration of embeddings requires:

- **Dimensionality reduction methods**: PCA, UMAP, t-SNE, NMF
- **LoRA-like adapters** for learned compression
- **Matched feature lengths**: Compare at 4, 8, 16, 32, 64, 96 dimensions
- **Especially 8 features**: Match handcrafted dimensionality for fair comparison

This systematic study could become its own paper - we explicitly did NOT do this, and future research should.

### 1.4 What We CAN Conclude About Embeddings

- Embeddings underperform handcrafted features **in this dataset**
- The magnitude of the gap is **uncertain** due to EPV violations
- Any embedding-based claims should be flagged as **hypothesis-generating**

---

## Section 2: Pilot Study vs Clinical Validation Framing

### 2.1 What This Study Is

| Study Type | Characteristics | Foundation PLR Status |
|------------|-----------------|----------------------|
| **Exploratory pilot** | Hypothesis-generating, limited sample | **APPLIES** |
| **Development study** | Internal validation, larger samples | PARTIAL |
| **Validation study** | External data, clinical deployment ready | DOES NOT APPLY |

**Correct framing**: "This proof-of-concept study demonstrates the feasibility of using foundation models for PLR preprocessing and identifies promising preprocessing configurations for future validation."

### 2.2 What We CAN Claim (Defensible)

1. **Foundation models achieve competitive preprocessing performance** compared to traditional methods on this dataset
2. **Handcrafted features outperform embedding features** for glaucoma classification with current sample sizes
3. **Preprocessing choice affects downstream AUROC** (though effect size estimates are uncertain)
4. **Ground truth preprocessing yields AUROC = 0.911** as an upper bound benchmark

### 2.3 What We CANNOT Claim

1. "Foundation models are superior/inferior to traditional methods" (requires larger samples + external validation)
2. "The 12.6pp embedding gap is real" (EPV violation precludes confident conclusion)
3. "AUROC of 0.91 will generalize" (no external validation)
4. "Model is ready for clinical deployment" (far from it)

---

## Section 3: STRATOS Compliance for Small Samples

### 3.1 Van Calster et al. 2024 on Development vs Validation

STRATOS guidance emphasizes:

> "Sample size for model development should ensure **prediction instability** is minimal, not just that aggregate metrics (AUROC) are stable."

**Implication**: Our c-statistic may appear stable (CI: 0.85-0.95), but individual predictions likely exhibit substantial instability (Riley 2023).

### 3.2 pminternal Implementation Status

**AUTHOR REQUEST** (mentioned multiple times): Implement pminternal instability plots in ggplot2 for supplementary materials.

**Current status**:
- Bootstrap predictions EXIST in MLflow: `(n_subjects × 1000 iterations)` per config
- Matplotlib prototype may exist
- **NEEDED**: ggplot2 implementation with Economist styling

**Data location**: `/home/petteri/mlruns/253031330985650090/*/artifacts/metrics/`
**Planning doc**: `docs/planning/pminternal-instability.md`

### 3.3 Internal vs External Validation Requirements

| Validation Type | Requirement | Foundation PLR Status |
|-----------------|-------------|----------------------|
| **Apparent performance** | Report (but do not trust) | Reported |
| **Internal (bootstrap)** | Required for any claims | Done (1000 iterations) |
| **Internal (cross-validation)** | Alternative to bootstrap | Available |
| **External (temporal)** | Highly recommended | NOT DONE |
| **External (geographic)** | Required for deployment | NOT DONE |

**Critical gap**: Without external validation, we cannot assess **transportability** or **generalizability**.

### 3.4 Bootstrap Instability Concerns (Riley 2023)

From Riley et al.'s "multiverse of madness" framework:

> "Instability is concerning as an individual's predicted value is used to guide their counselling, resource prioritisation, and clinical decision making. If different samples lead to different models with very different predictions for the same individual, then this should cast doubt into using a particular model for that individual."

**AUTHOR EMPHASIS**: DEFINITELY need instability plots! Already have matplotlib version, need ggplot2 for supplementary.

---

## Section 4: Honest Uncertainty Reporting

### 4.1 Required Uncertainty Quantification

| Metric | What We Report | What Should Be Acknowledged |
|--------|----------------|----------------------------|
| AUROC | 0.911 (CI: 0.90, 0.92) | CI is from bootstrap; true uncertainty may be wider |
| Calibration slope | 0.52 | Indicates overfitting (slope < 1) |
| Brier score | Point estimate | Scaled Brier (IPA) shows relative improvement |
| Net Benefit | At specific thresholds | DCA curves show threshold sensitivity |

### 4.2 Calibration Interpretation (CRITICAL)

**Correct interpretation of calibration slope = 0.52**:
- Slope < 1.0 indicates **overfitting/overconfidence**
- Predictions are more extreme than warranted
- At high predicted risk, observed risk is lower than predicted
- At low predicted risk, observed risk is higher than predicted

**AUTHOR FRAMING**:
> This is totally acceptable for a pilot study. We should explicitly say models are miscalibrated and highlight this as a data/sample size problem. We do NOT want to recalibrate with post-hoc techniques (temperature scaling, isotonic regression). The pilot nature must be ultra-clear so reviewers understand we are NOT claiming this is a deployment-ready classifier.

### 4.3 What Instability Means for Clinical Use

If we had generated instability plots (per Riley 2023), we would likely see:
- Wide prediction ranges for individuals near decision thresholds
- Classification instability > 15% for borderline cases
- Calibration curve instability in bootstrap samples

**AUTHOR EMPHASIS**: DEFINITELY THIS IS NEEDED! Create ggplot2 instability plots for supplementary.

---

## Section 5: Discussion Text Recommendations

### 5.1 Sample Size Limitations Paragraph (MANDATORY)

**Suggested text for Discussion**:

> "Several limitations warrant consideration. First, with 56 glaucoma events among 208 labeled subjects and 8 handcrafted features, our sample size yields an events-per-variable ratio of 7.0, meeting minimum but not optimal guidelines for prediction model development (Riley et al. 2020). For MOMENT embeddings (96 dimensions), the EPV of 0.58 precludes reliable inference; the observed performance gap between handcrafted and embedding features should therefore be interpreted as hypothesis-generating rather than confirmatory. Bootstrap resampling provides internal validation, but prediction instability at the individual level may be substantial (Riley et al. 2023), particularly for patients near clinical decision thresholds."

### 5.2 External Validation Paragraph (MANDATORY)

> "Second, all results reflect internal validation only. External validation in independent cohorts—ideally from different clinical settings, ethnicities, and device manufacturers—is essential before any clinical deployment could be considered. The single-center (Singapore National Eye Centre), predominantly Chinese-ethnicity population limits generalizability of calibration and discrimination estimates to other populations."

**AUTHOR ADDITION**: This study's core message is NOT about the classifier, but about the importance of preprocessing choices and uncertainty propagation for these pipelines. Often papers discuss downstream performance and brush input data uncertainty under the rug. As an example of this pattern in PLR literature, cite:
> Mure LS et al. (2009). "Melanopsin Bistability: A Fly's Eye Technology in the Human Retina." PLoS ONE 4(6): e5991.

### 5.3 Framing the FM Preprocessing Claim (RECOMMENDED)

> "Despite these limitations, our proof-of-concept study demonstrates that generic time-series foundation models can achieve preprocessing quality competitive with traditional methods specifically designed for biosignal artifact removal. The finding that MOMENT zero-shot performance approaches that of trained imputation methods (SAITS) suggests foundation models may offer practical benefits for PLR analysis pipelines, warranting validation in larger, multi-center studies."

### 5.4 Acknowledging the Calibration Problem (RECOMMENDED)

> "The calibration slope of 0.52 for our best-performing configuration indicates systematic overconfidence in model predictions, as expected for small development samples. While discrimination (AUROC = 0.91) appears strong, the model's predicted probabilities are miscalibrated and reflect the inherent limitations of small-sample classifier development. We explicitly chose not to apply post-hoc recalibration techniques (temperature scaling, isotonic regression) as such corrections would mask the fundamental sample size limitation."

---

## Part II: PLR in the Biosignal Foundation Model Landscape

### 2.1 PLR as an Autonomic Nervous System Window

The pupillary light reflex (PLR) occupies a unique position among biosignals:

| Biosignal | Primary System | Typical Frequency | Sample Rate | Foundation Models |
|-----------|---------------|-------------------|-------------|-------------------|
| ECG | Cardiac electrical | 0.05-50 Hz | 250-500 Hz | ECG-FM, MIRA, etc. (millions of samples) |
| PPG | Cardiovascular | 0.5-2 Hz | 25-125 Hz | PPG-GPT, PaPaGei (millions of samples) |
| EEG | Cortical electrical | 0.5-100 Hz | 256-1000 Hz | LaBraM, NeuroLM (millions of samples) |
| **PLR** | **Integrated ANS** | **<0.5 Hz** | **30-120 Hz** | **None specialized (hundreds of samples)** |

### 2.2 Creating a PLR Foundation Model - Validation Strategy

**AUTHOR NOTE**: For a theoretical PLR foundation model:

**Training data**: Raw PLR signals with all artifacts (blinks, noise, tracking failures)
**Validation data**: Hand-annotated dataset with ground truth outlier masks and imputed signals
**Self-supervised objective**: Learn to reconstruct "clean" PLR from "noisy" input without explicit labels
**Downstream validation**: Use labeled subset (N=208) to evaluate classification impact

This approach allows:
1. Leveraging ALL 507 subjects for pretraining (not just labeled ones)
2. Using ~1M timepoints for learning temporal dynamics
3. Reserving ground truth annotations for validation only

### 2.3 The Generalist vs Specialist Tradeoff

Our finding aligns with recent literature:

| Study | Signal | Finding |
|-------|--------|---------|
| Alchieri et al. 2025 | EDA | Generalist FMs "cannot outperform existing approaches" |
| Kataria et al. 2025 | PPG | Specialist 27% higher win score in full fine-tuning |
| **Our study** | PLR | MOMENT competitive for preprocessing, not classification |

**Pattern**: Foundation models excel at reconstruction (generic temporal understanding) but struggle with classification (domain-specific biomarkers).

### 2.4 Dimensionality Reduction - Critical Future Work

**AUTHOR CAVEAT**: We cannot derive generalizable insights about the embedding gap without systematic dimensionality reduction study.

**What's needed**:
- Compare MOMENT and other FMs (e.g., MIRA when available)
- Test various dimensionality reduction methods (PCA, UMAP, t-SNE, NMF)
- Compare at matched feature lengths: 4, 8, 16, 32, 64, 96
- Especially compare at 8 features (same as handcrafted)

This could become its own study - we explicitly did not do this systematic exploration.

---

## Part III: References

### Sample Size and EPV

| Citation | Year | Key Contribution |
|----------|------|------------------|
| Riley RD et al. | 2020 | "Calculating the sample size required for developing a clinical prediction model." BMJ 368:m441 |
| Peduzzi P et al. | 1996 | Classic EPV >= 10 rule |
| van Smeden M et al. | 2019 | EPV necessary but not sufficient |

### Model Instability

| Citation | Year | Key Contribution |
|----------|------|------------------|
| Riley RD et al. | 2023 | "Clinical prediction models and the multiverse of madness." BMC Medicine |
| Rhodes SA et al. | 2025 | pminternal R package |

### STRATOS and Reporting

| Citation | Year | Key Contribution |
|----------|------|------------------|
| Van Calster B et al. | 2024 | Performance measures for predictive AI (STRATOS TG6) |
| Collins GS et al. | 2024 | TRIPOD+AI reporting checklist |

### PLR Literature

| Citation | Year | Key Contribution |
|----------|------|------------------|
| Mure LS et al. | 2009 | Example of PLR study not accounting for preprocessing uncertainty |
| Najjar RP et al. | 2023 | Source dataset for this study |

---

## Appendix A: EPV Calculations

### Exact Calculations for Foundation PLR

```
N_total = 208 (labeled subjects)
N_events = 56 (glaucoma cases)
N_non_events = 152 (controls)

Feature set (featuresSimple.yaml):
  - 4 amplitude bins × 2 colors = 8 features
  - EPV = 56/8 = 7.0

MOMENT embeddings:
  - 96 dimensions
  - EPV = 56/96 = 0.58
```

**AUTHOR CONFIRMATION**: EPV = 7.0 is the correct value. There is no range.

---

*This addendum provides STRATOS-compliant guidance for honestly reporting the limitations of the Foundation PLR study given its sample size constraints, with author annotations integrated in v2.*
