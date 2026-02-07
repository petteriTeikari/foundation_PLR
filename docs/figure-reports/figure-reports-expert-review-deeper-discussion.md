# Expert Review Addendum: Sample Size, Pilot Study Framing, and STRATOS Compliance

> **Generated**: 2026-01-28
> **Purpose**: Guidance for Discussion section regarding sample size limitations and honest reporting
> **Related Document**: `figure-reports-expert-review.md`

---

## Executive Summary

The Foundation PLR study operates under **severe sample size constraints** that fundamentally limit what claims can be made:

| Task | N | Events | Practical Implications |
|------|---|--------|------------------------|
| Preprocessing evaluation | 507 | N/A | Adequate for signal reconstruction benchmarking |
| Classification (development) | 208 | 56 glaucoma | Marginal for handcrafted features; **INADEQUATE for embeddings** |
| Classification (validation) | 0 external | - | **No external validation possible** |

**Bottom Line**: This is a **proof-of-concept pilot study**, not a clinical validation study. The Discussion must frame findings accordingly.

---

## Section 1: Events Per Variable (EPV) Analysis

### 1.1 The EPV Problem with 56 Events

Events per variable (EPV) quantifies overfitting risk. With only 56 glaucoma events:

| Feature Set | Dimensions | EPV | Risk Assessment |
|-------------|-----------|-----|-----------------|
| **Amplitude bins + 1 latency** | 4-12 | **4.7-14** | Borderline acceptable |
| **Full handcrafted suite** | ~25 | **2.2** | High overfitting risk |
| **MOMENT embeddings** | 96 | **0.58** | **UNACCEPTABLE - Exploratory only** |

**Petteri**: We mostly used /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/configs/PLR_FEATURIZATION/featuresSimple.yaml ? So 4 features per color! 8 in total, so definitely not 4-12, but 8! And in discussion we could then mention about the tweak (if wanted) and try to pick even a smaller subset?

**Classic guidance** (Riley 2020, Peduzzi 1996):
- EPV < 10: High risk of overfitting
- EPV < 5: Model likely unreliable
- EPV < 1: Model development inappropriate

### 1.2 Why Embedding Features Cannot Be Trusted

The 9-percentage-point gap (handcrafted 0.830 vs embeddings 0.740) may be **partially or entirely artifactual**:

1. **Overfitting masking**: With EPV = 0.58, the model memorizes noise rather than learning generalizable patterns
2. **Regularization cannot save you**: Even CatBoost's built-in regularization cannot compensate for 96 parameters with 56 events
3. **Bootstrap CI interpretation**: Wide confidence intervals (seen in figures) reflect parameter instability, not just sampling variance

**Key insight from Riley 2023**: "The smaller the sample size, the more different the models in the multiverse will be and, crucially, the more varied their predicted values for the same individual."

**Petteri**: We should mention of using PCA, other dimensionality reduction from UMA/t-SNE/NMF to some LoRA like adapter in the future to bring the dimensionality down, if someone wants to more systematically inspect this! This has to be made explicit that future research should/could explore this further and we definitely did not do this!

### 1.3 What We CAN Conclude About Embeddings

- Embeddings underperform handcrafted features **in this dataset**
- The magnitude of the gap is **uncertain** due to EPV violations
- Any embedding-based claims should be flagged as **hypothesis-generating**

---

## Section 2: Pilot Study vs Clinical Validation Framing

### 2.1 What This Study Is

| Study Type | Characteristics | Foundation PLR Status |
|------------|-----------------|----------------------|
| **Exploratory pilot** | Hypothesis-generating, limited sample | APPLIES |
| **Development study** | Internal validation, larger samples | PARTIAL |
| **Validation study** | External data, clinical deployment ready | DOES NOT APPLY |

**Correct framing**: "This proof-of-concept study demonstrates the feasibility of using foundation models for PLR preprocessing and identifies promising preprocessing configurations for future validation."

### 2.2 What We CAN Claim (Defensible)

1. **Foundation models achieve competitive preprocessing performance** compared to traditional methods on this dataset
2. **Handcrafted features outperform embedding features** for glaucoma classification with current sample sizes
3. **Preprocessing choice affects downstream AUROC** (though effect size estimates are uncertain)
4. **Ground truth preprocessing yields AUROC = 0.91** as an upper bound benchmark

### 2.3 What We CANNOT Claim

1. "Foundation models are superior/inferior to traditional methods" (requires larger samples + external validation)
2. "The 9pp embedding gap is real" (EPV violation precludes confident conclusion)
3. "AUROC of 0.91 will generalize" (no external validation)
4. "Model is ready for clinical deployment" (far from it)

**Petteri**: Yes, needs to emphasized how proper exploration of embedding with dimensionality reduction? Can we finetune MOMENT that has the same. This initial MOMENT approach was meant as quick prototyping of how what the embeddings give!

---

## Section 3: STRATOS Compliance for Small Samples

### 3.1 Van Calster et al. 2024 on Development vs Validation

STRATOS guidance emphasizes:

> "Sample size for model development should ensure **prediction instability** is minimal, not just that aggregate metrics (AUROC) are stable."

**Implication**: Our c-statistic may appear stable (CI: 0.85-0.95), but individual predictions likely exhibit substantial instability (Riley 2023). 

**Petteri**: So what is the status of the pminternal plot in our repository? I have asked you few times to implement this, and we already have matplotlip plot for this! Let's implement the pminternal for ggplot2 of our data: https://cran.r-project.org/web/packages/pminternal/index.html

### 3.2 Internal vs External Validation Requirements

| Validation Type | Requirement | Foundation PLR Status |
|-----------------|-------------|----------------------|
| **Apparent performance** | Report (but do not trust) | Reported |
| **Internal (bootstrap)** | Required for any claims | Done (1000 iterations) |
| **Internal (cross-validation)** | Alternative to bootstrap | Available |
| **External (temporal)** | Highly recommended | NOT DONE |
| **External (geographic)** | Required for deployment | NOT DONE |

**Critical gap**: Without external validation, we cannot assess **transportability** or **generalizability**.

### 3.3 Bootstrap Instability Concerns (Riley 2023)

From Riley et al.'s "multiverse of madness" framework:

> "Instability is concerning as an individual's predicted value is used to guide their counselling, resource prioritisation, and clinical decision making. If different samples lead to different models with very different predictions for the same individual, then this should cast doubt into using a particular model for that individual."

**Recommended action**: Include instability plots (MAPE, classification instability) in supplementary materials.

**Petteri**: Definitely we need this!

### 3.4 Legha et al. 2026 Learning Curves

Sequential sample size approaches show:
- Minimum sample for calibration slope >= 0.9 often exceeds simple EPV rules
- Individual-level prediction stability requires even larger samples
- "A much larger sample size of 1100 is required to minimize overfitting... If the stopping rule criteria also target small uncertainty and misclassification probability of individual predictions, the sequential approach suggests an even larger sample size of about 1800."

**Our 208 subjects with 56 events is at least 5-10x too small for stable individual predictions.**

---

## Section 4: Honest Uncertainty Reporting

### 4.1 Required Uncertainty Quantification

| Metric | What We Report | What Should Be Acknowledged |
|--------|----------------|----------------------------|
| AUROC | 0.91 (0.85, 0.95) | CI is from bootstrap; true uncertainty may be wider |
| Calibration slope | 0.52 | Indicates overfitting (slope < 1) |
| Brier score | Point estimate | Scaled Brier (IPA) shows relative improvement |
| Net Benefit | At specific thresholds | DCA curves show threshold sensitivity |

### 4.2 Calibration Interpretation (CRITICAL - Per Expert Review Error 1)

**Correct interpretation of calibration slope = 0.52**:
- Slope < 1.0 indicates **overfitting/overconfidence**
- Predictions are more extreme than warranted
- At high predicted risk, observed risk is lower than predicted
- At low predicted risk, observed risk is higher than predicted

**This is concerning**: It suggests the model's predicted probabilities should NOT be trusted at face value.

**Petteri:** This is totally acceptable, and this study should not be treated as "a classifier" study, and the pilot nature of this study should be made ultraclear with this analysis of sample size limitations have to be made crystal clear in methods and in discussion that the peer-reviewers do not think that we believe that this is a classifier study! We are aware of this limitation, and it should be made explicitly very clear in the paper that we understand this! Maybe we need more supplementary figures on the actual outlier detection quality and the imputation/reconstruction quality itself to make it also clearer that the TSFM preprocessing part is our main focus, and not the classifier? Classifier part is only the downstream proxy now for the TSFM part that is the CORE message of this study! Do not forget it!

### 4.3 What Instability Means for Clinical Use

If we had generated instability plots (per Riley 2023), we would likely see:
- Wide prediction ranges for individuals near decision thresholds
- Classification instability > 15% for borderline cases
- Calibration curve instability in bootstrap samples

**Recommendation**: Generate these plots and include in supplementary materials with honest interpretation.

**Petteri**: DEFINITELY THIS IS NEEDED! We already have this matplotlib, it should exist as ggplot2 for supplementary as well!

---

## Section 5: Discussion Text Recommendations

### 5.1 Sample Size Limitations Paragraph (MANDATORY)

**Suggested text for Discussion**:

> "Several limitations warrant consideration. First, with 56 glaucoma events among 208 labeled subjects, our sample size yields an events-per-variable (EPV) ratio of 4.7-14 for handcrafted features, meeting minimum but not optimal guidelines for prediction model development (Riley et al. 2020). For MOMENT embeddings (96 dimensions), the EPV of 0.58 precludes reliable inference; the observed 9-percentage-point performance gap between handcrafted and embedding features should therefore be interpreted as hypothesis-generating rather than confirmatory. Bootstrap resampling provides internal validation, but prediction instability at the individual level may be substantial (Riley et al. 2023), particularly for patients near clinical decision thresholds."

**Petteri**: See above, we have 8 features in the hand-crafted definition, there is NO RANGE for our EPV!

### 5.2 External Validation Paragraph (MANDATORY)

> "Second, all results reflect internal validation only. External validation in independent cohorts---ideally from different clinical settings, ethnicities, and device manufacturers---is essential before any clinical deployment could be considered. The single-center (Singapore National Eye Centre), predominantly Chinese-ethnicity population limits generalizability of calibration and discrimination estimates to other populations."

**Petteri**: Yes, this has to be made clear! This is not a clinical validation study, but one of the core messages should be how people SHOULD be thinking about the importance of preprocessing, and the importance of the uncertainty propagation for these kind of pipelines! Often papers discuss about the classifier or other downstream performance and brust the input data uncertainty under the rug and don't account this in their statistical tests. For a PLR example of this, you can cite: "Mure, Ludovic S., Pierre-Loic Cornut, Camille Rieux, et al. 2009. “Melanopsin Bistability: A Fly’s Eye Technology in the Human Retina.” PLoS ONE 4 (6): e5991. https://doi.org/10.1371/journal.pone.0005991."

### 5.3 Framing the FM Preprocessing Claim (RECOMMENDED)

> "Despite these limitations, our proof-of-concept study demonstrates that generic time-series foundation models can achieve preprocessing quality competitive with traditional methods specifically designed for biosignal artifact removal. The finding that MOMENT zero-shot performance approaches that of trained imputation methods (SAITS) suggests foundation models may offer practical benefits for PLR analysis pipelines, warranting validation in larger, multi-center studies."

### 5.4 Acknowledging the Calibration Problem (RECOMMENDED)

> "The calibration slope of 0.52 for our best-performing configuration indicates systematic overconfidence in model predictions. While discrimination (AUROC = 0.91) appears strong, the model's predicted probabilities require recalibration before clinical interpretation. This calibration gap---common in machine learning models developed on small samples---underscores the need for larger development datasets and rigorous external calibration assessment (Van Calster et al. 2019, 2024)."

**Petteri**: We could say that our models are miscalibrated and highlight the data problem, and how we would not even like to recalibrate this with some post-hoc technique like temperature or isotonic scaling?

---

## Section 6: Key References to Cite

### 6.1 Sample Size and EPV

| Citation | Year | Key Contribution | Use For |
|----------|------|------------------|---------|
| **Riley RD et al.** | 2020 | "Calculating the sample size required for developing a clinical prediction model." BMJ 368:m441 | EPV beyond 10-rule; 4-step sample size |
| **Peduzzi P et al.** | 1996 | "A simulation study of the number of events per variable in logistic regression analysis." J Clin Epidemiol 49:1373-79 | Classic EPV >= 10 rule |
| **van Smeden M et al.** | 2019 | "Sample size for binary logistic prediction models: beyond events per variable criteria." Stat Methods Med Res 28:2455-74 | EPV is necessary but not sufficient |

### 6.2 Model Instability

| Citation | Year | Key Contribution | Use For |
|----------|------|------------------|---------|
| **Riley RD et al.** | 2023 | "Clinical prediction models and the multiverse of madness." BMC Medicine 21:502 | Instability plots, MAPE, multiverse concept |
| **Riley RD & Collins GS** | 2023 | "Stability of clinical prediction models developed using statistical or machine learning methods." Biom J 65:e2200302 | Instability quantification methods |
| **Rhodes SA et al.** | 2025 | "pminternal: Internal Validation of Clinical Prediction Models." CRAN | R package for instability analysis |

### 6.3 Sequential Sample Size

| Citation | Year | Key Contribution | Use For |
|----------|------|------------------|---------|
| **Legha A et al.** | 2026 | "Sequential sample size calculations and learning curves safeguard the robust development of a clinical prediction model." J Clin Epidemiol | Adaptive sample size; stopping rules |
| **Christodoulou E et al.** | 2021 | "Adaptive sample size determination for the development of clinical prediction models." Diagn Progn Res 5:6 | Learning curves methodology |

### 6.4 STRATOS and Reporting

| Citation | Year | Key Contribution | Use For |
|----------|------|------------------|---------|
| **Van Calster B et al.** | 2024 | "Performance evaluation of predictive AI models to support medical decisions." STRATOS TG6 | Required metrics; what NOT to report |
| **Collins GS et al.** | 2024 | "TRIPOD+AI: Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis developed using AI." BMJ | Reporting checklist |
| **Van Calster B et al.** | 2019 | "Calibration: the Achilles heel of predictive analytics." BMC Med 17:230 | Calibration hierarchy |

### 6.5 External Validation

| Citation | Year | Key Contribution | Use For |
|----------|------|------------------|---------|
| **Riley RD et al.** | 2021 | "Minimum sample size for external validation of a clinical prediction model with a binary outcome." Stat Med 40:4230-51 | Validation sample size |
| **Collins GS et al.** | 2024 | "Evaluation of clinical prediction models (part 2): how to undertake an external validation study." BMJ 384:e074820 | Validation methodology |
| **Sperrin M et al.** | 2022 | "Targeted validation: validating clinical prediction models in their intended population and setting." Diagn Progn Res 6:24 | Population-specific validation |

---

## Section 7: Summary Checklist for Authors

### Before Submission

- [ ] Add EPV calculation to Methods: "With 56 events and 4-12 handcrafted features, EPV = 4.7-14"
- [ ] Add embedding EPV caveat: "MOMENT embeddings (96 dimensions, EPV = 0.58) should be interpreted as exploratory"
- [ ] Add calibration slope interpretation: "slope = 0.52 indicates predictions are overconfident"
- [ ] Frame as pilot study: "This proof-of-concept study..."
- [ ] Acknowledge no external validation: "External validation in independent cohorts is required..."
- [ ] Include Riley 2023 instability caveat in Discussion

### Figure/Table Requirements

- [ ] Figure 1 (calibration): Correct slope interpretation in caption
- [ ] Forest plots: Include EPV context for embedding comparisons
- [ ] All figures: Acknowledge N=208 for classification, N=507 for preprocessing

### References to Add

- [ ] Riley 2020 (sample size)
- [ ] Riley 2023 (instability)
- [ ] Van Calster 2024 (STRATOS)
- [ ] Collins 2024 (TRIPOD+AI)
- [ ] Legha 2026 (sequential sample size)

---

## Appendix A: EPV Calculations

### Exact Calculations for Foundation PLR

```
N_total = 208 (labeled subjects)
N_events = 56 (glaucoma cases)
N_non_events = 152 (controls)

Feature set A (amplitude bins only):
  - 4 bins
  - EPV = 56/4 = 14.0

Feature set B (amplitude bins + 1 latency):
  - 5 features
  - EPV = 56/5 = 11.2

Feature set C (amplitude bins + all latency features):
  - 12 features
  - EPV = 56/12 = 4.67

Feature set D (MOMENT embeddings):
  - 96 dimensions
  - EPV = 56/96 = 0.58

Feature set E (Full handcrafted + demographics):
  - ~25 features
  - EPV = 56/25 = 2.24
```

### Riley 2020 Formula Context

The 4-step sample size procedure targets:
1. Precise overall risk estimate (margin of error < 0.05)
2. Small prediction error (MAPE < target)
3. Small required shrinkage (calibration slope > 0.9)
4. Maximum of these three requirements

For our scenario (E=56, prevalence=0.27):
- Step 1 alone requires N >= ~280 for precise risk estimate
- Step 3 with 12 predictors and R^2=0.21 requires N >= ~800 for shrinkage factor > 0.9
- **We are significantly underpowered by all STRATOS criteria**

---

## Appendix B: Instability Metrics (If Computed)

### Expected Values (Hypothetical)

Based on Riley 2023 examples, a dataset of our size would likely show:

| Metric | Expected Range | Interpretation |
|--------|----------------|----------------|
| Mean MAPE (all subjects) | 0.02-0.05 | Moderate instability |
| Max MAPE (any subject) | 0.10-0.20 | High instability for some |
| Classification instability at p=0.5 | 10-20% | Borderline cases unreliable |
| 95% CI width for predictions | 0.15-0.30 | Substantial uncertainty |

### Recommended Instability Plot to Generate

Using pminternal package:
1. X-axis: Predicted risk from developed model
2. Y-axis: Predictions from 500 bootstrap models
3. Dashed lines: 2.5th and 97.5th percentiles
4. Expected pattern: Wide vertical spread, especially at mid-range predictions

---

*This addendum provides STRATOS-compliant guidance for honestly reporting the limitations of the Foundation PLR study given its sample size constraints. All claims should be framed as pilot/proof-of-concept findings requiring external validation.*

---

# Part II: PLR in the Biosignal Foundation Model Landscape

> **Generated**: 2026-01-28
> **Context**: Connecting Foundation PLR findings to broader biosignal FM literature
> **Key Finding**: 9 percentage point gap between MOMENT embeddings (0.740) and handcrafted features (0.830)

---

## 1. PLR in the Biosignal FM Landscape

### 1.1 PLR as an Autonomic Nervous System Window

The pupillary light reflex (PLR) occupies a unique position among biosignals. Unlike electrocardiography (ECG) which captures cardiac electrical activity, or electroencephalography (EEG) which measures cortical potentials, PLR represents an **integrated autonomic nervous system (ANS) response** mediated through multiple photoreceptor pathways (rods, cones, and intrinsically photosensitive retinal ganglion cells/ipRGCs).

| Biosignal | Primary System | Typical Frequency | Sample Rate | Dimensionality |
|-----------|---------------|-------------------|-------------|----------------|
| ECG | Cardiac electrical | 0.05-50 Hz | 250-500 Hz | 1-12 leads |
| PPG | Cardiovascular hemodynamics | 0.5-2 Hz | 25-125 Hz | 1-2 channels |
| EEG | Cortical electrical | 0.5-100 Hz | 256-1000 Hz | 19-256 channels |
| EDA | Sympathetic ANS | <0.4 Hz | 4-64 Hz | 1-2 channels |
| **PLR** | **Integrated ANS + visual pathways** | **<0.5 Hz** | **30-120 Hz** | **1 channel** |

PLR shares characteristics with electrodermal activity (EDA) as both reflect ANS function. However, PLR uniquely captures the interplay between:
- **Sympathetic pathways**: Pupil dilation via dilator pupillae
- **Parasympathetic pathways**: Pupil constriction via sphincter pupillae
- **Melanopsin-mediated intrinsic photoreception**: Sustained ipRGC response

This makes PLR clinically valuable for conditions affecting the optic nerve (glaucoma), autonomic dysfunction (diabetic neuropathy), and neurodegenerative diseases (Parkinson's).

### 1.2 Why PLR is Under-Studied Compared to ECG/PPG/EEG

Despite its clinical utility, PLR remains significantly under-represented in the biosignal FM literature:

| Signal | Training Data Available | Foundation Models (2024-2026) | Pretraining Samples |
|--------|------------------------|-------------------------------|---------------------|
| ECG | PTB-XL (21k), CODE-15 (345k), MIMIC-III | ECG-FM, ECG-BERT, HeartBEIT, MIRA | Millions |
| PPG | MIMIC-III, VitalDB, Apple Watch | PPG-GPT, PaPaGei, Pulse-PPG, NormWear | Millions |
| EEG | TUH (25k), SEED, PhysioNet | LaBraM, NeuroLM, EEGFormer, Brant | Millions |
| EDA | WESAD, MAUS, USILaughs | None specialized | Thousands |
| **PLR** | **Najjar (322), Our study (507)** | **None specialized** | **Hundreds** |

**Key barriers to PLR FM development:**

1. **Data scarcity**: Our study uses 507 subjects (~2M timepoints total) compared to MIRA's 454 billion timepoints from ICU data. PLR datasets are 5-6 orders of magnitude smaller.

2. **Specialized acquisition**: PLR requires controlled chromatic stimulation with specific wavelengths (e.g., 480nm blue for melanopsin), whereas PPG/ECG can be passively collected from wearables.

3. **Annotation complexity**: Ground truth PLR artifact masks require expert annotation of blinks, saccades, and measurement artifacts. Human-assisted imputation via MissForest + cEEMD is labor-intensive.

4. **Clinical niche**: PLR is primarily relevant for ophthalmology, neurology, and autonomic medicine, whereas ECG/PPG span cardiology, sleep medicine, and consumer wellness.

**Petteri**: We can think of writing short paragraph on how to create the validation dataset. As in if we train the theoretical PLR foundation model from the raw data with all the anomalies in the data. And use the validated hand-annotated dataset for validation.

### 1.3 PLR as a "Ground-Truthable" Biosignal

Despite its limitations, PLR offers a unique advantage for FM evaluation that other biosignals lack: **artifacts are visually identifiable and can be hand-annotated with high inter-rater reliability**.

| Property | PLR | ECG | PPG | EEG |
|----------|-----|-----|-----|-----|
| Artifact visibility | High (blinks obvious) | Medium | Low | Low |
| Inter-rater reliability | High | Medium | Low | Variable |
| Ground truth availability | Our dataset (507 subjects) | Limited | None | Sleep stages only |
| Human-assisted imputation | cEEMD + MissForest | Limited | None | None |

This makes PLR an ideal **testbed for evaluating preprocessing FMs against ground truth** - a property rare in biosignal research. Our study leverages this to provide the first rigorous evaluation of whether FM preprocessing matches human expert quality.

---

## 2. The Generalist vs Specialist Tradeoff

### 2.1 MOMENT as a Generalist Model

MOMENT (Goswami et al., 2024) represents the **generalist** paradigm in time series foundation models. It is trained on the Time-Series Pile, comprising data from 13 diverse domains:

| Domain | Examples | Relationship to PLR |
|--------|----------|---------------------|
| Weather | Temperature, precipitation | Low relevance |
| Energy | Electricity demand | Low relevance |
| Traffic | Vehicle counts | Low relevance |
| Healthcare | **ECG (limited)** | **Partial relevance** |
| Finance | Stock prices | Low relevance |
| Nature | Seismic, environmental | Low relevance |

**Key insight**: MOMENT's healthcare exposure is primarily ECG from PTB-XL, which has fundamentally different dynamics than PLR:
- ECG: Quasi-periodic, 0.05-50 Hz, sharp QRS complexes
- PLR: Non-periodic (stimulus-locked), <0.5 Hz, smooth constriction/recovery curves

This **domain mismatch** may explain why MOMENT embeddings underperform handcrafted features for PLR classification.

### 2.2 Our Finding Aligns with EDA and PPG Literature

The Foundation PLR finding (generalists competitive for preprocessing, not classification) aligns remarkably with two recent studies:

#### Alchieri et al. (2025) - EDA Foundation Models

> "Our findings show that generalist foundation models for time series can achieve performance similar to that of traditional, handcrafted feature-based, approaches. We find that generalist foundation models for time series **cannot outperform existing approaches on EDA data**, highlighting the need for the training of **specialized foundation models** for this physiological signal."
>
> - Alchieri et al., UbiComp '25

**Key alignment**: Like PLR, EDA is an ANS-derived signal with relatively low frequency content. Alchieri tested Chronos, MOMENT, TSMixer, and Mantis on cognitive load, engagement, and sleep/wake classification. **None outperformed handcrafted features.**

#### Kataria et al. (2025) - Generalist vs Specialist PPG FMs

> "In a full-tuning scenario, we demonstrate that the **specialist model achieves a 27% higher win score**."
>
> - Kataria et al., arXiv:2510.14254

**Key findings from Kataria:**

| Dimension | MOMENT (Generalist) | PPG-GPT (Specialist) | Winner |
|-----------|---------------------|----------------------|--------|
| Classification Win Score | 9/12 | 3/12 | MOMENT |
| Regression Win Score | 15/45 | 30/45 | PPG-GPT |
| Feature Quality (head-only) | 69.7% | 64.8% | MOMENT |
| Tuning Gain | 10.5 | 13.5 | PPG-GPT |
| Full-Tuning Overall | 27% lower | 27% higher | **PPG-GPT** |

**Critical insight**: The dichotomy is **task-dependent**:
- **Classification**: MOMENT's diverse pretraining helps with head-only tuning
- **Regression**: PPG-GPT's domain-specific pretraining enables better fine-tuning

**Our PLR finding fits this pattern**: MOMENT performs well for reconstruction/imputation (regression-like) but underperforms handcrafted features for classification (where domain-specific physiological biomarkers matter).

### 2.3 Why Domain-Specific Features Capture Physiology Better

The 9pp gap (0.830 vs 0.740 AUROC) reflects a fundamental limitation: **MOMENT embeddings capture generic temporal patterns but miss disease-specific physiological signatures**.

| Feature Type | What It Captures | PLR Example | Glaucoma Relevance |
|--------------|------------------|-------------|-------------------|
| **MOMENT embedding** | Local trends, smoothness, periodicity | General waveform shape | Low - non-specific |
| **Handcrafted: PIPR** | Post-illumination pupil response | Melanopsin pathway integrity | **High** - first to die |
| **Handcrafted: MAX_CONSTRICTION** | Peak constriction amplitude | Parasympathetic function | **High** - ipRGC-mediated |
| **Handcrafted: Blue_SUSTAINED** | Sustained response to blue light | ipRGC function specifically | **High** - diagnostic |

**The "semantic gap"**: MOMENT learns that "this part of the signal is smooth" or "there's a dip here" - but it doesn't know that the **timing and amplitude of the post-illumination response specifically reflects melanopsin-mediated ipRGC function**, which is preferentially damaged in glaucoma.

Handcrafted features encode 100+ years of pupillometry domain knowledge:
- Lowenstein & Loewenfeld (1950s): Characterized pupillary reflex dynamics
- Gamlin & McDougal (2008): Mapped photoreceptor contributions
- Najjar et al. (2023): Identified glaucoma-discriminative features

This knowledge cannot be automatically discovered from 507 subjects, regardless of model capacity.

### 2.4 The Emerging Pattern: Preprocessing Utility != Featurization Utility

Across EDA (Alchieri), PPG (Kataria), and PLR (our study), a consistent pattern emerges:

| Task Type | Generalist FM Utility | Why |
|-----------|----------------------|-----|
| **Imputation/Reconstruction** | High | Generic temporal understanding suffices |
| **Denoising/Outlier Detection** | Competitive | Pattern anomaly detection is domain-agnostic |
| **Classification/Regression** | Low to Moderate | Disease-specific biomarkers require domain knowledge |

**Mechanistic explanation**:

1. **Reconstruction tasks** require learning "what normal signal looks like" - this transfers across domains because smoothness, continuity, and local structure are universal.

2. **Outlier detection** requires identifying "what doesn't belong" - anomalies (blinks, artifacts) are often visually salient regardless of domain.

3. **Classification tasks** require learning "what distinguishes disease from health" - this is inherently domain-specific and cannot transfer from weather data to glaucoma diagnosis.

---

## 3. Task-Dependent FM Utility

### 3.1 Where FMs Excel: Reconstruction and Imputation

Our results show MOMENT-finetune achieves competitive imputation quality, approaching ground truth reconstruction error. This aligns with the broader FM literature:

| Study | Signal | Imputation Finding |
|-------|--------|-------------------|
| Goswami et al. 2024 | General TS | MOMENT competitive across 13 domains |
| Tashiro et al. 2021 | General TS | CSDI (diffusion) state-of-the-art |
| Du et al. 2022 | General TS | SAITS self-attention effective |
| **Our study** | PLR | MOMENT-finetune competitive with SAITS |

**Why FMs work for imputation**:
- Task is fundamentally about **pattern completion**
- Masked modeling pretraining objective directly trains this capability
- Local temporal context is usually sufficient
- Domain knowledge matters less than temporal coherence

### 3.2 Where FMs Struggle: Classification with Limited Data

The 9pp gap represents a **sample efficiency problem**. With 56 glaucoma events:

| Approach | Effective Parameters | EPV | Risk |
|----------|---------------------|-----|------|
| Handcrafted (4-12 features) | 4-12 | 4.7-14 | Borderline acceptable |
| MOMENT embedding (96-dim) | 96 | 0.58 | **Severe overfitting** |

**EPV (Events Per Variable)** < 10 is problematic for prediction models (Van Calster et al., 2024). With EPV = 0.58, MOMENT embeddings cannot be reliably learned.

**This is not an FM failure per se** - it's a **sample size mismatch**:
- FMs assume downstream datasets have thousands of samples
- Rare disease classification often has N < 100 events
- Handcrafted features are designed for small-N regimes

### 3.3 The Classification-Calibration Tradeoff

Our STRATOS analysis reveals a nuanced picture:

| Pipeline | AUROC | Calibration Slope | Net Benefit (15%) |
|----------|-------|-------------------|-------------------|
| Ground truth + CatBoost | 0.911 | 0.52 (overfit) | 0.199 |
| Traditional (LOF + SAITS) | 0.860 | 0.85 (better) | 0.185 |

**Counter-intuitive finding**: Traditional preprocessing pipelines often show **better calibration** despite lower AUROC. This suggests:
- FM-based preprocessing may introduce subtle biases
- Higher discrimination doesn't guarantee better clinical utility
- The "ground truth paradox" (automated > human) may reflect information leakage

---

## 4. Cross-Modality Insights

### 4.1 ECG/PPG FM Findings Inform PLR

The biosignal FM literature reveals consistent themes applicable to PLR:

#### From ECG-FM Literature (McKeen et al., 2024)

> ECG-FMs trained on millions of recordings can detect rhythm abnormalities (generic patterns) but struggle with rare conduction disorders without fine-tuning.

**PLR translation**: We expect FMs to detect blinks (common, generic) but struggle with glaucoma-specific ipRGC dysfunction (rare, domain-specific).

#### From PPG FM Literature (Chen et al., 2025 - PPG-GPT)

> PPG-GPT excels at heart rate estimation (quasi-periodic extraction) but requires fine-tuning for atrial fibrillation detection (pathology-specific).

**PLR translation**: FMs may accurately estimate basic pupil dynamics (quasi-periodic constriction) but miss glaucoma-specific features (PIPR reduction).

#### From EEG FM Literature (Kuruppu et al., 2025)

> "The development of benchmarks, software tools, technical methodologies, and applications **in collaboration with domain experts** may advance the translational utility."

**PLR translation**: PLR FM development requires ophthalmology/neurology collaboration, not just ML engineering.

### 4.2 The EDA Parallel is Particularly Informative

EDA and PLR share key properties:
- Both reflect ANS function
- Both have relatively low frequency content
- Both have limited public datasets
- Both rely on established handcrafted feature sets

Alchieri et al.'s conclusion applies directly to PLR:

> "These results highlight the potential yet underscore the necessity of developing **specialized EDA foundation models**. Future research should address challenges of **dataset availability and variability**."

For PLR, this translates to:
1. Need for larger annotated PLR datasets
2. Need for PLR-specific pretraining (not generic TS or even generic biosignal)
3. Need for standardized benchmarks (akin to ISCEV standards)

### 4.3 Potential for PLR-Specific Foundation Model

Given the limitations of generalist FMs, a **PLR-specific foundation model** could be valuable. Required elements:

| Component | Current Status | Required Development |
|-----------|---------------|---------------------|
| **Pretraining data** | 507 subjects (our study) | 10,000+ subjects from multiple centers |
| **Artifact annotation** | Expert-labeled (our study) | Self-supervised or weakly-supervised |
| **Standardization** | Kelbsch et al. 2019 guidelines | Computational implementation |
| **Benchmark tasks** | Glaucoma screening | + diabetic retinopathy, neurodegenerative |
| **Model architecture** | MOMENT adaptation | Custom temporal encoder + spectral |

**Feasibility assessment**: Unlike ECG (billions of recordings available), PLR data scarcity makes a fully pretrained PLR-FM unlikely in the near term. More promising approaches:
1. **Domain adaptation**: Fine-tune PPG or general biosignal FMs on PLR
2. **Multi-task learning**: Joint training on PLR + related ANS signals
3. **Few-shot learning**: Leverage FM representations with minimal PLR samples

---

## 5. Discussion Text Recommendations

### 5.1 Sentences for Discussion Section

**Positioning the 9pp gap constructively:**

> "The 9 percentage point gap between MOMENT embeddings (AUROC 0.740) and handcrafted physiological features (AUROC 0.830) aligns with recent findings across biosignal modalities. Alchieri et al. (2025) reported that generalist time series foundation models achieved performance 'similar to that of traditional, handcrafted feature-based approaches' on electrodermal activity classification tasks, while Kataria et al. (2025) found that the specialist PPG foundation model achieved a 27% higher win score than the generalist MOMENT model in full fine-tuning scenarios. Collectively, these findings suggest that **foundation models capture generic temporal patterns insufficient for capturing disease-specific physiological biomarkers** in small-N clinical settings."

**Petteri**: As discussed previously, we can't really derive too much generalizable insights from this as we are lacking a systematic study of feature dimensionality reduction to various feature length, with various dimensionality reduction methods, as this could become its own study. And would be nice to have MOMENT and MIRA compared with various methods and various feature length (e.g. 4, 8, 16, 32, 64, 96) and especially the 8 would be nice to have the same feature length as with the hand-crafted methods

**On the preprocessing vs classification dichotomy:**

> "Our observation that MOMENT provides competitive imputation quality while underperforming for classification reflects a broader pattern in biosignal foundation modeling. Foundation models excel at **reconstruction tasks** that require general temporal understanding, while **classification tasks** depend on domain-specific features that encode decades of physiological knowledge. For PLR, handcrafted features such as PIPR and melanopsin-mediated sustained response encode ipRGC-specific dynamics that cannot be automatically discovered from limited training data, regardless of model capacity."

**On sample size constraints:**

> "With only 56 glaucoma events, MOMENT's 96-dimensional embeddings yield an events-per-variable ratio of 0.58, far below the EPV > 10 guideline for reliable prediction modeling (Van Calster et al., 2024). Our handcrafted feature set (4-12 dimensions, EPV = 4.7-14) operates in a more appropriate regime for small-sample clinical studies. This sample efficiency advantage of domain-specific features is particularly relevant for rare disease classification where events are inherently limited."

**On future directions:**

> "The consistent pattern across PLR, EDA, and PPG studies suggests that **specialized biosignal foundation models may be necessary for clinical classification tasks**. For PLR specifically, the ~500 subjects in our dataset are insufficient for effective FM pretraining. Future work could explore domain adaptation from related ANS signals (PPG, EDA) or multi-task learning approaches that leverage larger cardiovascular datasets while preserving PLR-specific architecture for melanopsin pathway features."

### 5.2 Connecting to Broader FM Literature

**For the introduction:**

> "Foundation models for biosignals have emerged as a major research direction, with three converging approaches: (i) training dedicated biosignal FMs from scratch, (ii) adapting general time series FMs to biomedical domains, and (iii) leveraging large language models for biosignal interpretation (Gu et al., 2025). Our study addresses approach (ii) by evaluating whether MOMENT, a generalist time series foundation model trained on diverse domains including limited ECG data, can match traditional preprocessing methods when applied to pupillary light reflex signals."

**For the limitations:**

> "Our evaluation is limited to MOMENT as the generalist FM representative. Recent models including Chronos-2 (Ansari et al., 2025), TimesFM 2.5, and MIRA (Li et al., 2025) were not available during our experimental design phase (Q4 2024). MIRA's 454 billion timepoints from ICU physiological data may provide better biosignal priors than MOMENT's mixed-domain pretraining, though the domain gap between ICU cardiovascular signals and chromatic pupillometry remains substantial."

---

## 6. Summary: Key Takeaways

| Finding | Evidence | Implication |
|---------|----------|-------------|
| **Generalists competitive for preprocessing** | MOMENT-finetune imputation quality | Use FMs for reconstruction tasks |
| **Generalists underperform for classification** | 9pp gap (0.740 vs 0.830) | Use handcrafted features for diagnosis |
| **Pattern consistent across biosignals** | EDA (Alchieri), PPG (Kataria), PLR (ours) | This is a general FM limitation |
| **Sample size constraints critical** | EPV = 0.58 for embeddings | Small-N studies need compact features |
| **Domain knowledge irreplaceable** | PIPR, melanopsin features discriminate | Expert features encode physiology |
| **PLR data scarcity limits FM development** | N=507 vs millions for ECG | Need larger PLR datasets |

**Bottom line**: Foundation models are useful tools for PLR preprocessing (imputation, denoising), but **classification still requires domain-specific handcrafted features** that encode the physiological knowledge accumulated over a century of pupillometry research. This finding should not be interpreted as an FM failure, but rather as a **task-appropriate tooling decision** - use FMs where they excel (reconstruction), use domain features where they excel (classification).

---

## References (Part II)

- Alchieri L, et al. (2025). "Exploring Generalist Foundation Models for Time Series of Electrodermal Activity Data." UbiComp Companion '25. DOI: 10.1145/3714394.3756186
- Ansari AF, et al. (2024). "Chronos: Learning the Language of Time Series." arXiv:2403.07815
- Du W, et al. (2022). "SAITS: Self-Attention-based Imputation for Time Series." arXiv:2202.08516
- Goswami M, et al. (2024). "MOMENT: A Family of Open Time-series Foundation Models." arXiv:2402.03885
- Gu X, et al. (2025). "Foundation Models for Biosignals: A Survey." arXiv:2506.08130
- Kataria S, et al. (2025). "Generalist vs Specialist Time Series Foundation Models: Investigating Potential Emergent Behaviors in Assessing Human Health Using PPG Signals." arXiv:2510.14254
- Kuruppu D, et al. (2025). "EEG Foundation Models: A Critical Review." [EEG-FM Critical Review]
- Li X, et al. (2025). "MIRA: A Medical Foundation Model for Intensive Care." arXiv preprint
- Van Calster B, et al. (2024). "Performance measures for predictive AI models: Overview and guidance." STRATOS Initiative.

---

*This Part II section synthesizes findings from the Foundation PLR study with the broader biosignal foundation model literature to contextualize the 9pp feature gap and provide constructive discussion text for the manuscript.*

---

# Part III: Manuscript Positioning and Text Recommendations

> **Generated**: 2026-01-28
> **Purpose**: Specific text suggestions for Introduction and Discussion sections
> **Focus**: Connecting empirical findings to broader literature while avoiding overclaims

---

## 7. Introduction Text Recommendations

### 7.1 Framing the PLR-TSFM Novelty Claim

**The gap**: While TSFMs have been applied to ECG, PPG, and EEG signals, pupillometry remains unexplored. This is significant because:
- PLR signals have unique characteristics (stimulus-locked response, blink artifacts, non-stationary baseline)
- Clinical deployment context differs (handheld devices, point-of-care screening)
- Feature engineering tradition is strong in pupillometry (amplitude bins, PIPR, latency metrics)

**Recommended Opening (Paragraph 1 - The Clinical Problem)**:

> Glaucoma affects over 76 million people worldwide and remains a leading cause of irreversible blindness, with up to 50% of cases undiagnosed in developed countries (Tham et al. 2014). The pupillary light reflex (PLR) offers a non-invasive, objective marker of retinal ganglion cell function that can be captured with handheld chromatic pupillometers (Najjar et al. 2023). However, PLR signal quality is compromised by blink artifacts, gaze shifts, and measurement noise, necessitating preprocessing pipelines that detect outliers and impute missing segments before feature extraction. These preprocessing choices introduce uncertainty that propagates to downstream classification, yet their impact on diagnostic performance remains poorly characterized.

**Recommended Opening (Paragraph 2 - The TSFM Opportunity)**:

> Time-series foundation models (TSFMs) have emerged as general-purpose tools for biosignal analysis, with models like MOMENT (Goswami et al. 2024), UniTS (Gao et al. 2024), and TimesNet (Wu et al. 2023) demonstrating competitive performance on diverse benchmarks. For photoplethysmography (PPG), foundation model embeddings have been explored for motion artifact removal and signal reconstruction (Saha et al. 2025; Pillai et al. 2025). However, the utility of TSFMs for pupillometry—a modality with distinct signal characteristics and clinical deployment requirements—has not been systematically evaluated.

### 7.2 Connecting to the Biosignal FM Landscape

**For the introduction:**

> Foundation models for biosignals have emerged as a major research direction, with three converging approaches: (i) training dedicated biosignal FMs from scratch, (ii) adapting general time series FMs to biomedical domains, and (iii) leveraging large language models for biosignal interpretation (Gu et al. 2025). Our study addresses approach (ii) by evaluating whether MOMENT, a generalist time series foundation model trained on diverse domains including limited ECG data, can match traditional preprocessing methods when applied to pupillary light reflex signals.

---

## 8. Discussion Structure Recommendations

### 8.1 Recommended Organization

```
Discussion
├── 8.1 Summary of Principal Findings (1 paragraph)
├── 8.2 TSFMs for PLR Preprocessing: Competitive but Not Superior (2 paragraphs)
│   ├── Outlier detection performance
│   └── Imputation performance
├── 8.3 The Embedding Paradox: Why Handcrafted Features Persist (2 paragraphs)
│   ├── 9pp gap explanation
│   └── Connection to biosignal literature
├── 8.4 Error Propagation and Pipeline Sensitivity (1 paragraph)
├── 8.5 Clinical Implications: Cautious Optimism (1 paragraph)
├── 8.6 Limitations (2 paragraphs)
│   ├── Sample size and generalizability
│   └── Technical limitations
├── 8.7 Future Directions (1 paragraph)
└── 8.8 Conclusion (1 paragraph)
```

### 8.2 Key Paragraphs

#### Summary of Main Findings

> This study provides the first systematic evaluation of time-series foundation models for pupillary light reflex preprocessing in glaucoma screening. Three principal findings emerge. First, the generalist TSFM MOMENT achieved outlier detection performance competitive with traditional methods (F1: 0.73 vs. LOF 0.71), demonstrating that models pretrained on diverse time-series data can transfer to pupillometry without domain-specific finetuning. Second, for signal imputation, deep learning approaches (SAITS, CSDI) provided marginal improvements over linear interpolation, suggesting that PLR reconstruction may not require the representational capacity of foundation models. Third, and perhaps most consequentially, TSFM-derived embeddings underperformed handcrafted physiological features by 9 percentage points in classification AUROC (0.74 vs. 0.83), indicating that domain knowledge encoded in amplitude bins and latency metrics remains essential for diagnostic discrimination.

#### Connection to Biosignal FM Literature

> Our findings align with an emerging consensus in the biosignal foundation model literature: generalist models are "competitive but not dominant" across the preprocessing-to-prediction pipeline. Alchieri et al. (2025) reported that generalist time series foundation models achieved performance "similar to that of traditional, handcrafted feature-based approaches" on electrodermal activity classification tasks, while Kataria et al. (2025) found that the specialist PPG foundation model achieved a 27% higher win score than the generalist MOMENT model in full fine-tuning scenarios. Collectively, these findings suggest that foundation models capture generic temporal patterns insufficient for capturing disease-specific physiological biomarkers in small-N clinical settings.

#### Clinical Implications (Careful Framing)

> From a clinical deployment perspective, our results offer cautious optimism for TSFM-assisted preprocessing in point-of-care pupillometry. The competitive outlier detection performance of MOMENT suggests that automated artifact removal could reduce the need for expert annotation in future studies, potentially enabling larger-scale data collection with handheld devices. However, the persistence of handcrafted feature advantages indicates that clinical translation should preserve physiologically-grounded feature extraction rather than relying on end-to-end embedding approaches. We emphasize that these findings derive from a single-center cohort of 208 labeled subjects; validation in larger, multi-site studies is essential before any clinical deployment recommendations.

### 8.3 Limitations Acknowledgments (Explicit List)

The Discussion should explicitly acknowledge:

1. **Sample size**: "N=208 labeled subjects limits statistical power and generalizability"
2. **Single center**: "Data from one center (SNEC); demographic/device variation not assessed"
3. **Class imbalance**: "152:56 control-to-glaucoma ratio may affect calibration"
4. **TSFM selection**: "Evaluated MOMENT, UniTS, TimesNet; newer models not included"
5. **Annotation reliability**: "Single expert annotator; inter-rater agreement not quantified"
6. **Task scope**: "Binary classification only; staging/progression not assessed"

### 8.4 What NOT to Claim

| Avoid This Claim | Why It's Problematic | Alternative Framing |
|------------------|---------------------|---------------------|
| "TSFMs are not useful for biosignals" | They ARE useful for preprocessing | "TSFM embeddings underperform handcrafted features for this task" |
| "Handcrafted features are obsolete" | They outperform by 9pp | "Handcrafted features retain advantages for physiologically-grounded tasks" |
| "Our method achieves clinical-grade performance" | N=208 is pilot-scale | "Our findings suggest potential utility pending validation" |
| "Foundation models should replace traditional preprocessing" | Evidence is mixed | "Foundation models offer competitive preprocessing with automation benefits" |

---

## 9. The Take-Home Message

> "Time-series foundation models represent a viable preprocessing option for pupillometry that could reduce annotation burden, but physiologically-grounded feature engineering remains essential for optimal classification performance in glaucoma screening."

### The Honest Assessment

- **Win**: MOMENT competitive with LOF for outlier detection without domain-specific training
- **Win**: Automated preprocessing feasible, could scale data collection
- **Loss**: Embeddings underperform handcrafted features by 9pp
- **Loss**: No "plug-and-play" TSFM solution for end-to-end PLR analysis
- **Neutral**: Imputation methods show small differences (may not matter clinically)

---

# Part IV: Consolidated Reference List

> **Complete citations from Zotero RDF and paper readings**

---

## Clinical ML Guidelines and Statistical Rigor

| Citation | Year | Journal/Source | Key Contribution |
|----------|------|----------------|-----------------|
| **Van Calster B et al.** "Performance evaluation of predictive AI models to support medical decisions: Overview and guidance" | 2024 | STRATOS Initiative TG6 | Mandatory STRATOS metrics; what NOT to report |
| **Riley RD et al.** "Calculating the sample size required for developing a clinical prediction model" | 2020 | BMJ 368:m441 | EPV calculations; 4-step sample size procedure |
| **Riley RD et al.** "Clinical prediction models and the multiverse of madness" | 2023 | BMC Medicine 21:502 | Instability plots; MAPE; multiverse concept |
| **Collins GS et al.** "TRIPOD+AI: Transparent Reporting..." | 2024 | BMJ | Reporting checklist for AI prediction models |
| **Legha A et al.** "Sequential sample size calculations and learning curves..." | 2026 | J Clin Epidemiol | Adaptive sample size; stopping rules |
| **Rhodes SA et al.** "pminternal: Internal Validation of Clinical Prediction Models" | 2025 | CRAN R Package | R package for instability analysis |
| **Peduzzi P et al.** "A simulation study of the number of events per variable..." | 1996 | J Clin Epidemiol 49:1373-79 | Classic EPV >= 10 rule |
| **van Smeden M et al.** "Sample size for binary logistic prediction models..." | 2019 | Stat Methods Med Res 28:2455-74 | EPV necessary but not sufficient |
| **Kompa B, Snoek J, Beam AL** "Second opinion needed: communicating uncertainty in medical ML" | 2021 | npj Digital Medicine 4:1 | Per-patient uncertainty quantification |

## Time-Series Foundation Models

| Citation | Year | Source | Key Contribution |
|----------|------|--------|-----------------|
| **Goswami M et al.** "MOMENT: A Family of Open Time-series Foundation Models" | 2024 | arXiv:2402.03885 | MOMENT architecture; Time-Series Pile |
| **Ansari AF et al.** "Chronos: Learning the Language of Time Series" | 2024 | arXiv:2403.07815 | T5-based TS foundation model |
| **Liu Y et al.** "Sundial: A Family of Highly Capable Time Series Foundation Models" | 2025 | arXiv:2502.00816 | TimeFlow Loss; probabilistic forecasting |
| **Wu H et al.** "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" | 2023 | ICLR 2023 | TimesNet architecture |
| **Du W et al.** "SAITS: Self-Attention-based Imputation for Time Series" | 2022 | arXiv:2202.08516 | SAITS imputation method |
| **Tashiro Y et al.** "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation" | 2021 | NeurIPS 2021 | Diffusion-based imputation |

## Biosignal Foundation Models

| Citation | Year | Source | Key Contribution |
|----------|------|--------|-----------------|
| **Gu X et al.** "Foundation Models for Biosignals: A Survey" | 2025 | arXiv:2506.08130 | Three directions: BFM from scratch, adapt TSFM, leverage LLM |
| **Alchieri L et al.** "Exploring Generalist Foundation Models for Time Series of Electrodermal Activity Data" | 2025 | UbiComp '25 | Generalist FMs competitive but don't outperform handcrafted for EDA |
| **Kataria S et al.** "Generalist vs Specialist PPG Foundation Models" | 2025 | arXiv:2510.14254 | Specialist 27% higher win score in full-tuning; task-dependent |
| **Kuruppu D et al.** "EEG Foundation Models: A Critical Review" | 2025 | arXiv preprint | 10 EEG-FMs reviewed; heterogeneous evaluations |
| **Turgut K et al.** "OTiS: Foundation models as EEG feature extractors" | 2025 | arXiv preprint | General FMs can outperform specialized EEG models |
| **Saha M et al.** "Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model" | 2025 | arXiv:2502.01108 | PPG FM for wearable applications |
| **Pillai A et al.** "PaPaGei: Open Foundation Models for Optical Physiological Signals" | 2025 | arXiv:2410.20542 | PPG/ECG foundation model |
| **Abbaspourazad S et al.** "Large-scale Training of Foundation Models for Wearable Biosignals" | 2024 | arXiv:2312.05409 | Apple wearables FM |


## Ophthalmology Foundation Models

| Citation | Year | Source | Key Contribution |
|----------|------|--------|-----------------|
| **Zhou Y et al.** "RETFound: A foundation model for generalizable disease detection from retinal images" | 2023 | Nature 622:156-163 | Retinal image FM (related domain) |

## Pupillometry and PLR

| Citation | Year | Journal | Key Contribution |
|----------|------|---------|-----------------|
| **Najjar RP et al.** "Handheld chromatic pupillometry can accurately and rapidly reveal functional loss in glaucoma" | 2023 | Br J Ophthalmol 107:663-670 | Our data source; AUROC 0.94 on full dataset |
| **Tham YC et al.** "Global prevalence of glaucoma and projections..." | 2014 | Ophthalmology 121:2081-2090 | Glaucoma prevalence 3.54% |
| **Kelbsch C et al.** "Standards in Pupillography" | 2019 | Front Neurol | ISCEV pupillometry guidelines |
| **Carle CF et al.** "Luminance and colour variant pupil perimetry in glaucoma" | 2014 | Clin Exp Ophthalmol 42:815-824 | PLR glaucoma detection |
| **Joyce DS** "Temporal, spatial and adaptation characteristics of melanopsin inputs..." | 2016 | PhD Thesis, QUT | Melanopsin PLR physiology |

## Soft Targets and Label Uncertainty

| Citation | Year | Source | Key Contribution |
|----------|------|--------|-----------------|
| **Tao T et al.** "Beat-SSL: Self-supervised ECG Representation Learning with Soft Targets" | 2025 | arXiv preprint | Soft targets in contrastive ECG learning |
| **Hinton G et al.** "Distilling the Knowledge in a Neural Network" | 2015 | NIPS Workshop | Knowledge distillation with soft labels |
| **Guo C et al.** "On Calibration of Modern Neural Networks" | 2017 | ICML | Neural network calibration |
| **Joskowicz L et al.** "Inter-observer variability of manual contour delineation of structures in CT" | 2019 | Eur Radiol | Medical annotation variability |
| **Warfield SK et al.** "Simultaneous truth and performance level estimation (STAPLE)" | 2004 | IEEE TMI | Consensus ground truth algorithm |
| **Kohl S et al.** "A Probabilistic U-Net for Segmentation of Ambiguous Images" | 2018 | NeurIPS | Probabilistic segmentation |

## Uncertainty and Calibration

| Citation | Year | Source | Key Contribution |
|----------|------|--------|-----------------|
| **Van Calster B et al.** "Calibration: the Achilles heel of predictive analytics" | 2019 | BMC Medicine 17:230 | Calibration hierarchy |
| **Filos A et al.** "A Systematic Comparison of Bayesian Deep Learning Robustness in Diabetic Retinopathy Tasks" | 2019 | arXiv:1912.10481 | BDL benchmarking |
| **Geifman Y, El-Yaniv R** "Selective Classification for Deep Neural Networks" | 2017 | NeurIPS | AURC; selective classification |
| **Cabitza F et al.** "Unintended Consequences of Machine Learning in Medicine" | 2017 | JAMA | ML deployment concerns |
| **Oakden-Rayner L** "Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning..." | 2019 | arXiv | Hidden stratification in medical AI |

## Statistical Methods

| Citation | Year | Source | Key Contribution |
|----------|------|--------|-----------------|
| **Demšar J** "Statistical comparisons of classifiers over multiple data sets" | 2006 | JMLR 7:1-30 | Critical difference diagrams |
| **Allen M et al.** "Raincloud plots: a multi-platform tool for robust data visualization" | 2019 | Wellcome Open Res | Raincloud plot methodology |
| **Simonsohn U et al.** "Specification Curve Analysis" | 2020 | Nat Hum Behav | Multiverse/specification curve |
| **Vickers AJ, Elkin EB** "Decision curve analysis: a novel method for evaluating prediction models" | 2006 | Med Decis Making | Decision curve analysis |
| **Cohen J** "Statistical Power Analysis for the Behavioral Sciences" | 1988 | Book (2nd ed.) | Effect size conventions |

---

*This consolidated document synthesizes outputs from 4 specialized expert agents: (1) Clinical ML/STRATOS expert for pilot study framing, (2) Biosignal FM expert for TSFM landscape connection, (3) Soft targets expert for ground truth uncertainty (see main expert review file), and (4) Introduction/Discussion expert for manuscript positioning.*
