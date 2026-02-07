# Translational Insights: PLR Preprocessing Beyond Ophthalmology

> **For researchers in other domains**: This page explains how the preprocessing challenges solved in this repository (pupillary light reflex signals) parallel problems in vibration analysis, seismology, audio processing, power grid monitoring, and other dense signal domains.

---

## Why This Matters to You

If you work with **dense, irregularly-corrupted time series**, the techniques here may transfer to your domain:

- **Vibration monitoring** (machinery health)
- **Seismic signals** (earthquake detection)
- **Audio processing** (speech, music)
- **Power grid monitoring** (anomaly detection)
- **Lung sounds** (source separation)

The PLR signal preprocessing pipeline demonstrates patterns that generalize across these domains.

---

## The Hype vs Reality of Time Series Foundation Models

![Comparison of time-series foundation model marketing claims versus research findings: where TSFMs genuinely help (anomaly detection, imputation, zero-shot prototyping) versus where they struggle (sparse data, event-driven patterns). GPT-4 performs at chance on real forecasting tasks (Schoenegger & Park 2023). Normalized metrics can hide poor predictions.](../repo-figures/assets/fig-trans-02-tsfm-hype-vs-reality.jpg)

**TSFM Hype vs Reality**

Foundation models are transforming NLP and vision, but time series is different. This diagram shows the realistic assessment: FMs excel with large datasets and multi-task scenarios, but traditional feature engineering often wins for domain-specific applications with limited data.

*Learn more: [MOMENT Paper](https://arxiv.org/abs/2402.03885) | [UniTS Paper](https://arxiv.org/abs/2403.00131)*

---

## Domain Fit Matrix: When Do FMs Help?

![Heat map of TSFM suitability across domains: dense signal domains (PLR, ECG, EEG, seismic, vibration) score well for anomaly detection and imputation. Sparse domains (EHR, business KPIs, logistics) score poorly, with X marks indicating imputation is harmful, not merely ineffective.](../repo-figures/assets/fig-trans-03-domain-fit-matrix.jpg)

**Domain Fit Matrix**

Not all time series are created equal. This matrix helps you assess whether foundation models are likely to help in your specific domain based on:
- Signal density (sparse vs dense)
- Sampling regularity
- Available training data
- Task complexity

---

## Sparse vs Dense Signals

![Visual comparison: PLR signal at 30 Hz (150 points in 5 seconds, smooth continuous curve where neighbors are correlated) versus supply chain data (11 points over 11 days, each point independent). Different sampling rates require fundamentally different mathematical approaches.](../repo-figures/assets/fig-trans-04-sparse-vs-dense.jpg)

**Sparse vs Dense Signals**

PLR is a **dense signal** - continuous measurements at regular intervals. This is fundamentally different from sparse event data (like network logs or transaction records). The preprocessing techniques here are designed for dense signals.

---

## Domain Parallels

### PLR ↔ Vibration Monitoring

![Side-by-side comparison of pupil signal artifact removal and vibration sensor dropout correction. Both involve detecting anomalies in dense, regularly-sampled signals and reconstructing the true underlying waveform. The preprocessing pipeline is domain-agnostic; what differs are the features and classification targets.](../repo-figures/assets/fig-trans-05-plr-vibration-parallel.jpg)

**Parallel to Vibration Analysis**

Industrial vibration monitoring faces similar challenges:
- Artifacts from sensor issues (like blinks in PLR)
- Feature extraction from amplitude patterns
- Classification of healthy vs faulty states

*Our outlier detection → imputation → feature extraction pipeline maps directly.*

---

### PLR ↔ Seismic Signals

![Comparison of PLR anomaly detection (blink artifacts at 30 Hz) with seismic event detection (earthquakes at 100 Hz). Shared methods: LOF, autoencoders, diffusion models, foundation models. Domain-specific: DASFormer for seismic (2025), MOMENT for PLR (Goswami et al. 2024).](../repo-figures/assets/fig-trans-06-plr-seismic-parallel.jpg)

**Parallel to Seismology**

Seismic event detection shares our core challenge: distinguishing **true signal** (pupil response / earthquake) from **noise** (blinks / background tremors).

---

### PLR ↔ Audio Processing

![Three-way comparison of denoising in PLR (30 Hz, time domain), speech (16 kHz, spectrogram), and music (48 kHz, spectrogram). Shared framework: observed signal equals true signal plus artifact. Same architecture families (autoencoders, masking, diffusion, transformers) apply across all three at different time scales.](../repo-figures/assets/fig-trans-07-plr-audio-parallel.jpg)

**Parallel to Audio**

Audio engineers deal with:
- Transient artifacts (clicks, pops) ↔ blink artifacts
- Signal reconstruction (inpainting) ↔ imputation
- Feature extraction (MFCCs, spectrograms) ↔ amplitude bins

---

### Source Separation in Lung Sounds

![Wearable lung sound source separation: a single microphone captures a mixture of lung sounds, heart sounds, ambient noise, and friction. Analogous to PLR preprocessing but with multiple signals to preserve. References Grooby et al. 2023, McLane et al. 2023, Rennoll et al. 2023.](../repo-figures/assets/fig-trans-08-source-separation-lung.jpg)

**Blind Source Separation**

In lung sound analysis, you must separate heart sounds from breath sounds. Similarly, we separate the true pupil response from artifact-induced variations.

---

### Power Grid Monitoring

![Parallel between PLR (30 Hz, blink artifacts) and power grid monitoring (60 Hz, voltage sags and transients). Both are dense, regularly-sampled signals with physically interpretable anomalies. The same anomaly detection algorithms apply.](../repo-figures/assets/fig-trans-09-power-grid-monitoring.jpg)

**Power Grid Parallels**

Smart grid monitoring requires:
- Real-time anomaly detection in dense signals
- Distinguishing equipment faults from normal load variations
- Imputing missing data from sensor dropouts

---

## The Dense Signal Club

![Membership diagram for signals where PLR preprocessing concepts transfer: requirements are greater than 1 sample/second, continuous underlying process, gaps represent errors, neighbors are correlated. Members: biosignals (PLR, ECG, EEG, PPG), engineering (grid, vibration, seismic), audio (speech, music). Not members: EHR, business KPIs, logistics.](../repo-figures/assets/fig-trans-10-dense-signal-club.jpg)

**You're in the Dense Signal Club if you have:**
- Regular sampling (even if there are gaps)
- Continuous underlying phenomenon
- Artifact contamination from measurement issues
- Need for feature extraction before classification

Members: ECG, EEG, EMG, vibration, audio, PLR, accelerometer data...

---

## Event-Conditioned Signals

![Architecture diagram for GMAN (Graph Mixing Additive Networks, Bechler-Speicher et al. 2025): for event-conditioned time series, GMAN conditions predictions on external events rather than imputing over them. Trajectories as directed graphs flow through ExtGNAN processing, then DeepSet aggregation, then interpretable prediction.](../repo-figures/assets/fig-trans-11-gman-event-conditioned.jpg)

**Event-Conditioned Signals**

PLR is **event-conditioned** - the signal is a response to a light stimulus. This differs from free-running signals (like ECG) and affects:
- How we align signals across subjects
- What "normal" looks like
- Feature extraction strategies

---

## When NOT to Impute

![Two-panel comparison. Left: PLR signal with a blink gap -- a measurement error that should be imputed because the signal existed but couldn't be measured. Right: logistics time series with a warehouse closure gap -- should NOT be imputed because the gap encodes real information. Decision tree: 'Is the gap a measurement error?' References Van Ness et al. 2023, McTavish et al. 2024 (NeurIPS Proposition 3.1).](../repo-figures/assets/fig-trans-01-when-not-to-impute.jpg)

**When Imputation is Wrong**

Imputation isn't always the answer. This decision tree helps you decide when to:
- Impute (fill in gaps)
- Delete (remove corrupted segments)
- Use model-based approaches
- Accept and model the missingness directly

---

## Handling Missing Values: The MGAM Approach

![M-GAM approach (McTavish et al. 2024, NeurIPS): when missingness is informative (store closed, patient didn't visit), M-GAM treats missing values as features rather than imputing them. Proposition 3.1: perfect imputation can reduce model performance when missingness carries information. Maintains GAM interpretability.](../repo-figures/assets/fig-trans-12-mgam-missing-values.jpg)

**Missing Values Strategy**

Different missing data patterns require different strategies. This figure shows how we characterize and handle various missingness patterns in PLR data.

---

## When Simple Baselines Win

![Decision framework for method selection: simple baselines (constant, linear, moving average) win when data is small, SNR is low, interpretability is required, or real-time constraints exist. Foundation models win when there's a large training corpus, complex temporal patterns, or zero-shot transfer is needed. References Zeng 2023, Makridakis et al. 2022.](../repo-figures/assets/fig-trans-13-when-simple-baselines-win.jpg)

**The Humble Baseline**

One of our key findings: **handcrafted features + simple imputation often beats foundation model embeddings**. This figure shows when simplicity wins:
- Small datasets (< 1000 samples)
- Domain-specific features available
- Interpretability required

*Don't assume complex = better. Test your baselines.*

---

## Domain-Specific vs Generic Approaches

![Trade-off between generic foundation models (MOMENT, TimesFM -- broad applicability, zero-shot, opaque) and domain-specific approaches (handcrafted features, EchoNet-Dynamic per Ouyang et al. 2020 -- interpretable, data-efficient, require expertise). High-stakes clinical applications favor domain-specific; prototyping can use generic. References Grinsztajn et al. 2022 on tabular data.](../repo-figures/assets/fig-trans-14-domain-specific-vs-generic.jpg)

**The Trade-off**

| Approach | Pros | Cons |
|----------|------|------|
| **Domain-specific features** | Interpretable, works with small data | Requires expertise, may miss patterns |
| **Generic embeddings** | No domain expertise needed, can discover new patterns | Needs large data, black box |

Our finding: For PLR with N=208 labeled subjects, **domain-specific wins by 9 percentage points**.

---

## How to Adapt This Code to Your Domain

### Step 1: Assess Domain Fit

Use the Domain Fit Matrix above to estimate whether our approach will transfer.

### Step 2: Map the Pipeline

```
Your Domain              PLR Pipeline
─────────────           ─────────────
Your artifacts    →     Blink detection (outlier methods)
Your gaps         →     Missing segments (imputation methods)
Your features     →     Amplitude bins + latency (featurization)
Your classifier   →     CatBoost (or your choice)
```

### Step 3: Fork and Modify

See `configs/` for the YAML structure. The registry pattern means you can add new methods without changing core code.

---

## Key Takeaways for Other Domains

1. **Foundation models aren't magic** - Test against domain-specific baselines
2. **Preprocessing matters** - 15% of downstream variance in our study
3. **Dense signals share patterns** - Vibration, audio, seismic, PLR all benefit from similar approaches
4. **Impute carefully** - Know when NOT to impute
5. **The pipeline is transferable** - Outlier → Impute → Feature → Classify

---

## References

### Time Series Foundation Models
- [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885)
- [UniTS: Building a Unified Time Series Model](https://arxiv.org/abs/2403.00131)
- [TimesNet: Temporal 2D-Variation Modeling](https://arxiv.org/abs/2210.02186)

### Domain-Specific Feature Engineering
- [Grinsztajn et al. 2022 - Why Tree-Based Models Beat Deep Learning on Tabular Data](https://arxiv.org/abs/2207.08815)
- [Christodoulou et al. 2019 - ML vs LR for Clinical Prediction](https://doi.org/10.1016/j.jclinepi.2019.02.004)

### Missing Data
- [Little & Rubin - Statistical Analysis with Missing Data](https://www.wiley.com/en-us/Statistical+Analysis+with+Missing+Data%2C+3rd+Edition-p-9780470526798)
- [Van Buuren - Flexible Imputation of Missing Data](https://stefvanbuuren.name/fimd/)
