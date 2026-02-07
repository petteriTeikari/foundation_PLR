# Lay Professional Figure Coverage Plan

**Status**: ✅ COMPLETE (20 of 20 figure plans detailed)
**Target Audience**: Professionals outside biomedical research
**Focus**: Translational value of TSFM preprocessing concepts
**Honest Framing**: Data quality challenges, not forecasting hype

---

## 1. Vision & Academic Honesty Statement

### 1.1 What This Plan IS

A set of figures explaining how **data quality challenges** in time series are universal across domains, and how the preprocessing concepts (anomaly detection → imputation/reconstruction) from this PLR repository can seed applications in:

- Engineering & industrial monitoring
- Audio signal processing
- Environmental/geophysical sensing
- Other biosignal domains

### 1.2 What This Plan is NOT

**We explicitly avoid:**

1. **Forecasting snakeoil** - We do NOT claim TSFMs solve all time series problems
2. **Lifestyle entrepreneurship vibes** - No Soho House energy, no "AI will revolutionize everything"
3. **Universal applicability claims** - TSFMs work poorly for sparse/irregular data (EHRs, business metrics)
4. **Classifier comparison hype** - The repo is about preprocessing effects, not "best model" contests

### 1.3 Academic Honesty: TSFM Limitations

From extensive literature review (see `/home/petteri/Dropbox/KnowledgeBase/Time Series/Deep - Time Series - Forecasting.md`):

| Criticism | Source | Implication |
|-----------|--------|-------------|
| "Difficult to take this paper seriously when they dismiss ARIMA" | HackerNews on TimeGPT | Many TSFM papers don't compare to proper baselines |
| "LLMs for TS underperform on datasets lacking periodicity" | Jin et al. 2024 | TSFMs need clear temporal structure |
| "GPT-4 forecasts are not significantly different from 50% on real-world tournaments" | Schoenegger & Park 2023 | LLM-based forecasting is underwhelming |
| "Transformers show poor accuracy when actual forecasts are plotted" | @predict_addict 2024 | Normalized metrics hide bad forecasts |
| "Informer paper compared to clearly unsuitable baselines" | Christoph Bergmeir | Cherry-picking is endemic |

**Our honest position**: TSFMs are useful for **specific preprocessing tasks** (anomaly detection, imputation) on **dense, regularly-sampled signals** where **temporal structure is learnable**. They are NOT a universal solution.

---

## 2. The Domain Spectrum: Where TSFMs Work (and Don't)

### 2.1 The TSFM Sweet Spot (Dense Signals)

| Domain | Sampling | Why TSFM Works |
|--------|----------|----------------|
| **Biomedical** (PLR, ECG, EEG) | 30-500 Hz | Clear temporal dynamics, hand-annotatable artifacts |
| **Industrial Vibration** | 5 kHz | Periodic patterns from rotating machinery |
| **Seismic** | 100-500 Hz | Earthquake signals have consistent waveforms |
| **Audio/Speech** | 16-48 kHz | Well-studied temporal-spectral structure |
| **Power Grid** | 30-100 Hz | 50/60 Hz fundamental with predictable harmonics |

### 2.2 Where TSFMs Fail (Sparse/Irregular Signals)

| Domain | Why TSFM Fails | Better Approach |
|--------|----------------|-----------------|
| **EHRs** | Structural missingness (gaps = information), irregular sampling | Neural ODEs, GMAN, M-GAM |
| **Business KPIs** | Daily/weekly = too sparse, driven by external events not physics | ARIMA, Prophet, human domain knowledge |
| **Supply Chain** | Event-driven, multivariate + text, external shocks | GMAN with event conditioning |
| **E-commerce Demand** | Seasonal priors, holiday effects, no "artifacts" per se | GAMs with seasonal decomposition |

### 2.3 Key Insight for Figures

> **The preprocessing concepts (anomaly detection, imputation) transfer across dense signal domains. The specific models and features do NOT.**

---

## 3. User Personas & Their Needs

### Persona 1: Humanitarian Logistics (UN WFP)

**Context**: Food supply chain monitoring at UN World Food Programme

| Aspect | PLR (30 Hz) | Logistics (daily updates) |
|--------|-------------|---------------------------|
| **Sampling** | 30 samples/second | 1 sample/day |
| **Artifact definition** | Blink = missing pupil data | Supply disruption = real event, not artifact |
| **Imputation need** | YES - reconstruct missing pupil trajectory | NO - gaps encode reality (warehouse closed, conflict zone) |
| **External knowledge** | Light stimulus timing (known) | Geopolitical events, weather, holidays (critical) |

**Why TSFMs fail here**: A "glitch" in supply data might be a missile strike in Kyiv. You don't want to impute over it - you want to **explain it** with external knowledge.

**Better approach**: GMAN (Graph Mixing Additive Networks) - represents each trajectory as a graph, allows conditioning on external events, provides interpretable attributions.

**Figures needed**:
- fig-trans-01: "When NOT to impute" - Logistics anomaly = real event
- fig-trans-02: GMAN architecture for event-conditioned time series

### Persona 2: E-Commerce Demand Forecasting

**Context**: Predicting grocery store demand

| Aspect | PLR | E-Commerce |
|--------|-----|------------|
| **Artifact definition** | Sensor failure, blink | No artifacts - all data is real |
| **Seasonality** | Light stimulus cycle (known, controlled) | Weekday/weekend, holidays, promotions (complex, external) |
| **Missing data** | Corruption to be fixed | Informative missingness (store closed = no sales) |
| **Imputation** | Essential for downstream features | NEVER - missing = meaningful |

**Why TSFMs fail here**: Demand forecasting needs:
- Seasonal priors (week patterns, holiday effects)
- External regressors (promotions, weather)
- Interpretable explanations (why did sales spike?)

**Better approach**: M-GAM (Missing-aware GAM) - interpretable, handles missing values as features not bugs, sparse regularization prevents overfitting.

**Figures needed**:
- fig-trans-03: "Missingness as information" - M-GAM vs impute-then-predict
- fig-trans-04: Seasonal decomposition for demand (STL, Prophet-style)

### Persona 3: Industrial Predictive Maintenance

**Context**: Monitoring turbine/engine sounds for anomalies

| Aspect | PLR | Industrial Acoustics |
|--------|-----|---------------------|
| **Sampling** | 30 Hz | 5-20 kHz |
| **Artifacts** | Blinks, tracking loss | Sensor noise, transmission dropouts |
| **Signal separation** | Not needed (single pupil trace) | Critical - separate engine from ambient noise |
| **Anomaly detection** | Find blinks to remove | Find faults to alert |

**Why TSFMs CAN work here**: Similar problem structure to PLR:
- Dense, regularly-sampled signal
- Artifacts are physically interpretable
- Ground truth can be hand-annotated
- Temporal dynamics are learnable

**Figures needed**:
- fig-trans-05: PLR ↔ Vibration parallel (artifact types comparison)
- fig-trans-06: Blind source separation for industrial audio
- fig-trans-07: Transfer learning potential across signal domains

### Persona 4: Wearable Health (Lung/Heart Sounds)

**Context**: Continuous acoustic monitoring for respiratory disease

| Aspect | PLR | Wearable Lung Sounds |
|--------|-----|---------------------|
| **Signal composition** | Pure pupil (mostly) | Lung + Heart + Ambient (mixture) |
| **Preprocessing goal** | Remove blinks, reconstruct | Separate sources, denoise |
| **Multi-sensor** | Single eye tracker | Multiple microphones (body + reference) |
| **Domain knowledge** | Pupillometry theory | Respiratory physiology, heart sounds |

**Why this is a GREAT translation domain**:
- Dense signal (audio @ 16 kHz)
- Clear artifact definition (ambient noise, friction)
- Blind source separation has decades of research
- Multi-sensor setup enables informed separation (not "blind")

**Figures needed**:
- fig-trans-08: Source separation diagram (lung/heart/ambient)
- fig-trans-09: Multi-sensor vs single-sensor artifact handling
- fig-trans-10: "Informed" separation using reference microphone

---

## 4. Figure Catalog

### Tier 1: Honest Limitations (4 figures)

| ID | Title | Purpose | Style |
|----|-------|---------|-------|
| fig-trans-01 | "When NOT to Impute" | Show logistics/EHR cases where gaps are information | Comparison diagram |
| fig-trans-02 | "The TSFM Hype vs Reality" | Honest comparison of claims vs evidence | Quote collage + metrics |
| fig-trans-03 | "Domain Fit Matrix" | Which domains suit TSFMs vs alternatives | Heat map |
| fig-trans-04 | "Sparse vs Dense: Different Beasts" | 30 Hz PLR vs daily business metrics | Waveform comparison |

### Tier 2: Translational Parallels (6 figures)

| ID | Title | Purpose | Style |
|----|-------|---------|-------|
| fig-trans-05 | "PLR ↔ Vibration: Same Problem, Different Domain" | Show preprocessing parallel | Side-by-side comparison |
| fig-trans-06 | "PLR ↔ Seismic: Anomaly Detection Transfers" | Earthquake detection as anomaly detection | Waveform + artifacts |
| fig-trans-07 | "PLR ↔ Audio: Denoising Concepts" | Speech enhancement parallel | Spectrogram comparison |
| fig-trans-08 | "Source Separation: Lung/Heart/Ambient" | Wearable acoustic monitoring | Component diagram |
| fig-trans-09 | "Power Grid Monitoring: 60 Hz is Regular Too" | Grid anomaly detection parallel | Frequency domain |
| fig-trans-10 | "The Dense Signal Club" | All domains where TSFMs work | Visual membership diagram |

### Tier 3: Alternative Approaches (4 figures)

| ID | Title | Purpose | Style |
|----|-------|---------|-------|
| fig-trans-11 | "GMAN for Event-Conditioned Time Series" | Logistics/humanitarian use case | Architecture + example |
| fig-trans-12 | "M-GAM: Missing Values as Features" | E-commerce/EHR approach | Model comparison |
| fig-trans-13 | "When Simple Baselines Win" | ARIMA beating deep learning | Performance comparison |
| fig-trans-14 | "Domain-Specific vs Generic Models" | Show when specialization wins | Trade-off diagram |

### Tier 4: Code Transferability (4 figures)

| ID | Title | Purpose | Style |
|----|-------|---------|-------|
| fig-trans-15 | "PLR Code: What's Domain-Specific?" | Show which code transfers | Module diagram |
| fig-trans-16 | "Configuration vs Hardcoding" | How Hydra configs enable domain switching | Config hierarchy |
| fig-trans-17 | "The Registry Pattern" | Single source of truth for experiments | Architecture diagram |
| fig-trans-18 | "From PLR to Your Domain: Fork Guide" | Step-by-step adaptation | Flowchart |

### Tier 5: Composite & Summary (2 figures)

| ID | Title | Purpose | Style |
|----|-------|---------|-------|
| fig-trans-19 | "The Data Quality Manifesto" | Universal preprocessing principles | Manifesto-style text |
| fig-trans-20 | "Choose Your Approach" | Decision tree for domain/method selection | Decision tree |

---

## 5. Literature Foundation

### 5.1 TSFM Criticisms & Honest Assessment

| Paper | Key Finding | Our Takeaway |
|-------|-------------|--------------|
| Hewamalage et al. 2022 | "Flawed evaluation practices in ML forecasting" | Need proper baselines |
| Zeng et al. 2022 | "Are Transformers Effective for Time Series?" Linear models competitive | Don't assume deep learning wins |
| Schoenegger & Park 2023 | GPT-4 ≈ 50% on real forecasting tournaments | LLMs don't generalize |
| Jin et al. 2024 | LLMs fail on datasets without clear periodicity | Need temporal structure |
| @predict_addict threads | Many papers use "normalization tricks" to hide bad forecasts | Check actual predictions |

### 5.2 Alternative Approaches

| Paper | Method | Use Case |
|-------|--------|----------|
| Bechler-Speicher 2025 | GMAN (Graph Mixing Additive Networks) | Sparse, irregular, multivariate with external events |
| McTavish 2024 | M-GAM (Missing-aware GAM) | Interpretable handling of missingness as features |
| Van Calster 2024 | STRATOS Guidelines | Proper performance evaluation (not just AUROC) |
| Riley 2023 | pminternal | Bootstrap stability analysis |

### 5.3 Dense Signal Domains

| Domain | Key Reference | TSFM Status |
|--------|---------------|-------------|
| ECG | ECG-FM, MIRA, HeartBEIT | Mature foundation models exist |
| EEG | LaBraM, NeuroLM, EEGFormer | Mature foundation models exist |
| PPG | PaPaGei (57k hours) | Mature foundation models exist |
| Seismic | DASFormer (2025) | Emerging TSFM for earthquake detection |
| Speech | DeepFilterNet, GTCRN | Domain-specific still wins |
| Vibration | Statistical + autoencoders | TSFMs being evaluated |

### 5.4 Sparse Signal Domains (Where TSFMs Fail)

| Domain | Key Reference | Why TSFMs Fail |
|--------|---------------|----------------|
| EHR | IGNITE (2024) | Structural missingness, irregular sampling |
| Business | Prophet, ARIMA | External drivers, sparse, no physics |
| Logistics | GMAN | Event-conditioned, multimodal |

---

## 6. Design Guidelines

### 6.1 Visual Style

- **75% manuscript style** (clean, academic, precise)
- **25% Economist aesthetics** (clear hierarchy, no chartjunk)
- **NO sci-fi effects** (no glows, no "AI brain" imagery)
- **Honest data** (show failures, not just successes)

### 6.2 Tone

| DO | DON'T |
|----|-------|
| "TSFMs are useful for specific preprocessing tasks" | "AI revolutionizes time series" |
| "On dense signals with learnable dynamics" | "On any time series data" |
| "When proper baselines are compared" | "State-of-the-art performance" |
| "Handcrafted features encode domain knowledge" | "End-to-end learning is always better" |

### 6.3 Accessibility

- Colorblind-safe palettes (Economist palette from configs)
- Alt text for all figures
- ELI5 captions alongside technical details

---

## 7. Key Messages by Persona

### For Humanitarian/Logistics

> "Your supply chain anomalies aren't artifacts to remove - they're events to explain. Use GMAN to condition on external knowledge, not TSFMs to impute over reality."

### For E-Commerce

> "Missing data in your demand forecasts isn't corruption - it's information. A store closed for holidays shouldn't be 'imputed'. Use M-GAM to treat missingness as a feature."

### For Industrial Monitoring

> "Your vibration/acoustic monitoring has the same structure as biomedical signals: dense, regular, with physically interpretable artifacts. The preprocessing concepts transfer directly."

### For Audio Engineers

> "Speech enhancement, music source separation, and biomedical denoising share the same core challenge: separating signal from noise/artifacts. The math is the same, the features differ."

### For Data Scientists (General)

> "Before applying TSFMs, ask: Is my signal dense and regular? Are artifacts physically interpretable? Does temporal structure exist? If no, consider alternatives."

---

## 8. Execution Plan

### Phase 1: Limitation Figures (fig-trans-01 to 04) ✅ PLANS COMPLETE
- ✅ Draft honest assessment of where TSFMs fail
- ✅ Create domain fit matrix
- Next: Generate figures, get feedback on tone

### Phase 2: Translation Figures (fig-trans-05 to 10) ✅ PLANS COMPLETE
- ✅ Visual parallels between PLR and other domains
- ✅ Highlight shared preprocessing challenges
- ✅ Show code transferability

### Phase 3: Alternative Approaches (fig-trans-11 to 14) ✅ PLANS COMPLETE
- ✅ GMAN for sparse/event-driven data
- ✅ M-GAM for missing-as-feature
- ✅ When baselines win

### Phase 4: Code Guide (fig-trans-15 to 18) ✅ PLANS COMPLETE
- ✅ What's domain-specific vs generic
- ✅ Configuration patterns
- ✅ Fork guide for other domains

### Phase 5: Synthesis (fig-trans-19 to 20) ✅ PLANS COMPLETE
- ✅ Data quality manifesto
- ✅ Decision tree for approach selection

**Status: All 20 figure plans detailed in `docs/repo-figures/figure-plans/fig-trans-*.md`**

---

## 9. Success Criteria

### Tone Check

- [ ] Would a UN WFP data scientist find this useful (not patronizing)?
- [ ] Would an audio engineer see parallels (not forced analogies)?
- [ ] Would a skeptic agree with limitations (not defensive)?
- [ ] Would a reader trust our honesty (not marketing)?

### Technical Check

- [ ] All claims backed by citations
- [ ] Failure cases shown alongside successes
- [ ] Alternative approaches given fair treatment
- [ ] Domain-specific caveats explicit

### Visual Check

- [ ] No sci-fi aesthetics
- [ ] Economist-level clarity
- [ ] Colorblind-safe
- [ ] Proper alt text

---

## Appendix A: Domain Details

### A.1 Dense Signal Domains (TSFM Applicable)

| Domain | Sampling Rate | Signal Type | Artifacts | TSFM Maturity |
|--------|---------------|-------------|-----------|---------------|
| PLR | 30 Hz | Autonomic response | Blinks, tracking | Early (this repo) |
| ECG | 250-500 Hz | Cardiac electrical | Noise, electrode | Mature (ECG-FM) |
| EEG | 256-1000 Hz | Neural electrical | Blinks, muscle | Mature (LaBraM) |
| PPG | 25-125 Hz | Cardiovascular | Motion, contact | Mature (PaPaGei) |
| Vibration | 5 kHz | Mechanical motion | Noise, dropouts | Intermediate |
| Seismic | 100-500 Hz | Ground motion | Instrument noise | Emerging (DASFormer) |
| Speech | 16-48 kHz | Acoustic | Background noise | Advanced |
| Music | 44.1-48 kHz | Audio channels | Recording noise | Advanced |
| Power Grid | 30-100 Hz | Voltage/Current | Harmonics | Intermediate |

### A.2 Sparse Signal Domains (TSFM Not Applicable)

| Domain | Sampling | Why TSFM Fails | Better Approach |
|--------|----------|----------------|-----------------|
| EHR | Hours-weeks | Structural missingness | Neural ODE, GMAN |
| Business KPIs | Daily-weekly | Event-driven, external | ARIMA, Prophet |
| Supply Chain | Irregular | Multimodal, shocks | GMAN + LLM |
| E-commerce | Daily | Seasonal, no artifacts | M-GAM, STL |

### A.3 Key References

**TSFM Limitations:**
- Hewamalage et al. (2022): "Forecast evaluation pitfalls and best practices"
- Zeng et al. (2022): "Are Transformers Effective for Time Series Forecasting?"
- @predict_addict Twitter threads on deceptive normalization tricks

**Alternative Approaches:**
- Bechler-Speicher et al. (2025): "GMAN: Graph Mixing Additive Networks" - for sparse/irregular
- McTavish et al. (2024): "M-GAM: Interpretable GAMs for Missing Values" - missingness as feature
- Van Ness et al. (2023): "Missingness indicators outperform imputation" - informative missingness

**Domain-Specific:**
- Apple ML Research (2024): "Large-Scale Training of Foundation Models for Wearable Biosignals"
- DASFormer (2025): "Self-Supervised TSFM for Earthquake Monitoring"
- Cold Diffusion (2024): "Seismic Denoising with Diffusion Models"

---

## Appendix B: Glossary for Lay Professionals

| Term | ELI5 | Technical |
|------|------|-----------|
| **TSFM** | AI trained on lots of time data | Time Series Foundation Model |
| **Anomaly** | Something weird in the data | Outlier requiring attention |
| **Artifact** | Garbage to remove | Measurement error, not real signal |
| **Imputation** | Filling in missing pieces | Reconstructing corrupted segments |
| **Dense signal** | Many measurements per second | High sampling rate (>10 Hz) |
| **Sparse signal** | Few measurements (daily/weekly) | Low, irregular sampling |
| **Domain** | Your area of work | Application context |
| **Transfer** | Using knowledge from one area in another | Cross-domain generalization |

---

*Last updated: 2026-02-01*
