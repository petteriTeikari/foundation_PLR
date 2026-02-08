# Reproducibility & MLSecOps Improvement Plan

**Date**: 2026-02-08
**Branch**: `chore/final-housekeeping`
**Scope**: Open-ended characterization of further improvements for foundation-PLR
**Status**: Enhancement roadmap (post-publication, not blocking handover)

---

## Executive Summary

This document maps the current foundation-PLR security and reproducibility posture against the latest (Jan 2025 -- Feb 2026) developments in MLSecOps, clinical AI governance, and reproducible ML research. It synthesizes insights from:

- **10 seed papers** (SecureAI-Flow, AIAppOps, Maria Platform, ML Monitoring MLR, event sourcing for reproducibility, biomedical AI reproducibility challenges, and more)
- **Clinical ML knowledge base** (IEC 62304, ISO 13485/14971, FDA SaMD, QMS, RegOps)
- **UAD-copilot MLSecOps reference** (CISO Assistant, SOC2, ISO 42001, model governance, Cards Trail)
- **2025-2026 web search** (OWASP ML Top 10, model signing, EU AI Act, NIST AI RMF, ML-BOMs)

**Current engineering maturity: strong** -- the repo is publication-ready with excellent CI/CD, pre-commit gating, privacy architecture, and reproducibility pipeline. Clinical governance maturity is lower (no model card, no demographic reporting, no FMEA). The improvements below are **aspirational enhancements** for long-term maintenance and potential clinical translation.

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Threat Landscape for Research ML Pipelines](#2-threat-landscape)
3. [Reproducibility Hardening](#3-reproducibility-hardening)
4. [Supply Chain Security](#4-supply-chain-security)
5. [Model & Data Governance](#5-model--data-governance)
6. [Clinical Translation Readiness](#6-clinical-translation-readiness)
7. [Monitoring & Drift Detection](#7-monitoring--drift-detection)
8. [Regulatory Alignment](#8-regulatory-alignment)
9. [Recommended GitHub Issues](#9-recommended-github-issues)
10. [References](#10-references)

---

## 1. Current State Assessment

### What's Already Excellent

| Domain | Status | Evidence | Key Gap |
|--------|--------|---------|---------|
| CI/CD pipeline | Excellent | 5 workflows, tiered testing, Docker reproducibility | -- |
| Pre-commit gating | Excellent | 9 hooks: registry anti-cheat, computation decoupling, R hardcoding | No `nbstripout` for notebooks |
| Privacy architecture | Good | PLRxxxx anonymization, gitignore enforcement, compliance scanning | No git history audit, no re-identification risk assessment |
| Reproducibility | Good | Two-block architecture, Docker, lockfiles, frozen experiment configs | No external validation, no data versioning |
| Testing | Excellent | Tiered (unit/integration/e2e), figure QA, registry validation | -- |
| Code quality | Excellent | ruff + mypy + guardrail tests | -- |
| Dependency management | Excellent | Dependabot plan, transitive floor bumps, uv lockfile | -- |
| Clinical governance | Weak | STRATOS metrics compliant | No model card, data card, FMEA, or demographic reporting |

### Gaps Identified

| Gap | Priority | Effort |
|-----|----------|--------|
| No `SECURITY.md` with vulnerability disclosure process | High | 30 min |
| No formal Model Card for CatBoost classifier | High | 1 hour |
| No formal Data Card for SERI PLR dataset | High | 1 hour |
| No demographic bias assessment (age, sex reporting) | High | 4 hours |
| `weights_only=True` NOT consistently used for `torch.load()` | High | 2 hours |
| 12+ bare `pickle.load()` calls without sandboxing | High | 2 hours |
| No IRB/ethics approval documentation in governance | High | 1 hour |
| No ML-BOM (ML Bill of Materials) | Medium | 2 hours |
| No model artifact signing/provenance | Medium | 4 hours |
| No data versioning (DVC or equivalent) | Medium | 4 hours |
| No explicit branch protection rules on GitHub | Medium | 15 min |
| No CODEOWNERS file | Low | 10 min |
| No signed commits enforcement | Low | 30 min |
| No runtime monitoring framework (drift detection) | Low | 8+ hours |

---

## 2. Threat Landscape for Research ML Pipelines {#2-threat-landscape}

### OWASP ML Top 10 (v0.3, 2023) Applied to Foundation PLR

| ID | Risk | Applicability | Current Mitigation |
|----|------|---------------|--------------------|
| ML01 | Input manipulation (adversarial examples) | LOW -- research pipeline, no live inference | N/A |
| ML02 | Data poisoning | LOW -- fixed dataset from Najjar 2023 | Data provenance documented |
| ML03 | Model inversion | LOW -- no public API | Models not exposed |
| ML04 | Membership inference | MEDIUM -- patient data | Anonymization (PLR→H/G codes) |
| ML05 | Model theft | LOW -- academic, MIT license | Open source by design |
| ML06 | **Supply chain attacks** | **HIGH** -- 50+ Python deps | Dependabot + lockfile |
| ML07 | Transfer learning attack | MEDIUM -- MOMENT fine-tuning | Fixed checkpoints |
| ML08 | Model skewing (train/serve mismatch) | LOW -- no serving | N/A |
| ML09 | Output integrity attack | LOW -- local pipeline | N/A |
| ML10 | Model poisoning | LOW -- own training only | MLflow tracking |

**Primary threat: ML06 (Supply Chain)**. The 2025 CVEs in MLflow validate this concern:
- **CVE-2025-11200** (auth bypass, CVSS 8.1): fixed in 2.22.0
- **CVE-2025-11201** (directory traversal RCE): fixed in 2.17.2
- **CVE-2025-52967** (SSRF): fixed in **3.1.0** (NOT covered by pyproject.toml floor of >=2.22.4)

Our `uv.lock` resolves mlflow to 3.9.0 which covers all three CVEs. However, the `pyproject.toml` floor of `>=2.22.4` does not protect against CVE-2025-52967 if a resolver were to pick a version <3.1.0. Consider raising the floor to `>=3.1.0`.

**Secondary threat: Pickle deserialization.** Multiple `torch.load()` calls lack `weights_only=True` (see Section 4.4) and there are 12+ bare `pickle.load()` calls across `src/orchestration/`, `src/data_io/`, `src/stats/`, `src/viz/`, and `src/log_helpers/`. The 542 MLflow pickle artifacts are the primary data transport format.

### SecureAI-Flow Threat Model (Rahman & Biplob 2025)

Maps ML-specific threats to CI/CD stages:

| Stage | Threat | Foundation PLR Status |
|-------|--------|-----------------------|
| Data collection | Data poisoning | Fixed dataset, provenance documented |
| Training | Metric spoofing | MLflow tracks all runs, registry validates |
| Model storage | Model tampering | No signing yet (gap) |
| Deployment | Drift exploitation | No monitoring yet (gap, but no deployment) |
| Dependencies | Supply chain compromise | Dependabot + uv lockfile + floor bumps |

---

## 3. Reproducibility Hardening {#3-reproducibility-hardening}

### 3.1 Sources of Irreproducibility (Han 2025)

Han (2025, BMC Medical Genomics) identifies 5 categories of irreproducibility in biomedical AI. Current status for foundation-PLR:

| Source | Risk Level | Current Mitigation | Improvement |
|--------|------------|-------------------|-------------|
| **Random seeds** | MEDIUM | Seeds set in config, but not all paths documented | Document ALL seed locations in a reproducibility checklist |
| **Hardware non-determinism** | LOW | Docker containers standardize environment | Add `torch.use_deterministic_algorithms(True)` option |
| **Data variations** | LOW | Fixed dataset, frozen configs | Add data hashing to verify dataset integrity |
| **Preprocessing randomness** | LOW | Deterministic pipeline (DuckDB queries) | Already mitigated by two-block architecture |
| **Framework differences** | LOW | Pinned via uv.lock | Add environment card documenting exact versions |

### 3.2 Event Sourcing for Reproducibility (Beber 2025)

Beber (2025) proposes event sourcing (immutable event logs) for perfect replication:

**Already implemented (analogous):**
- Git history = immutable event log of code changes
- MLflow = immutable event log of experiment runs
- Config versioning system = content-hashed configs with frozen protection

**Could improve:**
- **Standards compliance**: PMML/ONNX model export for interoperability (Rahrooh 2023)
- **Contribution tracking**: ORCID integration for multi-author provenance

### 3.3 Reproducibility Framework (Desai et al. 2025)

Desai et al. (2025, AI Magazine) propose a taxonomy:

| Level | Definition | Foundation PLR Status |
|-------|-----------|----------------------|
| **Repeatability** | Same team, same setup, same result | ACHIEVED (Docker + lockfiles) |
| **Dependent reproducibility** | Different team, same code/data | ACHIEVABLE (public repo + demo data) |
| **Independent reproducibility** | Different team, different implementation | PARTIALLY (frozen configs document choices) |
| **Direct replicability** | Same method, new data | NOT YET (no external dataset tested) |
| **Conceptual replicability** | Same concept, different method | OUT OF SCOPE |

**Recommendation**: Document the reproducibility level explicitly in a `REPRODUCIBILITY.md` or Model Card.

### 3.4 NeurIPS 2025 Reproducibility Checklist

The NeurIPS checklist provides a publication-ready standard:

- [ ] Code availability (public GitHub repo)
- [ ] Data availability (SERI dataset reference, demo data)
- [ ] Computing infrastructure documented (Docker, GPU requirements)
- [ ] Statistical significance (bootstrap CIs)
- [ ] Experiment repetition details (1000 bootstrap iterations)
- [ ] Random seed documentation
- [ ] Hyperparameter search details (Optuna/hyperopt configs)
- [ ] Train/val/test split methodology (patient-level, documented)

**Status**: Most items covered. Gap: formal checklist document linking to evidence.

---

## 4. Supply Chain Security {#4-supply-chain-security}

### 4.1 ML Bill of Materials (ML-BOM)

Two competing standards have matured in 2025:

| Standard | ML Support | Maturity | Tooling |
|----------|-----------|----------|---------|
| **CycloneDX v1.5+** | ML-BOM profile (models, datasets, training configs) | Production | `cyclonedx-bom` CLI |
| **SPDX 3.0.1** | AI/Dataset profiles | Production | `spdx-tools` |
| **OWASP AI-BOM** | Comprehensive (data + model + infra) | v0.1 (Nov 2025) | Early tooling |

**Recommended for foundation-PLR**: CycloneDX ML-BOM because:
1. Captures model architecture, training data provenance, preprocessing pipeline
2. Machine-readable (JSON/XML)
3. Compatible with CISA 2025 SBOM minimum elements
4. `cyclonedx-py` generates Python dependency SBOMs directly from `pyproject.toml`

### 4.2 Model Signing (OpenSSF OMS v1.0, April 2025)

The OpenSSF Model Signing specification reached v1.0 in April 2025. NVIDIA now signs all NGC models; Google is prototyping on Kaggle Hub.

**How it works:**
1. Hash model contents (weights, configs, tokenizers)
2. Sign with Sigstore (keyless) or PKI
3. Store detached signature in OMS format
4. Log signing event in Rekor transparency log
5. Verify at load time

**For foundation-PLR:** Sign MLflow artifacts and DuckDB checkpoint databases:
```bash
uv add model-signing
# Sign a model artifact
model-signing sign --model mlruns/253031330985650090/.../artifacts/model
# Verify before loading
model-signing verify --model mlruns/253031330985650090/.../artifacts/model
```

### 4.3 OpenSSF MLSecOps Whitepaper (August 2025)

The OpenSSF published a comprehensive whitepaper mapping security controls to ML pipeline stages using SLSA, Sigstore, and OpenSSF Scorecard. Key recommendations:

| Stage | Control | Foundation PLR Applicability |
|-------|---------|------------------------------|
| Data ingestion | Data provenance, integrity checks | Dataset hash verification |
| Training | Experiment tracking, reproducible builds | MLflow + Docker |
| Model storage | Signing, access control | Model signing (gap) |
| CI/CD | SLSA provenance, dependency scanning | Dependabot + lockfile |
| Deployment | Runtime monitoring, model verification | Not deployed (N/A for now) |

### 4.4 ModelScan (Protect AI)

Scans ML model files for unsafe code (pickle deserialization attacks, embedded scripts):

```bash
uv add modelscan
modelscan scan -p path/to/model.pkl
```

**Relevant for**: Any `torch.load()` or `pickle.load()` in the pipeline. **WARNING**: Contrary to project documentation, `weights_only=True` is NOT consistently used:
- `src/anomaly_detection/momentfm_outlier/moment_io.py`: `torch.load()` without `weights_only`
- `src/anomaly_detection/units/units_outlier.py`: `torch.load()` without `weights_only`
- `src/classification/tabpfn/model/loading.py`: `weights_only=None` (defaults to False)
- `src/classification/tabpfn_v1/scripts/model_builder.py`: `torch.load()` without `weights_only`
- 12+ bare `pickle.load()` calls across `src/` without sandboxing

ModelScan provides defense-in-depth by scanning serialized files for embedded code before loading.

---

## 5. Model & Data Governance {#5-model--data-governance}

### 5.1 Model Cards (Increasingly Mandated)

Model cards have evolved from a Google research proposal (Mitchell et al. 2019) to **effectively required** by regulation, though the "model card" format itself is not prescribed:
- **EU AI Act** (August 2025): GPAI providers must publish technical documentation (Annex IV) -- legally binding
- **FDA SaMD draft guidance** (January 2025): Recommends model description, data lineage, bias analysis -- draft, not yet legally binding
- **NIST AI RMF** (supplementary materials, 2025): Recommends model provenance and documentation -- voluntary framework

**Recommended Model Card for CatBoost classifier:**

```yaml
# configs/governance/model_card.yaml
model_name: "CatBoost Glaucoma Classifier"
version: "1.0 (publication freeze)"
model_type: "Gradient Boosted Decision Tree (CatBoost)"
task: "Binary classification (glaucoma vs control)"
intended_use:
  primary: "Research benchmark for preprocessing effect evaluation"
  out_of_scope: "Clinical diagnosis without further validation"
training_data:
  source: "Najjar et al. 2023, Br J Ophthalmol"
  n_subjects: 208
  class_distribution: "152 control (73.1%) + 56 glaucoma (26.9%)"
  geographic_origin: "Singapore (SNEC)"
  demographic_note: "Single-center, predominantly Asian population"
evaluation_metrics:
  auroc: "0.913 [95% CI from 1000 bootstrap iterations]"
  calibration_slope: "See DuckDB essential_metrics table"
  calibration_intercept: "See DuckDB essential_metrics table"
  net_benefit: "See DCA curves at 5-20% thresholds"
  brier_score: "See DuckDB essential_metrics table"
clinical_context:
  disease: "Open-angle glaucoma"
  clinical_task: "Population screening (secondary analysis of PLR recordings)"
  clinical_workflow: "Not integrated into any clinical workflow (research only)"
  decision_threshold: "Not established; DCA curves available for threshold selection"
  intended_use_statement: >
    Research tool for evaluating effect of preprocessing choices on classification
    performance. NOT for clinical decision-making without further validation.
regulatory_status:
  fda: "Not submitted; research use only"
  ce_mark: "Not applicable"
  irb_ethics: "Original SERI study approval [reference needed from Najjar 2023]"
sample_size_adequacy:
  n_total: 208
  n_events: 56
  events_per_variable: "TBD based on final feature count"
  note: "Borderline for stable AUROC (Riley: ~100 events recommended)"
validation:
  internal: "Stratified bootstrap (1000 iterations)"
  external: "None -- single-center dataset (primary scientific limitation)"
  temporal: "None -- cross-sectional"
limitations:
  - "Single-center dataset (Singapore/SNEC only, predominantly Asian population)"
  - "Limited to handcrafted PLR features"
  - "Small sample size (N=208, 56 events) limits generalizability"
  - "No external validation cohort"
  - "Disease prevalence in dataset (26.9%) >> population prevalence (3.54%)"
  - "All reported metrics are analytical performance only (not clinical performance)"
known_failure_modes:
  - "Poor signal quality PLR recordings (high artifact percentage)"
  - "Patients on pupil-affecting medications (miotics, mydriatics, alpha-agonists)"
  - "Non-glaucomatous pupil abnormalities (Adie's, Horner's, third nerve palsy)"
ethical_considerations:
  - "Not validated for clinical use -- screening only, not diagnostic"
  - "Potential bias toward Asian populations (Singapore single-center)"
  - "Subject privacy protected via anonymization pipeline"
  - "Generalizability to other ethnicities/populations is unknown"
  - "Known physiological variations in pupil dynamics across populations"
```

### 5.2 Data Cards (Gebru et al. 2021)

```yaml
# configs/governance/data_card.yaml
dataset_name: "SERI PLR Glaucoma Dataset (Subset)"
source: "Najjar et al. 2023, DOI: 10.1136/bjophthalmol-2021-319938"
collection_institution: "Singapore Eye Research Institute (SERI)"
subjects:
  total_preprocess: 507
  total_classify: 208
  controls: 152
  glaucoma: 56
data_format:
  raw: "DuckDB (SERI_PLR_GLAUCOMA.db)"
  processed: "DuckDB (foundation_plr_results.db)"
  features: "Handcrafted physiological PLR features"
ground_truth:
  outlier_masks: "Expert-annotated blink/artifact masks for 507 subjects"
  classification_labels: "Clinical diagnosis (glaucoma/control) for 208 subjects"
known_biases:
  geographic: "Singapore only (predominantly Asian population)"
  device: "Single pupillometer model"
  temporal: "Cross-sectional (single timepoint per subject)"
ethics:
  original_study_approval: "Reference: Najjar et al. 2023 (ethics approval details in original paper)"
  secondary_use: "Covered by original consent [verify with SERI DUA]"
  re_identification_risk: "Low (pseudonymized, no public individual-level outputs)"
privacy:
  anonymization: "Original PLRxxxx codes replaced with Hxxx/Gxxx codes"
  private_data: "data/private/ (gitignored), subject-level JSONs excluded"
  public_data: "Aggregate statistics only in data/public/"
  git_history: "Should be audited for any historical PLRxxxx leaks before making repo fully public"
demographics:
  age_distribution: "TBD -- document from original dataset"
  sex_distribution: "TBD -- document from original dataset"
  ethnicity: "Predominantly Chinese, Malay, Indian (Singapore population)"
  note: "Generalizability to non-Asian populations unknown"
splits:
  method: "Patient-level splitting (no data leakage)"
  validation: "Stratified bootstrap (1000 iterations)"
```

### 5.3 Extended Cards Trail (from Appendix-Cards + UAD-Copilot)

Full card taxonomy adapted for an academic biomedical ML repository. Sources: Appendix B of MCP Review, UAD-copilot Cards Trail methodology.

#### Essential Cards (Implement Pre-Publication)

| Card | Purpose | Content Source | Effort |
|------|---------|----------------|--------|
| **Data Card** | Dataset provenance, demographics, biases, privacy | 80% from `methods.tex` + `results.tex` (TRIPOD Item 20) | 1 hr (mostly extraction) |
| **Model Card** | CatBoost architecture, STRATOS metrics, limitations | 70% from `methods.tex` + `supplementary.tex` (Table S3) | 1.5 hrs |
| **Use Case Card** | Clinical decision context, risk taxonomy, stakeholders | 60% from `discussion.tex` (clinical translation section) | 1 hr |

#### Recommended Cards (Strengthen Reproducibility & Safety)

| Card | Purpose | Content Source | Effort |
|------|---------|----------------|--------|
| **Failure Notes** | Edge cases, failure modes, mitigations | `discussion.tex` (limitations), `results.tex` (calibration caveats, EPV issues) | 1 hr |
| **Environment Card** | Python/R/DuckDB versions, Docker digest, OS | `methods.tex` line 114-116, `pyproject.toml`, `renv.lock` | 30 min |
| **Reproducibility Checklist** | NeurIPS-style seeds/configs/harness checklist | Existing Hydra configs + `supplementary.tex` (Table S1 hyperparams) | 45 min |
| **Saliency/XAI Card** | Feature importance, SHAP with VIF caveats | `supplementary.tex` lines 199-233 (SHAP + VIF analysis) | 1 hr |

#### Optional Cards (Post-Publication or If Deployed)

| Card | Purpose | When Needed | Effort |
|------|---------|-------------|--------|
| **Feedback Card** | Stakeholder corrections/overrides (Zhou et al. 2023) | If deployed clinically with HITL | 2 hrs |
| **AI Usage Card** | Where/when AI used in pipeline (Luccioni et al. 2023) | Manuscript supplementary or audit | 30 min |
| **ESG Card** | Compute costs, equity impact (Binns et al. 2023) | Grant proposals, equity assessment | 2 hrs |
| **Schema Evolution Card** | DuckDB schema changes, config migrations | If pipeline is actively maintained | 1 hr |
| **Deployment Lineage Card** | Git SHA + DVC tag + CI metadata | If artifacts are distributed | 30 min |
| **Prompt Card** | Version control for LLM prompt templates | If using Claude for analysis generation | 30 min |

#### Why Feedback Cards Are Low Priority

Feedback Cards (Zhou et al. 2023) document stakeholder corrections, overrides, and continual learning feedback. They are essential for **production HITL systems** (e.g., Maria Platform's 81% approve/19% correct workflow). For foundation-PLR:
- No deployment = no stakeholder feedback loop
- Frozen model = no continual learning
- **Becomes relevant if**: Model is deployed in a virtual glaucoma clinic with ophthalmologist review

### 5.4 Manuscript Content Reuse Map

Most card content already exists in the manuscript -- the effort is **extraction and reformatting**, not creation from scratch.

| Card Field | Manuscript Source | Lines | Extraction Effort |
|------------|-------------------|-------|-------------------|
| Study population | `methods.tex` | 12-20 | Copy + reformat |
| Signal protocol | `methods.tex` | 18 | Copy |
| Ground truth limitations | `methods.tex` | 29-33 | Copy |
| Pipeline architecture | `methods.tex` | 35-66 | Summarize |
| 11 outlier methods | `methods.tex` | 39-50 | Copy from registry |
| 8 imputation methods | `methods.tex` | 51-59 | Copy from registry |
| STRATOS metrics | `methods.tex` | 104-112 | Copy |
| Software stack | `methods.tex` | 114-116 | Copy |
| Demographics (N=208) | `results.tex` | 13-16 | Copy (TRIPOD Item 20) |
| Calibration caveats | `results.tex` | 106-113 | Copy |
| DCA caveats | `results.tex` | 121-128 | Copy |
| Key AUROC results | `results.tex` | 62-72 | Extract numbers |
| Stability results | `results.tex` | 178-185 | Extract numbers |
| VIF/SHAP analysis | `supplementary.tex` | 224-233 | Summarize |
| Hyperparameter table | `supplementary.tex` | 242-271 | Copy |
| Age confounding | `discussion.tex` | 71-73 | Copy |
| Spectrum bias | `discussion.tex` | 79-80 | Copy |
| Single-annotator caveat | `discussion.tex` | 77 | Copy |
| Multimodal architecture | `discussion.tex` | 50-58 | Summarize |
| Clinical deployment context | `discussion.tex` | 41-58 | Summarize |

**Estimated total effort for all Essential + Recommended cards**: ~6 hours (mostly copy-edit from manuscript, not writing from scratch).

### 5.5 Manuscript-to-Repository Transfer Plan

When the manuscript is accepted for publication, governance artifacts should be transferred to the code repository:

```
Manuscript (sci-llm-writer)          Code Repository (foundation_PLR)
─────────────────────────────        ──────────────────────────────────
methods.tex (pipeline desc)    →     configs/governance/model_card.yaml
results.tex (TRIPOD Item 20)  →     configs/governance/data_card.yaml
discussion.tex (limitations)  →     configs/governance/failure_notes.md
supplementary.tex (Table S1)  →     configs/governance/hyperparameters.yaml
results-ground-truth.json     →     data/public/verified_results.json
DATA-SOURCE-MAP.md            →     docs/DATA-SOURCE-MAP.md
```

**Transfer protocol**:
1. After acceptance, create a `docs/manuscript/` directory with accepted PDF + supplementary
2. Extract governance YAML/MD cards from LaTeX source
3. Cross-reference card fields to manuscript section numbers for traceability
4. Add `manuscript_doi: "10.xxxx/..."` to all cards once published
5. Update README with publication reference

---

## 6. Clinical Translation Readiness {#6-clinical-translation-readiness}

### 6.1 FDA SaMD/AI Guidance (January 2025)

The FDA's Total Product Life Cycle (TPLC) approach for AI-enabled devices requires:

| Requirement | Foundation PLR Status | Gap |
|------------|----------------------|-----|
| Model description & architecture | Partial (code exists, no formal doc) | Create model card |
| Data lineage documentation | Good (two-block pipeline) | Formalize data card |
| Performance metrics tied to clinical claims | Excellent (STRATOS metrics) | None |
| Bias analysis & mitigation | Partial (acknowledged, not quantified) | Demographic analysis |
| Human-AI workflow documentation | N/A (research, not deployed) | Future |
| Post-market monitoring plan | N/A | Future |
| **SBOM for all software components** | Partial (pyproject.toml + lockfile) | Generate CycloneDX ML-BOM |
| **Predetermined Change Control Plan** | N/A (frozen for publication) | Future if model updates planned |

### 6.2 IEC 62304 Alignment (Software Life Cycle)

Although foundation-PLR is academic research (not a medical device), IEC 62304 thinking helps structure documentation:

| IEC 62304 Process | Foundation PLR Equivalent |
|-------------------|--------------------------|
| Development planning | `docs/planning/` + CLAUDE.md |
| Requirements specification | Research question doc + STRATOS metrics |
| Design specification | ARCHITECTURE.md + two-block pipeline |
| Implementation | `src/` with pre-commit quality gates |
| Integration testing | Tiered CI/CD (unit → integration → e2e) |
| Verification | `pytest` + figure QA + registry validation |
| Risk management (ISO 14971) | Partial: data leakage prevention, no formal FMEA |
| Configuration management | Config versioning system + frozen configs |
| Problem resolution | GitHub Issues + Dependabot |

**Safety class assessment** (open question -- depends on clinical workflow):
- **Class B** (non-serious injury possible): Defensible if the tool is one of multiple screening mechanisms and a missed case still gets other screening opportunities
- **Class C** (serious injury possible): If the tool is a first-line screener where a false negative means delayed treatment and progressive irreversible vision loss from untreated glaucoma

The determination depends on the software's role in the clinical workflow. This is a risk assessment question that should be formally resolved during any clinical translation effort.

### 6.3 Quality Management (QMS) Insights

From the Clinical ML QMS knowledge base:

- **Continuous Audit-Based Certification (CABC)**: Automated audits of MLOps artifacts > point-in-time audits. Foundation-PLR's CI/CD pipeline already provides this pattern.
- **Acceptance testing**: The `pytest tests/test_figure_qa/` suite is analogous to clinical acceptance testing -- verifying outputs meet quality standards before release.
- **Shared responsibility**: Clearly document that foundation-PLR is a **research tool**, not a clinical product. Clinical deployment requires additional validation by healthcare institutions.

### 6.4 Maria Platform Lessons (Lopes et al. 2026)

The Maria Platform (production healthcare AI in Brazil) provides reference patterns:

| Pattern | Maria Implementation | Foundation PLR Adaptation |
|---------|---------------------|--------------------------|
| Clean Architecture | Domain logic isolated from infra | Two-block pipeline (extraction vs analysis) |
| Event-Driven Architecture | Publish-subscribe for auditability | Git history + MLflow as event logs |
| Human-in-the-Loop | Clinician review + feedback | N/A (research) but documented for future |
| Shadow/canary deployment | Gradual rollout with auto-rollback | N/A |
| Golden set evaluation | Curated test cases | Ground truth subjects (507 with masks) |

**Key lesson**: Maria's 81% approve / 19% correct / 0% reject rate shows that **upstream engineering quality** (pre-commit hooks, CI/CD gating) prevents catastrophic failures in production. Foundation-PLR's strict gating already follows this pattern.

---

## 7. Monitoring & Drift Detection {#7-monitoring--drift-detection}

### 7.1 ML Monitoring Landscape (Naveed 2025)

Naveed's systematic review (136 papers) identifies 5 monitoring goals:

| Goal | Relevance to Foundation PLR |
|------|----------------------------|
| Detect model degradation (55.1%) | FUTURE: If deployed, track AUROC/calibration drift |
| Detect data quality issues (48.5%) | CURRENT: Pre-commit checks, figure QA |
| Ensure responsible ML (24.3%) | PARTIAL: Privacy enforced, fairness not quantified |
| Operational monitoring (23.0%) | LOW: Local pipeline, no latency/throughput concerns |
| Label scarcity handling (3.3%) | LOW: Fixed labeled dataset |

### 7.2 What Monitoring Would Look Like for PLR

If foundation-PLR were deployed clinically:

```yaml
# Hypothetical monitoring config
monitoring:
  data_drift:
    method: "Kolmogorov-Smirnov test on feature distributions"
    frequency: "daily"
    alert_threshold: 0.05
    features_tracked:
      - "constriction_velocity"
      - "redilation_t63"
      - "baseline_pupil_diameter"

  model_performance:
    method: "Rolling window AUROC on labeled subset"
    window_size: 100
    alert_threshold: 0.85  # Alert if AUROC drops below

  calibration_drift:
    method: "Calibration slope on rolling window"
    acceptable_range: [0.8, 1.2]

  fairness:
    method: "Demographic parity across age groups"
    groups: ["<40", "40-60", ">60"]
```

**Tools referenced in literature**: Evidently AI, WhyLabs, Alibi Detect, Great Expectations, Amazon SageMaker Model Monitor.

### 7.3 AIAppOps Framework (Jonsson et al. 2026)

Emphasizes monitoring as the **unifying mechanism** across the ML lifecycle:
- Statistical verification (distribution tests, performance metrics)
- Formal verification (invariant checking, constraint satisfaction)
- Runtime verification (safety-critical AI extensions)

**For foundation-PLR**: The existing pre-commit + CI/CD pipeline is already a form of "build-time monitoring." Runtime monitoring becomes relevant only if the pipeline is operationalized.

---

## 8. Regulatory Alignment {#8-regulatory-alignment}

### 8.1 EU AI Act Timeline

| Date | Milestone | Impact on Foundation PLR |
|------|-----------|--------------------------|
| Feb 2025 | Prohibited practices apply | N/A (Article 2(6) research exemption: "AI systems exclusively for scientific research and development") |
| Aug 2025 | GPAI obligations apply | N/A (not a GPAI provider) |
| Aug 2026 | Full applicability | If commercialized: definitively high-risk (Annex III, Section 1a -- medical device Class IIa+ under MDR) |

**Key requirement if commercialized**: Technical documentation making model development, training, and evaluation traceable. Foundation-PLR's frozen configs + MLflow tracking already provide this foundation.

### 8.2 NIST AI RMF (March 2025 Update)

The AI RMF four-function structure maps well:

| Function | Foundation PLR Implementation |
|----------|------------------------------|
| **GOVERN** | CLAUDE.md rules, pre-commit hooks, CI/CD gating |
| **MAP** | Research question document, STRATOS metrics scope |
| **MEASURE** | Bootstrap CIs, calibration metrics, DCA curves |
| **MANAGE** | Frozen configs, version control, Dependabot |

### 8.3 Cross-Framework Compliance Matrix

| Requirement | EU AI Act | FDA SaMD | NIST RMF | IEC 62304 | Status |
|------------|-----------|----------|----------|-----------|--------|
| Model documentation | Required | Required | Recommended | Required | GAP: Need model card |
| Data documentation | Required | Required | Recommended | Required | GAP: Need data card |
| Risk assessment | Required | Required | Required | Required | PARTIAL: Documented limitations |
| Performance metrics | Required | Required | Required | Required | DONE: STRATOS compliant |
| Bias assessment | Required | Required | Recommended | N/A | GAP: Not quantified |
| Traceability | Required | Required | Required | Required | DONE: MLflow + git + DuckDB |
| Post-market monitoring | Required | Required | Recommended | Required | N/A (research) |
| SBOM/ML-BOM | Required | Required | Recommended | Recommended | GAP: No formal BOM |
| Version control | Required | Required | Required | Required | DONE: git + uv.lock + frozen |
| Human oversight | Required | Required | Recommended | N/A | N/A (research) |
| Clinical validation evidence | Required (Annex IV) | Required | N/A | N/A | GAP: No clinical study |
| Demographic subgroup analysis | Required | Required | Recommended | N/A | GAP: Not quantified |

---

## 8b. Preliminary Failure Mode and Effects Analysis (FMEA)

If foundation-PLR were deployed clinically, these are the critical failure modes:

| Failure Mode | Effect | Severity | Likelihood | Detection | Risk |
|-------------|--------|----------|------------|-----------|------|
| False negative (missed glaucoma) | Delayed treatment, progressive irreversible vision loss | HIGH | Medium (AUROC 0.913) | LOW (patient may not return for screening) | **HIGH** |
| False positive (false alarm) | Unnecessary specialist referral, patient anxiety, cost | Low-Medium | Medium | HIGH (specialist exam reveals no disease) | **MEDIUM** |
| Signal quality failure (poor PLR) | Unreliable prediction on noisy input | Medium | Medium (artifact prevalence varies) | MEDIUM (if quality metrics checked) | **MEDIUM** |
| Calibration drift | Incorrect risk stratification if probabilities miscalibrated | HIGH | LOW (frozen model) | LOW (without monitoring) | **MEDIUM** |
| Prevalence mismatch | PPV/NPV invalid at population prevalence (3.54% vs 26.9% training) | HIGH | HIGH (deployment differs from training) | LOW | **HIGH** |
| Population mismatch | Unknown performance on non-Asian populations | HIGH | HIGH (if deployed outside Singapore) | LOW | **HIGH** |

**Note**: This is a qualitative preliminary FMEA for planning purposes. Formal ISO 14971 risk management requires a multi-disciplinary risk assessment team.

## 8c. Additional Security Tools Not Yet Mentioned

| Tool | Purpose | Applicability |
|------|---------|---------------|
| **gitleaks** / **trufflehog** | Scan git history for secrets (hundreds of patterns vs our 5 regex) | HIGH -- supplement `compliance_check.py` |
| **nbstripout** | Strip Jupyter notebook outputs to prevent patient data leakage | HIGH -- add to pre-commit |
| **pip-audit** / **uv audit** | Local vulnerability scanning of Python dependency tree | MEDIUM -- complement Dependabot |
| **SLSA** (Supply-chain Levels for Software Artifacts) | Documented, hosted build process provenance | MEDIUM -- Level 1 is nearly free with GitHub Actions |
| **Cosign** | Sign Docker container images (complement to OMS for models) | LOW -- relevant if images are distributed |
| **OWASP AI Testing Guide** (2026) | Concrete test cases for ML systems | LOW -- more actionable than ML Top 10 |

---

## 8d. Low-Hanging Fruit Triage

Practical assessment of what's trivial vs what needs real work:

### Trivial (< 30 min each, can do in one session)

| Fix | Time | What to Do |
|-----|------|------------|
| **SECURITY.md** | 15 min | Copy GitHub template, fill in email, supported versions, scope |
| **CODEOWNERS** | 5 min | `* @petteriTeikari` |
| **Branch protection on main** | 10 min | GitHub Settings → Branches → Add rule |
| **Environment Card** | 30 min | Extract versions from `pyproject.toml`, `renv.lock`, Dockerfiles into YAML |
| **`dependabot.yml`** | 10 min | Add weekly schedule for pip + npm ecosystems |

### Light Work (1-2 hours, mostly extraction from manuscript)

| Fix | Time | What to Do |
|-----|------|------------|
| **Reproducibility checklist** | 45 min | NeurIPS template + link to existing evidence (configs, Docker, lockfiles) |
| **Data Card** | 1 hr | 80% copy from `methods.tex` + `results.tex` → YAML |
| **Model Card** | 1.5 hr | 70% copy from manuscript → YAML + add clinical fields |
| **Data integrity checksums** | 30 min | `sha256sum data/public/*.db > data/_checksums.sha256` + CI step |
| **AI Usage Card** | 30 min | Document which pipeline stages use which methods |

### Moderate Work (2-4 hours)

| Fix | Time | What to Do |
|-----|------|------------|
| **Failure Notes** | 1 hr | Extract from `discussion.tex` limitations + add PLR-specific edge cases |
| **Use Case Card** | 1 hr | Clinical decision context from `discussion.tex` |
| **Demographic reporting** | 4 hrs | Query original dataset for age/sex distributions, compute subgroup AUROCs |
| **ML-BOM generation** | 2 hrs | Install `cyclonedx-py`, generate SBOM, add `make sbom` target |
| **Saliency/XAI Card** | 1 hr | Extract SHAP + VIF analysis from `supplementary.tex` |

### Defer / Low Priority

| Item | Why Defer | Revisit When |
|------|-----------|-------------|
| **Pickle safety (ModelScan, `weights_only`)** | All pickles are internal MLflow artifacts from own training. No 3rd party pickles loaded. Shared reproducibility artifact is DuckDB (not pickle). Risk is theoretical for this use case. | If pipeline accepts external model artifacts or is deployed as a service |
| **Model signing (Sigstore/OMS)** | No model distribution. Artifacts are local. | If models are published to a registry or shared externally |
| **Runtime monitoring** | No deployment. Academic publication. | If clinical translation begins |
| **DVC** | Dataset is frozen. SHA256 checksums provide equivalent integrity for fixed data. | If dataset changes or multi-version experiments begin |
| **Feedback Card** | No HITL loop. Frozen model. | If deployed with clinician review |

---

## 9. Recommended GitHub Issues {#9-recommended-github-issues}

### Tier 1: High Priority (Pre-Handover or Early Post-Publication)

#### Issue A: Add Reproducibility Checklist Document
**Labels**: `documentation`, `reproducibility`
**Effort**: 1 hour

Create `REPRODUCIBILITY.md` with NeurIPS-style checklist linking to evidence:

- [ ] Code availability: GitHub public repo (MIT license)
- [ ] Data availability: SERI dataset reference (Najjar 2023) + demo data
- [ ] Computing infrastructure: Docker images documented, GPU requirements
- [ ] Statistical significance: Bootstrap CIs (1000 iterations, 95% CI)
- [ ] Random seed documentation: All seed locations listed with values
- [ ] Hyperparameter search: Optuna/hyperopt configs in `configs/CLS_HYPERPARAMS/`
- [ ] Train/test split: Patient-level, stratified (no data leakage)
- [ ] Environment specification: Python 3.11, R 4.4+, exact versions in `uv.lock`/`renv.lock`
- [ ] Reproducibility level: Dependent reproducibility (Desai et al. 2025 taxonomy)

**Acceptance criteria**: Document exists, all items checked or explicitly marked as gaps, linked from README.

Rationale: Semmelrock et al. (2025, AI Magazine) identify poor documentation as a primary reproducibility barrier. NeurIPS, ICML, and all major venues now require this. Highest ROI for peer review.

---

#### Issue B: Create Model Card and Data Card (Governance Documentation)
**Labels**: `documentation`, `enhancement`
**Effort**: 2 hours

Create `configs/governance/model_card.yaml` and `configs/governance/data_card.yaml` following Mitchell et al. 2019 and Gebru et al. 2021 templates:

**Model Card** contents:
- CatBoost classifier architecture, hyperparameters, HPO method
- Training data demographics (N=208, Singapore, SNEC)
- STRATOS evaluation metrics with bootstrap CIs
- Limitations (single-center, small N, no external validation)
- Ethical considerations (not for clinical use without validation)

**Data Card** contents:
- SERI PLR dataset provenance (Najjar 2023)
- Subject counts per task (507 preprocess, 208 classify)
- Ground truth annotation methodology
- Known biases (geographic, device, temporal)
- Privacy architecture (anonymization, gitignore enforcement)

Additional fields to include (from clinical reviewer):
- Clinical context (disease, workflow, decision threshold)
- Regulatory status (research use only)
- Sample size adequacy (Riley criteria, events per variable)
- Known failure modes (poor signal quality, medications, non-glaucomatous pupil abnormalities)
- IRB/ethics approval reference
- Prevalence-adjusted PPV/NPV at population prevalence (3.54%)
- Feature importance rankings (SHAP or CatBoost feature importance)
- TRIPOD+AI checklist cross-references

**Acceptance criteria**: Both YAML files exist, all fields populated or explicitly marked "TBD", referenced from README.

Rationale: Model card content is effectively required by EU AI Act (Annex IV technical documentation) and recommended by FDA SaMD draft guidance. Even for academic work, they demonstrate rigor and facilitate future clinical translation.

---

#### Issue C: Demographic Reporting and Bias Assessment
**Labels**: `documentation`, `fairness`, `enhancement`
**Effort**: 4 hours

Even with N=208, document and report:
1. Age and sex distributions for the full cohort (controls vs glaucoma)
2. Age-stratified AUROC (even if CIs are wide, point estimates are informative)
3. Explicit statement that dataset is single-center, single-ethnicity (Singaporean: Chinese, Malay, Indian)
4. Known physiological variations in pupil dynamics across populations (pupil size varies with iris pigmentation, age, medication)
5. Generalizability limitations documented in model card

**Acceptance criteria**: Demographics table in data card, at minimum 2 subgroup analyses reported (age-stratified, sex-stratified), limitations documented.

Rationale: FDA SaMD guidance (2025) requires test data representative of intended use populations. EU AI Act requires bias analysis for high-risk systems. Journals increasingly flag the absence of demographic reporting as a deficiency regardless of sample size. Even acknowledging that the dataset is too small for robust subgroup analysis is itself valuable documentation.

---

### Tier 2: Medium Priority (Post-Publication Enhancements)

#### Issue D: Generate ML-BOM (Machine Learning Bill of Materials)
**Labels**: `security`, `enhancement`
**Effort**: 2 hours

Generate a CycloneDX ML-BOM capturing:
1. Python dependency SBOM from `pyproject.toml` (via `cyclonedx-py`)
2. Model component metadata (CatBoost version, training config hash)
3. Dataset provenance (Najjar 2023 DOI, preprocessing pipeline version)
4. R dependency listing from `renv.lock`
5. npm dependencies from `apps/visualization/package.json`

Add `make sbom` target and CI step to regenerate on dependency changes.

Rationale: CISA 2025 Minimum Elements for SBOM now recommended for all software. FDA SaMD guidance requires SBOM. CycloneDX ML-BOM extends traditional SBOM to cover ML-specific components (models, datasets, training configs).

References:
- [CycloneDX ML-BOM](https://cyclonedx.org/capabilities/mlbom/)
- [CISA 2025 SBOM Minimum Elements](https://www.cisa.gov/resources-tools/resources/2025-minimum-elements-software-bill-materials-sbom)
- [SPDX 3.0 AI Profile](https://spdx.dev/)

---

#### Issue E: Implement Model Artifact Signing with Sigstore/OMS
**Labels**: `security`, `enhancement`
**Effort**: 4 hours

Adopt OpenSSF Model Signing (OMS v1.0, April 2025) for model provenance:
1. Sign MLflow model artifacts at training completion
2. Sign DuckDB checkpoint database after extraction
3. Verify signatures before loading in analysis pipeline
4. Log signing events to Sigstore transparency log (Rekor)

```bash
uv add model-signing
# Sign: model-signing sign --model path/to/artifact
# Verify: model-signing verify --model path/to/artifact
```

Add `make sign-models` and `make verify-models` targets.

Rationale: NVIDIA signs all NGC models since March 2025. Google prototyping on Kaggle. OMS v1.0 is production-ready. Provides cryptographic proof that model artifacts haven't been tampered with -- essential for research integrity and future clinical translation.

References:
- [OpenSSF OMS Specification](https://github.com/ossf/model-signing-spec)
- [Google Security Blog: Model Signing](https://security.googleblog.com/2025/04/taming-wild-west-of-ml-practical-model.html)
- [NVIDIA NGC Model Signing](https://developer.nvidia.com/blog/bringing-verifiable-trust-to-ai-models-model-signing-in-ngc/)

---

#### Issue F: Add Data Integrity Verification
**Labels**: `enhancement`, `reproducibility`
**Effort**: 1 hour

For a frozen dataset, a lightweight checksum approach is more appropriate than full DVC:
1. Create `data/_checksums.sha256` with SHA256 hashes of all data files
2. Add CI step: `sha256sum -c data/_checksums.sha256` to verify integrity
3. Add `make verify-data` target
4. Document data provenance (Najjar 2023 DOI, extraction pipeline version) in data card

**Note**: Full DVC is recommended only if the dataset changes in future work. For the current frozen publication, checksums provide equivalent integrity guarantees with zero additional tooling.

Rationale: Code versioning (git) without data integrity verification leaves a reproducibility gap. SHA256 checksums are the simplest effective solution for fixed datasets.

---

#### Issue G: Document Pickle Safety Posture (Deferred Fix)
**Labels**: `security`, `documentation`
**Effort**: 1 hour

**Context**: All pickle/torch.load calls in this repo load **own MLflow artifacts** from own training runs -- no 3rd party or untrusted pickles. The shared reproducibility artifact is DuckDB (not pickle). Risk is theoretical for this use case.

**What to do now** (documentation):
1. Document the pickle safety posture in SECURITY.md: "All serialized model artifacts are from internal training. No external pickles are loaded."
2. Add `# SAFETY: internal artifact, no untrusted input` comments to key `torch.load()` and `pickle.load()` calls

**What to do later** (if the threat model changes):
1. Add `weights_only=True` to `torch.load()` calls where possible
2. Add ModelScan for defense-in-depth
3. Audit all serialization formats

Rationale: Pickle deserialization is a real ML supply chain threat (OWASP ML06), but the actual risk for this repo is LOW because all artifacts are self-generated. Document the posture now; implement technical controls if/when external artifacts are introduced.

---

#### Issue H: Add SECURITY.md with Vulnerability Disclosure Process
**Labels**: `documentation`, `security`
**Effort**: 30 minutes

Create a `SECURITY.md` following GitHub's standard template:
- Supported versions (current publication version)
- Reporting process (email, not public issues, for security vulnerabilities)
- Response timeline (90-day responsible disclosure)
- Known limitations (local pipeline, not production-hardened)

**Acceptance criteria**: File exists at repo root, linked from README.

Rationale: GitHub recommends this for all public repos. Signals security awareness to reviewers.

---

#### Issue I: GitHub Repository Hardening (Branch Protection)
**Labels**: `security`, `devops`
**Effort**: 1 hour

1. Enable branch protection rules on `main`:
   - Require PR review (1+ approval)
   - Require CI pass before merge
   - Dismiss stale reviews on new commits
   - Restrict force pushes
2. Add `CODEOWNERS` file (assign `@petteriTeikari` to all paths)
3. Configure `dependabot.yml` for automated weekly security scans

Rationale: Prevents accidental force-push to main (already happened once with PR #47). Low effort, high value for a publication-frozen repo.

---

### Tier 3: Low Priority (Long-Term / If Operationalized)

#### Issue J: Implement Runtime Monitoring Framework (If Deployed)
**Labels**: `enhancement`, `monitoring`
**Effort**: 8+ hours

Prototype monitoring for potential clinical deployment:
1. Data drift detection (KS test on feature distributions)
2. Model performance tracking (rolling AUROC on labeled subset)
3. Calibration drift monitoring (slope in [0.8, 1.2] range)
4. Alert pipeline (Evidently AI or custom dashboard)
5. Document monitoring SLAs and escalation procedures

Rationale: Naveed (2025) systematic review shows monitoring is the most common gap in ML systems. Jonsson et al. (2026) AIAppOps framework positions monitoring as the unifying mechanism for compliance. Not needed for academic publication but essential for translation.

---

#### Issue K: Environment Card and Docker Digest Pinning
**Labels**: `reproducibility`, `security`
**Effort**: 2 hours

1. Create `configs/governance/environment_card.yaml`:
   - Python version, R version, Node.js version
   - OS (Ubuntu version in Docker)
   - GPU requirements (optional, for training)
   - Docker image digest (SHA256)
2. Pin Docker base images with SHA256 digests (relates to existing GH#43)
3. Add CI check that environment card matches actual Docker build

Rationale: Han (2025) identifies hardware/software environment differences as a key source of irreproducibility. Environment cards (UAD-copilot Cards Trail) provide a formal record.

---

### Issue Priority Matrix

| Issue | Title | Tier | Effort | Primary Driver |
|-------|-------|------|--------|----------------|
| **A** | Reproducibility checklist | 1 (High) | 1 hr | NeurIPS/peer review |
| **B** | Model Card + Data Card | 1 (High) | 2 hrs | EU AI Act, FDA SaMD draft, TRIPOD+AI |
| **C** | Demographic bias assessment | 1 (High) | 4 hrs | FDA SaMD, EU AI Act, journal expectations |
| **D** | ML-BOM generation | 2 (Medium) | 2 hrs | CISA 2025, SBOM best practice |
| **E** | Model signing (Sigstore/OMS) | 2 (Medium) | 4 hrs | OpenSSF, supply chain integrity |
| **F** | Data integrity verification | 2 (Medium) | 1 hr | Reproducibility |
| **G** | Document pickle safety posture | 2 (Medium) | 1 hr | OWASP ML06, documentation |
| **H** | SECURITY.md | 2 (Medium) | 30 min | GitHub best practice |
| **I** | GitHub repo hardening | 2 (Medium) | 1 hr | Prevent force-push accidents |
| **J** | Runtime monitoring (if deployed) | 3 (Low) | 8+ hrs | Clinical translation |
| **K** | Environment card + Docker digest | 3 (Low) | 2 hrs | Reproducibility |

---

## 10. References {#10-references}

### Seed Papers (from biblio)

1. **Beber (2025)**. Total reproducibility via event sourcing in computational systems biology.
2. **Denniss et al. (2025)**. Pupil melanopsin biomarker glaucoma study (reproducible clinical protocols).
3. **Donkor et al. (2024)**. Computing in life sciences: algorithms to modern AI.
4. **Han (2025)**. Challenges of reproducible AI in biomedical data science. *BMC Medical Genomics*.
5. **Jonsson et al. (2026)**. AIAppOps: Socio-technical framework for data-driven AI operations.
6. **Larrazabal et al. (2022)**. Mitigating bias in radiology ML. *RSNA*.
7. **Lopes et al. (2026)**. Maria Platform: Clinical AI case study. *CAIN '26*.
8. **Naveed (2025)**. ML monitoring systems: Multivocal literature review (136 papers).
9. **Rahrooh et al. (2023)**. Interoperability and reproducibility framework (PREMIERE/PMML). *J Biomed Inform*.
10. **Rahman & Biplob (2025)**. SecureAI-Flow: CI/CD security framework (multi-agent DevSecOps).

### Standards & Frameworks

11. **OWASP ML Security Top 10 v0.3** (2023). [mltop10.info](https://mltop10.info/)
12. **OpenSSF Model Signing Spec v1.0** (April 2025). [github.com/ossf/model-signing-spec](https://github.com/ossf/model-signing-spec)
13. **OpenSSF MLSecOps Whitepaper** (August 2025). [openssf.org](https://openssf.org/blog/2025/08/05/visualizing-secure-mlops-mlsecops-a-practical-guide-for-building-robust-ai-ml-pipeline-security/)
14. **CycloneDX ML-BOM** (2025). [cyclonedx.org/capabilities/mlbom/](https://cyclonedx.org/capabilities/mlbom/)
15. **CISA 2025 Minimum Elements for SBOM**. [cisa.gov](https://www.cisa.gov/resources-tools/resources/2025-minimum-elements-software-bill-materials-sbom)
16. **SPDX 3.0.1 AI Profile** (2025). [spdx.dev](https://spdx.dev/)
17. **OWASP AI-BOM v0.1** (November 2025). [owasp.org](https://owasp.org/www-project-aibom/)

### Regulatory

18. **EU AI Act** (effective August 2024, full applicability August 2026). [digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
19. **FDA Draft Guidance: AI-Enabled DSFs** (January 2025). [fda.gov](https://www.fda.gov/media/166704/download)
20. **NIST AI RMF 1.0** (March 2025 update). [nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)
21. **NIST AI 600-1: GenAI Risk Profile** (July 2024). [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
22. **NIST IR 8596: Cyber AI Profile** (December 2025 draft). [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ir/2025/NIST.IR.8596.iprd.pdf)
23. **IEC 62304**: Medical device software lifecycle processes.
24. **ISO 14971**: Medical devices -- risk management.
25. **ISO/IEC 42001:2023**: AI management system requirements.

### Reproducibility

26. **Semmelrock et al. (2025)**. Reproducibility in ML-based research: Overview, barriers, and drivers. *AI Magazine*. DOI: 10.1002/aaai.70002
27. **Desai et al. (2025)**. What is reproducibility in AI/ML research? *AI Magazine*. DOI: 10.1002/aaai.70004
28. **NeurIPS 2025 Paper Checklist**. [neurips.cc](https://neurips.cc/public/guides/PaperChecklist)
29. **ML Reproducibility Challenge 2025** (Princeton). [reproml.org](https://reproml.org/)

### Model Documentation

30. **Mitchell et al. (2019)**. Model Cards for Model Reporting. *ACM FAccT*.
31. **Gebru et al. (2021)**. Datasheets for Datasets. *CACM*.

### Security Incidents

32. **CVE-2025-11200**: MLflow authentication bypass. [zeropath.com](https://zeropath.com/blog/mlflow-cve-2025-11200-authentication-bypass-summary)
33. **CVE-2025-11201**: MLflow directory traversal RCE. [zeropath.com](https://zeropath.com/blog/cve-2025-11201-mlflow-directory-traversal-rce)
34. **CVE-2025-52967**: MLflow SSRF. [wiz.io](https://www.wiz.io/vulnerability-database/cve/cve-2025-52967)

### Industry

35. **Protect AI ModelScan**. [github.com/protectai/modelscan](https://github.com/protectai/modelscan)
36. **Sigstore Model Transparency**. [github.com/sigstore/model-transparency](https://github.com/sigstore/model-transparency)
37. **NVIDIA NGC Model Signing** (March 2025). [developer.nvidia.com](https://developer.nvidia.com/blog/bringing-verifiable-trust-to-ai-models-model-signing-in-ngc/)

---

### Clinical & Methodological

38. **Collins GS et al. (2024)**. TRIPOD+AI statement. *BMJ*.
39. **Riley RD et al. (2021)**. Minimum sample size for developing a multivariable prediction model (pmsampsize). *Statistics in Medicine*.
40. **Van Calster B et al. (2019)**. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*.
41. **Vickers AJ, Elkin EB (2006)**. Decision curve analysis. *Medical Decision Making*.
42. **FDA (2021)**. Good Machine Learning Practice for Medical Device Development: Guiding Principles.
43. **Legha A et al. (2026)**. Sample size for clinical prediction model studies. *J Clin Epidemiol*.

---

## Appendix: Review Process

This document was reviewed by two specialized AI reviewer agents:

1. **Security reviewer** (MLSecOps engineer perspective): Identified factual errors (`weights_only=True` claim, workflow/hook counts, MLflow CVE coverage gap, `pip install` violations), missing threats (git history patient ID leaks, pickle deserialization exposure, Dropbox sync surface), and priority adjustments (ModelScan elevated, reproducibility checklist elevated).

2. **Clinical AI reviewer** (FDA/EU AI Act/IEC 62304 expert perspective): Identified missing clinical content (intended use statement, analytical vs clinical performance distinction, FMEA, TRIPOD+AI alignment, sample size adequacy, external validation planning), regulatory accuracy corrections (draft guidance vs legal requirement, safety class nuance), and priority adjustments (demographic reporting elevated to Tier 1, ML-BOM deprioritized).

Key corrections applied from reviews:
- Fixed false claim about `weights_only=True` being consistently used (it is not)
- Corrected "legally required" to "effectively required / recommended" for model cards
- Added clinical fields to model card template (failure modes, prevalence adjustment, clinical context)
- Added FMEA table and additional security tools section
- Elevated demographic reporting from Tier 3 to Tier 1
- Elevated reproducibility checklist from Tier 2 to Tier 1
- Added IRB/ethics and demographics fields to data card template
- Fixed all `pip install` references to `uv add`
- Corrected workflow count (5, not 6) and hook count (9, not 8)

---

*Generated 2026-02-08 by Claude Code with dual reviewer agent optimization. This document is a living roadmap -- priorities may shift based on publication timeline and clinical translation decisions.*
