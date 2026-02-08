# Reproducibility Checklist

NeurIPS-style reproducibility checklist for the Foundation PLR project.
Each item links to the evidence in this repository.

**Reproducibility level**: Dependent reproducibility (Desai et al. 2025 taxonomy) --
same code + same data + different environment is expected to match within bootstrap CI.

---

## Code Availability

- [x] **Source code**: GitHub public repo, [MIT license](LICENSE)
- [x] **Version control**: git with locked dependencies (`uv.lock`, `renv.lock`)
- [x] **Entry points**: `make reproduce` (full) / `make reproduce-from-checkpoint` (analysis only)
- [x] **Configuration**: All experiment configs in [`configs/`](configs/) via Hydra
- [x] **CI/CD**: 5 GitHub Actions workflows for automated testing

## Data Availability

- [x] **Original dataset reference**: Najjar et al. 2023, *Br J Ophthalmol* ([DOI: 10.1136/bjophthalmol-2021-319938](https://doi.org/10.1136/bjophthalmol-2021-319938))
- [x] **Demo data**: 8 stratified subjects in [`configs/demo_subjects.yaml`](configs/demo_subjects.yaml)
- [x] **Results database**: `data/public/foundation_plr_results.db` (DuckDB, 406 configurations)
- [x] **Data integrity**: SHA256 checksums in `data/_checksums.sha256`
- [ ] **Full raw data**: Available upon request (privacy-protected, requires IRB approval from SingHealth CIRB)

## Computing Infrastructure

- [x] **Docker images**: [`Dockerfile`](Dockerfile), [`Dockerfile.r`](Dockerfile.r), [`Dockerfile.shiny`](Dockerfile.shiny), [`Dockerfile.test`](Dockerfile.test)
- [x] **Python**: 3.11 (`uv` 0.9, lockfile: `uv.lock`)
- [x] **R**: 4.5.2 (`renv` 1.1.6, lockfile: `renv.lock`)
- [x] **Node.js**: 20 LTS (for `apps/visualization/`)
- [x] **Environment card**: [`configs/governance/environment_card.yaml`](configs/governance/environment_card.yaml)
- [ ] **GPU**: Optional for training (NVIDIA, any CUDA-capable). Not required for analysis from checkpoint.

## Statistical Methodology

- [x] **Bootstrap**: 1000 iterations, 95% CI, stratified resampling, 50% subsampling
- [x] **Random seeds documented**: data split = 42, CatBoost = 100, XGBoost = 44, foundation models = 13
- [x] **Train/test split**: 70/30, patient-level stratified, fixed across all 407 configurations
- [x] **STRATOS compliance**: All 5 domains reported (Van Calster et al. 2024) -- discrimination, calibration, clinical utility, distributions, overall
- [x] **No post-hoc recalibration**: Preserves diagnostic signal of overfitting
- [x] **Selection optimism**: Top-10 mean AUROC (0.911) reported alongside best config (0.913)

## Experiment Tracking

- [x] **MLflow**: All 542 runs tracked with full hyperparameters at `/home/petteri/mlruns/`
- [x] **DuckDB checkpoint**: Extracted metrics for 406 valid configurations
- [x] **Frozen configs**: Hydra configuration in [`configs/`](configs/), immutable after experiment completion
- [x] **MLflow registry**: [`configs/mlflow_registry/`](configs/mlflow_registry/) -- single source of truth for method names

## Model Documentation

- [x] **Model card**: [`configs/governance/model_card.yaml`](configs/governance/model_card.yaml)
- [x] **Data card**: [`configs/governance/data_card.yaml`](configs/governance/data_card.yaml)
- [x] **AI usage card**: [`configs/governance/ai_usage_card.yaml`](configs/governance/ai_usage_card.yaml)
- [x] **Pipeline methods**: 11 outlier detection, 8 imputation, 5 classifiers, 2 featurization approaches

## Known Limitations

- Single-center data (SNEC, Singapore) -- unknown generalizability
- Single annotator for ground truth (no inter-annotator reliability assessed)
- Sample size (N=208 classify, N=507 preprocess) limits subgroup analyses
- EPV = 7.0 for handcrafted features (below conventional threshold of 10)
- `numpy==1.25.2` hard-pinned for compatibility
- No external validation cohort
- Calibration metrics are descriptive (N=208 vs recommended 1000-2000)

## Verification Commands

```bash
# Reproduce full analysis from DuckDB checkpoint
make reproduce-from-checkpoint

# Run full test suite (~2042 tests)
make test

# Run fast smoke tests
make test-staging

# Verify data integrity
sha256sum -c data/_checksums.sha256

# Validate registry integrity
uv run pytest tests/test_guardrails/ -v
```

## Reproducibility Level (Desai et al. 2025 Taxonomy)

| Level | Definition | Status |
|-------|-----------|--------|
| **Repeatability** | Same code + same data + same environment = same results | Supported (Docker) |
| **Dependent reproducibility** | Same code + same data + different environment | Expected within bootstrap CI |
| **Independent reproducibility** | Different code + same data | Not tested |
| **Replicability** | Different data + similar methodology | Requires external validation |

## References

- Desai et al. (2025). What is reproducibility in AI/ML research? *AI Magazine*. DOI: 10.1002/aaai.70004
- Semmelrock et al. (2025). Reproducibility in ML-based research. *AI Magazine*. DOI: 10.1002/aaai.70002
- NeurIPS 2025 Paper Checklist. [neurips.cc](https://neurips.cc/public/guides/PaperChecklist)
- Van Calster et al. (2024). Performance evaluation of predictive AI models. STRATOS TG6.
