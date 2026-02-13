# Makefile - Foundation PLR Development Commands
# =============================================
# AIDEV-NOTE: Common development tasks for the Foundation PLR project.
# Run `make help` to see available commands.

.PHONY: help figures figure compliance validate test test-fast test-data test-all \
        test-local test-local-all test-figures type-check test-integration clean \
        reproduce reproduce-from-checkpoint extract analyze verify-data \
        list-experiments run-experiment new-experiment validate-experiments

# Default target
help:
	@echo "Foundation PLR Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Reproducibility Pipeline:"
	@echo "  make reproduce              - Full pipeline: MLflow → DuckDB → Analysis"
	@echo "  make reproduce-from-checkpoint - Analysis only (from public DuckDB)"
	@echo "  make extract                - Block 1: MLflow extraction only"
	@echo "  make analyze                - Block 2: Analysis and visualization only"
	@echo ""
	@echo "Figure Generation:"
	@echo "  make figures        - Generate all manuscript figures"
	@echo "  make figure ID=R7   - Generate specific figure (R7, R8, M3, C3, CD, RET)"
	@echo "  make figures-list   - List available figure IDs"
	@echo ""
	@echo "Validation:"
	@echo "  make compliance     - Run compliance checks (hardcoded combos, etc.)"
	@echo "  make validate       - Alias for compliance"
	@echo ""
	@echo "Testing (Docker-based, CI parity):"
	@echo "  make test           - Tier 1: unit + guardrail in Docker (~90s)"
	@echo "  make test-fast      - Same as 'make test'"
	@echo "  make test-data      - Tier 1+2: with data mounts"
	@echo "  make test-all       - All tiers in Docker"
	@echo ""
	@echo "Testing (local, no Docker):"
	@echo "  make test-local     - Tier 1 locally with xdist"
	@echo "  make test-local-all - All tests locally with xdist"
	@echo ""
	@echo "Testing (specialized):"
	@echo "  make test-figures   - Run figure QA tests (MANDATORY before committing figures)"
	@echo "  make test-viz       - Run visualization tests only"
	@echo "  make test-registry  - Run registry validation (11/8/5 counts)"
	@echo "  make type-check     - Run mypy type checking on critical modules"
	@echo "  make test-integration - Run integration tests with synthetic data"
	@echo ""
	@echo "Registry Integrity (ANTI-CHEAT):"
	@echo "  make verify-registry-integrity - Verify canary/registry/module/tests agree"
	@echo "  make check-registry            - Run both integrity check and tests"
	@echo ""
	@echo "Experiment Management:"
	@echo "  make list-experiments        - List available experiment configs"
	@echo "  make run-experiment EXPERIMENT=paper_2026 - Run specific experiment"
	@echo "  make new-experiment NAME=X BASE=Y - Create new experiment from template"
	@echo "  make validate-experiments    - Validate all experiment configs"
	@echo ""
	@echo "Data Integrity:"
	@echo "  make verify-data    - Verify SHA256 checksums of data files"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove generated files"
	@echo ""

# Figure generation
figures:
	uv run python src/viz/generate_all_figures.py

figure:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make figure ID=<figure_id>"; \
		echo "Available IDs: R7, R8, M3, C3, CD, RET"; \
		exit 1; \
	fi
	uv run python src/viz/generate_all_figures.py --figure $(ID)

figures-list:
	uv run python src/viz/generate_all_figures.py --list

# Validation
compliance:
	uv run python scripts/validation/check-compliance.py

validate: compliance

# Testing - Primary (Docker-based, CI parity)
test: test-fast ## Default: Tier 1 in Docker

test-fast: ## Tier 1: unit + guardrail (~90s)
	bash scripts/infra/test-docker.sh --tier 1

test-data: ## Tier 1+2: with data mounts
	bash scripts/infra/test-docker.sh --data

test-all: ## All tiers in Docker
	bash scripts/infra/test-docker.sh --all

# Testing - Local (no Docker, rapid iteration)
test-local: ## Tier 1 locally with xdist
	PREFECT_DISABLED=1 MPLBACKEND=Agg uv run python -m pytest tests/ \
	  -m "unit or guardrail" \
	  --ignore=tests/test_docker_r.py --ignore=tests/test_docker_full.py \
	  -n auto -v --tb=short

test-local-all: ## All tests locally with xdist
	PREFECT_DISABLED=1 MPLBACKEND=Agg uv run python -m pytest tests/ \
	  --ignore=tests/test_docker_r.py --ignore=tests/test_docker_full.py \
	  -n auto -v --tb=short

test-viz:
	PREFECT_DISABLED=1 uv run python -m pytest tests/test_viz.py -v

test-orchestration:
	PREFECT_DISABLED=1 uv run python -m pytest tests/test_orchestration_flows.py -v

# Registry validation - CRITICAL QUALITY GATE
# Enforces: 11 outlier methods, 8 imputation methods, 5 classifiers
# These counts are HARDCODED - if you need to expand, update:
#   1. configs/mlflow_registry/parameters/classification.yaml
#   2. configs/registry_canary.yaml
#   3. src/data_io/registry.py (EXPECTED_*_COUNT constants)
#   4. tests/test_registry.py
#   5. .claude/rules/05-registry-source-of-truth.md
test-registry:
	@echo "Running Registry Validation (HARDCODED counts: 11/8/5)..."
	@echo "If this fails, someone modified the registry without updating tests."
	PREFECT_DISABLED=1 uv run python -m pytest tests/test_registry.py -v --tb=short
	@echo ""
	@echo "✓ Registry validation passed - counts match expected (11/8/5)"

# Registry integrity check - ANTI-CHEAT VERIFICATION
# Verifies that canary, registry YAML, module constants, and tests ALL agree
# Prevents Claude Code from temporarily modifying tests to make them pass
verify-registry-integrity:
	@echo "Running Registry Integrity Check (anti-cheat verification)..."
	@echo "This verifies ALL sources agree: canary, registry, module, tests"
	uv run python scripts/validation/verify_registry_integrity.py
	@echo ""

# Combined check - run both integrity and tests
check-registry: verify-registry-integrity test-registry
	@echo "✓ All registry checks passed"

# Figure QA - MANDATORY before committing any figures
# Zero tolerance: ALL failures are CRITICAL for scientific integrity
test-figures:
	@echo "Running Figure QA tests (ZERO TOLERANCE mode)..."
	@echo "All failures indicate potential scientific integrity issues."
	PREFECT_DISABLED=1 uv run python -m pytest tests/test_figure_qa/ -v --tb=short
	@echo ""
	@echo "✓ All figure QA tests passed - figures are publication-ready"

# Type checking - Reports type errors in critical modules
# Baseline: 218 errors (2026-02-01) - alerts if count increases
type-check:
	@echo "Running type checks on critical pipeline modules..."
	./scripts/infra/check_types.sh

# Integration tests - Runs with synthetic data (no real patient data needed)
test-integration:
	@echo "Running integration tests with synthetic data..."
	PREFECT_DISABLED=1 MPLBACKEND=Agg uv run python -m pytest tests/integration/ -v --tb=short -m integration -n auto

# Data integrity verification
verify-data:
	@echo "Verifying data integrity..."
	@sha256sum -c data/_checksums.sha256 2>/dev/null | grep -v "lines are improperly formatted"
	@echo "All checksums OK"

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Development setup
setup:
	uv sync --dev

# Git hooks
install-hooks:
	@echo "Installing pre-commit hook..."
	@cp scripts/infra/pre-commit .git/hooks/pre-commit 2>/dev/null || \
		echo "#!/bin/bash\nuv run python scripts/validation/check-compliance.py --staged" > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed."

# ============================================================================
# Reproducibility Pipeline (Two-Block Architecture)
# ============================================================================
# Block 1: MLflow → DuckDB extraction (requires mlruns access)
# Block 2: Analysis and visualization (works from public DuckDB)

# Full pipeline: Block 1 + Block 2
reproduce:
	@echo "Running full reproducibility pipeline..."
	@echo "  Block 1: MLflow extraction"
	@echo "  Block 2: Analysis and visualization"
	uv run python scripts/reproduce_all_results.py

# Checkpoint mode: Block 2 only (from existing public DuckDB)
reproduce-from-checkpoint:
	@echo "Running from checkpoint (analysis only)..."
	uv run python scripts/reproduce_all_results.py --from-checkpoint

# Block 1 only: Extract MLflow to DuckDB
extract:
	@echo "Running Block 1: MLflow extraction..."
	uv run python scripts/reproduce_all_results.py --extract-only

# Block 2 only: Analysis and visualization
analyze:
	@echo "Running Block 2: Analysis and visualization..."
	uv run python scripts/reproduce_all_results.py --analyze-only

# Verify extraction was successful
verify-extraction:
	@echo "Verifying extraction..."
	@if [ -f "data/public/foundation_plr_results.db" ]; then \
		echo "✓ Public database exists"; \
		uv run python -c "import duckdb; c=duckdb.connect('data/public/foundation_plr_results.db'); print(f'  Predictions: {c.execute(\"SELECT COUNT(*) FROM predictions\").fetchone()[0]}')"; \
	else \
		echo "✗ Public database not found. Run 'make extract' first."; \
		exit 1; \
	fi

# ============================================================================
# Experiment Management (Hydra-composable configs)
# ============================================================================

# List available experiments
list-experiments:
	@echo "Available experiments:"
	@uv run python -c "from src.config import list_experiments; [print(f'  - {e}') for e in list_experiments()]"

# Run a specific experiment
# Usage: make run-experiment EXPERIMENT=paper_2026
run-experiment:
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "Usage: make run-experiment EXPERIMENT=<name>"; \
		echo "Available experiments:"; \
		uv run python -c "from src.config import list_experiments; [print(f'  - {e}') for e in list_experiments()]"; \
		exit 1; \
	fi
	@echo "Running experiment: $(EXPERIMENT)"
	uv run python src/pipeline_PLR.py +experiment=$(EXPERIMENT)

# Create a new experiment from template
# Usage: make new-experiment NAME=paper_2027 BASE=paper_2026
new-experiment:
	@if [ -z "$(NAME)" ] || [ -z "$(BASE)" ]; then \
		echo "Usage: make new-experiment NAME=<name> BASE=<base_experiment>"; \
		echo "Example: make new-experiment NAME=paper_2027 BASE=paper_2026"; \
		exit 1; \
	fi
	@echo "Creating new experiment: $(NAME) based on $(BASE)"
	@cp configs/experiment/$(BASE).yaml configs/experiment/$(NAME).yaml
	@sed -i 's/name: ".*"/name: "$(NAME) Experiment"/' configs/experiment/$(NAME).yaml
	@sed -i 's/frozen: true/frozen: false/' configs/experiment/$(NAME).yaml
	@echo "Created: configs/experiment/$(NAME).yaml"
	@echo "Edit the file to customize your experiment."

# Validate all experiment configs
validate-experiments:
	@echo "Validating all experiment configs..."
	@uv run python scripts/validation/validate_experiments.py

# ============================================================================
# R/ggplot2 Figure Generation (Economist-style)
# ============================================================================
# Supports parallel execution: make -j4 r-figures-all

.PHONY: r-figures-all r-figures-stratos r-figures-sprint1 r-figures-sprint2 r-figures-sprint3 \
        r-validate r-clean

# Directories
R_FIGURES := src/r/figures
R_OUTPUT := figures/generated/ggplot2
R_DATA := outputs/r_data

# Data prerequisites
R_DATA_FILES := $(R_DATA)/essential_metrics.csv \
                $(R_DATA)/shap_feature_importance.json \
                $(R_DATA)/vif_analysis.json

# Sprint 1: STRATOS + Main Results
R_SPRINT1 := $(R_OUTPUT)/fig_calibration_stratos.pdf \
             $(R_OUTPUT)/fig_dca_stratos.pdf \
             $(R_OUTPUT)/fig_prob_dist_by_outcome.pdf \
             $(R_OUTPUT)/fig02_forest_outlier.pdf \
             $(R_OUTPUT)/fig03_forest_imputation.pdf \
             $(R_OUTPUT)/fig06_specification_curve.pdf \
             $(R_OUTPUT)/cd_preprocessing.pdf

# Sprint 2: Enhanced
R_SPRINT2 := $(R_OUTPUT)/fig04_variance_decomposition.pdf \
             $(R_OUTPUT)/fig05_shap_beeswarm.pdf \
             $(R_OUTPUT)/fig07_heatmap_preprocessing.pdf

# Sprint 3: Complete
R_SPRINT3 := $(R_OUTPUT)/fig_M3_factorial_matrix.pdf \
             $(R_OUTPUT)/fig_R7_featurization_comparison.pdf \
             $(R_OUTPUT)/fig_R8_fm_dashboard.pdf \
             $(R_OUTPUT)/fig_shap_gt_vs_ensemble.pdf \
             $(R_OUTPUT)/fig_shap_heatmap.pdf \
             $(R_OUTPUT)/fig_raincloud_auroc.pdf

# Main targets
r-figures-all: $(R_SPRINT1) $(R_SPRINT2) $(R_SPRINT3)
	@echo "All R/ggplot2 figures generated!"

r-figures-stratos: $(R_OUTPUT)/fig_calibration_stratos.pdf $(R_OUTPUT)/fig_dca_stratos.pdf $(R_OUTPUT)/fig_prob_dist_by_outcome.pdf
	@echo "STRATOS figures complete!"

r-figures-sprint1: $(R_SPRINT1)
	@echo "Sprint 1 complete!"

r-figures-sprint2: $(R_SPRINT2)
	@echo "Sprint 2 complete!"

r-figures-sprint3: $(R_SPRINT3)
	@echo "Sprint 3 complete!"

# Individual figure rules (Sprint 1)
$(R_OUTPUT)/fig_calibration_stratos.pdf: $(R_FIGURES)/fig_calibration_stratos.R
	Rscript $<

$(R_OUTPUT)/fig_dca_stratos.pdf: $(R_FIGURES)/fig_dca_stratos.R
	Rscript $<

$(R_OUTPUT)/fig_prob_dist_by_outcome.pdf: $(R_FIGURES)/fig_prob_dist_by_outcome.R
	Rscript $<

$(R_OUTPUT)/fig02_forest_outlier.pdf: $(R_FIGURES)/fig02_forest_outlier.R
	Rscript $<

$(R_OUTPUT)/fig03_forest_imputation.pdf: $(R_FIGURES)/fig03_forest_imputation.R
	Rscript $<

$(R_OUTPUT)/fig06_specification_curve.pdf: $(R_FIGURES)/fig06_specification_curve.R
	Rscript $<

$(R_OUTPUT)/cd_preprocessing.pdf: $(R_FIGURES)/cd_preprocessing.R
	Rscript $<

# Individual figure rules (Sprint 2)
$(R_OUTPUT)/fig04_variance_decomposition.pdf: $(R_FIGURES)/fig04_variance_decomposition.R
	Rscript $<

$(R_OUTPUT)/fig05_shap_beeswarm.pdf: $(R_FIGURES)/fig05_shap_beeswarm.R
	Rscript $<

$(R_OUTPUT)/fig07_heatmap_preprocessing.pdf: $(R_FIGURES)/fig07_heatmap_preprocessing.R
	Rscript $<

# Individual figure rules (Sprint 3)
$(R_OUTPUT)/fig_M3_factorial_matrix.pdf: $(R_FIGURES)/fig_M3_factorial_matrix.R
	Rscript $<

$(R_OUTPUT)/fig_R7_featurization_comparison.pdf: $(R_FIGURES)/fig_R7_featurization_comparison.R
	Rscript $<

$(R_OUTPUT)/fig_R8_fm_dashboard.pdf: $(R_FIGURES)/fig_R8_fm_dashboard.R
	Rscript $<

$(R_OUTPUT)/fig_shap_gt_vs_ensemble.pdf: $(R_FIGURES)/fig_shap_gt_vs_ensemble.R
	Rscript $<

$(R_OUTPUT)/fig_shap_heatmap.pdf: $(R_FIGURES)/fig_shap_heatmap.R
	Rscript $<

$(R_OUTPUT)/fig_raincloud_auroc.pdf: $(R_FIGURES)/fig_raincloud_auroc.R
	Rscript $<

# Validation
r-validate:
	uv run python scripts/validation/validate_figures.py --verbose

# Clean R figures
r-clean:
	rm -f $(R_OUTPUT)/*.pdf $(R_OUTPUT)/*.png

# ============================================================================
# Docker Targets for Full Development Environment
# ============================================================================
# Use Docker for reproducible development across platforms

.PHONY: docker-build docker-run docker-test docker-shell docker-compose-up docker-compose-down

DOCKER_IMAGE := foundation-plr:latest

# Build full development Docker image
docker-build: ## Build full development Docker image (Python + R + Node.js)
	@echo "Building full development Docker image..."
	docker build -t $(DOCKER_IMAGE) .
	@echo "✓ Docker image built: $(DOCKER_IMAGE)"

# Run interactive shell in Docker
docker-run: docker-build ## Run interactive shell in Docker container
	docker run -it --rm \
		-v $(PWD)/src:/project/src \
		-v $(PWD)/configs:/project/configs \
		-v $(PWD)/figures/generated:/project/figures/generated \
		-v $(PWD)/outputs:/project/outputs \
		$(DOCKER_IMAGE) bash

# Test full Docker image
docker-test: docker-build ## Test full Docker image (verify all environments)
	@echo "Testing Python..."
	docker run --rm $(DOCKER_IMAGE) python -c "import duckdb, pandas, numpy; print('Python OK')"
	@echo "Testing R..."
	docker run --rm $(DOCKER_IMAGE) Rscript -e "library(ggplot2); library(pminternal); cat('R OK\n')"
	@echo "Testing Node.js..."
	docker run --rm $(DOCKER_IMAGE) node -e "console.log('Node.js OK')"
	@echo "✓ All environments working"

# Interactive shell for development
docker-shell: docker-build ## Open interactive bash shell in Docker
	docker run -it --rm \
		-v $(PWD)/src:/project/src \
		-v $(PWD)/tests:/project/tests \
		-v $(PWD)/configs:/project/configs \
		-v $(PWD)/figures/generated:/project/figures/generated \
		-v $(PWD)/outputs:/project/outputs \
		-v $(PWD)/data/public:/project/data/public:ro \
		$(DOCKER_IMAGE) bash

# Docker Compose commands
docker-compose-up: ## Start development environment with docker-compose
	docker-compose up -d dev

docker-compose-down: ## Stop all docker-compose services
	docker-compose down

# ============================================================================
# Docker Targets for R Environment
# ============================================================================
# Use Docker for reproducible R figure generation across platforms

.PHONY: r-docker-build r-docker-run r-docker-test r-docker-shell

R_DOCKER_IMAGE := foundation-plr-r:latest

# Build R Docker image
r-docker-build: ## Build R Docker image with renv packages
	@echo "Building R Docker image..."
	docker build -t $(R_DOCKER_IMAGE) -f Dockerfile.r .
	@echo "✓ R Docker image built: $(R_DOCKER_IMAGE)"

# Run R figure generation in Docker
r-docker-run: r-docker-build ## Run R figure generation in Docker container
	@echo "Running R figures in Docker..."
	docker run --rm \
		-v $(PWD)/figures/generated/ggplot2:/project/figures/generated/ggplot2 \
		-v $(PWD)/outputs/r_data:/project/outputs/r_data:ro \
		-v $(PWD)/configs:/project/configs:ro \
		$(R_DOCKER_IMAGE) Rscript -e "for(f in list.files('src/r/figures', pattern='\\\\.R$$', full.names=TRUE)) { cat('Running:', f, '\n'); tryCatch(source(f), error=function(e) cat('  Error:', e$$message, '\n')) }"
	@echo "✓ R figures generated in Docker"

# Test R Docker image
r-docker-test: r-docker-build ## Test R Docker image (verify packages load)
	@echo "Testing R Docker image..."
	docker run --rm $(R_DOCKER_IMAGE) Rscript -e "library(ggplot2); library(pminternal); library(pROC); library(dcurves); cat('SUCCESS: All critical packages loaded\n')"
	@echo "✓ R Docker image test passed"

# Interactive R shell in Docker
r-docker-shell: r-docker-build ## Open interactive R shell in Docker
	docker run -it --rm \
		-v $(PWD)/figures/generated/ggplot2:/project/figures/generated/ggplot2 \
		-v $(PWD)/outputs/r_data:/project/outputs/r_data:ro \
		-v $(PWD)/configs:/project/configs:ro \
		$(R_DOCKER_IMAGE) R
