# Dependabot Vulnerability Fix Plan

**Date**: 2026-02-08
**Branch**: `chore/final-housekeeping`
**Total open alerts**: 75 (4 critical, 23 high, 38 medium, 10 low)
**Context**: Codebase is being frozen for academic publication handover. Minimal-risk changes only.

## Executive Summary

Most vulnerabilities are in **transitive dependencies** pulled by heavy ML packages
(momentfm -> transformers, pypots -> torch, mlflow -> werkzeug/starlette/urllib3). This is
a local research pipeline — no web server, no untrusted input, no production deployment.
Most alerts are **low real-world risk** despite high CVSS scores.

**Strategy (handover-safe)**:
1. Remove unused code that drags in vulnerable deps (NuwaTS)
2. Bump only direct deps with safe patch-level upgrades
3. Dismiss remaining alerts with documented rationale
4. Do NOT attempt major version bumps (torch, transformers) that risk breaking the pipeline

---

## Alert Inventory by Package

### Python (pip) — 66 alerts across 20 packages

| Package | Installed | Alerts | Sev (C/H/M/L) | Direct dep? | Required by | Action |
|---------|-----------|--------|----------------|-------------|-------------|--------|
| **transformers** | 4.33.3 | 17 | 1/4/10/2 | No | momentfm | **Dismiss** — pinned by momentfm |
| **aiohttp** | 3.10.10 | 11 | 1/0/6/4 | No | torch-geometric | **Dismiss** — no HTTP server |
| **mlflow** | 2.17.2 | 9 | 5/0/3/1 | **Yes** | — | **Upgrade** to >=2.22.4 |
| **urllib3** | 2.2.3 | 5 | 3/2/0/0 | No | requests, docker | **Upgrade** (safe transitive) |
| **torch** | 2.5.1 | 3 | 1/0/1/1 | No | 15+ packages | **Dismiss** — risky to bump |
| **protobuf** | 5.28.3 | 2 | 0/2/0/0 | No | mlflow, tensorboard | **Upgrade** (safe transitive) |
| **starlette** | 0.41.2 | 2 | 1/0/1/0 | No | fastapi ← mlflow | **Auto-resolves** with mlflow bump |
| **filelock** | 3.16.1 | 2 | 0/0/2/0 | No | transformers, torch | **Upgrade** (safe transitive) |
| **jinja2** | 3.1.4 | 3 | 0/0/3/0 | No | flask ← mlflow | **Upgrade** (safe transitive) |
| **Werkzeug** | 3.1.3 | 2 | 0/0/2/0 | No | flask ← mlflow | **Auto-resolves** with mlflow bump |
| **lightgbm** | 4.5.0 | 1 | 0/1/0/0 | **Yes** | — | **Upgrade** to >=4.6.0 |
| **setuptools** | 75.3.0 | 1 | 0/1/0/0 | No | build system | **Upgrade** (safe transitive) |
| **h11** | 0.14.0 | 1 | 1/0/0/0 | No | httpcore, uvicorn | **Upgrade** (safe transitive) |
| **pyasn1** | 0.6.1 | 1 | 0/1/0/0 | No | google-auth | **Upgrade** (safe transitive) |
| **fonttools** | 4.54.1 | 1 | 0/0/1/0 | No | matplotlib | **Upgrade** (safe transitive) |
| **h2** | 4.1.0 | 1 | 0/0/1/0 | No | httpx | **Upgrade** (safe transitive) |
| **requests** | 2.32.3 | 1 | 0/0/1/0 | No | mlflow, transformers | **Upgrade** (safe transitive) |
| **orjson** | 3.10.11 | 1 | 0/0/1/0 | No | prefect, polars | **Dismiss** — no patch available |
| **virtualenv** | 20.27.1 | 1 | 0/0/1/0 | No | pre-commit | **Upgrade** (safe transitive) |
| **cryptography** | 43.0.3 | 1 | 0/0/0/1 | No | pyOpenSSL | **Upgrade** (safe transitive) |

### npm — 9 alerts across 3 packages

| Package | Installed | Alerts | Sev (C/H/M/L) | Direct dep? | Action |
|---------|-----------|--------|----------------|-------------|--------|
| **jspdf** | ^2.5.1 | 7 | 1/4/2/0 | **Yes** | **Dismiss** — dev-only, major version jump |
| **dompurify** | (transitive) | 1 | 0/0/1/0 | No | **Dismiss** — dev-only app |
| **esbuild** | (transitive) | 1 | 0/0/1/0 | No (via vite) | **Dismiss** — dev-only tooling |

---

## Real Risk Assessment

This is a **local research pipeline**, not a web service:

1. **No web server exposed** — MLflow runs locally, never internet-facing
2. **No untrusted input** — All data is from Najjar et al. 2023 dataset (SERI)
3. **No untrusted model loading** — We load our own trained models only
4. **npm app is dev-only** — `apps/visualization/` runs locally via `vite dev`
5. **Torch `weights_only=True`** — Already our practice for model loading

| Alert | CVSS | Real risk | Rationale |
|-------|------|-----------|-----------|
| torch RCE via `torch.load` | CRITICAL | **LOW** | Own models only; `weights_only=True` |
| transformers deserialization (x3) | CRITICAL/HIGH | **LOW** | momentfm uses known checkpoints |
| h11 chunked encoding | CRITICAL | **LOW** | Local dev server only |
| jspdf path traversal | CRITICAL | **LOW** | Dev-only viz app, no untrusted PDFs |
| mlflow RCE/SSRF/DNS rebinding | HIGH | **LOW** | Local-only, never exposed to network |
| lightgbm RCE | HIGH | **MEDIUM** | We load LightGBM models — upgrade is safe |

---

## Execution Plan

### Step 0: Remove dead code (NuwaTS)

NuwaTS is unused in the pipeline. It imports `transformers` directly and drags in dependencies.
Removing it is pure cleanup with zero risk.

```bash
# Delete NuwaTS directory and config
rm -rf src/imputation/nuwats/
rm -f configs/MODELS/NuwaTS.yaml

# Remove any NuwaTS imports from imputation_main.py (if conditional, just delete the branch)
# Verify: grep -r "nuwats\|NuwaTS" src/ --include="*.py"
```

**Expected impact**: Removes the only direct `from transformers import` in `src/`, though
transformers stays as a transitive dep via momentfm. Zero functional impact since NuwaTS
was never called by the pipeline.

### Step 1: Safe direct-dependency upgrades

Only bump packages we control in `pyproject.toml` with clear patch-level fixes:

```bash
uv add "lightgbm>=4.6.0"       # Fixes RCE (#12) — patch release, safe
uv add "mlflow>=2.22.4"        # Fixes #15,#16,#37,#43,#48,#65 — we only read experiments
```

**Verify**: `uv sync && uv run python -c "import mlflow, lightgbm; print('OK')"`

### Step 2: Safe transitive-dependency floor bumps

Add minimum version constraints for transitive deps. These don't change behavior,
just tell the resolver to pick patched versions:

```bash
uv add "urllib3>=2.6.3" "protobuf>=5.29.6" "jinja2>=3.1.6" \
       "requests>=2.32.4" "setuptools>=78.1.1" "cryptography>=44.0.1" \
       "h11>=0.16.0" "pyasn1>=0.6.2" "filelock>=3.20.3" \
       "fonttools>=4.60.2" "h2>=4.3.0" "virtualenv>=20.36.1"
```

**Verify**: `uv sync && uv run pytest tests/ -x -q --timeout=120 -p no:xdist`

**If any conflict**: Remove the offending constraint — it means an upstream package
pins an older version and we can't override it without risk.

### Step 3: Dismiss remaining alerts

Dismiss with documented rationale. Group by reason:

#### 3a. No patch available (dismiss as `no_bandwidth`)
```bash
for alert_num in 17 41 64; do
  gh api repos/petteriTeikari/foundation_PLR/dependabot/alerts/$alert_num \
    -X PATCH -f state=dismissed -f dismissed_reason=no_bandwidth \
    -f dismissed_comment="No patch available. Local research pipeline, no untrusted input."
done
```

| Alert | Package | Issue |
|-------|---------|-------|
| #17 | mlflow <=2.17.2 | Resource consumption — no patch |
| #41 | mlflow <=3.4.0 | Unsafe deserialization — no patch |
| #64 | orjson <=3.11.4 | Recursion depth — no patch |

#### 3b. Major version bump too risky for handover (dismiss as `tolerable_risk`)
```bash
for alert_num in 22 25 36; do
  gh api repos/petteriTeikari/foundation_PLR/dependabot/alerts/$alert_num \
    -X PATCH -f state=dismissed -f dismissed_reason=tolerable_risk \
    -f dismissed_comment="Torch major upgrade (2.5->2.6+) risks breaking momentfm/pypots. Local pipeline, no untrusted model loading."
done
```

| Alert | Package | Issue |
|-------|---------|-------|
| #22 | torch <2.7.1 | Local DoS |
| #25 | torch <=2.7.1 | Resource shutdown (needs 2.8!) |
| #36 | torch <2.6.0 | RCE via torch.load — mitigated by `weights_only=True` |

#### 3c. Pinned by upstream (transformers via momentfm)
```bash
for alert_num in 1 2 3 8 9 10 14 19 21 28 29 30 31 34 38 39 40; do
  gh api repos/petteriTeikari/foundation_PLR/dependabot/alerts/$alert_num \
    -X PATCH -f state=dismissed -f dismissed_reason=tolerable_risk \
    -f dismissed_comment="Pinned by momentfm dependency. No untrusted model loading. Local research pipeline."
done
```

#### 3d. aiohttp — no HTTP server (dismiss as `tolerable_risk`)
```bash
for alert_num in 4 5 32 50 51 52 53 54 55 56 57; do
  gh api repos/petteriTeikari/foundation_PLR/dependabot/alerts/$alert_num \
    -X PATCH -f state=dismissed -f dismissed_reason=tolerable_risk \
    -f dismissed_comment="aiohttp is a transitive dep (torch-geometric). No HTTP server in this pipeline."
done
```

#### 3e. npm dev-only (dismiss as `tolerable_risk`)
```bash
for alert_num in 67 68 69 70 71 72 73 74 75; do
  gh api repos/petteriTeikari/foundation_PLR/dependabot/alerts/$alert_num \
    -X PATCH -f state=dismissed -f dismissed_reason=tolerable_risk \
    -f dismissed_comment="Dev-only visualization app (apps/visualization/). Never deployed, runs locally only."
done
```

---

## Expected Outcome

| Category | Alerts | Action |
|----------|--------|--------|
| Fixed by upgrade (Steps 1-2) | ~30 | mlflow, lightgbm, transitive deps |
| Auto-resolved (starlette, werkzeug via mlflow) | ~4 | Comes with mlflow bump |
| Dismissed: no patch | 3 | #17, #41, #64 |
| Dismissed: torch too risky | 3 | #22, #25, #36 |
| Dismissed: transformers pinned | 17 | #1-#3, #8-#10, #14, #19, #21, #28-#31, #34, #38-#40 |
| Dismissed: aiohttp no server | 11 | #4, #5, #32, #50-#57 |
| Dismissed: npm dev-only | 9 | #67-#75 |
| Auto-dismissed by GitHub | 1 | #76 (brace-expansion) |
| **Total** | **75** | **All addressed** |

---

## Post-Fix Verification

- [ ] `uv sync` resolves without conflicts
- [ ] `uv run python -c "import mlflow, lightgbm; print('OK')"` succeeds
- [ ] `uv run pytest tests/ -x -q --timeout=120` passes (~2042 pass, 181 skip)
- [ ] `grep -r "nuwats\|NuwaTS" src/ --include="*.py"` returns nothing active
- [ ] `gh api repos/petteriTeikari/foundation_PLR/dependabot/alerts --jq '[.[] | select(.state=="open")] | length'` returns 0
- [ ] Dependabot page shows all alerts resolved or dismissed

## Notes

- `numpy==1.25.2` is hard-pinned — do NOT touch, it's a known compatibility constraint
- If mlflow 2.22.4 causes resolver conflicts with numpy, try `mlflow>=2.20.3` (fixes fewer but still patches CSRF)
- NuwaTS removal should be its own commit for clean git history
- The `mlflow>=2.17.1` floor in `pyproject.toml` gets raised to `>=2.22.4`
