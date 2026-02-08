# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main` branch (publication freeze) | Yes |
| Feature branches | Best-effort |

This is an academic research repository frozen for publication. Security patches are applied on a best-effort basis.

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Please email **petteri.teikari@gmail.com** with:
- Description of the vulnerability
- Steps to reproduce
- Affected components (file paths, dependencies)
- Severity estimate (critical / high / medium / low)

### Response Timeline

| Stage | Target |
|-------|--------|
| Acknowledgment | 7 days |
| Assessment | 30 days |
| Fix (if applicable) | 90 days |

We follow standard 90-day responsible disclosure.

## Scope and Limitations

This is a **local research pipeline**, not a production service:

- **No web-facing components** -- MLflow runs locally only, never internet-facing
- **No untrusted input** -- All data is from Najjar et al. 2023 (SERI, Singapore)
- **No untrusted model loading** -- All model artifacts are from own MLflow training runs
- **Shared reproducibility artifact is DuckDB** (not pickle)
- **npm app is dev-only** -- `apps/visualization/` runs locally via `vite dev`

### Known Gaps

- Some `torch.load()` calls in `src/imputation/` and `src/tools/tabpfn/` lack `weights_only=True` (low risk: only own artifacts loaded)
- 12+ bare `pickle.load()` calls exist in extraction code (low risk: all from own MLflow runs)
- See [Dependabot fix plan](docs/planning/dependabot-fix-plan.md) for full vulnerability inventory

## Pickle Safety Posture

All pickle and `torch.load` artifacts in this repository are **internally generated** from our own MLflow training runs. No external or third-party serialized objects are loaded anywhere in the pipeline.

- Internal artifacts: MLflow model checkpoints (`.pkl`, `.pth`)
- Shared artifact: `data/public/foundation_plr_results.db` (DuckDB, not pickle)
- Technical hardening (`ModelScan`, `weights_only=True`) is tracked in [GH#53](https://github.com/petteriTeikari/foundation_PLR/issues/53), deferred until the threat model changes (e.g., accepting external model artifacts)

## Security Controls

| Control | Status |
|---------|--------|
| Dependabot alerts | Enabled |
| Dependabot version updates | Configured (`.github/dependabot.yml`) |
| Branch protection on `main` | Enabled (restrict deletions) |
| Pre-commit hooks | 9 hooks (registry integrity, computation decoupling, patient ID scanning) |
| CISO Assistant workflow | PR compliance scanning (`.github/workflows/ciso-assistant-security.yml`) |
| CODEOWNERS | `@petteriTeikari` for all paths |

## Further Reading

- [Reproducibility & MLSecOps Improvement Plan](docs/planning/reproducibility-and-mlsecops-improvements.md)
- [Dependabot Fix Plan](docs/planning/dependabot-fix-plan.md)
