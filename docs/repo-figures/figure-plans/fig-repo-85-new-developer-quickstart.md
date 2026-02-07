# fig-repo-85: From Clone to First Test: 5-Minute Setup

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-85 |
| **Title** | From Clone to First Test: 5-Minute Setup |
| **Complexity Level** | L1 |
| **Target Persona** | All (new developer) |
| **Location** | `README.md`, `docs/onboarding/` |
| **Priority** | P1 (Critical) |

## Purpose

Show new developers that contributing to this project requires exactly two commands before running tests. The DevEx principle is "one command to set up, one command to verify." This figure eliminates the common barrier of complex multi-step installation procedures.

## Key Message

Clone, run the setup script, run tests -- you are contributing. No manual package installation, no configuration, no copy-paste walls of commands.

## Content Specification

### Panel 1: The 5-Step Linear Flowchart

```
STEP 1                    STEP 2                        STEP 3
git clone                 sudo ./scripts/               make test-local
                          setup-dev-environment.sh
┌─────────────┐          ┌───────────────────────┐     ┌──────────────────┐
│ $ git clone  │          │ The setup script does  │     │ PREFECT_DISABLED=1│
│   <repo-url> │──────────│ EVERYTHING for you:    │─────│ uv run pytest    │
│              │          │                        │     │ -m "unit or      │
│              │          │ 1. Installs uv         │     │  guardrail"      │
│              │          │    (if not present)     │     │ -n auto          │
│              │          │                        │     │                  │
│              │          │ 2. Creates Python 3.11  │     │ Result:          │
│              │          │    .venv virtual env    │     │ 2042 passed      │
│              │          │                        │     │ 0 failed         │
│              │          │ 3. Runs uv sync        │     │ 181 skipped      │
│              │          │    (all Python deps)    │     │                  │
│              │          │                        │     │ Skips are NORMAL │
│              │          │ 4. Installs pre-commit │     │ (need make       │
│              │          │    hooks                │     │  extract first)  │
│              │          │                        │     │                  │
│              │          │ 5. Verifies R >=4.4     │     └──────────────────┘
│              │          │    (optional, for R figs)│
└─────────────┘          └───────────────────────┘

                 STEP 4 (OPTIONAL)                       STEP 5
                 Generate production data                 Ready!
                 ┌───────────────────────┐              ┌──────────────────┐
                 │ make extract           │              │ You can now:     │
                 │   Block 1: MLflow→     │              │                  │
                 │   DuckDB extraction    │──────────────│ - Edit src/      │
                 │                        │              │ - Run tests      │
                 │ make analyze           │              │ - Generate figs  │
                 │   Block 2: DuckDB→     │              │ - Commit (hooks  │
                 │   figures/stats        │              │   run auto)      │
                 │                        │              │                  │
                 │ Result: 0 skips        │              │ Pre-commit hooks │
                 │ (all tests pass)       │              │ guard your code  │
                 └───────────────────────┘              └──────────────────┘
```

### Panel 2: What You Do NOT Need To Do

```
╳ pip install -r requirements.txt     → setup script uses uv sync
╳ conda create -n myenv python=3.11   → setup script creates .venv
╳ conda activate myenv                → source .venv/bin/activate
╳ pip install -e .                    → uv sync handles editable install
╳ Install pre-commit manually         → setup script installs hooks
╳ Configure environment variables     → Hydra config handles everything
╳ Copy-paste 20 lines of setup        → ONE script does it all
```

### Panel 3: Verification Checkpoints

```
After Step 2, verify:
  $ which python    → .venv/bin/python
  $ uv --version    → uv 0.x.x
  $ python --version → Python 3.11.x

After Step 3, verify:
  $ make test-local → "2042 passed, 0 failed"
  181 skips = EXPECTED (production data not yet generated)

After Step 4, verify:
  $ make verify-extraction → "Public database exists"
  $ make test-local-all    → "2042+ passed, 0 skips"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `pyproject.toml` | Python dependencies (managed by uv) |
| `renv.lock` | R package lockfile (for R figure generation) |
| `.pre-commit-config.yaml` | Pre-commit hook definitions |
| `Makefile` | All development commands (40+ targets) |

## Code Paths

| Module | Role |
|--------|------|
| `scripts/setup-dev-environment.sh` | One-command setup script |
| `Makefile` (target: `test-local`) | Tier 1 tests locally with xdist |
| `Makefile` (target: `extract`) | Block 1: MLflow extraction |
| `Makefile` (target: `analyze`) | Block 2: Analysis and visualization |
| `scripts/reproduce_all_results.py` | Full pipeline runner |
| `tests/conftest.py` | Root test fixtures (skip logic for missing data) |

## Extension Guide

To add a new setup dependency:
1. Python package: `uv add package_name` (updates `pyproject.toml`)
2. R package: Add to `renv.lock` via `renv::install("package")`
3. System tool: Add install step to `scripts/setup-dev-environment.sh`
4. Pre-commit hook: Add to `.pre-commit-config.yaml`

To modify test skip behavior:
1. Check `tests/conftest.py` for `skipif` decorators
2. Path constants: `RESULTS_DB`, `CD_DIAGRAM_DB`, `R_DATA_DIR`
3. Skips resolve when production data is generated (Step 4)

Note: This is a repo documentation figure - shows HOW to get started, NOT research results.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-85",
    "title": "From Clone to First Test: 5-Minute Setup"
  },
  "content_architecture": {
    "primary_message": "Clone, run setup script, run tests -- you are contributing. No manual package installation, no configuration, no copy-paste.",
    "layout_flow": "Left-to-right linear flowchart with 5 numbered steps, crossing-out panel below",
    "spatial_anchors": {
      "step1": {"x": 0.05, "y": 0.15},
      "step2": {"x": 0.25, "y": 0.15},
      "step3": {"x": 0.5, "y": 0.15},
      "step4": {"x": 0.25, "y": 0.55},
      "step5": {"x": 0.5, "y": 0.55},
      "dont_need": {"x": 0.75, "y": 0.15, "width": 0.2, "height": 0.7}
    },
    "key_structures": [
      {
        "name": "setup-dev-environment.sh",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["One command setup"]
      },
      {
        "name": "make test-local",
        "role": "healthy_normal",
        "is_highlighted": true,
        "labels": ["2042 passed, 0 failed"]
      },
      {
        "name": "181 skips",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Normal without production data"]
      }
    ],
    "callout_boxes": [
      {"heading": "DEVEX PRINCIPLE", "body_text": "One command to set up. One command to verify. Zero copy-paste."},
      {"heading": "181 SKIPS = NORMAL", "body_text": "Tests skip gracefully when production data is absent. Run make extract to resolve."}
    ]
  }
}
```

## Alt Text

Linear flowchart showing 5 steps from git clone to first contribution: clone, run setup script, run tests (2042 pass), optionally generate production data, then contribute. Crossed-out list shows what is NOT needed.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
