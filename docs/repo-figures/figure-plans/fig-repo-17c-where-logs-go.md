# fig-repo-17c: Where Do Logs Actually Go? (Hydra vs Our Approach)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-17c |
| **Title** | Where Do Logs Actually Go? |
| **Complexity Level** | L2-L3 (Technical overview) |
| **Target Persona** | ML Engineer, DevOps, Researchers debugging experiments |
| **Location** | ARCHITECTURE.md, docs/development/ |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain where log files actually end up in this repository, contrasting the "principled" Hydra logging approach with our "pragmatic" loguru + MLflow approach.

## Research Summary

### Hydra's Native Logging Approach (The "Principled" Way)

From [Hydra Logging Documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/logging/):

- Hydra automatically creates timestamped output directories
- Saves `{job_name}.log` using Python's standard `logging` module
- Configuration saved as `.hydra/config.yaml`, `.hydra/hydra.yaml`, `.hydra/overrides.yaml`
- All outputs co-located in single directory
- Configurable via `hydra/job_logging` and `hydra/hydra_logging`

### Our Approach (The "Pragmatic" Way)

We use loguru instead of Hydra's logging because:
1. Loguru is [3.5× faster](https://johal.in/logging-configuration-advanced-structured-logs-with-loguru-for-traceable-python-applications-2025/) and has better defaults
2. We need logs in MLflow for experiment tracking
3. Colorful console output helps during development
4. Thread-safe for parallel bootstrap iterations

Trade-off: Less integrated with Hydra's config system, but better for ML experiment workflows.

See also: [HydraFlow](https://github.com/daizutabi/hydraflow) for a more integrated solution.

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    WHERE DO LOGS ACTUALLY GO?                                            │
│                    Hydra's Way vs Our Approach                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ╔══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  HYDRA'S WAY (The "Principled" Approach)                                          ║  │
│  ╠══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║   Python code    Hydra takes over logging                                         ║  │
│  ║   ┌──────────┐   ┌───────────────────────────────────────────────────────────┐   ║  │
│  ║   │ @hydra   │   │                                                           │   ║  │
│  ║   │ .main()  │ → │  outputs/2026-02-01/15-30-42/                             │   ║  │
│  ║   │          │   │  ├── .hydra/                                               │   ║  │
│  ║   │ logging  │   │  │   ├── config.yaml        ← Your resolved config        │   ║  │
│  ║   │ .info()  │   │  │   ├── hydra.yaml         ← Hydra's internal config     │   ║  │
│  ║   │          │   │  │   └── overrides.yaml     ← CLI overrides you passed    │   ║  │
│  ║   └──────────┘   │  └── pipeline_PLR.log       ← All logging output          │   ║  │
│  ║                  └───────────────────────────────────────────────────────────┘   ║  │
│  ║                                                                                   ║  │
│  ║   ✅ Pro: Everything in ONE place (config + logs + outputs)                      ║  │
│  ║   ✅ Pro: Standard Python logging, integrates with libraries                     ║  │
│  ║   ❌ Con: No color in log files, plain text only                                 ║  │
│  ║   ❌ Con: Not integrated with MLflow experiment tracking                         ║  │
│  ║                                                                                   ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                         │
│  ╔══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  OUR WAY (The "Pragmatic" Approach)                                               ║  │
│  ╠══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║   Python code    Loguru → Multiple destinations                                   ║  │
│  ║   ┌──────────┐                                                                    ║  │
│  ║   │ loguru   │   ┌─────────────────────────────────────────────────────────┐     ║  │
│  ║   │ .info()  │   │  1️⃣ CONSOLE (stderr)                                    │     ║  │
│  ║   │ .error() │ → │     Colored output, visible during run                   │     ║  │
│  ║   │ .debug() │   │     🟢 INFO  🟡 WARNING  🔴 ERROR                        │     ║  │
│  ║   └──────────┘   └─────────────────────────────────────────────────────────┘     ║  │
│  ║        │                                                                          ║  │
│  ║        │         ┌─────────────────────────────────────────────────────────┐     ║  │
│  ║        ├───────→ │  2️⃣ LOCAL FILE                                          │     ║  │
│  ║        │         │     artifacts/hydra/pipeline_PLR.log                     │     ║  │
│  ║        │         │     Searchable, persistent, rotation = 10 MB             │     ║  │
│  ║        │         └─────────────────────────────────────────────────────────┘     ║  │
│  ║        │                                                                          ║  │
│  ║        │         ┌─────────────────────────────────────────────────────────┐     ║  │
│  ║        └───────→ │  3️⃣ MLFLOW ARTIFACTS                                    │     ║  │
│  ║                  │     mlruns/{exp_id}/{run_id}/artifacts/                  │     ║  │
│  ║                  │     ├── config/hydra_cfg.yaml    ← Hydra config          │     ║  │
│  ║                  │     └── hydra_logs/              ← Log file copies       │     ║  │
│  ║                  │                                                           │     ║  │
│  ║                  │  ACCESS: MLflow UI → Artifacts tab → hydra_logs/         │     ║  │
│  ║                  └─────────────────────────────────────────────────────────┘     ║  │
│  ║                                                                                   ║  │
│  ║   ✅ Pro: Logs attached to experiments (can review months later)                 ║  │
│  ║   ✅ Pro: Colored console, thread-safe, 3.5× faster than stdlib                  ║  │
│  ║   ✅ Pro: Find logs by experiment/run, not just by date                          ║  │
│  ║   ❌ Con: Not integrated with Hydra's directory structure                        ║  │
│  ║   ❌ Con: Logs duplicated (local + MLflow)                                       ║  │
│  ║                                                                                   ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  HOW TO FIND YOUR LOGS                                                                  │
│  ═════════════════════                                                                  │
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                                   │  │
│  │  Scenario                    │ Where to Look                                     │  │
│  │  ────────────────────────────┼─────────────────────────────────────────────────  │  │
│  │  During experiment run       │ Console (stderr) - colored output visible        │  │
│  │  ────────────────────────────┼─────────────────────────────────────────────────  │  │
│  │  After experiment completes  │ artifacts/hydra/pipeline_PLR.log                 │  │
│  │  (same machine)              │                                                   │  │
│  │  ────────────────────────────┼─────────────────────────────────────────────────  │  │
│  │  Reviewing old experiments   │ MLflow UI → Select run → Artifacts → hydra_logs  │  │
│  │  (any machine with mlruns)   │                                                   │  │
│  │  ────────────────────────────┼─────────────────────────────────────────────────  │  │
│  │  Prefect flow logs           │ Prefect UI → Run → Logs tab                      │  │
│  │                              │                                                   │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  THE KEY CODE                                                                           │
│  ════════════                                                                           │
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │  # src/log_helpers/log_utils.py                                                  │  │
│  │                                                                                   │  │
│  │  def setup_loguru():                                                              │  │
│  │      logger.remove()  # Remove default handler                                    │  │
│  │                                                                                   │  │
│  │      # Destination 1: Console with colors                                         │  │
│  │      logger.add(sys.stderr, colorize=True, level="INFO")                         │  │
│  │                                                                                   │  │
│  │      # Destination 2: Local file                                                  │  │
│  │      logger.add("artifacts/hydra/pipeline_PLR.log", level="INFO")                │  │
│  │                                                                                   │  │
│  │  # src/log_helpers/hydra_utils.py                                                 │  │
│  │                                                                                   │  │
│  │  def log_the_hydra_log_as_mlflow_artifact(hydra_log, ...):                       │  │
│  │      mlflow.log_artifact(hydra_log, artifact_path="hydra_logs")                  │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  WHY NOT USE HYDRA'S LOGGING?                                                           │
│  ═══════════════════════════                                                            │
│                                                                                         │
│  Hydra's logging is great for single-machine, single-run workflows.                     │
│  But for ML experiments, we need:                                                       │
│                                                                                         │
│  1. Logs attached to experiments  → MLflow artifacts                                    │
│  2. Thread-safe parallel logging  → Loguru handles 8 bootstrap threads                  │
│  3. Colored console output        → Loguru's defaults                                   │
│  4. Find logs by run ID           → MLflow UI search                                    │
│                                                                                         │
│  Trade-off: We don't use Hydra's directory-per-run structure for logs.                  │
│  Instead, logs go to a single file + MLflow artifacts.                                  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Hydra's way**: Single output directory with `.hydra/` subdirectory
2. **Our way**: Three destinations (console, local file, MLflow)
3. **How to find logs**: Table by scenario
4. **Key code snippets**: `setup_loguru()` and MLflow artifact logging
5. **Trade-off explanation**: Why we chose this approach

## Text Content

### Title Text
"Where Do Logs Actually Go?"

### Caption
This repository uses loguru instead of Hydra's native logging for better MLflow integration. Logs go to three places: (1) colored console output during runs, (2) local file at `artifacts/hydra/pipeline_PLR.log`, and (3) MLflow artifacts for long-term storage. The trade-off: we lose Hydra's neat directory-per-run structure but gain experiment-attached logs that can be reviewed months later via MLflow UI.

## Sources

- [Hydra Logging Documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/logging/)
- [Hydra Customizing Logging](https://hydra.cc/docs/configure_hydra/logging/)
- [HydraFlow - Integrate Hydra and MLflow](https://github.com/daizutabi/hydraflow)
- [Loguru vs Standard Logging Benchmarks](https://johal.in/logging-configuration-advanced-structured-logs-with-loguru-for-traceable-python-applications-2025/)
- [Python Logging: loguru vs logging](https://leapcell.io/blog/python-logging-vs-loguru)

## Prompts for Nano Banana Pro

### Style Prompt
Technical architecture diagram showing log flow paths. Two main sections comparing approaches. Directory trees and file paths. Code snippets in dark theme. Table showing scenarios and where to look. Professional documentation style.

### Content Prompt
Create a log destination diagram:

**SECTION 1 - Hydra's Way**:
- Single output directory structure
- Show .hydra/ contents
- Pros/cons list

**SECTION 2 - Our Way**:
- Three numbered destinations with arrows from loguru
- Console (colored), Local file, MLflow artifacts
- Pros/cons list

**TABLE - How to Find Logs**:
- During run → Console
- After run → Local file
- Old experiments → MLflow UI

**CODE - Key snippets**:
- setup_loguru() destinations
- MLflow artifact logging

**FOOTER - Why not Hydra?**:
- 4 reasons for MLflow integration

## Alt Text

Log destination diagram. Top shows Hydra's approach: single output directory with .hydra/ containing config files and job log. Bottom shows our approach: loguru sending to three destinations (1) colored console, (2) local file at artifacts/hydra/, (3) MLflow artifacts. Table shows where to find logs by scenario. Code snippets show setup_loguru() and mlflow.log_artifact() calls. Footer explains trade-off: lose Hydra directory structure, gain experiment-attached logs.

## Status

- [x] Draft created
- [x] Research completed (Hydra docs, loguru benchmarks)
- [ ] Generated
- [ ] Placed in ARCHITECTURE.md
