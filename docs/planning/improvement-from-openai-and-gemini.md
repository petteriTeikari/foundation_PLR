# External AI Feedback: Improving Claude Code DevEx
**Date**: 2026-01-22
**Purpose**: Synthesize feedback from Gemini and OpenAI to eliminate "context amnesia"

---

## PART 1: VERBATIM FEEDBACK

### 1.1 Gemini's Feedback

It is clear you have built a sophisticated research pipeline, but "context amnesia" in LLMs often stems from context fragmentation—where the instructions are present but spread across too many files for the model to "attend" to simultaneously without losing focus on the task at hand.

To make the DevEx with Claude Code (or any high-end LLM) truly exceptional for this repo, you need to transition from Progressive Disclosure (which requires the AI to remember to look) to Active Constraint Enforcement (which forces the AI to comply).

Here is how to optimize your setup to solve the amnesia problem:

#### 1. Implement a "Source of Truth" Validator
Claude Code performs best when it has a "judge" it can run locally. Since you already have scripts/validate_figures.py, expand this into a DevEx Guardian script.

Action: Create scripts/check-compliance.py that Claude must run before every commit.

What it checks: Does the new code hardcode a method name? Does the new plot script call setup_style()? Does it use more than 4 curves?

Instruction: Add this to your .claude/CLAUDE.md:

"Before finalizing any visualization or config change, you MUST run uv run scripts/check-compliance.py. If it fails, you must fix the code before reporting completion."

#### 2. Collapse the "Amnesia" Layers
Your current 4-level documentation is great for humans, but for an AI, "Level 3" and "Level 4" often get ignored unless explicitly prompted.

Recommendation: Consolidate GUARDRAILS.md and the "Critical Rules" from CLAUDE.md into a single .clauderules file or a highly condensed SUMMARY_FOR_AI.md.

The "Context Window" Strategy: A single file of 50 lines is 10x more effective than five files of 10 lines.

Move to Semantic Variables: Instead of telling Claude "use these colors," define a Python Theme class in src/viz/plot_config.py. Then, tell Claude: "Never use hex codes. Only use Theme.PRIMARY_FM." This moves the "memory" from documentation into the code itself.

#### 3. Use "Context Trigger" Files (The .claudeconf Pattern)
Claude Code is heavily influenced by files in the current working directory.

Action: Create a .claude/project-state.json that tracks the current phase of the Comprehensive Figure Recreation Plan.

When Claude starts a session, tell it: "Read .claude/project-state.json to see where we left off."

This prevents it from suggesting Phase 1 tasks when you are already in Phase 3.

#### 4. Refined Figure Registry Architecture
Your figure_registry.yaml is your strongest asset. To make it "exceptional," you should treat it as a DSL (Domain Specific Language).

Improvement: Add a "Validation Schema" to the registry.

```yaml
# Add this to figure_registry.yaml
constraints:
  main_manuscript:
    max_curves: 4
    required_method: "ground_truth"
    allowed_styles: ["economist", "ggplot2"]
```

Then, update your generate_all_figures.py to throw an error if a script tries to plot a 5th curve. When Claude hits that error, its "amnesia" is cured instantly by the traceback.

#### 5. Optimized "Claude Instructions"
Add these specific "Personas" to your .claude/CLAUDE.md behavior contract:

**The "Zero-Hallucination" Protocol**
"If you are unsure of a hyperparameter combo name, you are FORBIDDEN from guessing. You must run grep on configs/VISUALIZATION/plot_hyperparam_combos.yaml or query the MLflow directory using ls -R /home/petteri/mlruns. Evidence of verification is required in your thought process."

**The "Styling" Protocol**
"You are a data visualization expert at The Economist. Every plot you touch must adhere to src/viz/plot_config.py. If you see plt.plot() without a setup_style() call, consider it a critical bug."

#### 6. Visualization Discoverability
To help Claude "see" what it's doing, ensure your generation scripts always output a log of what was created.

Action: Modify generate_all_figures.py to update a figures/MANIFEST.md every time it runs.

The manifest should show: Figure ID | File Path | Last Generated | Data Source.

This gives Claude a "map" of the output artifacts without it having to hunt through directories.

#### Summary of Next Steps
- Merge GUARDRAILS.md and .claude/CLAUDE.md into a "Master Behavior Contract."
- Implement the scripts/check-compliance.py script to turn "guidelines" into "hard errors."
- Update the figure_registry.yaml to include strict constraints that the Python code actually enforces at runtime.

#### Gemini's Compliance Script (scripts/check-compliance.py)

```python
#!/usr/bin/env python3
import os
import re
import yaml
import sys
from pathlib import Path

# Load Configuration Source of Truth
CONFIG_DIR = Path("config")
FIG_REGISTRY_PATH = CONFIG_DIR / "figure_registry.yaml"
HYPERPARAM_PATH = CONFIG_DIR / "plot_hyperparam_combos.yaml"

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class ComplianceChecker:
    def __init__(self):
        self.registry = load_yaml(FIG_REGISTRY_PATH)
        self.combos = load_yaml(HYPERPARAM_PATH)
        self.errors = []

    def check_file_content(self, file_path):
        """Scan python files for hardcoding and styling violations."""
        content = file_path.read_text()
        rel_path = str(file_path)

        # Rule 1: Must call setup_style()
        if "setup_style(" not in content and "plot_config" in content:
            self.errors.append(f"[{rel_path}] Missing setup_style() call.")

        # Rule 2: No hardcoded Hex colors
        hex_colors = re.findall(r'#[0-9a-fA-F]{6}', content)
        if hex_colors:
            self.errors.append(f"[{rel_path}] Hardcoded hex colors found: {hex_colors}. Use plot_config.COLORS instead.")

        # Rule 3: Ground truth must be present in combo lists
        if "combos" in content.lower() and "ground_truth" not in content:
            self.errors.append(f"[{rel_path}] Potential comparison logic missing 'ground_truth' reference.")

    def check_registry_sync(self):
        """Verify generated files match the registry specifications."""
        gen_dir = Path("figures/generated")
        for fig_id, spec in self.registry.get('main_figures', {}).items():
            # Check for curve count violations
            max_curves = spec.get('max_curves', 0)
            if max_curves > 4:
                self.errors.append(f"[Registry: {fig_id}] Main figures cannot exceed 4 curves.")

            # Check Privacy/Gitignore Compliance
            json_path = gen_dir / "data" / spec['outputs']['json']
            if spec['outputs']['json_privacy'] == "PRIVATE":
                self.check_gitignore(spec['outputs']['json'])

    def check_gitignore(self, filename):
        """Ensure private files are not trackable by git."""
        gitignore = Path(".gitignore").read_text()
        if filename not in gitignore and f"**/{filename}" not in gitignore:
            self.errors.append(f"[Privacy] Private file '{filename}' is NOT in .gitignore.")

    def run(self):
        viz_dir = Path("src/viz")
        for py_file in viz_dir.glob("*.py"):
            self.check_file_content(py_file)

        self.check_registry_sync()

        if self.errors:
            print("❌ COMPLIANCE FAILURE:")
            for err in self.errors:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print("✅ All systems compliant with Manuscript specifications.")
            sys.exit(0)

if __name__ == "__main__":
    checker = ComplianceChecker()
    checker.run()
```

---

### 1.2 OpenAI's Feedback

You're already very close to a best-practice setup: you have progressive disclosure docs (README_PROGRESSIVE), a repo-wide behavior contract (CLAUDE), explicit guardrails (GUARDRAILS), and a "full context packet" for external AIs (CONTEXT_FOR_EXTERNAL_AI). The "context amnesia" you're seeing is usually caused by (a) the assistant not reliably loading the right memory/rules for a given task, and (b) "drift" where the same rule exists in multiple places and falls out of sync.

Below are concrete, repo-specific fixes that will noticeably reduce repeated explanations and improve discoverability.

#### 1) Make Claude Code load the right context every time (use the official mechanism)

Claude Code has an explicit hierarchy of memory locations (project, project rules, user, local) and it will auto-load these when launched.

**Do this:**

**A. Keep CLAUDE.md short, and "import" the rest**
Claude Code supports @path/to/file imports inside CLAUDE.md.
So instead of a large monolith, make the root CLAUDE.md a small dispatcher:
- "What this repo is"
- The 5–10 non-negotiable rules
- The canonical sources of truth (configs/registries)
- @ imports into detailed docs/rules

This reduces truncation risk and makes it easier for Claude to re-anchor quickly.

**B. Replace auto-context.yaml with .claude/rules/*.md (path-scoped rules)**
Claude Code natively supports modular rule files under .claude/rules/, including path-specific rules via YAML frontmatter paths: and glob patterns.

Recommended mapping (your repo):
- .claude/rules/general.md (always applies)
- .claude/rules/viz.md with paths: ["src/viz/**/*.py"]
- .claude/rules/frontend_viz.md with paths: ["apps/apps/visualization/**/*.tsx"]
- .claude/rules/mlflow.md with paths: ["**/*mlflow*.py"]
- .claude/rules/privacy.md (always applies)

This makes the "right rules at the right time" automatic and predictable.

**C. Put machine-specific stuff in CLAUDE.local.md**
Claude Code supports CLAUDE.local.md for local-only preferences (auto gitignored).
Use this for:
- local DB paths / mount points
- "where MLflow lives on this machine"
- any sandbox shortcuts

**D. Add a denylist for sensitive files**
Claude Code supports .claude/settings.json permissions.deny to make files invisible to the agent.

#### 2) Eliminate "drift": one source of truth, everything else references it

Right now you have a subtle but high-impact inconsistency:
- CLAUDE.md defines the standard 4 combos as ground_truth, best_ensemble, best_single_fm, traditional
- auto-context.yaml instead injects best_fm and simple_baseline, and even mentions "linear" (which isn't what your standard combos specify)

That kind of conflict is exactly what makes an agent appear "amnesiac": it's not forgetting—it's being handed contradictory anchors.

**Fix pattern (high leverage)**

Any "rule" text should name only:
- the canonical file, and
- the IDs/keys to use.

For example, in viz rules, say:
- "Load configs/VISUALIZATION/plot_hyperparam_combos.yaml and use standard_combos"
- "Plot exactly the IDs listed there, in the order listed"
- "Never hardcode method names"

...and avoid repeating the actual combo definitions in multiple docs.

#### 3) Move "memory" out of prose and into the repo's executable affordances

The fastest way to improve DevEx (for humans and Claude) is to make the correct path the easy path.

**Concrete upgrades:**

**A. Figure Registry → auto-generated discoverability**
Make configs/VISUALIZATION/figure_registry.yaml the canonical catalog. Then generate:
- FIGURES_INDEX.md (table of all figures, scripts, output paths, privacy)
- optionally a small JSON index for the React/D3 viewer

**B. Validation as a forcing function**
Implement scripts/validate_figures.py that checks:
- all expected outputs exist (pdf/png/json)
- JSON privacy rules are respected (PRIVATE never committed)
- comparison figures include ground_truth
- main figures don't exceed 4 curves

**C. Make common commands trivial**
Create wrappers:
- make figures
- make figure ID=R-RET
- make validate
- make clean-figures

#### 4) Teach Claude Code with "skills" (slash commands) instead of repeated prompts

Claude Code supports skills (custom slash commands) via SKILL.md files.

Examples you likely want:
- /figures:list → prints registry table, where outputs go
- /figures:run R-RET → runs the right generator command
- /figures:validate → runs validator and summarizes failures
- /combos:show standard → prints combo IDs from YAML and warns on mismatch
- /privacy:check → scans git status for forbidden JSON patterns

#### 5) A prompting template that reliably keeps Claude aligned

**Task brief template:**
- Goal: (one sentence)
- Files likely involved: (2–5 paths)
- Non-negotiables: "Load combos from YAML; don't hardcode; ground_truth always included; max curves; respect json_privacy"
- Definition of done: expected files under figures/generated/... + validator passes
- Repro command: the one command you expect to work at the end

#### Quick "highest ROI" checklist (if you do only 3 things)

1. **Resolve the combo drift**: stop duplicating combo definitions across docs; reference configs/VISUALIZATION/plot_hyperparam_combos.yaml only.
2. **Migrate to .claude/rules/*.md with paths**: so the right constraints load automatically for viz vs MLflow vs React work.
3. **Make figure_registry.yaml + validate_figures.py real** and use it to auto-generate an index of outputs.

#### OpenAI's .claude/rules/ Structure

**Directory layout:**
```
.claude/rules/
  00-core.md
  10-privacy.md
  20-figures.md
  30-mlflow-and-combos.md
  40-viz-style.md
  90-precommit-checklist.md
```

**00-core.md (always-on, minimal, non-negotiables)**
```md
---
# No paths => always applies
---

# Foundation PLR — Core Rules (Always On)

## Single source of truth (no drift)
- Hyperparameter combos MUST come from: `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
- Demo subjects MUST come from: `configs/demo_subjects.yaml`
- Figure inventory + outputs MUST come from: `configs/VISUALIZATION/figure_registry.yaml` (once present)
- Never duplicate combo definitions in rule text, docs, or code. Always reference the config file.

## Absolutely forbidden
- Do NOT hardcode method names / combo names in visualization code.
- Do NOT invent combo IDs. If a combo isn't in YAML/MLflow, it doesn't exist.
- Do NOT produce comparison figures that omit `ground_truth`.

## Defaults for "main" manuscript figures
- Max curves: 4 for main figures, 6 for supplementary.
- If a figure needs more, it must be split into multiple panels/figures.

## Canonical loading pattern
Use this exact pattern when you need combos or subjects:

```python
import yaml

cfg = yaml.safe_load(open("configs/VISUALIZATION/plot_hyperparam_combos.yaml"))
standard_combos = cfg["standard_combos"]

subjects_cfg = yaml.safe_load(open("configs/demo_subjects.yaml"))
demo_subjects = subjects_cfg["demo_subjects"]
```
```

**10-privacy.md (always-on safety rails)**
```md
---
# No paths => always applies
---

# Data / Privacy Rules (Always On)

## Subject-level artifacts
- Any artifact containing subject identifiers or subject-level traces is PRIVATE.
- PRIVATE artifacts must NOT be committed.
- If a figure produces JSON, it must declare whether it is `public` or `PRIVATE` in the figure registry.

## Safe outputs
- Manuscript figures (PDF/PNG/SVG) are typically safe.
- Derived aggregates (summary tables without subject IDs) are typically safe.
- When unsure: default to PRIVATE and ask the repo config/guardrails what to do.
```

**20-figures.md (only when editing figure-generation scripts)**
```md
---
paths:
  - "src/viz/**/*.py"
  - "configs/VISUALIZATION/figure_registry.yaml"
---

# Figure Generation Rules

## Figure registry is the contract
- Each figure must be declared in `configs/VISUALIZATION/figure_registry.yaml`
- Every comparison plot MUST include `ground_truth`.

## Combo requirements
- Main figures MUST use `standard_combos` from `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
- Never embed a combo list in code. Only load the YAML.

## Validation-first workflow
- After changes: run `scripts/validate_figures.py` and fix all failures.
```

**30-mlflow-and-combos.md (only when touching MLflow/DuckDB/data plumbing)**
```md
---
paths:
  - "**/*mlflow*.py"
  - "**/*duckdb*.py"
  - "src/**/*.py"
  - "configs/VISUALIZATION/plot_hyperparam_combos.yaml"
---

# MLflow / Combo Integrity Rules

## Never hallucinate combos
- Combos come from `configs/VISUALIZATION/plot_hyperparam_combos.yaml`.
- If code needs run IDs / metrics: query MLflow and confirm they exist.
- If an MLflow query returns no runs, fail loudly with which combo ID and which filter produced zero results.
- Do not silently plot fewer curves.
```

**40-viz-style.md (only when touching plotting / viz app)**
```md
---
paths:
  - "**/*plot*.py"
  - "**/*figure*.py"
  - "apps/visualization/**/*.ts"
  - "apps/apps/visualization/**/*.tsx"
---

# Visualization & Styling Rules

## Curves & comparability
- Main figures: max 4 curves. Supplementary: max 6.
- `ground_truth` must always be present in comparison figures.

## Colors
- Never hardcode hex colors.
- Use semantic CSS variables or plot_config.COLORS.

## Consistency
- Axis labels, fonts, gridlines, legend placement should be consistent across figures.
```

**90-precommit-checklist.md (always-on "definition of done")**
```md
---
# No paths => always applies
---

# Before You Consider a Task Done

- ✅ If figure-related: registry updated (or confirmed unchanged)
- ✅ Validator passes (or you explained exactly which checks fail and why)
- ✅ `ground_truth` present in every comparison figure
- ✅ No hardcoded combo/method names in plotting code
- ✅ No PRIVATE JSON staged for commit
- ✅ Max-curves rule respected (4 main / 6 supp)
```

---

### 1.3 OpenAI's Migration Recommendations from dpp-agents

#### P0 (highest ROI): Local anchoring (AIDEV tags) for "context amnesia"

The AIDEV tag spec + naming convention (human tags vs AI tags with -AI- infix).

**Where it helps most in your repo:**
- Figure generators / plotting utilities: add AIDEV-NOTE at the spot that loads configs/VISUALIZATION/plot_hyperparam_combos.yaml ("never hardcode combos; standard_combos only; must include ground_truth").
- Any subject filtering / privacy boundary code: mark as AIDEV-IMMUTABLE or AIDEV-NOTE with explicit rationale ("subject-level traces are PRIVATE; never commit JSON").
- Registry/validator code: add AIDEV-NOTE explaining expected outputs and why.

This is the single best "stop repeating yourself" tool because it travels with the code during refactors.

#### P0: "Critical rules" as enforced checks (not just prose)

Create a manuscript-repo version:
- No hardcoded combo lists/method strings (must load YAML)
- Every comparison includes ground_truth
- Main figures max 4 curves (supp 6)
- No PRIVATE JSON staged/committed
- Outputs must match configs/VISUALIZATION/figure_registry.yaml

**Enforcement mechanism:** Add a second validator pass for "critical rules" (grep-like checks for hardcoded combos). The point is: fail fast.

#### P1: Documentation taxonomy + TOC so discoverability becomes automatic

Add a TABLE_OF_CONTENTS.md that links:
- "How to reproduce figures" doc
- Registry spec (figure_registry.yaml explanation)
- Output locations & what's public/private
- "Context packet" for external AIs

#### P1: "Golden paths" but for reproduction

Create a repo-specific docs/golden-paths.md that defines 3 workflows:
1. "Update YAML → regenerate figures → validate"
2. "Add a new figure" (registry → generator → outputs → validation)
3. "Prepare submission bundle" (what outputs are expected, where they live, what's excluded)

#### P2: Session context / "episodes" (only if you do long-running feature work)

A simple .claude/CURRENT_CONTEXT.md (active task, last decision, "don't forget ground_truth", etc.).

#### P2: Skills / slash commands (only after registry+validator exist)

Once your figure_registry.yaml + validator are real, skills become a multiplier:
- /figures:list
- /figures:run <id>
- /figures:validate
- /privacy:check

#### What I would NOT migrate (low relevance / high maintenance)

- Stack-specific architecture contract (FastAPI, async DI, uv/bun)
- Cookiecutter/Jinja brace escaping and template-generation pitfalls
- The "no regex ever" rule — useful for code transformation projects, but likely overkill here

---

## PART 2: CLAUDE'S SYNTHESIS AND ACTION PLAN

### 2.1 Agreement Analysis (LLM-as-Judge)

| Recommendation | Gemini | OpenAI | Claude Agreement | Priority |
|----------------|--------|--------|------------------|----------|
| Compliance validator script | ✅ | ✅ | **STRONG AGREE** | P0 |
| Path-scoped rules (.claude/rules/*.md) | ❌ | ✅ | **AGREE** | P0 |
| Eliminate combo drift (single source of truth) | ✅ | ✅ | **STRONG AGREE** | P0 |
| AIDEV local anchors in code | ❌ | ✅ | **AGREE** | P1 |
| Consolidate docs (50 lines > 5×10 lines) | ✅ | ✅ | **PARTIAL AGREE** | P1 |
| Project state tracking (.claude/project-state.json) | ✅ | ❌ | **AGREE** | P2 |
| Auto-generated MANIFEST.md | ✅ | ✅ | **AGREE** | P2 |
| Skills/slash commands | ❌ | ✅ | **DEFER** | P3 |
| Makefile wrappers | ❌ | ✅ | **AGREE** | P2 |
| CLAUDE.local.md for machine-specific paths | ❌ | ✅ | **STRONG AGREE** | P1 |
| permissions.deny for sensitive files | ❌ | ✅ | **AGREE** | P1 |

### 2.2 Key Insights I Agree With

1. **"Active Constraint Enforcement > Progressive Disclosure"** (Gemini)
   - Documentation that requires the AI to "remember to look" is fragile
   - Runtime validation that fails loudly is 10x more effective
   - **Action**: Implement check-compliance.py as mandatory pre-commit

2. **"A single file of 50 lines is 10x more effective than five files of 10 lines"** (Gemini)
   - Context fragmentation is the root cause of "amnesia"
   - **Action**: Consolidate into path-scoped rules that auto-load

3. **"Drift is the enemy"** (OpenAI)
   - The auto-context.yaml vs CLAUDE.md mismatch is a real bug
   - **Action**: Delete auto-context.yaml, use only config/*.yaml as sources of truth

4. **"AIDEV tags travel with the code"** (OpenAI/dpp-agents)
   - Comments in the code itself are more durable than external docs
   - **Action**: Add AIDEV-NOTE comments at critical enforcement points

5. **"Move memory from prose into executable affordances"** (OpenAI)
   - Python classes/functions that enforce constraints > documentation
   - **Action**: Theme class in plot_config.py, runtime validation in generators

### 2.3 What I Partially Disagree With

1. **Skills/slash commands (OpenAI P2)**
   - Reason: Overkill for this repo's scope. The validator + Makefile covers 90% of use cases.
   - **Decision**: DEFER until core infrastructure is stable

2. **Project-state.json tracking (Gemini)**
   - Reason: Manual overhead without tooling support
   - **Decision**: Use simpler CURRENT_CONTEXT.md approach from dpp-agents

### 2.4 Final Prioritized Action Plan

#### PHASE 1: Foundation (Do First) - Critical Path

| # | Action | Source | Effort |
|---|--------|--------|--------|
| 1.1 | Create `scripts/check-compliance.py` with Gemini's template | Gemini | Medium |
| 1.2 | Create `.claude/rules/` directory with OpenAI's 6 rule files | OpenAI | Medium |
| 1.3 | Delete `auto-context.yaml` (eliminate drift source) | OpenAI | Trivial |
| 1.4 | Create `CLAUDE.local.md` with machine-specific paths | OpenAI | Small |
| 1.5 | Add `permissions.deny` in `.claude/settings.json` | OpenAI | Small |

#### PHASE 2: Code-Level Enforcement (Do Second)

| # | Action | Source | Effort |
|---|--------|--------|--------|
| 2.1 | Add AIDEV-NOTE comments to critical code sections | dpp-agents | Medium |
| 2.2 | Create `Theme` class in `plot_config.py` (semantic colors) | Gemini | Small |
| 2.3 | Add runtime curve-count validation to generators | Gemini | Small |
| 2.4 | Update `generate_all_figures.py` to output MANIFEST.md | Gemini | Small |

#### PHASE 3: Developer Experience (Do Third)

| # | Action | Source | Effort |
|---|--------|--------|--------|
| 3.1 | Create `Makefile` with common commands | OpenAI | Small |
| 3.2 | Create `docs/golden-paths.md` for reproduction workflows | OpenAI | Medium |
| 3.3 | Create `TABLE_OF_CONTENTS.md` linking all docs | OpenAI | Small |
| 3.4 | Create `.claude/CURRENT_CONTEXT.md` template | dpp-agents | Small |

#### PHASE 4: Polish (Do Last)

| # | Action | Source | Effort |
|---|--------|--------|--------|
| 4.1 | Auto-generate `FIGURES_INDEX.md` from registry | OpenAI | Medium |
| 4.2 | Add pre-commit hook for compliance check | Gemini | Small |
| 4.3 | Consider Skills if patterns emerge | OpenAI | Defer |

---

## PART 3: IMPLEMENTATION DETAILS

### 3.1 Phase 1.1: Compliance Script (Enhanced)

Based on Gemini's template, enhanced with OpenAI's suggestions:

```python
#!/usr/bin/env python3
"""
check-compliance.py - Mandatory validation before any commit

CRITICAL: Claude Code MUST run this before reporting any visualization
task as complete. Failures must be fixed before proceeding.

Usage:
    uv run scripts/check-compliance.py
    python scripts/check-compliance.py
"""
import re
import sys
import yaml
from pathlib import Path

CONFIG_DIR = Path("config")
VIZ_DIR = Path("src/viz")

class ComplianceChecker:
    def __init__(self):
        self.errors = []
        self.warnings = []

        # Load sources of truth
        self.combos = self._load_yaml(CONFIG_DIR / "plot_hyperparam_combos.yaml")
        self.registry = self._load_yaml(CONFIG_DIR / "figure_registry.yaml")

    def _load_yaml(self, path: Path) -> dict:
        if not path.exists():
            self.warnings.append(f"Config not found: {path}")
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def check_no_hardcoded_combos(self):
        """Rule: Never hardcode combo/method names in viz code."""
        forbidden_patterns = [
            r"outlier_method\s*=\s*['\"](?!pupil-gt)[^'\"]+['\"]",
            r"imputation_method\s*=\s*['\"][^'\"]+['\"]",
            # Allow loading from config, forbid inline definitions
        ]
        # Implementation continues...

    def check_setup_style_called(self):
        """Rule: All plot scripts must call setup_style()."""
        for py_file in VIZ_DIR.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            content = py_file.read_text()
            if "plt." in content or "matplotlib" in content:
                if "setup_style(" not in content and "apply_style(" not in content:
                    self.errors.append(
                        f"[{py_file.name}] Uses matplotlib but missing setup_style() call"
                    )

    def check_no_hardcoded_colors(self):
        """Rule: No hex colors in viz code."""
        for py_file in VIZ_DIR.glob("*.py"):
            content = py_file.read_text()
            # Skip the config file itself
            if py_file.name == "plot_config.py":
                continue
            hex_colors = re.findall(r"['\"]#[0-9a-fA-F]{6}['\"]", content)
            if hex_colors:
                self.errors.append(
                    f"[{py_file.name}] Hardcoded hex colors: {hex_colors}. "
                    "Use COLORS dict from plot_config.py"
                )

    def check_ground_truth_included(self):
        """Rule: Comparison figures must include ground_truth."""
        # Check figure registry
        for fig_id, spec in self.registry.get("main_figures", {}).items():
            required = spec.get("required_combos", [])
            if required and "ground_truth" not in required:
                self.errors.append(
                    f"[Registry: {fig_id}] Missing 'ground_truth' in required_combos"
                )

    def check_max_curves(self):
        """Rule: Main figures max 4 curves, supplementary max 6."""
        for fig_id, spec in self.registry.get("main_figures", {}).items():
            max_curves = spec.get("max_curves", 0)
            if max_curves > 4:
                self.errors.append(
                    f"[Registry: {fig_id}] Main figure exceeds 4 curves ({max_curves})"
                )
        for fig_id, spec in self.registry.get("supplementary_figures", {}).items():
            max_curves = spec.get("max_curves", 0)
            if max_curves > 6:
                self.errors.append(
                    f"[Registry: {fig_id}] Supplementary figure exceeds 6 curves ({max_curves})"
                )

    def check_private_not_staged(self):
        """Rule: PRIVATE JSON must not be in git staging."""
        import subprocess
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True
        )
        staged = result.stdout.strip().split("\n") if result.stdout.strip() else []

        for fig_id, spec in self.registry.get("main_figures", {}).items():
            if spec.get("outputs", {}).get("json_privacy") == "PRIVATE":
                json_file = spec["outputs"]["json"]
                for staged_file in staged:
                    if json_file in staged_file:
                        self.errors.append(
                            f"[Privacy] PRIVATE file staged for commit: {staged_file}"
                        )

    def run(self) -> int:
        """Run all checks and return exit code."""
        self.check_setup_style_called()
        self.check_no_hardcoded_colors()
        self.check_ground_truth_included()
        self.check_max_curves()
        self.check_private_not_staged()

        if self.warnings:
            print("⚠️  WARNINGS:")
            for w in self.warnings:
                print(f"   {w}")

        if self.errors:
            print("\n❌ COMPLIANCE FAILURE:")
            for e in self.errors:
                print(f"   {e}")
            print("\nFix these issues before proceeding.")
            return 1

        print("✅ All compliance checks passed.")
        return 0

if __name__ == "__main__":
    checker = ComplianceChecker()
    sys.exit(checker.run())
```

### 3.2 Phase 1.2: Path-Scoped Rules Directory

Create `.claude/rules/` with the 6 files from OpenAI's recommendation.

### 3.3 Phase 2.1: AIDEV Tags Placement

Add these comments to critical code sections:

```python
# In src/viz/plot_config.py
# AIDEV-NOTE: This is the ONLY place colors should be defined.
# AIDEV-NOTE: All viz scripts must call setup_style() before plotting.
# AIDEV-IMMUTABLE: Do not add hex colors elsewhere; reference COLORS dict.

# In src/viz/retained_metric.py (or similar)
# AIDEV-NOTE: Load combos from configs/VISUALIZATION/plot_hyperparam_combos.yaml
# AIDEV-NOTE: MUST include ground_truth in every comparison
# AIDEV-NOTE: Main figures max 4 curves, supplementary max 6
```

### 3.4 Phase 3.1: Makefile

```makefile
.PHONY: figures validate clean compliance

figures:
	python src/viz/generate_all_figures.py

figure:
	python src/viz/generate_all_figures.py --figure $(ID)

validate:
	python scripts/validate_figures.py

compliance:
	python scripts/check-compliance.py

clean:
	rm -f figures/generated/*.png figures/generated/*.pdf

all: compliance figures validate
```

---

## PART 4: NEXT STEPS

1. **Review this document** and approve/modify the action plan
2. **Execute Phase 1** (foundation) first - this unblocks everything else
3. **Run compliance check** after each phase to ensure no regressions
4. **Iterate** based on actual usage patterns

---

## PART 5: INSIGHTS FROM DPP-AGENTS REPO

### 5.1 Key Patterns to Migrate

| Pattern | Source | Applicability | Priority |
|---------|--------|---------------|----------|
| AIDEV local anchoring tags | `.claude/aidev-tags.md` | HIGH - stops "re-explaining" | P0 |
| Critical rules file | `.claude/critical-rules.md` | HIGH - enforced non-negotiables | P0 |
| No-improvisation rule | `.claude/rules/no-improvisation.md` | MEDIUM - prevents silent changes | P1 |
| Current context file | `.claude/current-context.md` | MEDIUM - session continuity | P2 |
| Golden paths | `.claude/golden-paths.md` | MEDIUM - documented workflows | P2 |

### 5.2 AIDEV Tags for This Repo

**Recommended placements:**

```python
# In src/viz/plot_config.py (line ~60)
# AIDEV-NOTE: This is the ONLY place colors should be defined.
# All viz scripts must use COLORS dict, never hardcode hex values.
# AIDEV-IMMUTABLE: Color palette is Paul Tol colorblind-safe - do not change.

# In src/viz/retained_metric.py (wherever combos are loaded)
# AIDEV-NOTE: Load combos from configs/VISUALIZATION/plot_hyperparam_combos.yaml
# AIDEV-NOTE: MUST include ground_truth in every comparison figure
# AIDEV-NOTE: Main figures max 4 curves, supplementary max 6

# In configs/VISUALIZATION/figure_registry.yaml (at top)
# AIDEV-NOTE: This is the SINGLE SOURCE OF TRUTH for figure specifications.
# All figure generation scripts must read from here, never hardcode.

# In any subject data loading code
# AIDEV-IMMUTABLE-START: Subject Privacy Boundary
# Any code that loads subject identifiers (PLRxxxx) must ensure
# resulting JSON files are marked as PRIVATE in figure_registry.yaml
# AIDEV-IMMUTABLE-END
```

### 5.3 Critical Rules for This Repo (from dpp-agents pattern)

Create `.claude/critical-rules.md`:

```markdown
# CRITICAL RULES - READ BEFORE EVERY VISUALIZATION ACTION

**WARNING**: Violating these rules wastes time. These are ENFORCED.

## 🚫 RULE 1: No Hardcoded Combos (ENFORCED)

### Combos MUST come from configs/VISUALIZATION/plot_hyperparam_combos.yaml

✅ **CORRECT**:
```python
import yaml
cfg = yaml.safe_load(open("configs/VISUALIZATION/plot_hyperparam_combos.yaml"))
combos = cfg["standard_combos"]
```

❌ **WRONG** (will be BLOCKED by compliance checker):
```python
combos = ["ground_truth", "best_ensemble", "best_single_fm", "traditional"]  # HARDCODED!
```

## 🚫 RULE 2: Ground Truth Required (ENFORCED)

Every comparison figure MUST include `ground_truth` combo.

## 🚫 RULE 3: Max Curves (ENFORCED)

- Main figures: MAX 4 curves
- Supplementary: MAX 6 curves

If more needed, split into multiple figures.

## 🚫 RULE 4: No Hex Colors in Viz Scripts (ENFORCED)

Colors MUST come from `plot_config.COLORS` dict.

## 🚫 RULE 5: setup_style() Required (ENFORCED)

All plotting scripts MUST call `setup_style()` before any matplotlib operations.

---

**Enforcement**: `scripts/check-compliance.py` BLOCKS commits that violate these rules.
```

### 5.4 No-Improvisation Rule Application

From dpp-agents `.claude/rules/no-improvisation.md`:

> "When the user explicitly specifies a parameter, Claude MUST use that exact value.
> No exceptions. No 'optimization'. No 'improvement'. No 'workaround'."

**Application to this repo:**
- If user says "4 combos", use exactly 4 - don't add a 5th "for comparison"
- If user says "use CatBoost", use CatBoost - don't suggest XGBoost
- If user says "ground_truth + best_ensemble", plot exactly those two

---

## PART 6: INSIGHTS FROM WEB RESEARCH (2026)

### 6.1 CLAUDE.md Best Practices (Anthropic Official)

Source: [Claude Code: Best practices for agentic coding](https://www.anthropic.com/engineering/claude-code-best-practices)

Key findings:
- **Less is more**: < 300 lines is best, shorter is better
- **Prefer pointers to copies**: Use `file:line` references instead of code snippets
- **Universally applicable only**: Only include instructions that apply to every session

### 6.2 Context Management

Source: [Context Management System](https://deepwiki.com/FlorianBruniaux/claude-code-ultimate-guide/3.2-the-compact-command)

Key findings:
- Watch `Ctx(u):` continuously - when it exceeds 70%, action is urgent
- Use subagents to verify details (preserves context)
- Split knowledge files strategically when they exceed 50KB (~12k tokens)

### 6.3 Path-Scoped Rules Bug (January 2026)

Source: [GitHub Issue #16299](https://github.com/anthropics/claude-code/issues/16299)

**Known bug**: Rules with `paths:` frontmatter still load globally at session start.

**Workaround**: Until fixed, use path-scoped rules anyway - they still provide semantic organization even if all load initially.

### 6.4 Rules Directory Best Practices

Source: [Claude Code Rules Directory](https://claudefa.st/blog/guide/mechanics/rules-directory)

- Keep rules focused: One concern per file
- Use descriptive filenames: `api-validation.md` beats `rules1.md`
- Rules load with same priority as CLAUDE.md

---

## PART 7: REVISED ACTION PLAN (Post-Research)

### 7.1 Immediate Actions (This Session)

| # | Action | Source | Status |
|---|--------|--------|--------|
| 1 | Create `.claude/critical-rules.md` | dpp-agents | TODO |
| 2 | Create `scripts/check-compliance.py` | Gemini | TODO |
| 3 | Create `.claude/rules/` directory structure | OpenAI | TODO |
| 4 | Add AIDEV-NOTE comments to critical files | dpp-agents | TODO |
| 5 | Delete `auto-context.yaml` if exists | OpenAI | TODO |

### 7.2 Near-Term Actions (Next Session)

| # | Action | Source |
|---|--------|--------|
| 6 | Create `CLAUDE.local.md` for machine paths | OpenAI |
| 7 | Create `Makefile` with common commands | OpenAI |
| 8 | Add `.claude/settings.json` with permissions.deny | OpenAI |
| 9 | Create `docs/golden-paths.md` | dpp-agents |

### 7.3 Deferred Actions

| # | Action | Reason for Deferral |
|---|--------|---------------------|
| Skills/slash commands | Wait for core infrastructure to stabilize |
| Auto-generated FIGURES_INDEX.md | Nice-to-have after validation works |
| Pre-commit hooks | After compliance script is tested |

---

## PART 8: SOURCES

### Web Sources (Mandatory Citation)

- [How I Use Every Claude Code Feature](https://blog.sshh.io/p/how-i-use-every-claude-code-feature)
- [Claude Code: Best practices for agentic coding](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Claude Skills and CLAUDE.md: a practical 2026 guide](https://www.gend.co/blog/claude-skills-claude-md-guide)
- [Writing a good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Context Management System](https://deepwiki.com/FlorianBruniaux/claude-code-ultimate-guide/3.2-the-compact-command)
- [Claude Code settings](https://code.claude.com/docs/en/settings)
- [Path-scoped rules bug](https://github.com/anthropics/claude-code/issues/16299)
- [Rules Directory - Mechanics](https://claudefa.st/blog/guide/mechanics/rules-directory)
- [The Complete Guide to CLAUDE.md](https://www.builder.io/blog/claude-md-guide)

### Internal Sources

- `/home/petteri/Dropbox/github-personal/dpp-agents/.claude/critical-rules.md`
- `/home/petteri/Dropbox/github-personal/dpp-agents/.claude/aidev-tags.md`
- `/home/petteri/Dropbox/github-personal/dpp-agents/.claude/golden-paths.md`
- `/home/petteri/Dropbox/github-personal/dpp-agents/.claude/current-context.md`
- `/home/petteri/Dropbox/github-personal/dpp-agents/.claude/rules/no-improvisation.md`

---

---

## PART 9: REVIEWER AGENT FEEDBACK

Two specialized reviewer agents evaluated this plan to identify risks and improvements.

### 9.1 Feasibility Reviewer Findings

**Major Risks Identified:**

1. **Over-Engineering Risk (HIGH)**
   - The plan proposes 15+ documentation/config files for a research manuscript repo
   - Most manuscripts need: README, CLAUDE.md, one validator script, one config
   - **Recommendation**: Simplify to 4 core files max:
     - `CLAUDE.md` (consolidated rules)
     - `configs/VISUALIZATION/figure_registry.yaml` (already exists)
     - `scripts/check-compliance.py` (new)
     - `.gitignore` (already updated)

2. **Path-Scoped Rules Bug (MEDIUM)**
   - GitHub Issue #16299 confirms rules load globally despite `paths:` frontmatter
   - Creating 6 separate rule files adds complexity without the intended benefit
   - **Recommendation**: Defer `.claude/rules/` directory until bug is fixed; use consolidated CLAUDE.md instead

3. **Compliance Script False Positives (MEDIUM)**
   - Regex pattern `["'][a-zA-Z]+["']` will match any quoted string
   - Will flag legitimate strings like `plt.xlabel("Time")` as "hardcoded combo"
   - **Recommendation**: Use allowlist of known safe patterns, not blocklist

**Simplified P0 Actions:**
1. Add critical rules to existing `.claude/CLAUDE.md` (don't create new files)
2. Create minimal `scripts/check-compliance.py` with allowlist approach
3. Add AIDEV-NOTE comments to 3-5 critical code locations only
4. Skip rules directory, skills, project-state.json for now

### 9.2 Compliance Script Reviewer Improvements

**False Positive Handling:**
```python
# Instead of:
if re.search(r'["\'](ground_truth|best_ensemble)...', line):
    # flag as hardcoded

# Use allowlist approach:
ALLOWED_PATTERNS = [
    r'yaml\.safe_load',          # Loading from config
    r'cfg\[.*combos.*\]',        # Accessing config dict
    r'registry\[.*\]',           # Registry access
    r'\.get\(["\']\w+["\']\)',   # Dict .get() calls
]
```

**Line Number Tracking:**
- Include file path and line number in all error messages
- Format: `ERROR: Hardcoded combo at src/viz/plot.py:42`

**Suggested Error Messages:**
```
ERROR: Hardcoded combo detected
  File: src/viz/retained_metric.py
  Line: 127
  Found: combos = ["ground_truth", "best_ensemble"]
  Fix: Load combos from configs/VISUALIZATION/plot_hyperparam_combos.yaml
       Example: cfg = yaml.safe_load(open("configs/VISUALIZATION/plot_hyperparam_combos.yaml"))
```

**Integration Recommendations:**
- Add as git pre-commit hook (optional, not mandatory)
- Add to CI pipeline for PR checks
- Run manually with `make compliance`

### 9.3 Revised Recommendations (Post-Review)

Based on reviewer feedback, the plan is revised:

| Original Action | Revised Action | Reason |
|-----------------|----------------|--------|
| Create 6 rule files in `.claude/rules/` | Defer | Bug #16299, over-engineering |
| Create `.claude/critical-rules.md` | Merge into `.claude/CLAUDE.md` | Reduce fragmentation |
| Create `TABLE_OF_CONTENTS.md` | Skip | Already have README_PROGRESSIVE.md |
| Create `project-state.json` | Skip | Manual overhead |
| Add AIDEV-NOTE to all files | Add to 3-5 critical files only | Proportional effort |

**Final P0 Actions:**
1. ✅ Create `scripts/check-compliance.py` (simplified, with allowlist)
2. ✅ Update `.claude/CLAUDE.md` with consolidated rules
3. ✅ Add AIDEV-NOTE to: `plot_config.py`, `retained_metric.py`, `figure_registry.yaml`
4. ⏸️ Defer: rules directory, skills, pre-commit hooks

---

---

## PART 10: CORRECTED RECOMMENDATIONS (Self-Reflection)

**Date**: 2026-01-22 (after user feedback)

The user correctly pushed back on my overly cautious recommendations. Here is my self-reflection:

### 10.1 Why My Previous Assumptions Were Wrong

| Original Assumption | Why It Was Wrong |
|---------------------|------------------|
| "Skills are overkill" | Skills are USEFUL for repetitive tasks like `/figures:run`, `/validate`. User has found them valuable. |
| "Defer rules due to bug #16299" | The path-scoped bug is a minor annoyance. The REAL problem is constant context loss about fundamental repo purpose. |
| "Over-engineering risk" | I applied a generic warning without recognizing the specific problem: **UNDER-documentation of persistent context** |
| "4 files is enough" | No - the problem is I keep forgetting WHY the repo exists, not that there are too many files |

### 10.2 What I Keep Forgetting (And Why It's Annoying)

The user is frustrated because I repeatedly forget:

1. **Research question**: End-to-end sensitivity analysis of preprocessing → classification
2. **What figures exist** and where they are
3. **Data provenance**: This uses a SUBSET of Najjar et al. 2023 data
4. **Sister repos**: Where the manuscript and literature review live
5. **Why we're not comparing classifiers** (that's not the research question)

These are HIGH-LEVEL VISION items that should NEVER be forgotten, yet I keep losing them between sessions.

### 10.3 Critical Data Provenance (MUST REMEMBER)

**Original Study**: Najjar et al. 2023, Br J Ophthalmol
- Paper: "Handheld chromatic pupillometry can accurately and rapidly reveal functional loss in glaucoma"
- DOI: 10.1136/bjophthalmol-2021-319938
- Sample: 173 controls + 149 glaucoma = **322 subjects** from Singapore National Eye Centre
- Their AUROC: **0.94** (95% CI 0.91-0.96)

**Our Study (Foundation PLR)**:
- Uses a **SUBSET** of the original dataset
- 507 subjects for preprocessing (outlier detection + imputation)
- 208 subjects for classification (152 control + 56 glaucoma)
- Our best AUROC: **0.913** with ground truth preprocessing

**CRITICAL**: We CANNOT directly compare our AUROC (0.913) to Najjar's AUROC (0.94) because:
- Different data subsets
- Different experimental design (we focus on preprocessing sensitivity, not max performance)
- Different feature extraction methods

### 10.4 Sister Repos (MUST BE DISCOVERABLE)

| Repo | Path | Contents |
|------|------|----------|
| **Manuscript** | `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/latent-methods-results/` | LaTeX methods + results |
| **Literature Review** | `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/appendix-literature-review/` | Background theory |
| **Biblio** | `/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/` | Reference papers including Najjar et al. |

### 10.5 Revised Recommendations (Actually Useful)

#### Skills: DO Implement (P1)

Skills are appropriate for this repo:

```yaml
# .claude/commands/figures.md
/figures:list    # List all figures from figure_registry.yaml
/figures:run     # Generate a specific figure by ID
/figures:all     # Generate all figures
/validate        # Run check-compliance.py
/manuscript      # Navigate to sister repo with manuscript context
```

#### Rules: DO Implement (P0)

High-level vision rules that MUST be present at all times:

```markdown
# .claude/rules/00-research-question.md
---
# No path restriction - always applies
---

# RESEARCH QUESTION (READ EVERY SESSION)

This repo studies: **How preprocessing affects downstream classification**

- NOT about comparing classifiers (CatBoost is fixed)
- NOT about maximizing AUROC (that's already done)
- IS about: outlier detection × imputation → classification performance

## Data Provenance

Uses SUBSET of Najjar et al. 2023 (Br J Ophthalmol).
- Original: 322 subjects, AUROC 0.94
- Our subset: 208 for classification, 507 for preprocessing
- DO NOT compare our AUROC directly to Najjar's

## Sister Repos

- Manuscript: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/latent-methods-results/`
- Literature: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/appendix-literature-review/`
```

#### Why Rules Don't Get Stale

The user asked: "do the rules get stale very fast?"

**Answer**: NO - these are **foundational rules** about the research question, not implementation details:

| Type | Example | Gets Stale? |
|------|---------|-------------|
| Research question | "Preprocessing → classification sensitivity" | NO - fixed for paper |
| Data provenance | "Subset of Najjar 2023" | NO - data is fixed |
| Classifier choice | "Use CatBoost" | NO - established |
| Figure registry | Which figures exist | RARELY - tracked in YAML |
| Method names | Available outlier/imputation methods | RARELY - from MLflow |

### 10.6 Final Revised Action Plan

#### P0: High-Level Vision (Do First)

1. **Add data provenance to root CLAUDE.md** - Najjar et al. reference, subset note
2. **Add sister repo links to root CLAUDE.md** - manuscript, literature review paths
3. **Create `.claude/rules/00-research-question.md`** - research question, data provenance
4. **Create Skills** for common tasks - /figures, /validate, /manuscript

#### P1: Enforcement (Do Second)

5. Create `scripts/check-compliance.py` with allowlist approach
6. Add AIDEV-NOTE comments to critical files (3-5 locations)

#### P2: Developer Experience (Do Third)

7. Create Makefile with common commands
8. Create CURRENT_CONTEXT.md template

### 10.7 Self-Correction Summary

I was wrong to defer Skills and Rules. The user correctly identified that:

1. **Skills ARE useful** for repetitive tasks
2. **Rules address the real problem** - context amnesia about fundamental research questions
3. **High-level vision rules don't get stale** - they're about the paper's purpose, not implementation
4. **The path-scoped bug is minor** - semantic organization still helps

---

## PART 11: USER FEEDBACK ON CLARITY (Verbatim)

> "You should have properly explicitly then mentioned that we need to improve our persistent context, rather than using some very unintuitive way to express this sentiment, no? '3. Over-engineering concern - I applied a generic warning without recognizing that your specific problem is under-documentation of persistent context, not over-documentation.' and let's nail this down which I am trying to do with you, the documentation of the repo must be excellent now so that we can work efficiently together, right? A shitload of time and tokens have been already wasted, so let's address the root cause and not live with the tech debt. I am confident that the time and tokens used now with good context engineering and persistent context, we save a lot of headache in the short and long-term to get this repo ready now for publication, and for some people continuing from this later on. So update the plan with these insights and save my prompt verbatim, and run after your synthesis once more and create a concise xml action plan that is crash-resistant with progress tracking from all this .md work so that it is easier for you to follow orders and leave all 'the pondering' to .md and make the XML focused on the actual execution! Does this seem clear? feel free to ask me the multi-choice question panel to clarify things if things are unclear"

### 11.1 The Simple Truth

**Goal**: Excellent documentation → Efficient collaboration → Faster publication

No jargon. No "under-documentation of persistent context." Just:
- Good docs = less wasted time
- Investment now = payoff immediately and for future contributors
- Fix the root cause, don't live with tech debt

### 11.2 Why This Matters

| Without good context | With good context |
|---------------------|-------------------|
| Repeat explanations every session | Context loads automatically |
| Waste tokens on "what is this repo about?" | Jump straight to work |
| Frustrating for user | Efficient collaboration |
| Tech debt accumulates | Clean foundation for publication |

### 11.3 Action Plan Philosophy

- **Pondering** → stays in .md files (like this one)
- **Execution** → concise XML with checkboxes
- **Progress tracking** → crash-resistant, resumable

---

## PART 12: EXECUTIVE SUMMARY

### What We've Done

1. ✅ Saved Gemini + OpenAI feedback verbatim (Part 1)
2. ✅ Synthesized as LLM-as-Judge (Part 2)
3. ✅ Reviewed dpp-agents patterns (Part 5)
4. ✅ Web research on Claude Code 2026 (Part 6)
5. ✅ Ran reviewer agents (Part 9)
6. ✅ Self-corrected after user feedback (Part 10, 11)
7. ✅ Added data provenance to CLAUDE.md (Najjar et al. 2023)
8. ✅ Added sister repo links
9. ✅ Created Skills: /figures, /validate, /manuscript
10. ✅ Created Rules: 00-research-question.md, 10-figures.md

### What Remains

See `ACTION_PLAN.xml` for executable checklist.

---

**Document Status**: Complete - all reasoning captured
**Execution**: See ACTION_PLAN.xml
