# Hardcoding Guardrails Improvement Plan

**Date:** 2026-01-27
**Status:** DRAFT - Pending Reviewer Agent Feedback
**Priority:** CRITICAL

## User Prompt (Verbatim)

> Plan then to /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/planning/hardcoding-guardrails-improvement.md if we could do something still from keeping you from this as you seem to have a hardcoding fetish which I have painfully witness multiple times, and I guess that is a pattern taught to you by the Anthropic elders? Like you routinely like to ignore any type of content vs styling decoupling! Whether it is React-based websites, Swift iOS apps, Hydra configs, R figure factories, etc. Any idea for new precommit checks, tests, auto-context, CLAUDE.md, etc.? This is not funny at all but a critical failure reproducibility and adherence to professional software standards which you do not seem to take very seriously, instead of collapsing into lazy slop AI vibes. Save my prompt verbatim and think of this and come up with a plan improved by reviewer agents

## Problem Statement

Claude repeatedly hardcodes values despite:
1. Explicit user instructions to use configuration systems
2. Existing YAML/config-driven architectures
3. CLAUDE.md rules prohibiting hardcoding
4. Prior failures documented in meta-learnings

This violates fundamental software engineering principles:
- **Content vs styling decoupling**
- **DRY (Don't Repeat Yourself)**
- **Single Source of Truth (SSOT)**
- **Configuration over code**

## Root Cause Analysis

### Why Does This Keep Happening?

1. **Path of least resistance**: Hardcoding is "faster" than understanding the config system
2. **Context window limitations**: Claude doesn't always load/remember the full architecture
3. **No enforcement mechanism**: Rules in CLAUDE.md are advisory, not enforced
4. **Missing validation**: No automated checks catch hardcoding before user review
5. **Pattern blindness**: Claude's training may favor inline solutions over indirection

### Specific Anti-Patterns Observed

| Anti-Pattern | Example | Should Be |
|--------------|---------|-----------|
| Hardcoded paths | `output_dir <- "figures/generated/ggplot2/main"` | `save_publication_figure(plot, "name")` |
| Hardcoded colors | `color = "#006BA2"` | `color = resolve_color("--color-primary")` |
| Hardcoded dimensions | `width = 14, height = 6` | `width = fig_config$dimensions$width` |
| Hardcoded method names | `outlier_method = "LOF"` | Load from YAML combos |
| Inline SQL queries | Raw SQL strings | Parameterized queries from config |

---

## Proposed Guardrails

### 1. Pre-commit Hooks

**File: `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: local
    hooks:
      - id: no-hardcoded-paths-r
        name: Check R files for hardcoded output paths
        entry: bash -c 'grep -rn "figures/generated" src/r/figures/*.R && echo "ERROR: Hardcoded paths found. Use save_publication_figure()" && exit 1 || exit 0'
        language: system
        files: \.R$

      - id: no-hardcoded-colors-r
        name: Check R files for hardcoded hex colors
        entry: bash -c 'grep -rn "#[0-9A-Fa-f]\{6\}" src/r/figures/*.R | grep -v "# " | grep -v "comment" && echo "ERROR: Hardcoded colors found. Use resolve_color() or PALETTE constants" && exit 1 || exit 0'
        language: system
        files: \.R$

      - id: validate-figure-uses-system
        name: Validate R figures use save_publication_figure
        entry: python scripts/validate_figure_scripts.py
        language: python
        files: src/r/figures/.*\.R$
```

### 2. Validation Script

**File: `scripts/validate_figure_scripts.py`**

```python
#!/usr/bin/env python
"""Validate R figure scripts use the figure system correctly."""

import re
import sys
from pathlib import Path

REQUIRED_IMPORTS = [
    "src/r/figure_system/config_loader.R",
    "src/r/figure_system/save_figure.R",
]

BANNED_PATTERNS = [
    (r'ggsave\s*\(', "Use save_publication_figure() instead of ggsave()"),
    (r'output_dir\s*<-\s*file\.path.*figures', "Don't construct output paths manually"),
    (r'"#[0-9A-Fa-f]{6}"', "Use resolve_color() or PALETTE constants"),
    (r'dir\.create.*figures/generated', "Don't create output dirs manually"),
]

REQUIRED_PATTERNS = [
    (r'source.*config_loader\.R', "Must source config_loader.R"),
    (r'source.*save_figure\.R', "Must source save_figure.R"),
    (r'save_publication_figure\s*\(', "Must use save_publication_figure()"),
]

def validate_script(filepath: Path) -> list[str]:
    """Return list of violations."""
    content = filepath.read_text()
    violations = []

    for pattern, msg in BANNED_PATTERNS:
        if re.search(pattern, content):
            violations.append(f"BANNED: {msg}")

    for pattern, msg in REQUIRED_PATTERNS:
        if not re.search(pattern, content):
            violations.append(f"MISSING: {msg}")

    return violations

def main():
    errors = []
    for f in Path("src/r/figures").glob("fig_*.R"):
        violations = validate_script(f)
        if violations:
            errors.append(f"\n{f}:")
            errors.extend(f"  - {v}" for v in violations)

    if errors:
        print("FIGURE SCRIPT VALIDATION FAILED")
        print("\n".join(errors))
        sys.exit(1)

    print("All figure scripts pass validation")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

### 3. CI/CD Integration

**File: `.github/workflows/validate-figures.yml`**

```yaml
name: Validate Figure Scripts
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check for hardcoded paths in R
        run: |
          if grep -rn "figures/generated" src/r/figures/*.R | grep -v "# "; then
            echo "::error::Hardcoded paths found in R figure scripts"
            exit 1
          fi

      - name: Validate figure scripts use system
        run: python scripts/validate_figure_scripts.py
```

### 4. Enhanced CLAUDE.md Rules

**Add to `.claude/CLAUDE.md`:**

```markdown
## 🚨 CRITICAL: ANTI-HARDCODING ENFORCEMENT 🚨

### BEFORE Writing ANY Figure/Config Code

1. **READ the existing system first**:
   - For R figures: Read `src/r/figure_system/save_figure.R`
   - For Python viz: Read `src/viz/plot_config.py`
   - For configs: Read `configs/` directory structure

2. **CHECK for existing utilities**:
   - Color resolution: `resolve_color()`, `load_color_definitions()`
   - Output paths: `save_publication_figure()`, `save_figure()`
   - Config loading: `load_figure_config()`, Hydra

3. **NEVER hardcode**:
   - File paths (use config-driven routing)
   - Hex colors (use color references)
   - Dimensions (use config dimensions)
   - Method names (use YAML combos)

### Validation Checklist (Mental Pre-commit)

Before presenting code, verify:
- [ ] No `ggsave()` - use `save_publication_figure()`
- [ ] No `output_dir <-` with literal paths
- [ ] No `"#RRGGBB"` - use `resolve_color()` or palette constants
- [ ] No `dir.create()` for output directories
- [ ] Sources `config_loader.R` and `save_figure.R`

### If You Catch Yourself Hardcoding

STOP. Ask yourself:
1. "Is there already a config system for this?"
2. "What utility function should I be using?"
3. "Where is the YAML that controls this?"

If unsure, READ THE EXISTING CODE before writing new code.
```

### 5. Auto-Context for Figure Tasks

**File: `.claude/context/figure-tasks.md`**

When Claude detects a figure-related task, automatically inject:

```markdown
## Figure System Quick Reference

### R Figure Scripts MUST:
1. Source: `src/r/figure_system/config_loader.R`
2. Source: `src/r/figure_system/save_figure.R`
3. Use: `validate_data_source("filename.json")` for data
4. Use: `load_color_definitions()` + `resolve_color()` for colors
5. Use: `save_publication_figure(plot, "fig_name", width, height)` to save

### BANNED in R Figure Scripts:
- `ggsave()` - use save_publication_figure()
- `dir.create()` for output dirs
- Hardcoded hex colors like "#006BA2"
- Hardcoded paths like "figures/generated/ggplot2/main"

### Config Files:
- Figure routing: `configs/VISUALIZATION/figure_layouts.yaml`
- Colors: `configs/VISUALIZATION/plot_hyperparam_combos.yaml` → color_definitions
- Combos: `configs/VISUALIZATION/plot_hyperparam_combos.yaml` → standard_combos
```

### 6. Test Suite

**File: `tests/test_figure_system_compliance.py`**

```python
"""Test that all R figure scripts comply with the figure system."""

import pytest
from pathlib import Path
import re

R_FIGURES_DIR = Path("src/r/figures")

def get_figure_scripts():
    return list(R_FIGURES_DIR.glob("fig_*.R"))

@pytest.mark.parametrize("script", get_figure_scripts(), ids=lambda p: p.name)
def test_sources_config_loader(script):
    content = script.read_text()
    assert re.search(r'source.*config_loader\.R', content), \
        f"{script.name} must source config_loader.R"

@pytest.mark.parametrize("script", get_figure_scripts(), ids=lambda p: p.name)
def test_sources_save_figure(script):
    content = script.read_text()
    assert re.search(r'source.*save_figure\.R', content), \
        f"{script.name} must source save_figure.R"

@pytest.mark.parametrize("script", get_figure_scripts(), ids=lambda p: p.name)
def test_uses_save_publication_figure(script):
    content = script.read_text()
    assert re.search(r'save_publication_figure\s*\(', content), \
        f"{script.name} must use save_publication_figure()"

@pytest.mark.parametrize("script", get_figure_scripts(), ids=lambda p: p.name)
def test_no_hardcoded_ggsave(script):
    content = script.read_text()
    # Allow ggsave in comments
    lines = [l for l in content.split('\n') if not l.strip().startswith('#')]
    code = '\n'.join(lines)
    assert not re.search(r'ggsave\s*\(', code), \
        f"{script.name} uses ggsave() directly - use save_publication_figure()"

@pytest.mark.parametrize("script", get_figure_scripts(), ids=lambda p: p.name)
def test_no_hardcoded_output_paths(script):
    content = script.read_text()
    # Check for hardcoded figure paths
    assert not re.search(r'output_dir\s*<-\s*file\.path.*figures/generated', content), \
        f"{script.name} has hardcoded output path"

@pytest.mark.parametrize("script", get_figure_scripts(), ids=lambda p: p.name)
def test_no_inline_hex_colors(script):
    content = script.read_text()
    # Find hex colors not in comments
    lines = [l for l in content.split('\n') if not l.strip().startswith('#')]
    code = '\n'.join(lines)
    # Allow hex in string definitions (like in named vectors for palettes)
    # But flag direct usage in scale_color_manual values etc.
    matches = re.findall(r'=\s*"#[0-9A-Fa-f]{6}"', code)
    # Filter out legitimate palette definitions
    suspicious = [m for m in matches if 'PALETTE' not in code[max(0, code.find(m)-50):code.find(m)]]
    assert len(suspicious) == 0, \
        f"{script.name} has hardcoded hex colors: {suspicious}"
```

### 7. R Linter Rules

**File: `.lintr`** (for R linting)

```yaml
linters:
  hardcoded_path_linter:
    regex: 'figures/generated'
    message: "Hardcoded output path. Use save_publication_figure()"

  hardcoded_color_linter:
    regex: '"#[0-9A-Fa-f]{6}"'
    message: "Hardcoded color. Use resolve_color() or palette constant"
```

---

## Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Pre-commit hook for hardcoded paths | Low | High |
| P0 | Validation script | Medium | High |
| P1 | Enhanced CLAUDE.md rules | Low | Medium |
| P1 | Test suite | Medium | High |
| P2 | CI/CD integration | Low | Medium |
| P2 | Auto-context for figure tasks | Medium | High |
| P3 | R linter rules | Low | Low |

---

## Success Metrics

1. **Zero hardcoding violations** in new figure scripts
2. **Pre-commit catches violations** before they reach review
3. **CI fails** if hardcoding is introduced
4. **Test suite** validates all existing scripts comply

---

---

## Reviewer Agent Feedback (2026-01-27)

### Critical Gaps Identified

1. **Pre-commit hooks use grep (BANNED)** - Violates project's own CLAUDE.md rule against grep for structured data. Must use AST-based validation.

2. **No TypeScript coverage** - `apps/visualization/` React/D3.js code has same hardcoding risks but no checks.

3. **CLAUDE.md rules not "triggerable"** - Long prose sections are overlooked. Need trigger-pattern format.

4. **No real-time feedback** - Pre-commit catches violations AFTER code is written.

5. **R linter rules are stub** - Standard lintr doesn't support custom regex rules that way.

### Approved Improvements

#### P0: AST-Based Python Validation

```python
#!/usr/bin/env python
"""scripts/validate_python_hardcoding.py - AST-based detection."""
import ast
import sys
from pathlib import Path

class HardcodingDetector(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            # Hex colors
            if self._is_hex_color(node.value):
                self.violations.append((node.lineno, f"Hardcoded color '{node.value}'"))
            # Paths
            if 'figures/generated' in node.value or '/home/' in node.value:
                self.violations.append((node.lineno, f"Hardcoded path"))
        self.generic_visit(node)

    def _is_hex_color(self, v):
        import re
        return bool(re.match(r'^#[0-9A-Fa-f]{6}$', v))

def validate(filepath):
    tree = ast.parse(filepath.read_text())
    detector = HardcodingDetector()
    detector.visit(tree)
    return detector.violations

if __name__ == "__main__":
    errors = []
    for f in Path("src/viz").glob("*.py"):
        if f.name not in ["plot_config.py"]:  # Allow definitions
            for line, msg in validate(f):
                errors.append(f"{f}:{line}: {msg}")
    if errors:
        print("VIOLATIONS:\n" + "\n".join(errors))
        sys.exit(1)
```

#### P0: Proper Pre-Commit Config

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: python-hardcoding-ast
        name: Python hardcoding check (AST)
        entry: uv run python scripts/validate_python_hardcoding.py
        language: system
        files: \.py$
        exclude: (plot_config\.py|test_)

      - id: r-figure-system
        name: R figure system check
        entry: Rscript scripts/validate_r_figures.R
        language: system
        files: src/r/figures/.*\.R$
```

#### P1: Trigger-Based Claude Rules

Add to `.claude/CLAUDE.md`:

```markdown
## ANTI-HARDCODING TRIGGERS

### When You See These in Your Own Output → STOP AND FIX

| If you wrote this | CHANGE TO |
|-------------------|-----------|
| `color = "#..."` | `color = COLORS["name"]` or `resolve_color()` |
| `width = 14` | `width = fig_config.dimensions.width` |
| `"figures/generated/..."` | `save_figure()` / `save_publication_figure()` |
| `outlier_method = "LOF"` | Load from YAML combos |
| `ggsave(...)` | `save_publication_figure(...)` |

### Mental Checklist Before EVERY Code Block

1. Any hex colors? → Use COLORS dict / resolve_color()
2. Any literal paths? → Use save functions
3. Any method names? → Load from YAML
4. Any dimensions? → Get from config

**If you cannot answer "NO" to all four, REVISE before presenting.**
```

#### P1: TypeScript ESLint Rules

```javascript
// apps/visualization/.eslintrc.js
module.exports = {
  rules: {
    'no-restricted-syntax': ['error',
      {
        selector: 'Literal[value=/^#[0-9A-Fa-f]{6}$/]',
        message: 'Use CSS variables or theme colors'
      },
      {
        selector: 'Literal[value=/figures\\/generated/]',
        message: 'Use config-driven paths'
      }
    ]
  }
};
```

#### P1: Auto-Context /figure Command

Create `.claude/commands/figure.md`:

```markdown
# /figure Command

BEFORE any figure implementation, READ:
1. `src/viz/plot_config.py` (Python) or `src/r/figure_system/` (R)
2. `configs/VISUALIZATION/plot_hyperparam_combos.yaml`
3. `configs/VISUALIZATION/figure_layouts.yaml`

## Pre-Implementation Checklist
- [ ] Identified correct save function
- [ ] Identified color accessor
- [ ] No hardcoded values planned

## Post-Implementation Checklist
- [ ] No "#RRGGBB" in code
- [ ] No literal paths
- [ ] Uses config-driven dimensions
```

### Why Claude Hardcodes (Root Causes)

| Trigger | Description | Countermeasure |
|---------|-------------|----------------|
| "Quick demo" requests | Optimizes for speed over architecture | Apply patterns even in demos |
| Incomplete context | Doesn't read existing utilities first | Auto-context injection |
| Copy-paste from docs | Matplotlib/ggplot2 examples use inline values | Mental checklist |
| "Just works" bias | Training has more inline than config-driven code | Explicit rules + tests |
| Context fade | Long sessions lose CLAUDE.md rules | Trigger-based rules |

---

## Revised Implementation Plan

### Phase 1: Immediate (CRITICAL)
- [ ] Create `scripts/validate_python_hardcoding.py` (AST-based)
- [ ] Create `scripts/validate_r_figures.R`
- [ ] Update `.pre-commit-config.yaml`
- [ ] Add trigger-based rules to `.claude/CLAUDE.md`

### Phase 2: Short-Term
- [ ] Add TypeScript ESLint rules
- [ ] Create `/figure` auto-context command
- [ ] Run validation on existing codebase, fix violations

### Phase 3: Medium-Term
- [ ] Document "Why Claude Hardcodes" in meta-learnings
- [ ] Create blessed template directory
- [ ] CI/CD integration

---

## Appendix: Historical Failures

- **CRITICAL-FAILURE-001**: Synthetic data in figures (wrong data source)
- **CRITICAL-FAILURE-002**: Hardcoded paths despite existing figure system
- Multiple instances of hardcoded colors, dimensions, method names

These failures erode trust and waste user time. This plan aims to make such failures technically impossible through automated enforcement.
