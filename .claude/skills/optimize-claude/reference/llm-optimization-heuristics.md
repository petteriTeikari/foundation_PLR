# LLM Instruction Optimization Heuristics

## Core Principles

### 1. Front-Load Critical Rules

LLMs weight early tokens more heavily. Place MUST/NEVER/BANNED rules in the first 25% of each file.

### 2. Imperative Over Descriptive

| Weak | Strong |
|------|--------|
| "You should consider using" | "USE" |
| "It is recommended to" | "MUST" |
| "Try to avoid" | "NEVER" |
| "It would be best if" | "ALWAYS" |
| "preferably" | Remove qualifier |

### 3. Tables Over Prose

Rules are scannable in tables, buried in paragraphs. Convert:

**Before** (paragraph):
> When generating figures, you should always use the setup_style function first, then load combos from the YAML config, and make sure to include ground truth...

**After** (table):
| Step | Action |
|------|--------|
| 1 | `setup_style()` |
| 2 | Load from `plot_hyperparam_combos.yaml` |
| 3 | Include ground_truth combo |

### 4. One Rule, One Location

Duplication causes:
- Token waste (same tokens loaded 3x)
- Drift risk (update one copy, forget others)
- Confusion (which version is canonical?)

Solution: canonical location + 1-line cross-references elsewhere.

### 5. CORRECT/WRONG Pattern

Most effective format for behavioral rules:

```python
# CORRECT
from src.data_io.registry import get_valid_outlier_methods

# WRONG - BANNED
methods = run_name.split("__")[3]
```

Keep exactly 1 CORRECT/WRONG pair per rule. Remove duplicate examples.

### 6. Severity Signaling

Use consistent markers:
- `CRITICAL` / `BANNED` / `NEVER` → Zero tolerance
- `MUST` / `ALWAYS` / `REQUIRED` → Mandatory
- `SHOULD` / `PREFER` → Strong recommendation
- No marker → Informational

### 7. Cross-Reference Format

Standard format for moved content:
```
**[Topic]**: See `path/to/file.md`
```

### 8. File Size Sweet Spots

| File Type | Ideal Size | Max Size |
|-----------|-----------|----------|
| Root CLAUDE.md | 10-15K | 20K |
| .claude/CLAUDE.md | 5-8K | 10K |
| rules/*.md | 1-1.5K | 2K |
| GUARDRAILS.md | 500-700 | 1K |

### 9. Remove Dead Weight

Content types with low instruction value:
- "Why this matters" explanations (reference meta-learnings instead)
- Historical context of bugs (keep rule, remove story)
- Multiple code examples for same pattern (keep 1)
- User quotes (paraphrase into rules)
- Section headers without content
