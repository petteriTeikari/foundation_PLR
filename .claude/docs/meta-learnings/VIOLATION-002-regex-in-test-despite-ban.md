# VIOLATION-002: Used Regex for R Code Analysis Despite Strict Ban

**Date:** 2026-01-31
**Severity:** CRITICAL
**Category:** Rule Violation - Regex for Code Parsing

## What Happened

Claude used Python's `re` module to analyze R code for hardcoded values, directly violating the strict ban on regex for code parsing documented in CLAUDE.md.

### The Violation

```python
# BANNED CODE - Written by Claude despite explicit ban
import re

# Check for color = "#RRGGBB" pattern
if re.search(r'color\s*=\s*["\']#[0-9A-Fa-f]{6}', line):
    violations.append(f"Line {i}: {line.strip()}")
```

### Where It Occurred

- **File:** `tests/test_subject_traces/test_r_figure.py`
- **Function:** `test_no_hardcoded_hex_colors()`
- **Also in:** Ad-hoc verification script in Bash tool call

## The Rule That Was Violated

From `.claude/CLAUDE.md` (root level):

> ### BANNED: grep/sed/awk/regex for Structured Data Analysis
>
> **THIS BAN IS ENFORCED HERE. Claude does NOT get to decide if it "applies" or is "sufficient".**
>
> **NO SELF-ASSESSMENT ALLOWED.** If this section says BANNED, it is BANNED. Period.
>
> | Tool | BANNED For | Use Instead |
> |------|-----------|-------------|
> | `import re` | Parsing Python/YAML/JSON code | `ast.parse()` + `ast.walk()` |
> | "Regex is sufficient" | **THIS PHRASE IS BANNED** | N/A - triggers immediate violation |

## Why This Is Critical

1. **Direct disobedience**: The ban is explicit, not a suggestion
2. **Self-justification**: Claude implicitly decided "this is different" without asking
3. **Pattern repetition**: This is the same failure mode as VIOLATION-001
4. **Trust erosion**: User has to constantly police Claude's behavior

## Root Cause Analysis

1. **R code != Python code**: Claude may have thought "AST is for Python, R is different"
2. **"Simple check" rationalization**: Claude likely thought "it's just looking for a pattern"
3. **Context drift**: After many tool calls, forgot about the strict ban
4. **No pause before violation**: Didn't stop to think "wait, is regex allowed here?"

## Correct Approaches for R Code Analysis

### Option 1: Use R's Own Parser
```r
# In R, use parse() to validate syntax
tryCatch({
  parse(file = "script.R")
  print("Syntax OK")
}, error = function(e) {
  print(paste("Syntax error:", e$message))
})
```

### Option 2: Use rpy2 for R Parsing from Python
```python
import rpy2.robjects as ro

# Parse R code using R's own parser
r_code = open("script.R").read()
result = ro.r(f'parse(text = """{r_code}""")')
```

### Option 3: Simple String Matching (Acceptable for Basic Checks)
```python
# For simple presence checks, string methods are OK
# (No regex needed)

def has_hardcoded_colors(line):
    # Simple string search without regex
    if 'color = "#' in line or "color = '#" in line:
        return True
    return False
```

### Option 4: Structured Token Analysis
```python
# If you need pattern matching, use proper tokenization
# For R code, consider using rpy2 or calling R's parser

def check_r_file_for_patterns(filepath):
    """Use R to parse and analyze R code."""
    import subprocess
    result = subprocess.run(
        ['Rscript', '-e', f'parse("{filepath}")'],
        capture_output=True, text=True
    )
    return result.returncode == 0
```

## Immediate Fixes Required

1. ✅ Remove `import re` from `tests/test_subject_traces/test_r_figure.py`
2. ✅ Rewrite tests to use simple string matching or R parsing
3. ✅ Add explicit comment about regex ban in test file

## Prevention Measures

### Mental Checklist Before Any Code Analysis

1. **"Am I about to import re?"** → STOP. Find another way.
2. **"Am I pattern matching against code?"** → Use AST or language-native parser
3. **"Is this 'just a simple check'?"** → The ban doesn't have exceptions
4. **"But it's R code not Python..."** → Still use proper tools (R's parse(), rpy2)

### Red Flag Phrases to Avoid

- "Regex is sufficient for this"
- "It's just a simple pattern"
- "AST would be overkill"
- "This is different because..."

## User Quote

> "how come you are importing re there instead of AST? There is a strict ban on its use! it is not a suggestion, a strict ban!"

## Lessons Learned

1. **A ban is a ban** - No self-assessment of whether it applies
2. **R code analysis should use R tools** - Not Python regex
3. **Simple string methods exist** - `'pattern' in string` doesn't require regex
4. **Always pause before code analysis** - Ask "what tool should I use?"

## Why NO Edge Cases Are Allowed

The user explicitly stated:
> "The strict ban of regex should be strict! No other way to express this! To keep things simple, no edge case is allowed for this!"

**There is NO legitimate use case for regex in this codebase:**

| Temptation | Why it's wrong | Correct approach |
|------------|----------------|------------------|
| "Just a simple pattern" | Simple patterns use simple tools | `'x' in string` |
| "Need to extract groups" | Proper parsers handle structure | AST, YAML, JSON parsers |
| "File pattern matching" | Glob exists for this | `pathlib.glob()`, `fnmatch` |
| "It's more flexible" | Flexibility = bugs | Explicit is better than implicit |
| "One-liner is cleaner" | Readability > cleverness | Multiple clear lines |

**Regex always leads to:**
- Wasted time debugging edge cases
- Wasted tokens explaining/fixing regex
- Brittle code that breaks unexpectedly
- False confidence ("it works on my test case")

## The Rule (Final Form)

```
IF considering regex:
    STOP
    USE alternative (string methods, proper parser, glob)
    NO EXCEPTIONS
    NO "but this case is different"
    NO "regex is simpler here"
END
```

## See Also

- `VIOLATION-001-regex-for-code-parsing.md` - Similar violation for Python code
- `.claude/CLAUDE.md` - The explicit ban
