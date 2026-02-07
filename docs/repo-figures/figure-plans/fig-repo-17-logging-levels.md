# fig-repo-17: Logging Levels: Why Not Just print()?

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-17 |
| **Title** | Logging Levels: Why Not Just print()? |
| **Complexity Level** | L1 (Concept explanation) |
| **Target Persona** | Research Scientists (Jupyter users) |
| **Location** | docs/concepts-for-researchers.md, CONTRIBUTING.md |
| **Priority** | P1 (Important - debugging foundation) |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain why this repository uses loguru instead of print() statements, especially for scientists who are used to Jupyter notebooks.

## Key Message

"When debugging 1000 bootstrap iterations, print() loses messages in noise. Loguru captures everything with timestamps, levels, and colors—so you find bugs in seconds, not hours."

## Visual Concept

**Side-by-side terminal output comparison:**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LOGGING LEVELS: WHY NOT JUST print()?                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PROBLEM: Debugging 1000 Bootstrap Iterations                               │
│  ════════════════════════════════════════════════                               │
│                                                                                 │
│  ❌ WITH print()                     ✅ WITH loguru                             │
│  ────────────────                    ──────────────                             │
│  ┌──────────────────────────┐        ┌──────────────────────────────────────┐  │
│  │Starting iteration        │        │2026-01-31 14:23:01 │ INFO │ Starting │  │
│  │Processing...             │        │2026-01-31 14:23:01 │DEBUG │ Iter 1   │  │
│  │Processing...             │        │2026-01-31 14:23:02 │DEBUG │ Iter 2   │  │
│  │Error                     │        │2026-01-31 14:23:03 │ERROR │ Failed   │  │
│  │Processing...             │        │  ↳ File: boot.py:42                  │  │
│  │Processing...             │        │  ↳ Subject: PLR0123                  │  │
│  │Done                      │        │  ↳ Traceback: ValueError...         │  │
│  └──────────────────────────┘        │2026-01-31 14:23:04 │ INFO │ Retry OK │  │
│                                      └──────────────────────────────────────┘  │
│  "Which iteration failed?"           "Iteration 847, subject PLR0123, line 42" │
│  "When?"  "What was the context?"    "Everything is here!"                     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE LOG LEVEL HIERARCHY                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  ┌─────────┐                                                                    │
│  │  DEBUG  │  Detailed internal state (for developers)                          │
│  │─────────│  "Processing subject PLR0001, feature count: 20"                   │
│  │  INFO   │  Normal operations (milestones)                                    │
│  │─────────│  "Bootstrap iteration 500/1000 complete"                           │
│  │ WARNING │  Something unexpected but recoverable                              │
│  │─────────│  "Missing value in subject PLR0042, imputing..."                   │
│  │  ERROR  │  Something failed                                                  │
│  │─────────│  "Classification failed for MOMENT+SAITS combo"                    │
│  │CRITICAL │  System-wide failure                                               │
│  └─────────┘  "Database connection lost, aborting"                              │
│                                                                                 │
│  In production: Show INFO and above (hide DEBUG)                                │
│  In debugging: Show DEBUG and above (everything)                                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LOGURU BONUS FEATURES                                                          │
│  ════════════════════                                                           │
│                                                                                 │
│  🎨 Colored output     → Errors pop out in red                                  │
│  📁 Automatic rotation → Logs rotate daily, keep 7 days                         │
│  🧵 Thread-safe        → Works with parallel bootstrap                          │
│  🔍 Exception capture  → Full stack trace when errors occur                     │
│  📊 JSON export        → Machine-readable for analysis                          │
│                                                                                 │
│  Setup: ONE LINE!                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ from loguru import logger                                               │   │
│  │ logger.info("Processing subject {}", subject_id)  # That's it!          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. **Side-by-side terminal comparison**: print() vs loguru output
2. **Log level pyramid**: DEBUG → INFO → WARNING → ERROR → CRITICAL
3. **Feature list with icons**: colors, rotation, thread-safety, etc.
4. **One-line setup example**: Show simplicity of loguru

### Optional Elements
1. Jupyter vs script comparison
2. Log file location diagram
3. Filter demonstration (showing only ERRORs)

## Text Content

### Title Text
"Logging Levels: Why Not Just print()?"

### Labels/Annotations
- Problem: "1000 bootstrap iterations, one error somewhere—where?"
- print() side: "Lost in noise, no context, no timestamps"
- loguru side: "Timestamped, leveled, colored, traceable"
- Hierarchy: "Control verbosity: DEBUG (everything) to ERROR (problems only)"
- Setup: "One import, zero configuration"

### Caption (for embedding)
print() statements disappear into noise during long-running experiments like 1000 bootstrap iterations. Loguru provides structured logging with timestamps, severity levels (DEBUG/INFO/WARNING/ERROR), and automatic exception capture—all with a one-line setup. We use loguru across 139 source files in this repository. When something fails at iteration 847, you'll know exactly when, where, and why.

## Prompts for Nano Banana Pro

### Style Prompt
Developer debugging experience comparison. Split-screen terminal output style—left side messy/chaotic (print), right side organized/clean (loguru). Use dark terminal backgrounds with syntax highlighting. Include a pyramid diagram for log levels. Feature list with small icons. Economist-style data presentation for the comparison. Matte, professional, no glowing effects.

### Content Prompt
Create a debugging experience comparison infographic:

**TOP - Terminal Comparison**:
- LEFT: Dark terminal with messy, undifferentiated print() output
- RIGHT: Dark terminal with colorful, timestamped loguru output
  - Green for INFO, blue for DEBUG, red for ERROR
  - Show context info (file, line, subject ID)
- Caption below comparing "which iteration failed?" confusion vs clear answer

**MIDDLE - Level Pyramid**:
- Five stacked boxes: DEBUG (bottom) → INFO → WARNING → ERROR → CRITICAL (top)
- Example message next to each level
- Arrow showing "verbosity control"

**BOTTOM - Features**:
- 5 icons with labels: colored output, file rotation, thread-safe, exception capture, JSON export
- Code snippet showing one-line setup: `from loguru import logger`

### Refinement Notes
- The print() side should look genuinely frustrating/chaotic
- The loguru side should look like "relief" and "clarity"
- Make the RED error messages pop out
- Include the specific number: "139 files use loguru in this repo"

## Alt Text

Comparison of print() vs loguru logging. Left terminal shows chaotic print() output with no timestamps or context. Right terminal shows organized loguru output with timestamps, colored levels (green INFO, red ERROR), and file/line information. Middle shows log level hierarchy pyramid from DEBUG (most verbose) to CRITICAL (least verbose). Bottom lists loguru features: colored output, automatic rotation, thread safety, exception capture, JSON export.

## Technical Notes

### Verification in Codebase
- 139 files import loguru (verified via grep)
- Standard setup in `src/log_helpers/log_utils.py`
- Log rotation configured for 7-day retention

### Web Search Sources
- [GitHub: Loguru](https://github.com/Delgan/loguru)
- [Real Python: Loguru Tutorial](https://realpython.com/python-loguru/)
- [Better Stack: Loguru Guide](https://betterstack.com/community/guides/logging/loguru/)

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated (16:10 aspect ratio)
- [ ] Placed in docs/concepts-for-researchers.md
