# fig-repo-17b: Log Levels and Thread-Safe Debugging (Expert)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-17b |
| **Title** | Log Levels and Thread-Safe Debugging |
| **Complexity Level** | L3 (Expert - Technical deep-dive) |
| **Target Persona** | Software Engineers, DevOps |
| **Location** | CONTRIBUTING.md, docs/development/ |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain log levels, thread-safety for parallel bootstrap, and loguru's production features.

## Key Message

"DEBUG for development, INFO for production. Thread-safe logging captures parallel bootstrap iterations without race conditions."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LOG LEVELS AND THREAD-SAFE DEBUGGING                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  LOG LEVEL HIERARCHY                                                            │
│  ═══════════════════                                                            │
│                                                                                 │
│  ┌─────────────┐                                                                │
│  │  CRITICAL   │  System-wide failure, abort immediately                        │
│  │─────────────│  "Database connection lost"                                    │
│  │   ERROR     │  Operation failed, needs attention                             │
│  │─────────────│  "Classification failed for MOMENT+SAITS"                      │
│  │  WARNING    │  Unexpected but recoverable                                    │
│  │─────────────│  "Subject PLR0042 has 15% missing data, imputing"              │
│  │   INFO      │  Normal operation milestones                                   │
│  │─────────────│  "Bootstrap iteration 500/1000 complete"                       │
│  │   DEBUG     │  Detailed internal state (dev only)                            │
│  └─────────────┘  "Processing subject PLR0001, features=[20]"                   │
│                                                                                 │
│  Production: logger.level("INFO")  →  Shows INFO, WARNING, ERROR, CRITICAL     │
│  Debugging:  logger.level("DEBUG") →  Shows everything                          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THREAD-SAFETY FOR PARALLEL BOOTSTRAP                                           │
│  ════════════════════════════════════                                           │
│                                                                                 │
│  1000 bootstrap iterations × 8 CPU cores = 8 parallel threads logging           │
│                                                                                 │
│  ❌ WITHOUT thread-safety:           ✅ WITH loguru (thread-safe):             │
│  ┌─────────────────────────────┐     ┌─────────────────────────────┐           │
│  │ Iter 5Iter 3Iter 7 don...   │     │ [T1] Iter 500 complete      │           │
│  │ ...e Iter 2 completeIter... │     │ [T2] Iter 501 complete      │           │
│  │ 4 complete ERROR at It...   │     │ [T3] ERROR at Iter 502      │           │
│  │ (garbled, interleaved)      │     │ [T4] Iter 503 complete      │           │
│  └─────────────────────────────┘     └─────────────────────────────┘           │
│                                                                                 │
│  Race condition: messages overlap    Atomic writes: clean separation            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LOGURU PRODUCTION FEATURES                                                     │
│  ══════════════════════════                                                     │
│                                                                                 │
│  🎨 Colored output     Color-coded by level (red=ERROR, green=INFO)             │
│  📁 File rotation      logger.add("file.log", rotation="1 day", retention="7d") │
│  🧵 Thread-safe        All sinks are thread-safe by default                     │
│  🔍 Exception capture  @logger.catch() decorates functions for auto-logging     │
│  📊 JSON serialization logger.add(sink, serialize=True) for machine parsing     │
│  ⚡ Lazy evaluation    logger.debug("Heavy: {}", expensive_fn) - only if DEBUG  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CODE EXAMPLES                                                                  │
│  ═════════════                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │ from loguru import logger                                                  ││
│  │                                                                            ││
│  │ # Basic usage                                                              ││
│  │ logger.debug("Processing subject {}", subject_id)                          ││
│  │ logger.info("Bootstrap iteration {}/{} complete", i, total)                ││
│  │ logger.warning("Missing data in {}, imputing", subject_id)                 ││
│  │ logger.error("Classification failed: {}", error_msg)                       ││
│  │                                                                            ││
│  │ # Production setup with rotation                                           ││
│  │ logger.add(                                                                ││
│  │     "logs/pipeline_{time}.log",                                            ││
│  │     rotation="1 day",                                                      ││
│  │     retention="7 days",                                                    ││
│  │     level="INFO"                                                           ││
│  │ )                                                                          ││
│  │                                                                            ││
│  │ # Auto exception capture                                                   ││
│  │ @logger.catch()                                                            ││
│  │ def bootstrap_iteration(i):                                                ││
│  │     ...  # Any exception is automatically logged with full traceback       ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  VERIFIED: 139 source files use loguru across this repository                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Log level pyramid**: DEBUG → INFO → WARNING → ERROR → CRITICAL with examples
2. **Verbosity control**: Production vs debugging settings
3. **Thread-safety diagram**: Garbled output vs clean atomic writes
4. **Feature list**: rotation, thread-safety, exception capture, JSON, lazy eval
5. **Code examples**: Basic usage, production setup, @logger.catch()

## Text Content

### Title Text
"Log Levels and Thread-Safe Debugging"

### Caption
Loguru provides structured logging with five severity levels (DEBUG to CRITICAL) and thread-safe output for parallel bootstrap iterations. Production logs show INFO and above; debugging shows everything. Features include automatic file rotation, exception capture via @logger.catch(), and JSON serialization for machine parsing. Used in 139 files across this repository.

## Prompts for Nano Banana Pro

### Style Prompt
Technical logging documentation with hierarchy diagrams and code blocks. Log level pyramid with color coding. Thread-safety comparison showing garbled vs clean output. Feature list with icons. Multiple code blocks in dark theme. Economist-style clean layout. Matte, professional colors.

### Content Prompt
Create a technical logging documentation figure:

**SECTION 1 - Log Level Pyramid**:
- Five-level pyramid: DEBUG (bottom) to CRITICAL (top)
- Each level with example message
- Arrow showing "verbosity control"

**SECTION 2 - Thread-Safety**:
- Two terminal boxes side-by-side
- LEFT: Garbled, interleaved text (race condition)
- RIGHT: Clean, organized with thread IDs

**SECTION 3 - Features**:
- Six icons with labels: colors, rotation, thread-safe, exception, JSON, lazy eval

**SECTION 4 - Code**:
- Two dark code blocks: basic usage and production setup
- Include @logger.catch() example

## Alt Text

Technical logging documentation. Log level pyramid from DEBUG (most verbose) to CRITICAL (least verbose) with example messages for each level. Thread-safety comparison showing garbled output without proper logging vs clean atomic writes with loguru. Feature list: colored output, file rotation, thread-safety, exception capture, JSON serialization, lazy evaluation. Code examples showing basic usage and production setup with file rotation.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in CONTRIBUTING.md
