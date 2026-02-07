# fig-repo-17a: Finding Your Error in 1000 Runs (ELI5)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-17a |
| **Title** | Finding Your Error in 1000 Runs |
| **Complexity Level** | L0 (ELI5 - Concept only) |
| **Target Persona** | Research Scientist, Jupyter user |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Show why structured logging helps find errors in long-running experiments—NO technical log levels, NO API details.

## Key Message

"When something fails at iteration 847 out of 1000, colored logs help you find it instantly."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FINDING YOUR ERROR IN 1000 RUNS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PROBLEM: Your experiment ran 1000 bootstrap iterations.                    │
│               Something failed. But where?                                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ❌ WITH print()                       ✅ WITH STRUCTURED LOGGING               │
│  ─────────────────                     ──────────────────────────               │
│                                                                                 │
│  ┌──────────────────────────┐          ┌────────────────────────────────────┐  │
│  │ Processing...            │          │ ✓ 14:23:01 Starting iteration 1    │  │
│  │ Processing...            │          │ ✓ 14:23:02 Starting iteration 2    │  │
│  │ Processing...            │          │ ✓ ...                              │  │
│  │ Processing...            │          │ ✓ 14:31:45 Starting iteration 846  │  │
│  │ Processing...            │          │ ✓ 14:31:46 Starting iteration 847  │  │
│  │ Error                    │          │ ❌ 14:31:47 ERROR at iteration 847 │  │
│  │ Processing...            │          │    → Subject: PLR0123              │  │
│  │ Processing...            │          │    → Problem: Missing data         │  │
│  │ Done                     │          │ ✓ 14:31:48 Recovered, continuing   │  │
│  └──────────────────────────┘          └────────────────────────────────────┘  │
│                                                                                 │
│  😰 "Which iteration failed?"          😊 "Iteration 847, subject PLR0123!"   │
│  😰 "When did it happen?"              😊 "At 14:31:47, took 8 minutes"       │
│  😰 "What was the problem?"            😊 "Missing data in that subject"      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE COLORS HELP                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  ✓ Green  = Everything is fine                                                  │
│  ⚠ Yellow = Something unusual (but handled)                                    │
│  ❌ Red    = Error! Look here!                                                  │
│                                                                                 │
│  The RED error stands out in a sea of green checkmarks!                         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  139 of our Python files use this logging system.                               │
│  We catch errors in seconds, not hours.                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements (MAX 5 CONCEPTS)

1. **Problem statement**: 1000 iterations, something failed, where?
2. **Side-by-side terminal comparison**: Messy print vs organized logs
3. **Three questions answered**: Which iteration? When? What problem?
4. **Color coding**: Green/Yellow/Red simple explanation
5. **Adoption stat**: 139 files use this system

## Text Content

### Title Text
"Finding Your Error in 1000 Runs"

### Labels/Annotations
- NO log level hierarchy (DEBUG/INFO/WARNING/ERROR)
- NO thread-safety or technical features
- Simple: "green = OK, red = error"

### Caption
When your experiment runs 1000 iterations and something fails, structured logging helps you find it instantly. The error is highlighted in red with the exact iteration number, timestamp, and subject ID. No more scrolling through walls of "Processing..." to find what went wrong.

## Prompts for Nano Banana Pro

### Style Prompt
Debugging experience comparison for researchers. Two terminal windows side-by-side. Left terminal is cluttered and confusing. Right terminal is organized with color-coded messages (green checkmarks, red errors). Include question/answer format showing what information you can find. Simple traffic light color system. Medical research context.

### Content Prompt
Create a debugging comparison:

**TOP - Problem Statement**:
- Text: "1000 iterations. Something failed. Where?"

**MIDDLE - Terminal Comparison**:
- LEFT: Gray terminal with repetitive "Processing..." and one buried "Error"
- RIGHT: Colorful terminal with timestamps, green checkmarks, one red ERROR with details

**BOTTOM LEFT - Questions**:
- Three sad face emoji with questions: "Which iteration?", "When?", "What problem?"

**BOTTOM RIGHT - Answers**:
- Three happy face emoji with answers: "Iteration 847", "14:31:47", "Missing data"

**FOOTER - Color Guide**:
- Traffic light: Green = OK, Yellow = Warning, Red = Error
- Stat: "139 files use this system"

NO log levels, NO technical jargon, NO code.

## Alt Text

Debugging comparison: Left shows cluttered terminal with repetitive "Processing..." messages and buried error. Right shows organized terminal with timestamps, green checkmarks for success, and red highlighted error showing "Iteration 847, Subject PLR0123, Missing data". Simple color guide: green=OK, yellow=warning, red=error.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md
