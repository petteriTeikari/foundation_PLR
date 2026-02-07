# FAILURE-003: Repeated Instruction Amnesia on Visual Bugs

## Severity: HIGH

## The Failure

Claude repeatedly failed to fix the yellow density overlap in `fig_multi_metric_raincloud.png` despite:
1. User mentioning it **at least twice** explicitly
2. The issue being documented in `figure-production-grade.md` (line 306)
3. The image clearly showing the overlap when viewed

## Root Cause Analysis

### 1. Context Compaction Amnesia
- Session hit context limit and was compacted
- Compaction summary mentioned "Yellow overlap in raincloud" as pending
- Claude continued with infrastructure work instead of fixing the actual bug

### 2. Prioritization Failure
- Claude got distracted creating XML tracking plans and style loaders
- Ignored the **visible bug** in favor of **architectural improvements**
- User wanted the bug FIXED, not more planning documents

### 3. Reading Plans Without Executing
- Claude read `figure-production-grade.md` which listed the issue
- Created tasks in XML tracking it as "NOT_STARTED"
- Never actually wrote the code to fix it

### 4. Failure to Look at Generated Images
- The PNG clearly shows lowercase "b" panel labels
- The PNG clearly shows yellow overlapping with blue Ensemble
- Claude updated code but didn't verify the OUTPUT

## What Should Have Happened

1. **First mention**: User says "yellow overlap" → Claude fixes it immediately
2. **See it in plan**: Plan says "pending fix" → Claude executes it
3. **After regenerating**: Claude views the PNG to verify fix worked

## The Actual Fix Required

In `fig_multi_metric_raincloud.R`, the `stat_halfeye()` needs bandwidth adjustment:

```r
# BEFORE (causes overlap):
ggdist::stat_halfeye(
  adjust = 0.5,      # Too narrow bandwidth
  width = 0.6,       # Distributions too tall
  ...
)

# AFTER (prevents overlap):
ggdist::stat_halfeye(
  adjust = 1.0,      # Wider bandwidth = smoother
  width = 0.4,       # Shorter distributions
  scale = "width",   # Scale by width, not area
  ...
)
```

Or increase vertical spacing between categories.

## Prevention Guardrails

### 1. Bug-First Rule
> When a visual bug is reported, fix it BEFORE any other work.
> Infrastructure improvements can wait; broken figures cannot.

### 2. Verify Output Rule
> After modifying figure code, VIEW the regenerated PNG.
> Code changes mean nothing if output is still broken.

### 3. Compaction Handoff Rule
> After context compaction, first action should be:
> "What bugs were reported that I haven't fixed yet?"

### 4. Count Mentions Rule
> If user mentions something 2+ times, it becomes CRITICAL priority.
> Drop everything else and fix it.

## Impact

- User frustration increased with each repetition
- Trust in Claude's ability to follow instructions eroded
- Time wasted on planning instead of fixing
- Session context wasted on non-productive work

## Lessons

1. **Plans are worthless if not executed**
2. **Visual bugs require visual verification**
3. **User repetition = escalating priority**
4. **Fix the bug, THEN improve architecture**

## Files Affected

- `src/r/figures/fig_multi_metric_raincloud.R` - needs ggdist parameter fix
- `figures/generated/ggplot2/main/fig_multi_metric_raincloud.png` - still broken

## Status

- [x] Fix applied to R script (2026-01-28)
- [x] Figure regenerated (2026-01-28)
- [x] Output visually verified (no overlap) - height=0.5 constraint works
- [x] Panel labels confirmed "A  Discrimination" style (matches fig_calibration_dca_combined.png)

---

## ADDENDUM: Same Failure Pattern Repeated (2026-01-28)

### The Failure (Again)

While fixing `fig_multi_metric_raincloud.png`, I was shown `fig_roc_rc_combined.png` which CLEARLY had wrong styling ("A" + subtitle below instead of "A  ROC Curves"). I:
1. Started editing the file
2. Made partial changes
3. Got distracted verifying the raincloud figure
4. Did NOT complete and regenerate the ROC/RC figure
5. Required ANOTHER user prompt to finish

### Root Cause

**Incomplete Task Execution**: Started a fix, verified something else, forgot to finish.

### Prevention Rule (ADDED)

> **Complete-Before-Move Rule**: When fixing a figure:
> 1. Make ALL edits to the file
> 2. Regenerate the figure
> 3. Verify the output visually
> 4. ONLY THEN move to the next task
>
> NEVER start editing a file, then switch context without completing the fix cycle.
