# CRITICAL-FAILURE-002: No Visual Verification of Generated Figures

**Date:** 2026-01-27
**Severity:** CRITICAL
**Category:** Figure QA / Quality Assurance

## The Incident

Claude generated R code for CD diagrams (Demšar-style Critical Difference diagrams), claimed the code was "fixed" with Economist styling, but **never visually verified the actual output**.

The generated `fig_cd_diagrams.png` had multiple severe problems:
1. **Method names clipped** - Labels extend beyond figure boundaries and are cut off
2. **Unreadable text** - Panel C has extremely long pipeline names that overlap
3. **Wrong output directory** - Script saved to `main/` instead of `supplementary/`
4. **Extraneous files** - Script created 4 files when only 1 was requested

## Evidence

User feedback:
> "the remaining fig_cd_diagrams.png looks awful with missing data!"
> "Can you open a metalearning doc for this failure to create proper visual verification testing"

The actual output showed:
- Panel C "Combined Preprocessing Pipelines" with names like "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune + C..." being truncated
- Method labels running off the right edge of the figure
- Abbreviation system failing to catch long pipeline combination names

## Root Cause

1. **No visual verification step** - Claude generated code and assumed it worked
2. **Testing in isolation** - The `create_cd_diagram()` function was developed without testing the full pipeline with real data
3. **Overconfidence** - Multiple "fix" iterations were claimed complete without actual validation
4. **Missing feedback loop** - No automated check that output figures are actually readable

## Why This Is Critical

- **Scientific integrity** - Unreadable figures cannot be used for publication
- **Wasted iteration cycles** - User had to manually identify problems Claude should have caught
- **Trust erosion** - Claiming fixes that don't work damages credibility

## Mandatory Remediation

### Immediate Actions

1. **Always view generated figures** - Use the Read tool on PNG files to visually inspect output
2. **Check for clipping** - Verify all text labels are fully visible within figure bounds
3. **Validate with real data** - Don't test with toy examples only

### Systematic Prevention

Add to figure generation workflow:

```r
# MANDATORY: Visual verification checklist
# After generating ANY figure:
# 1. Open the PNG and visually inspect
# 2. Check: Are all labels fully visible?
# 3. Check: Is text readable at intended size?
# 4. Check: Does the figure match the specification?
# 5. Check: Is output going to correct directory?
```

### Test Harness Requirements

The existing `tests/test_figure_qa/` should be extended with:

1. **Bounding box validation** - Ensure no plot elements extend beyond figure margins
2. **Text legibility check** - Verify minimum font sizes
3. **Output path validation** - Confirm figures go to configured directories
4. **Single-file validation** - Ensure scripts don't create unwanted byproduct files

## Correct Behavior Going Forward

**BEFORE claiming any figure generation is complete:**

1. Use `Read` tool to view the actual PNG output
2. Check that ALL text is readable and unclipped
3. Verify output is in the correct directory per `figure_layouts.yaml`
4. Confirm no extraneous files were created
5. Compare against user requirements explicitly

## Related Failures

- **CRITICAL-FAILURE-001**: Synthetic data used instead of real predictions
- Both failures share a common theme: **insufficient validation of outputs**

## Lessons Learned

> "Trust but verify" is insufficient. The correct approach is "verify then report."

Never claim a fix is complete without visual confirmation of the actual output.
