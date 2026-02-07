#!/bin/bash
# Type checking script for CI - reports but doesn't block
# See docs/planning/pipeline-robustness-plan.md for context

set -e

echo "=== Type Checking Critical Pipeline Modules ==="
echo ""

CRITICAL_MODULES=(
    "src/ensemble/ensemble_utils.py"
    "src/classification/classifier_log_utils.py"
    "src/classification/weighing_utils.py"
    "src/featurization/feature_utils.py"
)

# Run mypy and capture output
OUTPUT=$(uv run mypy "${CRITICAL_MODULES[@]}" 2>&1 || true)

# Count errors in critical files only (not transitive deps)
CRITICAL_ERRORS=$(echo "$OUTPUT" | grep -E "^src/(ensemble/ensemble_utils|classification/(classifier_log_utils|weighing_utils)|featurization/feature_utils)" | grep -c "error:" || echo "0")

TOTAL_ERRORS=$(echo "$OUTPUT" | grep -c "error:" || echo "0")

echo "Critical module errors: $CRITICAL_ERRORS"
echo "Total errors (including transitive): $TOTAL_ERRORS"
echo ""

# Baseline established 2026-02-01: 218 total errors
BASELINE=218

if [ "$TOTAL_ERRORS" -gt "$BASELINE" ]; then
    echo "⚠️  Error count increased from baseline ($BASELINE → $TOTAL_ERRORS)"
    echo "Please review new type errors before committing."
    exit 1
elif [ "$TOTAL_ERRORS" -lt "$BASELINE" ]; then
    echo "✅ Error count decreased from baseline ($BASELINE → $TOTAL_ERRORS)"
    echo "Consider updating baseline in this script."
fi

echo ""
echo "=== Type Check Complete ==="
