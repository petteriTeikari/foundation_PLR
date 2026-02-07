#!/bin/bash
# Generate PLR Decomposition 5×5 Grid Figure
# Run this after extraction completes (data/private/preprocessed_signals_per_subject.db)

set -e

DB_PATH="data/private/preprocessed_signals_per_subject.db"

# Check if extraction is complete (no WAL file or empty WAL)
if [ -f "${DB_PATH}.wal" ] && [ -s "${DB_PATH}.wal" ]; then
    WAL_SIZE=$(stat -c%s "${DB_PATH}.wal" 2>/dev/null || stat -f%z "${DB_PATH}.wal" 2>/dev/null)
    if [ "$WAL_SIZE" -gt 1000 ]; then
        echo "WARNING: Extraction may still be running (WAL file exists: ${WAL_SIZE} bytes)"
        echo "Check with: ps aux | grep decomposition"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check if DB exists
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database not found at $DB_PATH"
    echo "Run the extraction first with:"
    echo "  uv run python src/orchestration/tasks/decomposition_extraction.py"
    exit 1
fi

echo "=== Generating PLR Decomposition 5×5 Grid ==="
echo "Database: $DB_PATH"
echo "Bootstrap iterations: 100"
echo ""

# Generate the figure
uv run python src/viz/fig_decomposition_grid.py --db "$DB_PATH" --bootstrap 100

echo ""
echo "=== Done ==="
echo "Figure saved to: figures/generated/fig_decomposition_grid.png"
echo "JSON data saved to: figures/generated/data/fig_decomposition_grid.json"
