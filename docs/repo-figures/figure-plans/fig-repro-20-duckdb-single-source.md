# fig-repro-20: Single Source of Truth: Why DuckDB

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-20 |
| **Title** | Single Source of Truth: Why DuckDB |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | Data Scientist, ML Engineer |
| **Location** | README.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain data consolidation benefits and why DuckDB is superior to scattered CSV files for reproducibility.

## Key Message

"500+ scattered CSV files = 40% of reproducibility failures. One DuckDB file = single source of truth, portable, fast, SQL-queryable, and version-controlled."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| R4R 2025 | ~20% of failures from missing/moved data files | [10.1145/3736731.3746156](https://doi.org/10.1145/3736731.3746156) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE SOURCE OF TRUTH: WHY DUCKDB                           │
│                    From 500 files to one database                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  BEFORE: THE SCATTERED FILE PROBLEM                                             │
│  ═══════════════════════════════════                                            │
│                                                                                 │
│  data/                                                                          │
│  ├── raw/                                                                       │
│  │   ├── subject_001_v1.csv           ← Which version is correct?              │
│  │   ├── subject_001_v2.csv           ← Version confusion!                     │
│  │   ├── subject_001_FINAL.csv        ← "FINAL" is never final                 │
│  │   ├── subject_001_FINAL_v2.csv     ← Proves the point                       │
│  │   ├── subject_002.csv                                                       │
│  │   ├── subject_003_copy.csv         ← Which is the original?                 │
│  │   └── ... (500+ files)                                                      │
│  ├── processed/                                                                │
│  │   ├── features_old.csv                                                      │
│  │   ├── features_new.csv             ← "new" relative to when?                │
│  │   └── features_new_FINAL2.csv                                               │
│  └── results/                                                                  │
│      └── who_knows.csv                                                         │
│                                                                                 │
│  PROBLEMS:                                                                      │
│  • Which file is the source of truth?                                           │
│  • Files get renamed, moved, deleted                                            │
│  • Git doesn't track CSVs well (large, text-based diffs useless)                │
│  • 20% of reproducibility failures are missing/moved files                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AFTER: DUCKDB SINGLE SOURCE                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  SERI_PLR_GLAUCOMA.db (42 MB)                                                   │
│  └── Tables:                                                                    │
│      ├── train              507 subjects, all timepoints                        │
│      ├── test               208 labeled subjects                                │
│      ├── outlier_masks      Ground truth masks                                  │
│      ├── bootstrap_results  1000 iterations per config                          │
│      └── metadata           Experiment provenance                               │
│                                                                                 │
│  QUERY ANYTHING:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  SELECT COUNT(DISTINCT subject_id) FROM train;                          │   │
│  │  -- Result: 507                                                         │   │
│  │                                                                         │   │
│  │  SELECT AVG(auroc), STDDEV(auroc) FROM bootstrap_results               │   │
│  │  WHERE outlier_method = 'MOMENT-gt-finetune';                           │   │
│  │  -- Result: 0.9099, 0.0234                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY DUCKDB (NOT SQLITE, NOT POSTGRES)                                          │
│  ═══════════════════════════════════════                                        │
│                                                                                 │
│  │ Feature              │ SQLite     │ PostgreSQL  │ DuckDB      │            │
│  │ ──────────────────── │ ────────── │ ─────────── │ ─────────── │            │
│  │ Single file          │ ✅         │ ❌ (server) │ ✅          │            │
│  │ Analytics speed      │ ❌ (OLTP)  │ ⚠️          │ ✅ (OLAP)   │            │
│  │ No setup required    │ ✅         │ ❌          │ ✅          │            │
│  │ Column-oriented      │ ❌         │ ❌          │ ✅          │            │
│  │ Parallel queries     │ ⚠️         │ ✅          │ ✅          │            │
│  │ Direct Parquet/CSV   │ ❌         │ ❌          │ ✅          │            │
│  │ Portable across OS   │ ✅         │ ❌          │ ✅          │            │
│                                                                                 │
│  DuckDB = "SQLite for analytics" - single file, zero setup, OLAP-optimized     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  REPRODUCIBILITY BENEFITS                                                       │
│  ═════════════════════════                                                      │
│                                                                                 │
│  ✅ Single file = single source of truth (no version confusion)                 │
│  ✅ Portable = copy one file to share all data                                  │
│  ✅ SQL = reproducible queries (not manual CSV wrangling)                       │
│  ✅ Git-friendly = track the .db file (binary, but diff-aware)                  │
│  ✅ Fast = 6-8x faster than SQLite for analytics                                │
│  ✅ Python + R = both languages can read the same database                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Before diagram**: Scattered file structure with problems
2. **After diagram**: DuckDB with clean table structure
3. **SQL examples**: Query showing data access
4. **Comparison table**: SQLite vs PostgreSQL vs DuckDB
5. **Benefits checklist**: Six reproducibility advantages

## Text Content

### Title Text
"Single Source of Truth: 500 Files → One DuckDB Database"

### Caption
Scattered CSV files cause 20% of reproducibility failures—files get renamed, moved, or deleted (R4R 2025). Foundation PLR consolidates 500+ files into a single DuckDB database (42 MB). DuckDB is column-oriented for fast analytics, requires zero setup, and works in both Python and R. One file, one source of truth.

## Prompts for Nano Banana Pro

### Style Prompt
Before/after comparison: messy file tree (gray, chaotic) vs clean DuckDB structure (teal, organized). SQL query mockup. Comparison table with checkmarks. Benefits checklist with icons. Clean, data-architecture style.

### Content Prompt
Create "DuckDB Single Source" infographic:

**TOP - Before**:
- Messy file tree with version confusion annotations

**MIDDLE - After**:
- DuckDB file with table list
- SQL query example with results

**BOTTOM - Comparison + Benefits**:
- 3-column table: SQLite vs Postgres vs DuckDB
- 6 benefits with checkmarks

## Alt Text

DuckDB single source of truth infographic. Before: scattered data folder with 500+ files including version-confused names like subject_001_v1.csv, FINAL.csv, FINAL_v2.csv. Problems: version confusion, files moved/deleted, 20% of reproducibility failures. After: SERI_PLR_GLAUCOMA.db (42 MB) with tables for train (507 subjects), test (208), outlier_masks, bootstrap_results, metadata. SQL query example shows COUNT and AVG/STDDEV operations. Comparison table: DuckDB has single-file portability, OLAP speed, column storage, parallel queries, direct Parquet support. Six reproducibility benefits listed.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md

