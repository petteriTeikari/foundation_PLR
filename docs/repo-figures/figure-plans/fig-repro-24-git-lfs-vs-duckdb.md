# fig-repro-24: Git LFS vs DuckDB for Large Data

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-24 |
| **Title** | Git LFS vs DuckDB for Large Data |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | ML Engineer, Data Scientist |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P3 |
| **Aspect Ratio** | 16:10 |

## Purpose

Compare data storage strategies for reproducibility: Git LFS (Large File Storage) vs DuckDB consolidation.

## Key Message

"Git LFS tracks large files but doesn't solve data sprawl. DuckDB consolidates data into a queryable, portable single file. We chose DuckDB for analytical reproducibility."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GIT LFS VS DUCKDB FOR LARGE DATA                             │
│                    Two approaches to data reproducibility                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PROBLEM: LARGE DATA IN REPOSITORIES                                        │
│  ═════════════════════════════════════════                                      │
│                                                                                 │
│  Git wasn't designed for large binary files:                                    │
│  • CSVs > 100MB slow down clone                                                 │
│  • Binary diffs are useless                                                     │
│  • Repository bloats over time                                                  │
│  • GitHub limits: 100MB per file, 1GB total                                     │
│                                                                                 │
│  Two solutions: Git LFS or consolidation                                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  OPTION 1: GIT LFS                          OPTION 2: DUCKDB                    │
│  ══════════════════                         ════════════════                    │
│                                                                                 │
│  ┌─────────────────────────┐                ┌─────────────────────────┐        │
│  │                         │                │                         │        │
│  │  git lfs track "*.csv"  │                │  All CSVs → DuckDB      │        │
│  │                         │                │                         │        │
│  │  data/                  │                │  data/                  │        │
│  │  ├── file1.csv (LFS)    │                │  └── all_data.db        │        │
│  │  ├── file2.csv (LFS)    │                │                         │        │
│  │  ├── file3.csv (LFS)    │                │  42 MB, single file     │        │
│  │  └── ... (500 files)    │                │                         │        │
│  │                         │                │  SQL-queryable!         │        │
│  │  Pointer files in git   │                │                         │        │
│  │  Actual data on server  │                │                         │        │
│  │                         │                │                         │        │
│  └─────────────────────────┘                └─────────────────────────┘        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMPARISON                                                                     │
│  ══════════                                                                     │
│                                                                                 │
│  │ Feature                │ Git LFS        │ DuckDB          │                 │
│  │ ────────────────────── │ ────────────── │ ─────────────── │                 │
│  │ Tracks file changes    │ ✅ Yes         │ ⚠️ Whole DB     │                 │
│  │ Keeps file structure   │ ✅ Yes         │ ❌ Consolidated │                 │
│  │ Queryable              │ ❌ No          │ ✅ SQL          │                 │
│  │ Reduces file count     │ ❌ No          │ ✅ 500 → 1      │                 │
│  │ Self-documenting       │ ❌ No          │ ✅ Schema       │                 │
│  │ Cross-language         │ ⚠️ Read files  │ ✅ Python + R   │                 │
│  │ Requires server        │ ✅ LFS server  │ ❌ No           │                 │
│  │ Offline access         │ ⚠️ Needs fetch │ ✅ Full         │                 │
│  │ Version confusion      │ ⚠️ Still files │ ✅ Eliminated   │                 │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHEN TO USE EACH                                                               │
│  ═════════════════                                                              │
│                                                                                 │
│  USE GIT LFS:                              USE DUCKDB:                          │
│  ─────────────                             ──────────                           │
│  • Raw data archives (don't query)         • Analytical data (query often)     │
│  • Model weights (opaque binaries)         • Structured results                │
│  • Media files (images, videos)            • Multiple related tables           │
│  • Need per-file version history           • Need SQL joins/aggregations       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR CHOICE: DUCKDB                                                  │
│  ═════════════════════════════                                                  │
│                                                                                 │
│  We chose DuckDB because:                                                       │
│                                                                                 │
│  1. ANALYTICAL WORKFLOW                                                         │
│     Our data is meant to be queried, not just stored                            │
│     → SQL is the universal query language                                       │
│                                                                                 │
│  2. ELIMINATING VERSION CONFUSION                                               │
│     500 CSVs = 500 opportunities for "which file?"                              │
│     1 DuckDB = 1 source of truth                                                │
│                                                                                 │
│  3. PYTHON + R INTEROP                                                          │
│     Both languages read the same .db file                                       │
│     → No data format translation                                                │
│                                                                                 │
│  4. PORTABLE REPRODUCIBILITY                                                    │
│     Copy one file, get all data                                                 │
│     No LFS server needed                                                        │
│                                                                                 │
│  Trade-off: We lose per-CSV version history, but gain:                          │
│  • Schema documentation                                                         │
│  • Referential integrity                                                        │
│  • 6-8x faster analytics                                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Problem statement**: Git + large files issues
2. **Two approaches**: Visual comparison of Git LFS vs DuckDB
3. **Feature comparison table**: 9 dimensions compared
4. **Use case recommendations**: When to use each
5. **Foundation PLR rationale**: Four reasons for DuckDB choice

## Text Content

### Title Text
"Git LFS vs DuckDB: Two Approaches to Large Data"

### Caption
Git LFS tracks large files externally while keeping directory structure. DuckDB consolidates files into a queryable database. Foundation PLR chose DuckDB: 500 CSVs → 1 database, SQL-queryable, Python+R compatible, no server required. Trade-off: less per-file history, but eliminates version confusion and enables analytical reproducibility.

## Prompts for Nano Banana Pro

### Style Prompt
Two-panel comparison: Git LFS (file tree with LFS markers) vs DuckDB (single database file). Feature comparison table. Use case recommendations. Foundation PLR rationale section. Clean, architectural style.

### Content Prompt
Create "Git LFS vs DuckDB" infographic:

**TOP - Two Panels**:
- Left: Git LFS with 500 tracked files
- Right: DuckDB single 42 MB file

**MIDDLE - Comparison Table**:
- 9 features compared with checkmarks/X

**BOTTOM - Recommendations + Choice**:
- When to use each
- Foundation PLR: 4 reasons for DuckDB

## Alt Text

Git LFS vs DuckDB comparison for large data. Problem: Git wasn't designed for large files (100MB limit, binary diffs useless). Git LFS: tracks 500 CSV files with pointers, requires LFS server, keeps file structure. DuckDB: consolidates to single 42 MB file, SQL-queryable, no server needed. Feature table compares 9 dimensions: DuckDB wins on queryability, file reduction, cross-language support, offline access, version confusion elimination. Foundation PLR chose DuckDB for analytical workflow, eliminating version confusion, Python+R interop, and portable reproducibility.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

