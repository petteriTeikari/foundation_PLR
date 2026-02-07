# fig-repo-16: DuckDB: Your Portable Data Warehouse

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-16 |
| **Title** | DuckDB: Your Portable Data Warehouse |
| **Complexity Level** | L1 (Concept explanation) |
| **Target Persona** | All (especially PIs and Biostatisticians) |
| **Location** | docs/concepts-for-researchers.md, Root README |
| **Priority** | P0 (Critical - central data architecture) |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain what DuckDB is and why it replaces scattered CSV files as the single source of truth for this project.

## Key Message

"DuckDB is like having a portable Excel that can handle millions of rows and complex SQL queries in milliseconds—no server required."

## Visual Concept

**Before/After comparison with query example:**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DUCKDB: YOUR PORTABLE DATA WAREHOUSE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  BEFORE: The CSV Nightmare                AFTER: Single Source of Truth         │
│  ═════════════════════════════            ══════════════════════════════        │
│                                                                                 │
│  📁 /data/                                🗄️ SERI_PLR_GLAUCOMA.db              │
│  ├── PLR0001.csv                          ┌────────────────────────┐            │
│  ├── PLR0002.csv                          │ • 507 subjects         │            │
│  ├── PLR0003.csv                          │ • 1,981 timepoints each│            │
│  ├── ...                                  │ • 1M+ total rows       │            │
│  └── PLR0507.csv                          │ • Single file: ~150 MB │            │
│                                           └────────────────────────┘            │
│  ❌ 507 separate files                    ✅ One portable file                  │
│  ❌ Manual joining                        ✅ SQL queries                        │
│  ❌ Slow loading                          ✅ Fast analytics                     │
│  ❌ No relationships                      ✅ Indexed, relational                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY DUCKDB (not SQLite or PostgreSQL)?                                         │
│  ──────────────────────────────────────                                         │
│                                                                                 │
│  SQLite         PostgreSQL        DuckDB                                        │
│  ┌─────────┐    ┌─────────┐       ┌─────────┐                                   │
│  │ OLTP    │    │ OLTP+   │       │ OLAP    │ ← Optimized for analytics!       │
│  │ (rows)  │    │ OLAP    │       │ (cols)  │                                   │
│  └─────────┘    └─────────┘       └─────────┘                                   │
│                                                                                 │
│  Good for:      Good for:         Good for:                                     │
│  • Mobile apps  • Web apps        • Data science                                │
│  • Simple CRUD  • Enterprise      • Analytics                                   │
│  • Transactions • Full server     • Single-file                                 │
│                                                                                 │
│  Speed for "SELECT AVG(auroc) GROUP BY method":                                 │
│  SQLite:      ████████████████░░░░░░░░  800ms                                  │
│  PostgreSQL:  ████████████░░░░░░░░░░░░  600ms (requires server!)               │
│  DuckDB:      ██░░░░░░░░░░░░░░░░░░░░░░  100ms ← 6-8× faster!                   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXAMPLE QUERY                                                                  │
│  ─────────────                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ SELECT outlier_method, AVG(auroc) as mean_auroc                          │  │
│  │ FROM essential_metrics                                                   │  │
│  │ WHERE classifier = 'CatBoost'                                            │  │
│  │ GROUP BY outlier_method                                                  │  │
│  │ ORDER BY mean_auroc DESC;                                                │  │
│  │                                                                          │  │
│  │ ➜ Returns top methods in 100ms!                                          │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  For Excel users: Think of SQL as "advanced pivot tables with formulas"         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. **Before/After visualization**: 507 CSVs → 1 DuckDB file
2. **Database comparison table**: SQLite vs PostgreSQL vs DuckDB
3. **Speed comparison**: Query times for analytical operations
4. **Example SQL query**: Simple aggregate showing typical usage
5. **"For Excel users" callout**: Analogy to pivot tables

### Optional Elements
1. Schema diagram showing table relationships
2. DuckDB logo/icon
3. "No server required" badge
4. File size comparison (150MB single file vs scattered)

## Text Content

### Title Text
"DuckDB: Your Portable Data Warehouse"

### Labels/Annotations
- Before: "507 separate CSV files = reproducibility nightmare"
- After: "Single DuckDB file = portable, fast, relational"
- Comparison: "DuckDB is optimized for analytics (OLAP), not transactions (OLTP)"
- Speed: "6-8× faster than SQLite for aggregate queries"
- Analogy: "For Excel users: SQL is like advanced pivot tables"

### Caption (for embedding)
DuckDB replaces 507 scattered CSV files with a single, portable database file. Unlike SQLite (optimized for transactions) or PostgreSQL (requires a server), DuckDB is designed for analytical queries—exactly what we need for research. Aggregate queries like "average AUROC by method" complete in ~100ms. Think of it as "Excel for a million rows"—all the power of SQL without server setup.

## Prompts for Nano Banana Pro

### Style Prompt
Data architecture comparison infographic. Clean before/after split layout. Use folder icons for CSV files, cylinder database icon for DuckDB. Include a small SQL code block on dark background. Comparison table with database icons. Speed comparison as horizontal bars. Economist-style data visualization. Matte, professional colors. Medical/clinical research context.

### Content Prompt
Create a data architecture comparison infographic:

**TOP HALF - Before/After**:
- LEFT: Folder icon with many small CSV file icons spilling out, label "507 CSVs"
- RIGHT: Single clean database cylinder icon, label "One DuckDB file"
- Arrow between them labeled "Consolidation"
- Checkmarks/X marks for features below each

**MIDDLE - Database Comparison**:
- Three columns: SQLite, PostgreSQL, DuckDB
- Icons: Simple database, server rack, laptop database
- Labels: "OLTP (transactions)", "OLTP+OLAP (server required)", "OLAP (analytics)"
- DuckDB highlighted as "Our choice"

**BOTTOM - Speed Demo**:
- Three horizontal bars showing query times: 800ms, 600ms, 100ms
- SQL code snippet in a dark box
- Caption: "DuckDB is 6-8× faster for analytics"

### Refinement Notes
- The CSV→DuckDB transition should feel like "relief from chaos"
- Emphasize "no server required" prominently
- The Excel analogy is important for PI audience
- Show that SQL is approachable, not scary

## Alt Text

Before/after comparison of data storage: Left shows 507 scattered CSV files in a folder (reproducibility nightmare), right shows single DuckDB database file (portable, fast, relational). Middle section compares three databases: SQLite (OLTP, mobile apps), PostgreSQL (needs server), DuckDB (OLAP, analytics). Bottom shows query speed comparison: SQLite 800ms, PostgreSQL 600ms, DuckDB 100ms with example SQL query.

## Technical Notes

### Database Choice Rationale
- DuckDB is columnar (OLAP) vs SQLite row-based (OLTP)
- No external server needed (unlike PostgreSQL)
- Native Polars/Pandas integration
- SQL interface familiar to researchers

### Files in Repository
- Input: `SERI_PLR_GLAUCOMA.db` (raw data consolidated)
- Output: `foundation_plr_results.db` (extracted metrics)

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated (16:10 aspect ratio)
- [ ] Placed in docs/concepts-for-researchers.md, README
