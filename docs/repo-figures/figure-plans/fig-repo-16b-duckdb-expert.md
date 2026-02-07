# fig-repo-16b: DuckDB vs SQLite vs PostgreSQL (Expert)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-16b |
| **Title** | DuckDB vs SQLite vs PostgreSQL: Why OLAP? |
| **Complexity Level** | L3 (Expert - Technical deep-dive) |
| **Target Persona** | Biostatisticians, Data Scientists |
| **Location** | docs/development/, ARCHITECTURE.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain the technical reasons for choosing DuckDB—OLAP optimization, columnar storage, and query performance.

## Key Message

"DuckDB is optimized for analytical queries (OLAP), not transactions (OLTP). Aggregates run 6-8× faster than SQLite."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DUCKDB vs SQLITE vs POSTGRESQL: WHY OLAP?                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  DATABASE COMPARISON                                                            │
│  ═══════════════════                                                            │
│                                                                                 │
│  ┌─────────────────┬─────────────────┬─────────────────┐                        │
│  │     SQLite      │   PostgreSQL    │     DuckDB      │                        │
│  ├─────────────────┼─────────────────┼─────────────────┤                        │
│  │ Type: OLTP      │ Type: OLTP/OLAP │ Type: OLAP      │ ← Analytics focused!  │
│  │ Storage: Row    │ Storage: Row    │ Storage: Column │ ← Columnar!           │
│  │ Setup: File     │ Setup: Server   │ Setup: File     │ ← No server!          │
│  │ Best: Mobile    │ Best: Web apps  │ Best: Analytics │                        │
│  └─────────────────┴─────────────────┴─────────────────┘                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  OLTP vs OLAP                                                                   │
│  ════════════                                                                   │
│                                                                                 │
│  OLTP (Online Transaction Processing)     OLAP (Online Analytical Processing)   │
│  ─────────────────────────────────────    ────────────────────────────────────  │
│  "Get me user #12345's order"             "Average AUROC grouped by method"     │
│  Single row lookups                       Aggregate across millions of rows     │
│  Frequent writes                          Mostly reads                          │
│  Row-oriented storage (fast inserts)      Column-oriented (fast aggregates)     │
│                                                                                 │
│  SQLite, PostgreSQL optimized for ↑        DuckDB optimized for ↑               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  QUERY PERFORMANCE                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  Query: "SELECT AVG(auroc) FROM metrics GROUP BY outlier_method"                │
│                                                                                 │
│  SQLite:      ████████████████████████████████████░░░░░░░  800ms                │
│  PostgreSQL:  ██████████████████████████████░░░░░░░░░░░░░  600ms (server!)     │
│  DuckDB:      ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  100ms ✓             │
│                                                                                 │
│               6-8× faster for analytical queries!                               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COLUMNAR STORAGE                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  Row Storage (SQLite):              Column Storage (DuckDB):                    │
│  ┌─────────────────────────┐        ┌─────────────────────────┐                 │
│  │ Row1: [id, method, auroc]│        │ id:     [1,2,3,4,5...]  │                 │
│  │ Row2: [id, method, auroc]│        │ method: [A,B,A,C,B...]  │                 │
│  │ Row3: [id, method, auroc]│        │ auroc:  [0.9,0.8,0.9...]│ ← Contiguous!  │
│  └─────────────────────────┘        └─────────────────────────┘                 │
│                                                                                 │
│  To compute AVG(auroc):             To compute AVG(auroc):                      │
│  Jump across memory for each row    Sequential read of auroc column            │
│  Cache misses, slow                 Cache-friendly, fast                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SQL EXAMPLE                                                                    │
│  ═══════════                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │ -- Top preprocessing methods by AUROC                                      ││
│  │ SELECT                                                                     ││
│  │     outlier_method,                                                        ││
│  │     imputation_method,                                                     ││
│  │     AVG(auroc) as mean_auroc,                                              ││
│  │     COUNT(*) as n_runs                                                     ││
│  │ FROM essential_metrics                                                     ││
│  │ WHERE classifier = 'CatBoost'                                              ││
│  │ GROUP BY outlier_method, imputation_method                                 ││
│  │ ORDER BY mean_auroc DESC                                                   ││
│  │ LIMIT 10;                                                                  ││
│  │                                                                            ││
│  │ -- Executes in ~100ms on our 410+ runs dataset                             ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Three-way comparison table**: SQLite vs PostgreSQL vs DuckDB
2. **OLTP vs OLAP explanation**: Transaction vs analytical workloads
3. **Query performance bars**: 800ms vs 600ms vs 100ms
4. **Columnar storage diagram**: Row vs column orientation
5. **SQL example**: Real query from our codebase

## Text Content

### Title Text
"DuckDB vs SQLite vs PostgreSQL: Why OLAP?"

### Caption
DuckDB is an OLAP database optimized for analytical queries, not transactions. Columnar storage makes aggregates (AVG, COUNT, GROUP BY) 6-8× faster than row-oriented SQLite. No server setup required—it's a single file like SQLite but with analytics performance approaching PostgreSQL.

## Prompts for Nano Banana Pro

### Style Prompt
Technical database comparison with tables, flowcharts, and code blocks. Three-column comparison table for databases. OLTP vs OLAP side-by-side explanation. Performance bars with exact millisecond values. Storage diagram showing row vs column layout. Dark-mode SQL code block. Economist-style clean data visualization.

### Content Prompt
Create a technical database comparison:

**SECTION 1 - Comparison Table**:
- Three columns: SQLite, PostgreSQL, DuckDB
- Rows: Type (OLTP/OLAP), Storage (Row/Column), Setup (File/Server), Best Use

**SECTION 2 - OLTP vs OLAP**:
- Two boxes with definitions
- Example queries for each
- Arrows pointing to which databases optimize for which

**SECTION 3 - Performance**:
- Three horizontal bars: SQLite 800ms, PostgreSQL 600ms, DuckDB 100ms
- Caption: "6-8× faster for analytical queries"

**SECTION 4 - Columnar Storage**:
- Two mini-diagrams showing row vs column memory layout
- Highlight why column is faster for aggregates

**SECTION 5 - SQL Example**:
- Dark code block with GROUP BY query
- Comment showing execution time

## Alt Text

Technical comparison of DuckDB vs SQLite vs PostgreSQL. Table shows DuckDB is OLAP-optimized with columnar storage and file-based setup. OLTP vs OLAP explanation with example queries. Performance bars: SQLite 800ms, PostgreSQL 600ms, DuckDB 100ms (6-8× faster). Diagram shows row vs columnar storage and why columnar is faster for aggregates. SQL example query.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/development/
