# fig-repo-30: Python-R Interoperability

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-30 |
| **Title** | Python-R Interoperability |
| **Complexity Level** | L2 (Technical) |
| **Target Persona** | ML Engineer, Biostatistician |
| **Location** | CONTRIBUTING.md, docs/development/ |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain why both Python and R are used, and how they communicate via DuckDB and JSON.

## Key Message

"Python handles ML training and extraction. R handles statistical visualization (ggplot2) and pminternal instability analysis. DuckDB is the bridge."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PYTHON-R INTEROPERABILITY                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHY BOTH LANGUAGES?                                                            │
│  ═══════════════════                                                            │
│                                                                                 │
│  Each language excels at different tasks:                                       │
│                                                                                 │
│  ┌──────────────────────────────┐     ┌──────────────────────────────┐         │
│  │         PYTHON               │     │           R                  │         │
│  │   ═══════════════════        │     │   ═══════════════════        │         │
│  │                              │     │                              │         │
│  │   ✅ ML training             │     │   ✅ ggplot2 visualizations  │         │
│  │   ✅ Deep learning           │     │   ✅ pminternal (Riley 2023) │         │
│  │   ✅ Foundation models       │     │   ✅ Statistical analysis    │         │
│  │   ✅ MLflow integration      │     │   ✅ Publication-quality     │         │
│  │   ✅ Data extraction         │     │      figures                 │         │
│  │                              │     │                              │         │
│  │   Used for: Stages 1-4      │     │   Used for: Visualization    │         │
│  │   (outlier→impute→feat→cls) │     │   + instability analysis     │         │
│  │                              │     │                              │         │
│  └──────────────────────────────┘     └──────────────────────────────┘         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE BRIDGE: DuckDB                                                             │
│  ══════════════════                                                             │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   PYTHON                        DuckDB                         R        │   │
│  │   ═══════                    ════════════                    ═══        │   │
│  │                                                                         │   │
│  │   MLflow    ─────────────▶  foundation_plr_   ────────────▶  ggplot2   │   │
│  │   pickles                   results.db                        figures   │   │
│  │                                                                         │   │
│  │   Extract &                 Stores all                       Reads &    │   │
│  │   compute                   metrics +                        visualizes │   │
│  │   STRATOS                   bootstrap                                   │   │
│  │   metrics                   predictions                                 │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Both Python and R can read from DuckDB natively!                               │
│  • Python: import duckdb; conn.execute("SELECT * FROM metrics")                 │
│  • R: library(duckdb); dbGetQuery(conn, "SELECT * FROM metrics")                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FILE FORMATS FOR EXCHANGE                                                      │
│  ═════════════════════════                                                      │
│                                                                                 │
│  │ Format  │ Direction      │ Contents                      │                  │
│  │ ─────── │ ───────────── │ ──────────────────────────── │                   │
│  │ DuckDB  │ Python → R     │ All metrics, bootstrap data  │                  │
│  │ JSON    │ Python → R     │ Figure-specific data         │                  │
│  │ CSV     │ Either way     │ Simple tabular data          │                  │
│  │ Parquet │ Either way     │ Large columnar data          │                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PMINTERNAL: WHY R IS REQUIRED                                                  │
│  ═════════════════════════════                                                  │
│                                                                                 │
│  pminternal (Rhodes 2025, Riley 2023) is an R package for:                      │
│  • Model instability analysis                                                   │
│  • Bootstrap prediction variance                                                │
│  • Calibration instability plots                                                │
│                                                                                 │
│  NO Python equivalent exists! We must use R for this STRATOS-required analysis. │
│                                                                                 │
│  Integration: Python extracts predictions → DuckDB → R pminternal               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CODE EXAMPLE                                                                   │
│  ════════════                                                                   │
│                                                                                 │
│  # Python: Extract to DuckDB                                                    │
│  conn = duckdb.connect("data/foundation_plr_results.db")                        │
│  conn.execute("CREATE TABLE metrics AS SELECT * FROM df")                       │
│                                                                                 │
│  # R: Read and visualize                                                        │
│  library(duckdb)                                                                │
│  conn <- dbConnect(duckdb(), "data/foundation_plr_results.db")                  │
│  df <- dbGetQuery(conn, "SELECT * FROM metrics")                                │
│  ggplot(df, aes(x=method, y=auroc)) + geom_boxplot()                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Language comparison**: What Python vs R excels at
2. **DuckDB bridge diagram**: Data flow Python → DuckDB → R
3. **File format table**: Which format for what purpose
4. **pminternal explanation**: Why R is required
5. **Code examples**: Basic Python and R snippets

## Text Content

### Title Text
"Python-R Interoperability: DuckDB as the Bridge"

### Caption
The pipeline uses Python for ML training and data extraction, and R for statistical visualization (ggplot2) and instability analysis (pminternal). DuckDB serves as the lingua franca—both languages read from the same database. This separation leverages each language's strengths: Python's ML ecosystem and R's statistical rigor.

## Prompts for Nano Banana Pro

### Style Prompt
Two-language comparison with DuckDB bridge in center. Python (blue) and R (orange) columns. Data flow arrows through central database. Code snippets at bottom. Clean, technical documentation aesthetic.

### Content Prompt
Create a Python-R interop diagram:

**TOP - Why Both**:
- Two columns: Python strengths vs R strengths
- Bullet points for each

**MIDDLE - DuckDB Bridge**:
- Flow diagram: Python → DuckDB → R
- Labels showing what each step does

**BOTTOM LEFT - File Formats**:
- Table: Format | Direction | Contents

**BOTTOM RIGHT - pminternal**:
- Note: "R-only package for instability analysis"

## Alt Text

Python-R interoperability diagram. Python handles ML training, deep learning, foundation models, MLflow. R handles ggplot2 visualization, pminternal instability analysis. DuckDB bridges the two: Python extracts MLflow data to DuckDB, R reads from DuckDB for visualization. File formats: DuckDB and JSON for Python→R, CSV and Parquet bidirectional.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in CONTRIBUTING.md
