# fig-repo-15b: Query Optimization: Why Polars is Fast (Expert)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-15b |
| **Title** | Query Optimization: Why Polars is Fast |
| **Complexity Level** | L3 (Expert - Technical deep-dive) |
| **Target Persona** | Data Engineers, Computational Scientists |
| **Location** | docs/development/, ARCHITECTURE.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain the technical reasons why Polars outperforms Pandas—lazy evaluation, query optimization, and multi-threading.

## Key Message

"Polars is fast because it optimizes your query plan before execution, then runs it across all CPU cores."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    QUERY OPTIMIZATION: WHY POLARS IS FAST                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  EAGER vs LAZY EXECUTION                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  PANDAS (Eager)                          POLARS (Lazy)                          │
│  ┌─────────────────┐                     ┌─────────────────┐                    │
│  │ df.filter(...)  │ ← Execute now!      │ df.lazy()       │ ← Build plan     │
│  └────────┬────────┘                     │   .filter(...)  │                    │
│           ▼ Full data scan               │   .groupby(...) │                    │
│  ┌─────────────────┐                     │   .agg(...)     │                    │
│  │ df.groupby(...) │ ← Execute now!      └────────┬────────┘                    │
│  └────────┬────────┘                              │                             │
│           ▼ Another full scan                     ▼                             │
│  ┌─────────────────┐                     ┌─────────────────┐                    │
│  │ df.agg(...)     │ ← Execute now!      │ .collect()      │ ← Execute once!   │
│  └─────────────────┘                     └─────────────────┘                    │
│                                                                                 │
│  3 passes through data                   1 optimized pass                       │
│  (redundant work)                        (fused operations)                     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PARALLELISM                                                                    │
│  ═══════════                                                                    │
│                                                                                 │
│  Pandas (NumPy backend):     Polars (Rust + Apache Arrow):                      │
│  ┌─────────────────────┐     ┌─────────────────────┐                            │
│  │ [█]░░░░░░░░░░░░░░░░ │     │ [█][█][█][█][█][█][█][█] │                       │
│  │ Single-threaded     │     │ Multi-threaded (all cores) │                     │
│  └─────────────────────┘     └─────────────────────┘                            │
│                                                                                 │
│  On 8-core CPU: 1× speed     On 8-core CPU: up to 8× speed                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  MEMORY LAYOUT                                                                  │
│  ═════════════                                                                  │
│                                                                                 │
│  NumPy (Row-oriented):       Apache Arrow (Column-oriented):                    │
│  ┌────────────────────┐      ┌────────────────────┐                             │
│  │ Row 1: [A, B, C]   │      │ Col A: [1,2,3,4,5] │ ← Contiguous in memory     │
│  │ Row 2: [D, E, F]   │      │ Col B: [a,b,c,d,e] │                             │
│  │ Row 3: [G, H, I]   │      │ Col C: [x,y,z,w,v] │                             │
│  └────────────────────┘      └────────────────────┘                             │
│                                                                                 │
│  Aggregates require jumping  Aggregates are cache-friendly                      │
│  across memory (slow)        (fast sequential access)                           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CODE EXAMPLE                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │ # Polars lazy API                                                          ││
│  │ result = (                                                                 ││
│  │     pl.scan_parquet("data.parquet")  # Don't load yet                      ││
│  │     .filter(pl.col("method") == "CatBoost")                                ││
│  │     .groupby("outlier_method")                                             ││
│  │     .agg(pl.col("auroc").mean())                                           ││
│  │     .collect()  # NOW execute optimized plan                               ││
│  │ )                                                                          ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  VERIFIED: 42 source files use Polars in this repository                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Eager vs Lazy flowchart**: 3 passes vs 1 optimized pass
2. **CPU parallelism diagram**: Single-thread vs multi-thread
3. **Memory layout comparison**: Row vs column orientation
4. **Code example**: Lazy API with comments
5. **Technical terms**: Apache Arrow, query optimization, cache-friendly

## Text Content

### Title Text
"Query Optimization: Why Polars is Fast"

### Caption
Polars outperforms Pandas through three mechanisms: (1) Lazy evaluation builds a query plan that's optimized before execution, reducing 3 data passes to 1; (2) Multi-threading uses all CPU cores via Rust's parallelism; (3) Apache Arrow's columnar memory layout makes aggregations cache-friendly. Code uses `.lazy()` to build plans and `.collect()` to execute.

## Prompts for Nano Banana Pro

### Style Prompt
Technical architecture diagram with flowcharts and code blocks. Dark-mode code snippets. CPU core visualization with filled/empty blocks. Memory layout diagrams showing row vs column orientation. Economist-style clean lines. Matte, professional colors. No glowing effects.

### Content Prompt
Create a technical deep-dive infographic:

**SECTION 1 - Eager vs Lazy**:
- Two flowcharts side-by-side
- Pandas: 3 boxes with "Execute now!" labels, arrows down
- Polars: 1 box with "Build plan", then "Execute once!"
- Caption: "3 passes vs 1 optimized pass"

**SECTION 2 - Parallelism**:
- Two CPU diagrams
- Pandas: 1 filled core, 7 empty
- Polars: 8 filled cores
- Caption: "Single-threaded vs all cores"

**SECTION 3 - Memory Layout**:
- Two small grid diagrams
- Row-oriented: rows highlighted
- Column-oriented: columns highlighted (faster for aggregates)

**SECTION 4 - Code**:
- Dark code block with Polars lazy API example
- Comments explaining each step

## Alt Text

Technical explanation of Polars performance. Three mechanisms: (1) Flowchart comparing Pandas eager execution (3 passes) vs Polars lazy execution (1 optimized pass); (2) CPU diagram showing Pandas single-threaded vs Polars multi-threaded across all cores; (3) Memory layout comparing NumPy row-oriented vs Apache Arrow column-oriented storage. Includes code example of Polars lazy API.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/development/
