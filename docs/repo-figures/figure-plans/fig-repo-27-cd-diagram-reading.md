# fig-repo-27: How to Read Critical Difference Diagrams

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-27 |
| **Title** | How to Read Critical Difference Diagrams |
| **Complexity Level** | L2 (Statistical concept) |
| **Target Persona** | All |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Teach readers how to interpret Critical Difference (CD) diagrams—a key visualization used to compare preprocessing methods.

## Key Message

"CD diagrams show which methods are statistically equivalent (connected by horizontal bars) and which are significantly different. Lower rank = better performance."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    HOW TO READ CRITICAL DIFFERENCE DIAGRAMS                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS A CD DIAGRAM?                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  A visualization for comparing multiple methods across multiple datasets/folds  │
│  Based on: Friedman test + Nemenyi post-hoc test                                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │     CD = 1.24                                                           │   │
│  │     ┌──────────────────────────────────────────────────────┐            │   │
│  │     1         2         3         4         5         6                 │   │
│  │     ├─────────┼─────────┼─────────┼─────────┼─────────┤                 │   │
│  │     │                                                                   │   │
│  │     │  ┌─────────────────────┐                                          │   │
│  │     ├──┤   Method A ─────────┼───── 1.5                                 │   │
│  │     │  └─────────────────────┘                                          │   │
│  │     │  ┌─────────────────────┐                                          │   │
│  │     ├──┤   Method B ─────────┼───── 2.1        ← Connected = NOT        │   │
│  │     │  └─────────────────────┘                   significantly different│   │
│  │     │                                                                   │   │
│  │     │                    ┌─────────────────────┐                        │   │
│  │     │                    │   Method C ─────────┼───── 4.2               │   │
│  │     │                    └─────────────────────┘                        │   │
│  │     │                              ← Gap = significantly WORSE          │   │
│  │     │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  READING THE DIAGRAM                                                            │
│  ═══════════════════                                                            │
│                                                                                 │
│  1. RANK AXIS (top)                                                             │
│     ───────────────                                                             │
│     • Numbers 1-6 show average ranks across folds                               │
│     • LOWER rank = BETTER performance                                           │
│     • Rank 1 = best, Rank 6 = worst                                             │
│                                                                                 │
│  2. METHOD MARKERS                                                              │
│     ───────────────                                                             │
│     • Each method has a marker at its average rank                              │
│     • Marker position shows relative performance                                │
│                                                                                 │
│  3. HORIZONTAL BARS                                                             │
│     ────────────────                                                            │
│     • Connect methods that are NOT significantly different                      │
│     • Methods connected by a bar = statistically equivalent                     │
│     • Methods NOT connected = significantly different (p < 0.05)                │
│                                                                                 │
│  4. CRITICAL DIFFERENCE (CD)                                                    │
│     ───────────────────────                                                     │
│     • The bar at top shows CD width                                             │
│     • Methods within CD of each other are equivalent                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXAMPLE INTERPRETATION                                                         │
│  ══════════════════════                                                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │     1         2         3         4         5         6                 │   │
│  │     ├─────────┼─────────┼─────────┼─────────┼─────────┤                 │   │
│  │     │                                                                   │   │
│  │     ├─●── pupil-gt (1.2)                                                │   │
│  │     │  └───────────────┐                                                │   │
│  │     │      ●── ensemble (1.8)                                           │   │
│  │     │      └───────────┼─┐                                              │   │
│  │     │          ●── MOMENT (2.4)                                         │   │
│  │     │                  └─┘                                              │   │
│  │     │                                                                   │   │
│  │     │                              ●── LOF (4.5)                        │   │
│  │     │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Interpretation:                                                                │
│  • pupil-gt, ensemble, and MOMENT are statistically equivalent (connected)      │
│  • LOF is significantly worse (not connected to the others)                     │
│  • Ground truth is best (rank 1.2) but ensemble nearly ties (rank 1.8)          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON MISTAKES                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  ❌ "Rank 1.5 is clearly better than rank 1.8"                                  │
│     → Check if they're connected! If yes, difference is NOT significant         │
│                                                                                 │
│  ❌ "Method with lowest rank is always best"                                    │
│     → True for rank, but may not be clinically meaningful                       │
│                                                                                 │
│  ❌ "All methods on left side are equivalent"                                   │
│     → Only methods connected by bars are equivalent                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **CD diagram anatomy**: Labeled example showing axis, markers, bars
2. **Reading guide**: 4 numbered steps to interpret the diagram
3. **Example interpretation**: Real example with conclusions
4. **Common mistakes**: What NOT to conclude

## Text Content

### Title Text
"How to Read Critical Difference Diagrams"

### Caption
Critical Difference (CD) diagrams compare multiple methods using Friedman tests and Nemenyi post-hoc analysis. The horizontal axis shows average ranks (lower = better). Methods connected by horizontal bars are NOT significantly different from each other (p > 0.05). Methods without a connecting bar are significantly different. The CD bar at top shows the minimum rank difference needed for statistical significance.

## Prompts for Nano Banana Pro

### Style Prompt
Educational diagram explaining CD plot interpretation. Annotated example with callouts pointing to key elements. Numbered reading guide. Example interpretation panel. Common mistakes with X marks. Clean, instructional aesthetic.

### Content Prompt
Create a CD diagram reading guide:

**TOP - Anatomy**:
- CD diagram with labeled parts: rank axis, method markers, connecting bars, CD bar
- Callouts explaining each element

**MIDDLE - Reading Steps**:
- 4 numbered steps with icons
- 1: Rank axis (lower = better)
- 2: Method markers
- 3: Connecting bars (= equivalent)
- 4: CD threshold

**BOTTOM - Example**:
- Small CD diagram showing 4 methods
- Interpretation text below

**SIDEBAR - Mistakes**:
- 3 common misinterpretations with X marks

## Alt Text

Educational diagram explaining Critical Difference plot interpretation. Shows CD diagram anatomy with labeled rank axis (1-6, lower is better), method markers at average rank positions, and horizontal bars connecting statistically equivalent methods. Reading guide: (1) rank axis, (2) markers, (3) connecting bars mean no significant difference, (4) CD bar shows threshold. Example shows pupil-gt, ensemble, MOMENT connected (equivalent) while LOF is separate (significantly worse).

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md
