# fig-repo-16a: One Database Instead of 500 Files (ELI5)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-16a |
| **Title** | One Database Instead of 500 Files |
| **Complexity Level** | L0 (ELI5 - Concept only) |
| **Target Persona** | PI, Clinician, Non-technical |
| **Location** | Root README, docs/concepts-for-researchers.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Show the practical benefit of consolidating 507 scattered CSV files into one DuckDB database—NO SQL, NO technical details.

## Key Message

"All your data in one place. Like moving from a messy filing cabinet to a single organized folder."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ONE DATABASE INSTEAD OF 500 FILES                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ❌ BEFORE                                 ✅ AFTER                             │
│  ════════                                  ═══════                              │
│                                                                                 │
│  📁 /data/                                 🗄️ SERI_PLR_GLAUCOMA.db             │
│  ├── PLR0001.csv                                                                │
│  ├── PLR0002.csv                           ┌────────────────────────┐           │
│  ├── PLR0003.csv                           │                        │           │
│  ├── PLR0004.csv                           │   507 subjects         │           │
│  ├── PLR0005.csv                           │   1 MILLION+ data      │           │
│  ├── ...                                   │   points               │           │
│  ├── ...                                   │                        │           │
│  ├── ...                                   │   One single file!     │           │
│  └── PLR0507.csv                           │                        │           │
│                                            └────────────────────────┘           │
│  507 separate files!                                                            │
│  Scattered, hard to search                 Organized, fast, portable            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ANALOGY                                                                        │
│  ═══════                                                                        │
│                                                                                 │
│  📂 Filing Cabinet (CSV files)            📁 Digital Folder (DuckDB)           │
│                                                                                 │
│  • Papers scattered in drawers            • Everything searchable               │
│  • Can't search across files              • Instant answers                     │
│  • Slow to find anything                  • One file to backup                  │
│  • Hard to share                          • Easy to share                       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ✨ BONUS: No software installation needed!                                     │
│     Works like a regular file. Share it via email, USB, or cloud.              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements (MAX 5 CONCEPTS)

1. **Before/After visual**: Scattered files vs single database
2. **File count**: 507 files → 1 file
3. **Data scale**: 1 million+ data points
4. **Analogy**: Filing cabinet vs digital folder
5. **Portability benefit**: Share via email/USB/cloud

## Text Content

### Title Text
"One Database Instead of 500 Files"

### Labels/Annotations
- NO SQL queries
- NO "OLAP" or "columnar" terminology
- Simple icons: folders, files, database cylinder

### Caption
Instead of managing 507 separate CSV files, all our data lives in one DuckDB file. It's like moving from a messy filing cabinet to a single organized folder—everything is searchable and fast. The file is portable: share it via email, USB, or cloud storage.

## Prompts for Nano Banana Pro

### Style Prompt
Simple before/after comparison for non-technical audience. Messy folder icon with papers spilling out on left. Clean, organized database icon on right. Filing cabinet analogy with real-world objects. Friendly, reassuring design. NO code, NO technical terms. Green checkmarks for benefits. Medical research context.

### Content Prompt
Create a before/after comparison:

**LEFT (Before - red/gray tint)**:
- Messy folder icon with 507 small file icons spilling out
- Label: "507 separate CSV files"
- Sad/stressed icon
- List: "Scattered, hard to search, slow"

**RIGHT (After - green/blue tint)**:
- Clean database cylinder icon
- Label: "1 file, 1 million+ data points"
- Happy/relieved icon
- List: "Organized, fast, portable"

**MIDDLE - Filing Cabinet Analogy**:
- Physical filing cabinet (left) → Digital folder (right)
- Simple bullet points comparing benefits

**BOTTOM**:
- Star icon: "No software installation needed!"
- Share icons: email, USB, cloud

NO SQL, NO technical jargon.

## Alt Text

Before/after comparison: Left shows 507 scattered CSV files in a messy folder. Right shows one clean DuckDB database containing 1 million+ data points. Analogy compares filing cabinet (hard to search, slow) to digital folder (organized, fast, portable). Note that no software installation is needed.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README
