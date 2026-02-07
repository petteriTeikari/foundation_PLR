# fig-repo-34: README Hierarchy: Finding Documentation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-34 |
| **Title** | README Hierarchy: Finding Documentation |
| **Complexity Level** | L1 (Navigation guide) |
| **Target Persona** | All |
| **Location** | README.md, CONTRIBUTING.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Show where all the documentation lives and which README to read for different needs.

## Key Message

"Multiple READMEs serve different purposes: root for overview, docs/ for guides, configs/ for configuration, src/r/ for R code. Start at root, drill down as needed."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    README HIERARCHY: FINDING DOCUMENTATION                      │
│                    "Which README should I read?"                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  DOCUMENTATION MAP                                                              │
│  ════════════════                                                               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │                         📄 README.md (root)                             │   │
│  │                         ════════════════════                            │   │
│  │                         Start here! Project overview,                   │   │
│  │                         quickstart, key findings                        │   │
│  │                                    │                                    │   │
│  │              ┌────────────────────┼────────────────────┐               │   │
│  │              │                    │                    │               │   │
│  │              ▼                    ▼                    ▼               │   │
│  │      📁 docs/              📁 configs/           📁 src/              │   │
│  │      ══════════            ═══════════           ═══════              │   │
│  │                                                                        │   │
│  │   ├── getting-started/     ├── README.md          ├── r/              │   │
│  │   │   └── README.md        │   Hydra config       │   └── README.md   │   │
│  │   │       ↳ Installation   │   structure          │       ↳ R figure  │   │
│  │   │                        │                      │         scripts   │   │
│  │   ├── user-guide/          ├── VISUALIZATION/     │                   │   │
│  │   │   └── README.md        │   └── README.md      ├── viz/            │   │
│  │   │       ↳ Running        │       ↳ Figure       │   └── README.md   │   │
│  │   │         experiments    │         registry     │       ↳ Python    │   │
│  │   │                        │                      │         plots     │   │
│  │   ├── concepts/            └── mlflow_registry/   │                   │   │
│  │   │   └── README.md            └── README.md      └── data_io/        │   │
│  │   │       ↳ STRATOS,               ↳ Method           └── README.md   │   │
│  │   │         bootstrap,               names               ↳ Registry   │   │
│  │   │         calibration                                    module    │   │
│  │   │                                                                    │   │
│  │   └── tutorials/                                                       │   │
│  │       └── README.md                                                    │   │
│  │           ↳ End-to-end                                                 │   │
│  │             workflows                                                  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHICH README FOR WHICH QUESTION?                                               │
│  ════════════════════════════════                                               │
│                                                                                 │
│  │ Question                              │ Read this                          │ │
│  │ ───────────────────────────────────── │ ────────────────────────────────── │ │
│  │ "What is this project?"               │ README.md (root)                   │ │
│  │ "How do I install?"                   │ docs/getting-started/README.md     │ │
│  │ "How do I run experiments?"           │ docs/user-guide/README.md          │ │
│  │ "What is STRATOS? Bootstrap?"         │ docs/concepts/README.md            │ │
│  │ "How are configs structured?"         │ configs/README.md                  │ │
│  │ "What methods are valid?"             │ configs/mlflow_registry/README.md  │ │
│  │ "How do R figures work?"              │ src/r/README.md                    │ │
│  │ "How do I create a figure?"           │ configs/VISUALIZATION/README.md   │ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  OTHER KEY DOCUMENTATION                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  📄 ARCHITECTURE.md     Technical architecture, pipeline stages                 │
│  📄 CONTRIBUTING.md     Development workflow, code standards                    │
│  📄 CLAUDE.md           AI assistant instructions, rules                        │
│  📄 .claude/CLAUDE.md   Behavior contract, figure rules                         │
│  📄 CITATION.cff        How to cite this work                                   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  RECOMMENDED READING PATH                                                       │
│  ════════════════════════                                                       │
│                                                                                 │
│  New user:                                                                      │
│  README.md → docs/getting-started/ → docs/user-guide/ → docs/concepts/          │
│                                                                                 │
│  New contributor:                                                               │
│  README.md → ARCHITECTURE.md → CONTRIBUTING.md → .claude/CLAUDE.md              │
│                                                                                 │
│  Figure creator:                                                                │
│  configs/VISUALIZATION/README.md → src/r/README.md → src/viz/README.md          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Documentation tree**: Visual hierarchy of READMEs
2. **Question/Answer table**: Which README for which question
3. **Other docs list**: ARCHITECTURE, CONTRIBUTING, CLAUDE, CITATION
4. **Reading paths**: Recommended sequences for different roles

## Text Content

### Title Text
"README Hierarchy: Your Map to Documentation"

### Caption
Documentation is distributed across multiple READMEs, each serving a specific purpose. Start at root README.md for project overview, then drill down: docs/getting-started/ for installation, docs/concepts/ for STRATOS and bootstrap explanations, configs/README.md for Hydra configuration, configs/mlflow_registry/ for valid method names. Contributors should also read ARCHITECTURE.md and CONTRIBUTING.md.

## Prompts for Nano Banana Pro

### Style Prompt
Documentation tree diagram with folder icons. Table mapping questions to READMEs. Reading paths as horizontal arrows. Clean, navigational aesthetic with clear hierarchy.

### Content Prompt
Create a README hierarchy diagram:

**TOP - Tree**:
- Root README.md at top
- Branches down to docs/, configs/, src/
- Sub-branches with descriptions

**MIDDLE - Question Table**:
- Two columns: Question | Read this
- 8 common questions mapped to READMEs

**BOTTOM - Reading Paths**:
- Three paths: New user, Contributor, Figure creator
- Arrow sequences showing order

## Alt Text

README hierarchy diagram showing documentation structure. Root README for project overview branches to docs/ (getting-started, user-guide, concepts, tutorials), configs/ (Hydra config, VISUALIZATION, mlflow_registry), and src/ (r, viz, data_io). Question table maps common questions to specific READMEs. Three reading paths: new users (overview → install → guide → concepts), contributors (overview → architecture → contributing → claude), figure creators (visualization → r → viz).

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md
