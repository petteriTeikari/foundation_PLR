# fig-repo-32: Virtual Environments Explained (.venv)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-32 |
| **Title** | Virtual Environments Explained (.venv) |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | Biologist, First-time Python user |
| **Location** | docs/getting-started/, README |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain what .venv is and why we isolate Python packages per project—for users who've never used virtual environments.

## Key Message

"A .venv is like a separate toolbox for each project. Your system Python stays clean, and this project gets exactly the packages it needs."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VIRTUAL ENVIRONMENTS EXPLAINED                               │
│                    Why does this project have a .venv folder?                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE ANALOGY: SEPARATE TOOLBOXES                                                │
│  ═══════════════════════════════                                                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   SYSTEM PYTHON                          PROJECT .venv                  │   │
│  │   (Your computer's Python)               (This project's Python)        │   │
│  │                                                                         │   │
│  │   ┌───────────────────┐                  ┌───────────────────┐         │   │
│  │   │  🧰 Main Toolbox   │                  │  🧰 Project Toolbox │         │   │
│  │   │                   │                  │                   │         │   │
│  │   │  🔨 Python 3.11   │                  │  🔨 Python 3.11   │         │   │
│  │   │  🔧 basic tools   │                  │  📊 pandas 2.1    │         │   │
│  │   │                   │                  │  🦆 duckdb 0.9    │         │   │
│  │   │                   │                  │  📈 polars 0.19   │         │   │
│  │   │                   │                  │  🔬 scikit-learn  │         │   │
│  │   │                   │                  │  ...200 more...   │         │   │
│  │   └───────────────────┘                  └───────────────────┘         │   │
│  │                                                                         │   │
│  │   Shared across ALL                      ONLY for foundation_PLR        │   │
│  │   your Python projects                                                  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY SEPARATE TOOLBOXES?                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  ❌ WITHOUT virtual environment:                                                │
│                                                                                 │
│  Project A needs pandas 1.5  ─┬─▶  CONFLICT!  ◀─┬─  Project B needs pandas 2.1 │
│                               │     💥          │                               │
│                               └────────────────┘                               │
│                                                                                 │
│  Both install to same place, one breaks the other.                              │
│                                                                                 │
│  ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│  ✅ WITH virtual environments:                                                  │
│                                                                                 │
│  Project A's .venv ─▶ pandas 1.5  ✓                                             │
│                                      No conflict! Different folders.            │
│  Project B's .venv ─▶ pandas 2.1  ✓                                             │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHAT'S IN .venv?                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  foundation_PLR/                                                                │
│  ├── src/                    ← Your code                                        │
│  ├── .venv/                  ← Virtual environment (DON'T EDIT!)                │
│  │   ├── bin/                ← Python executable                                │
│  │   │   ├── python          ← The Python for this project                      │
│  │   │   └── pip             ← Package installer                                │
│  │   └── lib/                ← Installed packages                               │
│  │       └── python3.11/                                                        │
│  │           └── site-packages/                                                 │
│  │               ├── pandas/                                                    │
│  │               ├── duckdb/                                                    │
│  │               └── ...                                                        │
│  ├── pyproject.toml          ← List of required packages                        │
│  └── uv.lock                 ← Exact versions locked                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  HOW TO USE IT                                                                  │
│  ═════════════                                                                  │
│                                                                                 │
│  # First time: Create .venv and install packages                                │
│  uv sync                                                                        │
│                                                                                 │
│  # Before working: Activate the environment                                     │
│  source .venv/bin/activate                                                      │
│                                                                                 │
│  # Your prompt changes to show you're in the venv                               │
│  (foundation_PLR) $ python script.py   ← Uses project's Python                  │
│                                                                                 │
│  # When done: Deactivate                                                        │
│  deactivate                                                                     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KEY POINTS                                                                     │
│  ══════════                                                                     │
│                                                                                 │
│  ✓ .venv is created automatically by `uv sync`                                  │
│  ✓ .venv is in .gitignore (not shared, recreated from uv.lock)                  │
│  ✓ Every developer gets identical packages via uv.lock                          │
│  ✓ You can safely delete .venv and recreate with `uv sync`                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Toolbox analogy**: System Python vs Project .venv
2. **Why separate**: Version conflict diagram without vs with venvs
3. **Folder structure**: What's inside .venv/
4. **Usage commands**: uv sync, source activate, deactivate
5. **Key points**: Recreatable, gitignored, locked versions

## Text Content

### Title Text
"Virtual Environments: A Separate Toolbox for Each Project"

### Caption
A virtual environment (.venv) isolates this project's Python packages from your system Python and other projects. This prevents version conflicts—if Project A needs pandas 1.5 and Project B needs pandas 2.1, virtual environments keep them separate. The .venv folder is created by `uv sync` and contains all 200+ packages this project needs. It's gitignored because any developer can recreate it from the locked uv.lock file.

## Prompts for Nano Banana Pro

### Style Prompt
ELI5 toolbox analogy diagram. Two toolboxes side by side: system (sparse) and project (full). Version conflict illustration showing broken vs working. Folder tree showing .venv contents. Command examples at bottom. Friendly, accessible aesthetic.

### Content Prompt
Create a virtual environment explanation:

**TOP - Toolbox Analogy**:
- Two toolbox illustrations: "System Python" (mostly empty) vs "Project .venv" (full of packages)
- Icons representing packages inside

**MIDDLE - Why Separate**:
- Conflict diagram: Two projects fighting over pandas
- Solution: Separate folders, no conflict

**BOTTOM LEFT - Folder Structure**:
- Tree showing .venv/bin/python, lib/site-packages/

**BOTTOM RIGHT - Commands**:
- uv sync (create)
- source .venv/bin/activate (use)
- deactivate (exit)

## Alt Text

Virtual environment explanation using toolbox analogy. System Python is a sparse main toolbox. Project .venv is a separate full toolbox with pandas, duckdb, polars, etc. Without venvs, projects conflict over package versions. With venvs, each project has isolated packages. Folder structure shows .venv containing bin/python and lib/site-packages. Commands: uv sync to create, source activate to use, deactivate to exit.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/getting-started/
