# Software Concepts for Researchers

> **For:** Ophthalmology PIs, biostatisticians, research scientists, and clinical collaborators who may not have a software engineering background.

This page explains the software tools used in this project using familiar analogies.

## Quick Reference Table

| Tool | What It Does | Think of It Like... |
|------|--------------|---------------------|
| **MLflow** | Tracks experiments | Lab notebook that never forgets |
| **Hydra** | Manages settings | Smart Excel settings sheet |
| **DuckDB** | Stores data | Excel on steroids |
| **pytest** | Checks code works | Automated reviewer |
| **pre-commit** | Catches mistakes | Spell-checker for code |
| **Git** | Tracks changes | "Track Changes" in Word |
| **uv** | Installs packages | App store for Python |

---

## MLflow: Your Digital Lab Notebook

### What It Does

MLflow is like a **laboratory notebook that automatically records everything** about each experiment you run.

### Why Researchers Need It

Imagine running 500+ experiments with different preprocessing methods. Without MLflow, you'd need to:
- Manually record every parameter
- Remember which CSV file has which results
- Hope you didn't accidentally overwrite something

**With MLflow:**
- Every experiment is automatically logged
- Parameters, results, and even model files are stored
- You can compare experiments side-by-side
- Results are searchable and reproducible

### Analogy

> Think of MLflow as a **research assistant** who sits next to you and writes down:
> - What settings you used
> - When you ran the experiment
> - What results you got
> - Where the output files are
>
> ...for every single run, automatically.

---

## Hydra: Smart Configuration

### What It Does

Hydra manages **configuration files** (settings) for experiments.

### Why Researchers Need It

In a large experiment, you might have 50+ parameters:
- Bootstrap iterations
- Disease prevalence
- Model hyperparameters
- File paths

Without Hydra, these would be scattered across your code. Changing one setting means hunting through multiple files.

**With Hydra:**
- All settings in one place (`configs/defaults.yaml`)
- Override any setting from command line
- No code changes needed to try new parameters

### Analogy

> Think of Hydra as a **settings spreadsheet** where:
> - Each row is a parameter
> - You can change values without touching the analysis code
> - Different "sheets" for different experiment types

### Example

```bash
# Run with default settings
python src/pipeline_PLR.py

# Change bootstrap iterations (no code change!)
python src/pipeline_PLR.py CLS_EVALUATION.BOOTSTRAP.n_iterations=500
```

---

## DuckDB: Excel on Steroids

### What It Does

DuckDB is a **database** that stores and queries data, similar to Excel but much faster and more powerful.

### Why Researchers Need It

Excel struggles with:
- Large datasets (>100,000 rows)
- Complex queries across multiple tables
- Reproducible analysis

**DuckDB:**
- Handles millions of rows easily
- SQL queries are reproducible scripts
- No "accidentally sorted wrong column" disasters

### Analogy

> Imagine Excel, but:
> - It can handle your entire dataset, not just a sample
> - Queries are saved as text (reproducible)
> - Multiple people can use it simultaneously
> - It fits in a single file (like `.xlsx`)

### Example Query

```sql
-- Get average AUROC by preprocessing method
SELECT outlier_method, AVG(auroc) as mean_auroc
FROM results
GROUP BY outlier_method
ORDER BY mean_auroc DESC;
```

---

## pytest: Automated Code Reviewer

### What It Does

pytest **automatically checks that code works correctly**.

### Why Researchers Need It

When you modify code:
- Did you accidentally break something else?
- Does the new feature actually work?
- Will it still work next month?

**pytest:**
- Runs hundreds of checks in seconds
- Catches bugs before they reach your results
- Gives confidence that changes are safe

### Analogy

> Think of pytest as a **thorough reviewer** who:
> - Checks every calculation
> - Tests edge cases
> - Never gets tired or forgets
> - Runs in 30 seconds instead of 30 minutes

### Example

```bash
# Run all tests
uv run pytest tests/ -v

# Output:
# ✅ test_auroc_range_valid ... PASSED
# ✅ test_no_missing_subjects ... PASSED
# ❌ test_method_count ... FAILED (expected 11, got 17)
```

---

## pre-commit: Spell-Checker for Code

### What It Does

pre-commit runs **automatic checks before you save your work**.

### Why Researchers Need It

Common mistakes:
- Inconsistent formatting
- Hardcoded values that should be in config
- Forgetting to update documentation

**pre-commit:**
- Catches these automatically
- Won't let you save until fixed
- Keeps code quality consistent

### Analogy

> Like how Word underlines spelling mistakes in red, pre-commit highlights code problems before they become permanent.

---

## Git: Track Changes for Code

### What It Does

Git tracks **every change ever made** to the code.

### Why Researchers Need It

- See what changed between versions
- Undo mistakes safely
- Collaborate without overwriting each other's work
- Prove exactly what code produced published results

### Analogy

> Git is like "Track Changes" in Word, but:
> - For the entire project, not just one document
> - With an unlimited undo history
> - That multiple authors can use simultaneously

---

## uv: Fast Package Manager

### What It Does

uv **installs Python packages** (libraries of code).

### Why Researchers Need It

Python needs additional libraries (pandas, numpy, etc.). uv:
- Installs them quickly (10-100x faster than alternatives)
- Ensures everyone has the same versions
- Avoids "it works on my computer" problems

### Analogy

> Think of uv as an **app store for Python**:
> - Search for what you need
> - Install with one command
> - Automatic updates and compatibility checks

---

## Why This Matters for Your Research

These tools ensure:

| Goal | How Tools Help |
|------|----------------|
| **Reproducibility** | MLflow logs everything, Git tracks changes |
| **Reliability** | pytest catches bugs, pre-commit enforces standards |
| **Efficiency** | Hydra enables easy experiments, DuckDB handles big data |
| **Collaboration** | Everyone uses same tools, same settings |

### The Alternative

Without these tools, researchers often:
- Lose track of which settings produced which results
- Accidentally break working code
- Spend hours debugging "Excel errors"
- Struggle to reproduce results from 6 months ago

---

## Getting Help

If you're a clinical collaborator or PI and need help:

1. **Ask about the data** - We can export results to Excel/CSV for your analysis
2. **Ask about figures** - We can regenerate any figure from the paper
3. **Ask about methods** - The code documents exactly what was done

You don't need to learn these tools to benefit from the research - but understanding them helps you ask better questions about reproducibility.
