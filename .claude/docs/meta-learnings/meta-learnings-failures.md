# Meta-Learnings: Failure Analysis

## Index of Documented Failures

| ID | Severity | Summary | Date |
|----|----------|---------|------|
| CRITICAL-FAILURE-001 | CRITICAL | Synthetic data used in scientific figures | 2026-01-25 |
| CRITICAL-FAILURE-002 | CRITICAL | R version mismatch causing reproducibility issues | 2026-01-26 |
| VIOLATION-001 | HIGH | Regex used for Python code parsing (explicit ban violation) | 2026-01-26 |

---

## Root Causes of LaTeX Layout Failures

### 1. Incomplete Verification
**Problem**: Made changes without compiling and visually verifying the PDF output.
**Fix**: Always compile and check output after batch changes.

### 2. Superficial Fixes Instead of Root Cause Analysis
**Problem**: Changed `table` to `table*` but didn't change the inner `tabular` to span full width.
**Fix**: When fixing layout issues, trace the full rendering chain:
- Float environment (`table*`)
- Inner content (`tabularx{\textwidth}`)
- Column specifications

### 3. Inconsistent Application
**Problem**: Fixed some occurrences but not all (e.g., some `\columnwidth` remained).
**Fix**: Use systematic search-and-replace across ALL files, then verify count matches.

### 4. Font Size Oversight for Code/Diagrams
**Problem**: lstlisting and verbatim blocks overflow columns because font is too large.
**Fix**: Define styles with appropriate font sizes upfront:
- `\fontsize{5}{6}\selectfont` for ASCII diagrams
- `\fontsize{6}{7}\selectfont` for code listings

### 5. Two-Column Layout Specific Issues
**Problem**: Didn't account for two-column constraints:
- `tabular` without width spec renders at natural width
- `\textwidth` in table* = full page width
- `\columnwidth` in table (single) = column width
- Code blocks need explicit width constraints

**Fix**: In two-column with `table*`:
- Use `tabularx{\textwidth}` for all tables
- Use `\small` or `\footnotesize` for table content
- Use tiny fonts for code listings

## Systematic Fixes Required

1. ALL `\begin{tabular}` → `\begin{tabularx}{\textwidth}` with appropriate column specs
2. ALL code listings need `basicstyle=\ttfamily\fontsize{5}{6}\selectfont`
3. ALL ASCII diagrams need verbatim with tiny font
4. Verify by compiling after each batch of changes

## Prevention

- Create checklist for two-column LaTeX documents
- Always verify visual output, not just compilation success
- Trace full rendering chain for any layout element
