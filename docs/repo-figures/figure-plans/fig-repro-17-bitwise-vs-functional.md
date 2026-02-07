# fig-repro-17: Bitwise vs Functional Reproducibility

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-17 |
| **Title** | Bitwise vs Functional Reproducibility |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | ML Engineer, Biostatistician |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Distinguish between bitwise reproducibility (SHA-256 matches) and functional reproducibility (same conclusions), explaining when each matters.

## Key Message

"Bitwise = exact bytes match (hard, sometimes unnecessary). Functional = same scientific conclusions (usually sufficient). Know which you need and which you're achieving."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    BITWISE VS FUNCTIONAL REPRODUCIBILITY                        │
│                    Two standards, different requirements                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  DEFINITIONS                                                                    │
│  ═══════════                                                                    │
│                                                                                 │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐      │
│  │     BITWISE REPRODUCIBILITY     │  │   FUNCTIONAL REPRODUCIBILITY    │      │
│  │                                 │  │                                 │      │
│  │  SHA-256(output_A) ==           │  │  conclusions_A ==               │      │
│  │  SHA-256(output_B)              │  │  conclusions_B                  │      │
│  │                                 │  │                                 │      │
│  │  Every byte is identical        │  │  Results are scientifically     │      │
│  │                                 │  │  equivalent                     │      │
│  │                                 │  │                                 │      │
│  │  Example: Same PNG file         │  │  Example: AUROC 0.9110 vs       │      │
│  │  to the last pixel              │  │  0.9111 (within CI)             │      │
│  │                                 │  │                                 │      │
│  └─────────────────────────────────┘  └─────────────────────────────────┘      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY BITWISE IS HARD                                                            │
│  ═══════════════════                                                            │
│                                                                                 │
│  Sources of non-determinism (even with same inputs):                            │
│                                                                                 │
│  │ Source                  │ Example                      │ Impact        │    │
│  │ ─────────────────────── │ ──────────────────────────── │ ───────────── │    │
│  │ Floating-point order    │ sum([1e-15, 1e15, -1e15])    │ Different sum │    │
│  │ Thread scheduling       │ Parallel operations          │ Order varies  │    │
│  │ Timestamps              │ File creation time           │ Metadata      │    │
│  │ Memory addresses        │ Pointer values in output     │ Different IDs │    │
│  │ GPU non-determinism     │ cuDNN convolutions           │ ±1e-6         │    │
│  │ Dictionary ordering     │ Python <3.7 dict iteration   │ Key order     │    │
│                                                                                 │
│  Getting bitwise reproducibility requires controlling ALL of these!             │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHEN EACH MATTERS                                                              │
│  ══════════════════                                                             │
│                                                                                 │
│  BITWISE REQUIRED:                     FUNCTIONAL SUFFICIENT:                   │
│  ─────────────────                     ─────────────────────                    │
│                                                                                 │
│  • Cryptographic applications          • Scientific conclusions                 │
│  • Audit compliance                    • Machine learning models                │
│  • Legal evidence                      • Statistical analyses                   │
│  • Package distribution hashes         • Figure generation                      │
│                                                                                 │
│  Effort: Very high                     Effort: Moderate                         │
│  Achievable: <1% of projects           Achievable: ~60% with good practices     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR TARGET                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  Primary goal: FUNCTIONAL REPRODUCIBILITY                                       │
│  ───────────────────────────────────────                                        │
│                                                                                 │
│  ✅ Same AUROC (within bootstrap CI)                                            │
│  ✅ Same calibration metrics                                                    │
│  ✅ Same scientific conclusions                                                 │
│  ⚠️ Figures may differ in metadata (timestamps, order)                          │
│  ⚠️ Model weights may differ slightly (GPU non-determinism)                     │
│                                                                                 │
│  Where we DO achieve bitwise:                                                   │
│  • uv.lock → identical package installations                                    │
│  • JSON sidecar data → exact metric values                                      │
│  • DuckDB queries → deterministic results                                       │
│                                                                                 │
│  We verify: JSON metrics match, scientific conclusions match                    │
│  We accept: PNG metadata may differ, training order may vary                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Two definitions**: Side-by-side boxes with formulas
2. **Non-determinism sources**: Table of what breaks bitwise
3. **When each matters**: Two columns with use cases
4. **Foundation PLR target**: What we aim for and why

## Text Content

### Title Text
"Bitwise vs Functional Reproducibility: Know What You Need"

### Caption
Bitwise reproducibility (SHA-256 matches exactly) requires controlling floating-point order, thread scheduling, timestamps, and more—achievable by <1% of projects. Functional reproducibility (same scientific conclusions) is often sufficient and achievable with good practices. Foundation PLR targets functional reproducibility for ML results while achieving bitwise for dependency installs and metric data.

## Prompts for Nano Banana Pro

### Style Prompt
Two definition boxes with code/formula style. Table of non-determinism sources. Two-column comparison of when each matters. Target section with checkmarks and warnings. Clean, technical style.

### Content Prompt
Create "Bitwise vs Functional" infographic:

**TOP - Definitions**:
- Left box: SHA-256 equality, "every byte identical"
- Right box: conclusions equality, "scientifically equivalent"

**MIDDLE - Non-determinism Table**:
- 6 sources with examples and impacts

**BOTTOM - Foundation PLR**:
- Target: functional reproducibility
- Checklist of what we achieve (metrics, conclusions)
- Warnings for acceptable variation (PNG metadata, GPU weights)

## Alt Text

Bitwise vs functional reproducibility comparison. Bitwise: SHA-256 output matches exactly (e.g., identical PNG files). Functional: scientific conclusions match (e.g., AUROC 0.9110 vs 0.9111 within CI). Non-determinism sources: floating-point order, thread scheduling, timestamps, memory addresses, GPU operations, dictionary ordering. Bitwise needed for cryptography/audit/legal; functional sufficient for science/ML/statistics. Foundation PLR targets functional reproducibility (same AUROC, calibration, conclusions) while achieving bitwise for lockfiles and JSON metrics.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

