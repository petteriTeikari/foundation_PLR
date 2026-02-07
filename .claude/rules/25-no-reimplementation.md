# RULE: No Reimplementation of Verified Code

**Use existing verified implementations via interop. NEVER reimplement statistical methods.**

## Verified Implementations

| Method | Source | Interop |
|--------|--------|---------|
| pminternal (model stability) | R package (Rhodes 2025) | rpy2 or subprocess |
| relplot/SmoothECE (calibration) | Python (Apple) | Direct import |
| dcurves (DCA) | R package | rpy2 or subprocess |
| pmcalibration | R package | rpy2 or subprocess |

## Correct Pattern

```python
# Wrap R packages - ALLOWED
import subprocess
subprocess.run(['Rscript', 'analysis.R', 'input.csv', 'output.json'])

# Or use rpy2 - ALLOWED
from rpy2.robjects.packages import importr
pminternal = importr('pminternal')
```

## BANNED

Creating `src/stats/X_reimplementation.py` is FORBIDDEN.
Creating `src/stats/X_wrapper.py` that calls verified libraries is ALLOWED.

## Why

1. Reimplementing complex statistics introduces subtle bugs
2. Original authors have validated their code
3. Reviewers expect canonical implementations
4. Using standard tools ensures published-method reproducibility
