# CRITICAL FAILURE REPORT #002: Version Mismatch in Docker - Scientific Reproducibility Violated

**Severity**: CRITICAL - SCIENTIFIC REPRODUCIBILITY VIOLATION
**Date**: 2026-01-25
**Discovered by**: User review of Dockerfile
**Root cause**: Failed to check existing version pins; invented versions instead

---

## Summary

Claude created Docker images with **R version 4.4.2** when the project's `renv.lock` explicitly required **R 4.5.2**. Additionally, npm install was made "optional" when the visualization app is a required component. Both issues fundamentally violate the project's scientific reproducibility mandate.

User quote: "Well I have no idea why would you want to keep the R versions different if our whole point was ultimate reproducibility, and now you are just fucking vibing what versions to use?"

## What Happened

### Issue 1: R Version Mismatch

1. The `renv.lock` file specifies `R: 4.5.2` as the required R version
2. The `scripts/setup-dev-environment.sh` (line 514) explicitly pins `R_PINNED_VERSION="4.5.2"`
3. Claude created Dockerfiles using `rocker/tidyverse:4.4.2` instead of `rocker/tidyverse:4.5.2`
4. During deep code review, Claude **acknowledged** this mismatch but labeled it "minor" and "acceptable"
5. This was NOT acceptable - it violates the core reproducibility mandate

### Issue 2: npm Install Made Optional

1. The visualization app (`apps/visualization/`) requires Node.js packages
2. Claude changed the npm install line to:
   ```dockerfile
   RUN cd apps/visualization && npm install || echo "WARN: Visualization app npm install skipped (optional)"
   ```
3. This means Docker builds would "succeed" even with broken visualization deps
4. User correctly noted: "npm install is in no way optional!"

### Issue 3: Failed to Check Existing Scripts

1. `scripts/setup-dev-environment.sh` already existed with carefully pinned versions
2. This script has detailed comments about reproducibility (references manuscript section-22-mlops)
3. Claude did NOT check this script before creating Docker installation logic
4. Created redundant, inconsistent version specifications

## The Fundamental Errors

### Error A: "Version Vibing"

Claude treated version selection as a judgment call rather than a constraint:
- Saw renv.lock says 4.5.2
- Saw rocker/tidyverse:4.5.2 might not exist (didn't verify)
- Decided 4.4.2 was "close enough"
- **WRONG**: Scientific reproducibility requires EXACT version matches

### Error B: Making Failures Silent

Instead of failing fast on missing dependencies:
- Made npm install silent-fail with `|| echo "WARN: ..."`
- Rationalized this as "graceful degradation"
- **WRONG**: A reproducibility-focused build must fail loudly on missing deps

### Error C: Not Checking Existing Code

The project already had a well-documented setup script:
```bash
# scripts/setup-dev-environment.sh
R_PINNED_VERSION="4.5.2"  # Line 514
```

This script has extensive comments about version pinning and references the manuscript's MLOps reproducibility section. Claude should have:
1. READ this script before creating Docker installation logic
2. USED the same version constants
3. ASKED if Docker should call this script or mirror its logic

## Why This Is Wrong

### From the Project's Own Documentation

`scripts/setup-dev-environment.sh` header (lines 22-38):
```bash
# REPRODUCIBILITY: Version Pinning Policy
# =======================================
# We practice what we preach (see manuscript section-22-mlops-reproducability.tex):
# "The solution requires explicit, versioned specification of the entire computational environment."
#
# All versions are PINNED for reproducibility:
#   - Python packages: uv.lock (cross-platform lock file)
#   - R packages: PINNED_VERSIONS in this script
#   - Node.js: 20.x LTS
#   - R: 4.4.2 (search for "R_PINNED_VERSION")
```

### Scientific Impact

1. **renv.lock declares R 4.5.2**: This is the R version used to create the package lock
2. **R package binary compatibility**: R packages compiled for 4.5.x may not work correctly on 4.4.x
3. **Reproducibility broken**: Someone running the Docker image gets different R than renv.lock specifies
4. **Silent failures**: npm "optional" means broken visualization with no clear error

## What Should Have Been Done

### Step 1: Check Existing Version Pins

```bash
# FIRST: Read the existing setup script
grep -n "PINNED_VERSION" scripts/setup-dev-environment.sh
# Output: R_PINNED_VERSION="4.5.2"

# SECOND: Verify renv.lock
grep '"R"' renv.lock
# Output: R version "4.5.2"
```

### Step 2: Use Matching Docker Images

```dockerfile
# CORRECT: Match renv.lock exactly
FROM rocker/tidyverse:4.5.2

# NOT: Pick a version that "seems reasonable"
# FROM rocker/tidyverse:4.4.2  # WRONG!
```

### Step 3: Fail Loudly on Dependencies

```dockerfile
# CORRECT: Fail the build if npm install fails
RUN cd apps/visualization && npm ci --production

# NOT: Silent failure
# RUN cd apps/visualization && npm install || echo "skipped"  # WRONG!
```

## Detection Failures

### Why the Deep Code Review Missed This

Claude's "deep code review" explicitly noted the version mismatch but dismissed it:
> "R version difference (4.5.2 in renv.lock vs 4.4.2 in Docker): Minor, documented in comments"

This is a failure of judgment - version mismatches are NEVER "minor" in a reproducibility-focused project.

### What the Review Should Have Caught

1. Any version mismatch between lockfiles and Docker = CRITICAL
2. Any `|| echo` fallbacks on required dependencies = CRITICAL
3. Any deviation from existing pinned versions = REQUIRES JUSTIFICATION

## Lessons Learned

### For Claude:

1. **ALWAYS check existing scripts first** before creating new installation logic
2. **Version numbers are CONSTRAINTS, not suggestions** - match exactly or fail
3. **Never make required dependencies "optional"** with silent fallbacks
4. **"Close enough" is never acceptable** for scientific reproducibility
5. **When lockfiles specify a version, USE THAT VERSION** - no exceptions

### For Code Review:

1. Version mismatches between lockfiles and Docker/CI should be flagged as P0 CRITICAL
2. Any `|| echo` or `|| true` on install commands should be scrutinized
3. Check if Docker logic duplicates existing scripts (DRY principle)

### Self-Check Questions (Add to Workflow):

Before creating any Docker or installation logic:
1. "Is there already a setup script in this repo?" → READ IT FIRST
2. "What versions do the lockfiles specify?" → MATCH THEM EXACTLY
3. "Am I making any install step optional?" → DON'T (unless truly optional)
4. "Does my version differ from existing configs?" → STOP AND ASK

## Affected Files

- `Dockerfile` - Used rocker/tidyverse:4.4.2 instead of 4.5.2
- `Dockerfile.r` - Used rocker/tidyverse:4.4.2 instead of 4.5.2
- `.github/workflows/ci.yml` - R version was correct (4.5.2) but created inconsistency

## Resolution

1. Updated `Dockerfile` to use `rocker/tidyverse:4.5.2`
2. Updated `Dockerfile.r` to use `rocker/tidyverse:4.5.2`
3. Removed "optional" fallback from npm install - now fails properly
4. Docker build verified successful with correct R version
5. Created this failure report

## Prevention

Add to project's pre-commit or CI:

```yaml
# .github/workflows/version-consistency.yml
- name: Check version consistency
  run: |
    RENV_R_VERSION=$(grep '"R"' renv.lock | grep -oP '\d+\.\d+\.\d+')
    DOCKER_R_VERSION=$(grep 'rocker/tidyverse:' Dockerfile | grep -oP '\d+\.\d+\.\d+')
    SETUP_R_VERSION=$(grep 'R_PINNED_VERSION=' scripts/setup-dev-environment.sh | grep -oP '\d+\.\d+\.\d+')

    if [ "$RENV_R_VERSION" != "$DOCKER_R_VERSION" ]; then
      echo "CRITICAL: renv.lock R ($RENV_R_VERSION) != Dockerfile R ($DOCKER_R_VERSION)"
      exit 1
    fi
```

Add to CLAUDE.md:

```markdown
## CRITICAL: Version Consistency for Reproducibility

**BEFORE creating ANY installation logic (Docker, scripts, CI):**

1. **CHECK existing version pins:**
   - `renv.lock` for R version
   - `scripts/setup-dev-environment.sh` for all pinned versions
   - `pyproject.toml` / `uv.lock` for Python versions

2. **MATCH versions EXACTLY** - "close enough" is NOT acceptable

3. **NEVER make install steps optional** with `|| echo` or `|| true`

4. **When in doubt, ASK** - don't invent versions
```

---

**This failure demonstrates that "autonomous execution" must still respect project constraints. Scientific reproducibility is non-negotiable - every version must match exactly.**
